"""Pure-PyTorch foothold selection helpers (Isaac-Sim-free).

This module serves two roles:

**Legacy selector —** :func:`select_foothold_by_cost`
    Lightweight circular-window cost minimizer over ray-hit cells. Uses
    per-cell max-min roughness (not foot-polygon geometry). Returns **terrain
    z** from the chosen ray hit. Kept bit-stable for existing DTC/DTCLite envs
    and as the matched A/B control (``PlannerV2-Legacy``). Do **not** extend
    this into Planner V2.

**Planner V2 building blocks —** :func:`reachability_mask`,
    :func:`step_height_mask`, :func:`roughness_cap_mask`
    Hard-filter primitives composed by :func:`select_foothold_v2` (Task 4),
    which uses ``generate_candidates`` + ``foot_patch_stats`` + soft costs.

Legacy cost (per candidate cell c within radius ``window_m``):

    cost(c) = alpha * roughness(c) + beta * slope(c)
            + gamma * obstacle_flag(c) + delta * ||xy_c - xy_raibert||²

``roughness`` is max-min z over neighbours with ``||xy|| <= window_m``
(circular neighbourhood, not a square axis-aligned window).

Fast grid path
--------------
When ``grid_shape`` / ``grid_resolution`` are supplied, roughness uses
``max_pool2d`` over a replicate-padded height grid (avoids zero-padding bias
at terrain edges). See :func:`_local_roughness_grid`.

Memory at B=4096, K=693: grid path ~11 MB vs brute-force ~5.9 GB.

Ray ordering: IsaacLab ``grid_pattern`` ``ordering="xy"`` → row-major
``(H, W)`` flatten; ``size=[1.6, 1.0]``, ``resolution=0.05`` → H=21, W=33.
"""
from __future__ import annotations

import torch
import torch.nn.functional as _F

from .foot_geometry_constants import (
    G1_FOOT_LENGTH,
    G1_FOOT_N_LAT,
    G1_FOOT_N_LONG,
    G1_FOOT_WIDTH,
    G1_SOLE_Z_OFFSET,
)
from .planner import _yaw_rotation

__all__ = [
    "foothold_quality_at_centers",
    "reachability_mask",
    "roughness_cap_mask",
    "score_foothold_candidates_v2",
    "select_foothold_by_cost",
    "select_foothold_v2",
    "step_height_mask",
]


def _coerce_per_foot_body_frame(
    body_pos_w: torch.Tensor,
    body_yaw: torch.Tensor,
    num_feet: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast shared ``(B, 3)`` / ``(B,)`` pose to per-foot ``(B, F, …)`` if needed."""
    batch = body_pos_w.shape[0]
    if body_pos_w.dim() == 2:
        body_pos_w = body_pos_w.unsqueeze(1).expand(batch, num_feet, 3)
    elif body_pos_w.shape[1] != num_feet:
        raise ValueError(f"body_pos_w must be (B, 3) or (B, F, 3) with F={num_feet}, got {tuple(body_pos_w.shape)}")
    if body_yaw.dim() == 1:
        body_yaw = body_yaw.unsqueeze(1).expand(batch, num_feet)
    elif body_yaw.shape[1] != num_feet:
        raise ValueError(f"body_yaw must be (B,) or (B, F) with F={num_feet}, got {tuple(body_yaw.shape)}")
    return body_pos_w, body_yaw


def _world_to_body(
    pos_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_yaw: torch.Tensor,
) -> torch.Tensor:
    """Rotate world-frame offsets into the body frame.

    ``body_pos_w`` / ``body_yaw`` may be shared across feet ``(B, 3)`` / ``(B,)`` or
    per-foot landing pose ``(B, F, 3)`` / ``(B, F)`` — the latter is required when
    Raibert priors use per-foot landing horizons (stance foot at ``t_step``).
    """
    if pos_w.dim() == 4:
        batch, num_feet, _, _ = pos_w.shape
        body_pos_w, body_yaw = _coerce_per_foot_body_frame(body_pos_w, body_yaw, num_feet)
        rel_w = pos_w - body_pos_w.unsqueeze(2)
        rot_wb = _yaw_rotation(-body_yaw.reshape(batch * num_feet)).reshape(batch, num_feet, 3, 3)
        return torch.einsum("bfij,bfkj->bfki", rot_wb, rel_w)
    if pos_w.dim() == 3:
        batch = body_pos_w.shape[0]
        rel_w = pos_w - body_pos_w.view(batch, 1, 3)
        rot_wb = _yaw_rotation(-body_yaw.reshape(-1)[:batch])
        return torch.einsum("bij,bkj->bki", rot_wb, rel_w)
    raise ValueError(f"_world_to_body expects 3D or 4D pos_w, got shape {tuple(pos_w.shape)}")


def _in_range(values: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return (values >= lo) & (values <= hi)


def reachability_mask(
    candidates_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_yaw: torch.Tensor,
    *,
    x_range: tuple[float, float] = (-0.12, 0.35),
    y_range_left: tuple[float, float] = (0.06, 0.28),
    y_range_right: tuple[float, float] = (-0.28, -0.06),
) -> torch.Tensor:
    """Body-frame xy reachability per foot, including cross-leg y separation.

    ``candidates_w`` stores foot-body targets in world frame. Vertical feasibility
    is handled separately by :func:`step_height_mask` (terrain Δz vs stance foot),
    not here — so climbing/descending stairs does not false-reject on absolute
    body-frame foot z.

    Left foot (F=0) must fall in ``y_range_left``; right foot (F=1) in
    ``y_range_right``. Forward-only DTC_FORWARD uses these defaults; omni-range
    turning may need wider y gaps (see planner upgrade plan Task 9 note).
    """
    pos_b = _world_to_body(candidates_w, body_pos_w, body_yaw)
    valid_x = _in_range(pos_b[..., 0], *x_range)
    valid_y_left = _in_range(pos_b[:, 0, :, 1], *y_range_left)
    valid_y_right = _in_range(pos_b[:, 1, :, 1], *y_range_right)
    valid_y = torch.stack([valid_y_left, valid_y_right], dim=1)
    return valid_x & valid_y


def step_height_mask(
    candidates_w: torch.Tensor,
    stance_terrain_z_w: torch.Tensor,
    *,
    max_dz: float = 0.20,
    sole_z_offset: float = G1_SOLE_Z_OFFSET,
) -> torch.Tensor:
    """Reject candidates whose terrain height differs too much from the stance foot.

    Operates on terrain/sole z (see planner contract), not foot-body link z.
    """
    candidate_terrain_z = candidates_w[..., 2] + sole_z_offset
    dz = (candidate_terrain_z - stance_terrain_z_w.unsqueeze(-1)).abs()
    return dz <= max_dz


def roughness_cap_mask(
    dz_omega: torch.Tensor,
    *,
    max_dz_omega: float = 0.10,
) -> torch.Tensor:
    """Reject candidates whose foot-patch roughness exceeds the cap."""
    return dz_omega <= max_dz_omega


def _local_roughness_grid(z_grid: torch.Tensor, radius_cells: int) -> torch.Tensor:
    """Per-cell max-min z over a circular neighbourhood (grid fast path).

    Uses edge **replicate** padding so boundary cells are not biased toward 0.
    """
    ksize = 2 * radius_cells + 1
    pad = radius_cells
    z4 = z_grid.unsqueeze(1)  # (B, 1, H, W)
    z4 = _F.pad(z4, (pad, pad, pad, pad), mode="replicate")
    z_max = _F.max_pool2d(z4, kernel_size=ksize, stride=1, padding=0).squeeze(1)
    z_min = -_F.max_pool2d(-z4, kernel_size=ksize, stride=1, padding=0).squeeze(1)
    return z_max - z_min


def select_foothold_by_cost(
    raibert_xy_w: torch.Tensor,
    ray_hits_w: torch.Tensor,
    *,
    window_m: float = 0.10,
    alpha: float = 2.0,
    beta: float = 1.0,
    gamma: float = 5.0,
    delta: float = 1.0,
    obstacle_height_threshold: float = 0.15,
    grid_shape: tuple[int, int] | None = None,
    grid_resolution: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy selector: lowest-cost ray hit within a **circular** xy window.

    Returns ``(selected_xyz_w, debug_cost)`` where ``selected_xyz_w[..., 2]`` is
    **terrain z** from the ray hit (not foot-body z). Planner V2 uses
    :func:`select_foothold_v2` instead, which applies ``terrain_z - sole_z_offset``.

    When no ray falls inside the window, falls back to the globally nearest ray
    in xy (never silent index-0 on tied ``1e6`` costs).
    """
    B, F, _ = raibert_xy_w.shape
    _, K, _ = ray_hits_w.shape
    device = raibert_xy_w.device

    # 1) Distance from each ray to each Raibert center (xy only). (B, F, K)
    diff = raibert_xy_w[:, :, None, :2] - ray_hits_w[:, None, :, :2]
    d2 = (diff * diff).sum(dim=-1)
    in_window = d2 <= (window_m * window_m)

    # 2) Per-cell roughness -- choose fast grid path or legacy brute-force path.
    if grid_shape is not None and grid_resolution is not None:
        # ---- Fast path: O(B*H*W) via max_pool2d --------------------------------
        H, W = grid_shape
        assert K == H * W, (
            f"select_foothold_by_cost: grid_shape={grid_shape} implies K={H * W} "
            f"but ray_hits_w has K={K}. Check grid_shape=(H,W) matches scanner."
        )
        # Reshape flat K into spatial grid (B, H, W).
        # Ray ordering: outer loop y (rows=H), inner loop x (cols=W) -- matches
        # IsaacLab grid_pattern with default ordering="xy".
        z_grid = ray_hits_w[:, :, 2].reshape(B, H, W)

        radius_cells = max(1, int(round(window_m / grid_resolution)))
        roughness = _local_roughness_grid(z_grid, radius_cells).reshape(B, K)
    else:
        # ---- Legacy brute-force path: O(B*K^2) --------------------------------
        # Kept for backward-compat when grid kwargs are absent.
        cell_diff = ray_hits_w[:, :, None, :2] - ray_hits_w[:, None, :, :2]
        cell_d2 = (cell_diff * cell_diff).sum(dim=-1)
        cell_nbr = cell_d2 <= (window_m * window_m)
        z_all = ray_hits_w[..., 2]  # (B, K)
        z_neg = torch.where(cell_nbr, z_all.unsqueeze(1).expand(B, K, K),
                            torch.full_like(z_all.unsqueeze(1).expand(B, K, K), -float("inf")))
        z_pos = torch.where(cell_nbr, z_all.unsqueeze(1).expand(B, K, K),
                            torch.full_like(z_all.unsqueeze(1).expand(B, K, K), float("inf")))
        z_max = z_neg.max(dim=-1).values  # (B, K)
        z_min = z_pos.min(dim=-1).values  # (B, K)
        roughness = z_max - z_min  # (B, K)

    slope = roughness  # 1-ring approximation: same as roughness here.
    obstacle_flag = (roughness > obstacle_height_threshold).float()

    # 3) Cost per (env, foot, cell). Broadcast scalar terms over F.
    cost_terrain = alpha * roughness + beta * slope + gamma * obstacle_flag  # (B, K)
    cost = cost_terrain.unsqueeze(1) + delta * d2  # (B, F, K)
    # Cells outside the window: large penalty.
    cost = torch.where(in_window, cost, torch.full_like(cost, 1e6))

    # 4) Pick argmin per (env, foot); empty window → nearest ray in xy.
    has_candidate = in_window.any(dim=-1)
    best_idx = cost.argmin(dim=-1)
    nearest_idx = d2.argmin(dim=-1)
    best_idx = torch.where(has_candidate, best_idx, nearest_idx)
    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, F)
    selected = ray_hits_w[b_idx, best_idx]  # (B, F, 3)
    debug_cost = cost.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    return selected, debug_cost


_SLOPE_SCORE_SIGMA = 0.2094


def foothold_quality_at_centers(
    centers_foot_body_w: torch.Tensor,
    foot_yaws: torch.Tensor,
    ray_hits_w: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    grid_resolution: float,
    grid_center_w: torch.Tensor,
    grid_yaw: torch.Tensor,
    foot_length: float = G1_FOOT_LENGTH,
    foot_width: float = G1_FOOT_WIDTH,
    n_long: int = G1_FOOT_N_LONG,
    n_lat: int = G1_FOOT_N_LAT,
    support_threshold: float = 0.03,
) -> torch.Tensor:
    """Task 0.5 quality score for committed foot-body centers: ``support × exp(-0.5(slope/σ)²)``."""
    from . import foothold_geometry

    stats = foothold_geometry.foot_patch_stats(
        centers_foot_body_w,
        foot_yaws,
        ray_hits_w,
        foot_length=foot_length,
        foot_width=foot_width,
        n_long=n_long,
        n_lat=n_lat,
        support_threshold=support_threshold,
        grid_shape=grid_shape,
        grid_resolution=grid_resolution,
        grid_center_w=grid_center_w,
        grid_yaw=grid_yaw,
    )
    raw = stats["support_ratio"] * torch.exp(-0.5 * (stats["slope_angle"] / _SLOPE_SCORE_SIGMA) ** 2)
    return raw.clamp(0.0, 1.0)


def _nearest_ray_xyz(xy_w: torch.Tensor, ray_hits_w: torch.Tensor) -> torch.Tensor:
    """Snap ``xy_w`` ``(B, F, 2)`` to nearest scanner hit; return ``(B, F, 3)`` terrain xyz."""
    diff = xy_w.unsqueeze(-2) - ray_hits_w[:, None, :, :2]
    d2 = (diff * diff).sum(dim=-1)
    hit_idx = d2.argmin(dim=-1)
    b_idx = torch.arange(ray_hits_w.shape[0], device=ray_hits_w.device).view(-1, 1).expand_as(hit_idx)
    return ray_hits_w[b_idx, hit_idx]


def _under_hip_fallback_w(
    body_pos_w: torch.Tensor,
    body_yaw: torch.Tensor,
    ray_hits_w: torch.Tensor,
    *,
    fallback_hip_y: float,
    fallback_leg_length: float,
    sole_z_offset: float,
) -> torch.Tensor:
    """Conservative under-hip foot-body target snapped to the elevation grid."""
    device, dtype = body_pos_w.device, body_pos_w.dtype
    num_feet = 2
    body_pos_w, body_yaw = _coerce_per_foot_body_frame(body_pos_w, body_yaw, num_feet)
    batch = body_pos_w.shape[0]
    offs_b = torch.tensor(
        [[0.0, fallback_hip_y, -fallback_leg_length], [0.0, -fallback_hip_y, -fallback_leg_length]],
        device=device,
        dtype=dtype,
    )
    rot = _yaw_rotation(body_yaw.reshape(batch * num_feet)).reshape(batch, num_feet, 3, 3)
    world_off = torch.einsum("bfij,fj->bfi", rot, offs_b)
    hip_w = body_pos_w + world_off
    snapped = _nearest_ray_xyz(hip_w[..., :2], ray_hits_w)
    out = snapped.clone()
    out[..., 2] = snapped[..., 2] - sole_z_offset
    return out


def score_foothold_candidates_v2(
    nominal_xy_w: torch.Tensor,
    candidate_terrain_xyz_w: torch.Tensor,
    candidate_mask: torch.Tensor,
    ray_hits_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_yaw: torch.Tensor,
    target_yaw: torch.Tensor,
    stance_terrain_z_w: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    grid_resolution: float,
    grid_center_w: torch.Tensor,
    grid_yaw: torch.Tensor,
    foot_length: float = 0.18,
    foot_width: float = 0.065,
    n_long: int = 4,
    n_lat: int = 3,
    support_threshold: float = 0.03,
    sole_z_offset: float = G1_SOLE_Z_OFFSET,
    reach_x_range: tuple[float, float] = (-0.12, 0.35),
    reach_y_range_left: tuple[float, float] = (0.06, 0.28),
    reach_y_range_right: tuple[float, float] = (-0.28, -0.06),
    max_step_dz: float = 0.20,
    max_dz_omega: float = 0.10,
    w_terrain: float = 1.0,
    w_nominal: float = 0.5,
    w_reach: float = 1.0,
    w_height: float = 0.5,
    w_edge: float = 1.0,
    w_slope: float = 0.3,
    lambda_support: float = 1.0,
    lambda_overhang: float = 2.0,
    sigma_nominal: float = 0.10,
    sigma_rough: float = 0.05,
    sigma_h: float = 0.10,
    sigma_theta_rad: float = _SLOPE_SCORE_SIGMA,
    fallback_hip_y: float = 0.12,
    fallback_leg_length: float = 0.78,
    enable_fallback: bool = True,
    return_mask_debug: bool = False,
) -> dict[str, torch.Tensor]:
    """Score pre-generated Planner V2 terrain candidates and select one per foot.

    ``candidate_terrain_xyz_w`` stores scanner terrain xyz. The returned
    ``selected_xyz_w`` stores foot-body targets, using the same sole-z convention
    as :func:`select_foothold_v2`.
    """
    if grid_center_w is None or grid_yaw is None:
        raise ValueError("score_foothold_candidates_v2 requires grid_center_w and grid_yaw.")
    if nominal_xy_w.shape[-1] not in (2, 3):
        raise ValueError(
            f"nominal_xy_w last dim must be 2 (xy) or 3 (xyz prior); got shape {tuple(nominal_xy_w.shape)}."
        )
    if candidate_terrain_xyz_w.ndim != 4 or candidate_terrain_xyz_w.shape[-1] != 3:
        raise ValueError(
            "candidate_terrain_xyz_w must be (B, F, K, 3), "
            f"got shape {tuple(candidate_terrain_xyz_w.shape)}."
        )
    if candidate_mask.shape != candidate_terrain_xyz_w.shape[:3]:
        raise ValueError(
            f"candidate_mask must be {tuple(candidate_terrain_xyz_w.shape[:3])}, "
            f"got {tuple(candidate_mask.shape)}."
        )

    from . import foothold_costs, foothold_geometry

    nominal_xy = nominal_xy_w[..., :2]
    b, num_feet, _ = nominal_xy.shape
    if candidate_terrain_xyz_w.shape[:2] != (b, num_feet):
        raise ValueError("nominal_xy_w and candidate_terrain_xyz_w must share (B, F).")
    if ray_hits_w.shape[0] != b:
        raise ValueError("ray_hits_w batch must match nominal_xy_w.")
    if target_yaw.shape != (b, num_feet):
        raise ValueError(f"target_yaw must be (B, F), got {target_yaw.shape}.")

    k_c = candidate_terrain_xyz_w.shape[2]
    if k_c <= 0:
        raise ValueError("candidate_terrain_xyz_w must contain at least one candidate.")

    valid_candidates = candidate_mask.bool()
    candidates_fb = candidate_terrain_xyz_w.clone()
    candidates_fb[..., 2] = candidate_terrain_xyz_w[..., 2] - sole_z_offset

    flat_centers = candidates_fb.reshape(b, num_feet * k_c, 3)
    flat_yaws = target_yaw.unsqueeze(-1).expand(b, num_feet, k_c).reshape(b, num_feet * k_c)
    stats = foothold_geometry.foot_patch_stats(
        flat_centers,
        flat_yaws,
        ray_hits_w,
        foot_length=foot_length,
        foot_width=foot_width,
        n_long=n_long,
        n_lat=n_lat,
        support_threshold=support_threshold,
        grid_shape=grid_shape,
        grid_resolution=grid_resolution,
        grid_center_w=grid_center_w,
        grid_yaw=grid_yaw,
    )

    def _reshape(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(b, num_feet, k_c)

    dz_omega = _reshape(stats["dz_omega"])
    support_ratio = _reshape(stats["support_ratio"])
    overhang_ratio = _reshape(stats["overhang_ratio"])
    slope_angle = _reshape(stats["slope_angle"])
    z_q70 = _reshape(stats["z_q70"])

    reach_valid = reachability_mask(
        candidates_fb,
        body_pos_w,
        body_yaw,
        x_range=reach_x_range,
        y_range_left=reach_y_range_left,
        y_range_right=reach_y_range_right,
    )
    height_valid = step_height_mask(
        candidates_fb, stance_terrain_z_w, max_dz=max_step_dz, sole_z_offset=sole_z_offset
    )
    rough_valid = roughness_cap_mask(dz_omega, max_dz_omega=max_dz_omega)
    valid = reach_valid & height_valid & rough_valid & valid_candidates
    valid_count = valid.sum(dim=-1).long()
    used_fallback = valid_count == 0

    j_nom = foothold_costs.cost_nominal(candidates_fb, nominal_xy, sigma_nominal=sigma_nominal)
    j_reach = foothold_costs.cost_reach(candidates_fb, body_pos_w, body_yaw)
    j_terrain = foothold_costs.cost_terrain(dz_omega, sigma_rough=sigma_rough)
    j_edge = foothold_costs.cost_edge(
        support_ratio, overhang_ratio, lambda_support=lambda_support, lambda_overhang=lambda_overhang
    )
    j_height = foothold_costs.cost_height(
        candidates_fb, stance_terrain_z_w, sigma_h=sigma_h, sole_z_offset=sole_z_offset
    )
    j_slope = foothold_costs.cost_slope(slope_angle, sigma_theta_rad=sigma_theta_rad)

    total = (
        w_nominal * j_nom
        + w_reach * j_reach
        + w_terrain * j_terrain
        + w_edge * j_edge
        + w_height * j_height
        + w_slope * j_slope
    )
    total = torch.where(valid, total, torch.full_like(total, float("inf")))
    best_k = total.argmin(dim=-1)

    idx_exp = best_k.unsqueeze(-1).unsqueeze(-1).expand(b, num_feet, 1, 3)
    selected_xyz_w = candidates_fb.gather(2, idx_exp).squeeze(2)
    selected_z_q70 = z_q70.gather(2, best_k.unsqueeze(-1)).squeeze(-1)
    selected_xyz_w[..., 2] = selected_z_q70 - sole_z_offset

    raw_quality = support_ratio * torch.exp(-0.5 * (slope_angle / _SLOPE_SCORE_SIGMA) ** 2)
    foothold_score = raw_quality.gather(2, best_k.unsqueeze(-1)).squeeze(-1).clamp(0.0, 1.0)

    per_cost = torch.stack([j_terrain, j_nom, j_reach, j_height, j_edge, j_slope], dim=-1)
    per_cost_debug = per_cost.gather(
        2, best_k.view(b, num_feet, 1, 1).expand(b, num_feet, 1, 6)
    ).squeeze(2)

    if enable_fallback and used_fallback.any():
        fb_xyz = _under_hip_fallback_w(
            body_pos_w,
            body_yaw,
            ray_hits_w,
            fallback_hip_y=fallback_hip_y,
            fallback_leg_length=fallback_leg_length,
            sole_z_offset=sole_z_offset,
        )
        selected_xyz_w = torch.where(used_fallback.unsqueeze(-1), fb_xyz, selected_xyz_w)
        foothold_score = torch.where(used_fallback, torch.zeros_like(foothold_score), foothold_score)
        per_cost_debug = torch.where(
            used_fallback.unsqueeze(-1),
            torch.zeros_like(per_cost_debug),
            per_cost_debug,
        )

    result: dict[str, torch.Tensor] = {
        "selected_xyz_w": selected_xyz_w,
        "foothold_score": foothold_score,
        "valid_count": valid_count,
        "used_fallback": used_fallback,
        "per_cost_debug": per_cost_debug,
    }
    if return_mask_debug:
        result["mask_debug"] = {
            "reach_valid_count": reach_valid.sum(dim=-1),
            "step_height_valid_count": height_valid.sum(dim=-1),
            "roughness_valid_count": rough_valid.sum(dim=-1),
            "in_bounds_valid_count": valid_candidates.sum(dim=-1),
            "combined_valid_count": valid_count,
        }
    return result


def select_foothold_v2(
    raibert_xy_w: torch.Tensor,
    ray_hits_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_yaw: torch.Tensor,
    target_yaw: torch.Tensor,
    stance_terrain_z_w: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    grid_resolution: float,
    grid_center_w: torch.Tensor,
    grid_yaw: torch.Tensor,
    half_width_cells: int = 2,
    foot_length: float = 0.18,
    foot_width: float = 0.065,
    n_long: int = 4,
    n_lat: int = 3,
    support_threshold: float = 0.03,
    sole_z_offset: float = G1_SOLE_Z_OFFSET,
    reach_x_range: tuple[float, float] = (-0.12, 0.35),
    reach_y_range_left: tuple[float, float] = (0.06, 0.28),
    reach_y_range_right: tuple[float, float] = (-0.28, -0.06),
    max_step_dz: float = 0.20,
    max_dz_omega: float = 0.10,
    w_terrain: float = 1.0,
    w_nominal: float = 0.5,
    w_reach: float = 1.0,
    w_height: float = 0.5,
    w_edge: float = 1.0,
    w_slope: float = 0.3,
    lambda_support: float = 1.0,
    lambda_overhang: float = 2.0,
    sigma_nominal: float = 0.10,
    sigma_rough: float = 0.05,
    sigma_h: float = 0.10,
    sigma_theta_rad: float = _SLOPE_SCORE_SIGMA,
    fallback_hip_y: float = 0.12,
    fallback_leg_length: float = 0.78,
    enable_fallback: bool = True,
    return_mask_debug: bool = False,
) -> dict[str, torch.Tensor]:
    """Planner V2 foothold selector: fixed grid candidates + patch stats + hard filters + soft costs.

    ``body_pos_w`` / ``body_yaw`` should be the **landing-time** pelvis pose per foot
    ``(B, F, 3)`` / ``(B, F)`` when horizons differ across feet; using the current
    pelvis with long-horizon Raibert priors falsely rejects stance-foot candidates.
    """
    if grid_center_w is None or grid_yaw is None:
        raise ValueError("select_foothold_v2 requires grid_center_w and grid_yaw (Task 0.1 contract).")
    if raibert_xy_w.shape[-1] not in (2, 3):
        raise ValueError(
            f"raibert_xy_w last dim must be 2 (xy) or 3 (xyz prior); got shape {tuple(raibert_xy_w.shape)}."
        )

    from . import foothold_candidates

    raibert_xy = raibert_xy_w[..., :2]
    candidates_terrain, in_bounds = foothold_candidates.generate_candidates(
        raibert_xy,
        ray_hits_w,
        grid_center_w,
        grid_yaw,
        grid_shape=grid_shape,
        grid_resolution=grid_resolution,
        half_width_cells=half_width_cells,
    )
    return score_foothold_candidates_v2(
        raibert_xy,
        candidates_terrain,
        in_bounds,
        ray_hits_w,
        body_pos_w=body_pos_w,
        body_yaw=body_yaw,
        target_yaw=target_yaw,
        stance_terrain_z_w=stance_terrain_z_w,
        grid_shape=grid_shape,
        grid_resolution=grid_resolution,
        grid_center_w=grid_center_w,
        grid_yaw=grid_yaw,
        foot_length=foot_length,
        foot_width=foot_width,
        n_long=n_long,
        n_lat=n_lat,
        support_threshold=support_threshold,
        sole_z_offset=sole_z_offset,
        reach_x_range=reach_x_range,
        reach_y_range_left=reach_y_range_left,
        reach_y_range_right=reach_y_range_right,
        max_step_dz=max_step_dz,
        max_dz_omega=max_dz_omega,
        w_terrain=w_terrain,
        w_nominal=w_nominal,
        w_reach=w_reach,
        w_height=w_height,
        w_edge=w_edge,
        w_slope=w_slope,
        lambda_support=lambda_support,
        lambda_overhang=lambda_overhang,
        sigma_nominal=sigma_nominal,
        sigma_rough=sigma_rough,
        sigma_h=sigma_h,
        sigma_theta_rad=sigma_theta_rad,
        fallback_hip_y=fallback_hip_y,
        fallback_leg_length=fallback_leg_length,
        enable_fallback=enable_fallback,
        return_mask_debug=return_mask_debug,
    )
