"""Foot-polygon terrain statistics for Planner V2 foothold selection."""

from __future__ import annotations

import torch

from .gridmap_utils import gather_grid_by_row_col, world_xy_to_grid_float

__all__ = ["foot_patch_stats"]


def _foot_sample_offsets_body(
    foot_length: float,
    foot_width: float,
    n_long: int,
    n_lat: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n_long <= 0 or n_lat <= 0:
        raise ValueError("n_long and n_lat must be positive.")
    xs = torch.linspace(-foot_length * 0.5, foot_length * 0.5, n_long, device=device, dtype=dtype)
    ys = torch.linspace(-foot_width * 0.5, foot_width * 0.5, n_lat, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def _sample_xy_world(
    centers_w: torch.Tensor,
    yaws: torch.Tensor,
    offsets_body: torch.Tensor,
) -> torch.Tensor:
    """Return ``(B, M, S, 2)`` footprint sample xy in world frame."""
    c = torch.cos(yaws)
    s = torch.sin(yaws)
    gx = offsets_body[..., 0]
    gy = offsets_body[..., 1]
    x_w = c.unsqueeze(-1) * gx - s.unsqueeze(-1) * gy
    y_w = s.unsqueeze(-1) * gx + c.unsqueeze(-1) * gy
    return torch.stack([x_w, y_w], dim=-1) + centers_w[..., :2].unsqueeze(-2)


def _lookup_terrain_z_grid(
    sample_xy_w: torch.Tensor,
    ray_hits_w: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    grid_resolution: float,
    grid_center_w: torch.Tensor,
    grid_yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_float = world_xy_to_grid_float(
        sample_xy_w,
        grid_center_w,
        grid_yaw,
        grid_shape=grid_shape,
        grid_resolution=grid_resolution,
    )
    row = torch.round(grid_float[..., 0]).long()
    col = torch.round(grid_float[..., 1]).long()
    terrain_xyz, in_bounds = gather_grid_by_row_col(
        ray_hits_w,
        row,
        col,
        grid_shape=grid_shape,
    )
    return terrain_xyz[..., 2], in_bounds


def _lookup_terrain_z_brute(sample_xy_w: torch.Tensor, ray_hits_w: torch.Tensor) -> torch.Tensor:
    diff = sample_xy_w[..., None, :] - ray_hits_w[:, None, None, :, :2]
    d2 = (diff * diff).sum(dim=-1)
    nearest = d2.argmin(dim=-1)
    batch = sample_xy_w.shape[0]
    b_idx = torch.arange(batch, device=sample_xy_w.device).view(batch, 1, 1).expand_as(nearest)
    return ray_hits_w[b_idx, nearest, 2]


def _fit_plane_normals(
    sample_xy_w: torch.Tensor,
    terrain_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = sample_xy_w[..., 0]
    y = sample_xy_w[..., 1]
    z = terrain_z
    ones = torch.ones_like(x)

    sum_x = x.sum(dim=-1)
    sum_y = y.sum(dim=-1)
    sum_z = z.sum(dim=-1)
    sum_x2 = (x * x).sum(dim=-1)
    sum_y2 = (y * y).sum(dim=-1)
    sum_xy = (x * y).sum(dim=-1)
    sum_xz = (x * z).sum(dim=-1)
    sum_yz = (y * z).sum(dim=-1)

    lhs = torch.stack(
        [
            torch.stack([sum_x2, sum_xy, sum_x], dim=-1),
            torch.stack([sum_xy, sum_y2, sum_y], dim=-1),
            torch.stack([sum_x, sum_y, ones.sum(dim=-1)], dim=-1),
        ],
        dim=-2,
    )
    rhs = torch.stack([sum_xz, sum_yz, sum_z], dim=-1)

    det = torch.linalg.det(lhs)
    singular = det.abs() < 1e-8
    coeff = torch.zeros(*lhs.shape[:-2], 3, device=lhs.device, dtype=lhs.dtype)
    if (~singular).any():
        coeff[~singular] = torch.linalg.solve(lhs[~singular], rhs[~singular].unsqueeze(-1)).squeeze(-1)

    a, b, _c = coeff[..., 0], coeff[..., 1], coeff[..., 2]

    normal = torch.stack([-a, -b, torch.ones_like(a)], dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    slope_angle = torch.acos(normal[..., 2].clamp(-1.0, 1.0))

    ez = torch.tensor([0.0, 0.0, 1.0], device=normal.device, dtype=normal.dtype)
    normal = torch.where(singular.unsqueeze(-1), ez, normal)
    slope_angle = torch.where(singular, torch.zeros_like(slope_angle), slope_angle)
    return normal, slope_angle


def foot_patch_stats(
    centers_w: torch.Tensor,
    yaws: torch.Tensor,
    ray_hits_w: torch.Tensor,
    *,
    foot_length: float,
    foot_width: float,
    n_long: int,
    n_lat: int,
    support_threshold: float,
    grid_shape: tuple[int, int] | None = None,
    grid_resolution: float | None = None,
    grid_center_w: torch.Tensor | None = None,
    grid_yaw: torch.Tensor | None = None,
    return_debug: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute terrain statistics for rectangular foot patches around candidate centers."""
    if centers_w.ndim != 3 or centers_w.shape[-1] != 3:
        raise ValueError(f"centers_w must be (B, M, 3), got {centers_w.shape}.")
    if yaws.shape != centers_w.shape[:2]:
        raise ValueError(f"yaws must be (B, M), got {yaws.shape}.")

    batch, num_candidates, _ = centers_w.shape
    if ray_hits_w.shape[0] != batch:
        raise ValueError("ray_hits_w batch must match centers_w.")

    grid_args = (grid_shape, grid_resolution, grid_center_w, grid_yaw)
    has_grid = all(arg is not None for arg in grid_args)
    if num_candidates > 2 and not has_grid:
        raise ValueError("grid_shape, grid_resolution, grid_center_w, and grid_yaw are required when M > 2.")

    offsets_body = _foot_sample_offsets_body(
        foot_length,
        foot_width,
        n_long,
        n_lat,
        device=centers_w.device,
        dtype=centers_w.dtype,
    )
    sample_xy_w = _sample_xy_world(centers_w, yaws, offsets_body)

    if has_grid:
        terrain_z, sample_in_bounds = _lookup_terrain_z_grid(
            sample_xy_w,
            ray_hits_w,
            grid_shape=grid_shape,
            grid_resolution=grid_resolution,
            grid_center_w=grid_center_w,
            grid_yaw=grid_yaw,
        )
    else:
        terrain_z = _lookup_terrain_z_brute(sample_xy_w, ray_hits_w)
        sample_in_bounds = torch.ones(
            batch, num_candidates, offsets_body.shape[0], device=centers_w.device, dtype=torch.bool
        )

    z_median = terrain_z.median(dim=-1).values
    z_q70 = torch.quantile(terrain_z, 0.70, dim=-1)
    z_max = terrain_z.max(dim=-1).values
    z_min = terrain_z.min(dim=-1).values
    dz_omega = z_max - z_min

    supported = torch.abs(terrain_z - z_median.unsqueeze(-1)) <= support_threshold
    support_ratio = supported.float().mean(dim=-1)
    overhang_ratio = (terrain_z < (z_median.unsqueeze(-1) - support_threshold)).float().mean(dim=-1)
    footprint_in_bounds_ratio = sample_in_bounds.float().mean(dim=-1)

    slope_normal, slope_angle = _fit_plane_normals(sample_xy_w, terrain_z)

    out: dict[str, torch.Tensor] = {
        "z_median": z_median,
        "z_q70": z_q70,
        "z_max": z_max,
        "z_min": z_min,
        "dz_omega": dz_omega,
        "support_ratio": support_ratio,
        "overhang_ratio": overhang_ratio,
        "sample_in_bounds": sample_in_bounds,
        "footprint_in_bounds_ratio": footprint_in_bounds_ratio,
        "slope_normal": slope_normal,
        "slope_angle": slope_angle,
    }

    if return_debug:
        grid_float = world_xy_to_grid_float(
            sample_xy_w,
            grid_center_w if grid_center_w is not None else torch.zeros(batch, 2, device=centers_w.device),
            grid_yaw if grid_yaw is not None else torch.zeros(batch, device=centers_w.device),
            grid_shape=grid_shape if grid_shape is not None else (21, 33),
            grid_resolution=grid_resolution if grid_resolution is not None else 0.05,
        )
        out["sample_xy_w"] = sample_xy_w
        out["sample_grid_row_col"] = torch.round(grid_float).long()
    return out
