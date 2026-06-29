"""Pure-PyTorch cost-based foothold selection over a local elevation patch.

Designed so the same code runs in sim (ray-cast hits) and on real hardware
(grid_map_builder elevation cells). Isaac-Sim-free for unit-test friendliness.

Cost composition (per candidate cell c):
    cost(c) = alpha * roughness(c)            # max-min over neighborhood
            + beta  * slope(c)                # |grad z| approximated by 1-ring
            + gamma * obstacle_flag(c)        # 1 if roughness > threshold
            + delta * ||xy_c - xy_raibert||^2 # stay-close-to-prior

Selection: argmin over candidate cells within ``window_m`` of the Raibert center.
"""
from __future__ import annotations

import torch

__all__ = ["select_foothold_by_cost"]


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the lowest-cost foothold in a square window around ``raibert_xy_w``.

    Args:
        raibert_xy_w: ``(B, F, 3)`` Raibert prior; only ``xy`` is read.
        ray_hits_w: ``(B, K, 3)`` candidate cells (e.g., RayCaster hits or
            grid_map cells in world frame).
        window_m: half-side of the candidate window, metres.
        alpha, beta, gamma, delta: cost weights (see module docstring).
        obstacle_height_threshold: roughness above this triggers ``gamma``.

    Returns:
        selected_xyz_w: ``(B, F, 3)`` chosen foothold (xy + grid z).
        debug_cost: ``(B, F)`` cost at the selected cell, for logging.
    """
    B, F, _ = raibert_xy_w.shape
    _, K, _ = ray_hits_w.shape
    device = raibert_xy_w.device

    # 1) Distance from each ray to each Raibert center (xy only). (B, F, K)
    diff = raibert_xy_w[:, :, None, :2] - ray_hits_w[:, None, :, :2]
    d2 = (diff * diff).sum(dim=-1)
    in_window = d2 <= (window_m * window_m)

    # 2) Per-cell roughness via K-nearest neighbours of each cell itself.
    # We approximate "neighborhood" by the same window radius: for each
    # candidate cell, look at all OTHER cells within window_m of IT.
    # (B, K, K) is feasible because K ≤ ~700 for the existing height scanner.
    cell_diff = ray_hits_w[:, :, None, :2] - ray_hits_w[:, None, :, :2]
    cell_d2 = (cell_diff * cell_diff).sum(dim=-1)
    cell_nbr = cell_d2 <= (window_m * window_m)
    z_all = ray_hits_w[..., 2]  # (B, K)
    # Mask non-neighbours with +/- inf so they don't affect max/min.
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

    # 4) Pick argmin per (env, foot).
    best_idx = cost.argmin(dim=-1)  # (B, F)
    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, F)
    selected = ray_hits_w[b_idx, best_idx]  # (B, F, 3)
    debug_cost = cost.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    return selected, debug_cost
