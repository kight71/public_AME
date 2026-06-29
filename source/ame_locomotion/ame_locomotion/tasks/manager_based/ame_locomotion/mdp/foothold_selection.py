"""Pure-PyTorch cost-based foothold selection over a local elevation patch.

Designed so the same code runs in sim (ray-cast hits) and on real hardware
(grid_map_builder elevation cells). Isaac-Sim-free for unit-test friendliness.

Cost composition (per candidate cell c):
    cost(c) = alpha * roughness(c)            # max-min over neighborhood
            + beta  * slope(c)                # |grad z| approximated by 1-ring
            + gamma * obstacle_flag(c)        # 1 if roughness > threshold
            + delta * ||xy_c - xy_raibert||^2 # stay-close-to-prior

Selection: argmin over candidate cells within ``window_m`` of the Raibert center.

Fast path (grid kwargs supplied)
---------------------------------
When ``grid_shape=(H, W)`` and ``grid_resolution`` are provided the function
reshapes the flat K=H*W ray-hit tensor into a 2-D height grid and uses a pair
of ``_F.max_pool2d`` passes (one for z_max, one for z_min via negation) to
compute local roughness in O(B*H*W) memory.

Memory comparison at B=4096, K=693 (H=21, W=33, resolution=0.05 m):
  Brute-force: two (B, K, K, 2) tensors -> ~5.9 GB (OOM at 16 GB)
  Grid path:   two (B, 1, H, W) tensors -> ~11 MB

Ray-ordering note
-----------------
IsaacLab's ``grid_pattern`` with default ``ordering="xy"`` calls
``torch.meshgrid(x, y, indexing="xy")``, which produces a grid of shape
``(len(y), len(x)) = (H, W)``. After ``flatten()`` the scan order is
row-major over ``(H, W)`` -- outer loop over y (rows), inner loop over x
(columns). Therefore ``ray_hits_w[:, :, 2].reshape(B, H, W)`` is correct
when ``grid_shape = (H, W) = (len(y_cells), len(x_cells))``.

For ``size=[1.6, 1.0]``, ``resolution=0.05``:
  x: 33 points -> W=33
  y: 21 points -> H=21
  K = H*W = 693
"""
from __future__ import annotations

import torch
import torch.nn.functional as _F

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
    grid_shape: tuple[int, int] | None = None,
    grid_resolution: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the lowest-cost foothold in a square window around ``raibert_xy_w``.

    Args:
        raibert_xy_w: ``(B, F, 3)`` Raibert prior; only ``xy`` is read.
        ray_hits_w: ``(B, K, 3)`` candidate cells (e.g., RayCaster hits or
            grid_map cells in world frame).
        window_m: half-side of the candidate window, metres.
        alpha, beta, gamma, delta: cost weights (see module docstring).
        obstacle_height_threshold: roughness above this triggers ``gamma``.
        grid_shape: optional ``(H, W)`` tuple. When provided together with
            ``grid_resolution``, enables the fast max_pool2d path.
            K must equal H*W. H = len(y-cells), W = len(x-cells).
        grid_resolution: cell size in metres (e.g. 0.05). Required when
            ``grid_shape`` is provided.

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
        ksize = 2 * radius_cells + 1  # pool kernel size
        # padding=radius_cells gives same spatial size as input (with edge
        # zero-padding; edges see a smaller effective neighbourhood which is
        # equivalent to the brute-force "in_window" mask clipping at the grid edge).
        z4 = z_grid.unsqueeze(1)  # (B, 1, H, W)
        z_max = _F.max_pool2d(z4, kernel_size=ksize, stride=1, padding=radius_cells).squeeze(1)
        z_min = -_F.max_pool2d(-z4, kernel_size=ksize, stride=1, padding=radius_cells).squeeze(1)
        roughness = (z_max - z_min).reshape(B, K)  # (B, K)
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

    # 4) Pick argmin per (env, foot).
    best_idx = cost.argmin(dim=-1)  # (B, F)
    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, F)
    selected = ray_hits_w[b_idx, best_idx]  # (B, F, 3)
    debug_cost = cost.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    return selected, debug_cost
