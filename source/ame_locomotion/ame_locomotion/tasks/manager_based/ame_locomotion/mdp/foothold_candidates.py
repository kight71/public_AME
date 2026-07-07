"""Fixed-window candidate generation for the Planner V2 foothold selector."""

from __future__ import annotations

import torch

from .gridmap_utils import (
    gather_grid_by_row_col,
    grid_float_to_row_col,
    world_xy_to_grid_float,
)

__all__ = ["generate_candidates", "generate_terrain_window_candidates"]


def generate_candidates(
    raibert_xy_w: torch.Tensor,
    ray_hits_w: torch.Tensor,
    grid_center_w: torch.Tensor,
    grid_yaw: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    grid_resolution: float,
    half_width_cells: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a fixed odd window of elevation candidates around each Raibert prior.

    Args:
        raibert_xy_w: ``(B, F, 2)`` or ``(B, F, 3)`` Raibert prior xy in world frame.
        ray_hits_w: ``(B, K, 3)`` privileged scanner hits, row-major ``K = H * W``.
        grid_center_w: ``(B, 2)`` world xy of the scanner/grid center.
        grid_yaw: ``(B,)`` yaw of the real scanner grid.
        grid_shape: ``(H, W)`` height-grid shape.
        grid_resolution: cell size in metres.
        half_width_cells: half-width of the candidate window in grid cells.
            ``2`` yields a ``5 x 5 = 25`` candidate set.

    Returns:
        candidate_xyz_w: ``(B, F, K_c, 3)`` world-frame candidate positions.
        in_bounds_mask: ``(B, F, K_c)`` bool; False when the cell is outside the grid.
    """
    if raibert_xy_w.shape[-1] == 3:
        raibert_xy = raibert_xy_w[..., :2]
    elif raibert_xy_w.shape[-1] == 2:
        raibert_xy = raibert_xy_w
    else:
        raise ValueError(f"raibert_xy_w must have 2 or 3 trailing dims, got shape {raibert_xy_w.shape}.")

    batch, num_feet, _ = raibert_xy.shape
    if ray_hits_w.shape[0] != batch:
        raise ValueError("raibert_xy_w and ray_hits_w must share batch dimension.")
    if grid_center_w.shape != (batch, 2):
        raise ValueError(f"grid_center_w must be (B, 2), got {grid_center_w.shape}.")
    if grid_yaw.shape != (batch,):
        raise ValueError(f"grid_yaw must be (B,), got {grid_yaw.shape}.")

    height, width = grid_shape
    expected_cells = height * width
    if ray_hits_w.shape[1] != expected_cells:
        raise ValueError(
            f"ray_hits_w has K={ray_hits_w.shape[1]} but grid_shape={grid_shape} implies K={expected_cells}."
        )

    grid_float = world_xy_to_grid_float(
        raibert_xy,
        grid_center_w,
        grid_yaw,
        grid_shape=grid_shape,
        grid_resolution=grid_resolution,
    )
    center_row, center_col = grid_float_to_row_col(grid_float)

    offsets = torch.arange(
        -half_width_cells,
        half_width_cells + 1,
        device=raibert_xy.device,
        dtype=center_row.dtype,
    )
    drow, dcol = torch.meshgrid(offsets, offsets, indexing="ij")
    drow = drow.reshape(-1)
    dcol = dcol.reshape(-1)

    row = center_row.unsqueeze(-1) + drow.view(1, 1, -1)
    col = center_col.unsqueeze(-1) + dcol.view(1, 1, -1)

    candidate_xyz_w, in_bounds_mask = gather_grid_by_row_col(
        ray_hits_w,
        row,
        col,
        grid_shape=grid_shape,
    )
    return candidate_xyz_w, in_bounds_mask


def generate_terrain_window_candidates(
    nominal_xy_w: torch.Tensor,
    ray_hits_w: torch.Tensor,
    grid_center_w: torch.Tensor,
    grid_yaw: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    grid_resolution: float,
    half_width_cells: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a larger terrain-aware candidate window around each nominal foothold.

    This is intentionally terrain-class agnostic: it only gathers scanner grid
    cells around the nominal xy prior and leaves steppability decisions to the
    Planner V2 scoring/filtering layer. The default ``4`` yields a
    ``9 x 9 = 81`` candidate set.
    """
    return generate_candidates(
        nominal_xy_w,
        ray_hits_w,
        grid_center_w,
        grid_yaw,
        grid_shape=grid_shape,
        grid_resolution=grid_resolution,
        half_width_cells=half_width_cells,
    )
