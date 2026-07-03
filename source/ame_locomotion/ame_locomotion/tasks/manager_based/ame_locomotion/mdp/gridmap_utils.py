"""Pure-PyTorch helpers for robot-centered, yaw-aligned height grids."""

from __future__ import annotations

import torch


def grid_center_indices(grid_shape: tuple[int, int], *, device=None, dtype=None) -> torch.Tensor:
    """Return ``(center_row, center_col)`` for a row-major ``(H, W)`` grid."""
    height, width = grid_shape
    return torch.tensor([(height - 1) * 0.5, (width - 1) * 0.5], device=device, dtype=dtype)


def grid_center_w_from_scanner_pos(scanner_pos_w: torch.Tensor) -> torch.Tensor:
    """Return the world xy grid center from scanner world positions."""
    return scanner_pos_w[..., :2]


def _batch_view(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """View a ``(B, ...)`` tensor so it broadcasts across target dims after B."""
    if x.ndim > target_ndim:
        raise ValueError(f"Cannot broadcast tensor with ndim={x.ndim} to target_ndim={target_ndim}.")
    return x.reshape((x.shape[0],) + (1,) * (target_ndim - x.ndim) + x.shape[1:])


def world_xy_to_grid_float(
    xy_w: torch.Tensor,
    grid_center_w: torch.Tensor,
    grid_yaw: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    grid_resolution: float,
) -> torch.Tensor:
    """Map world xy to floating ``(row, col)`` in a robot-centered yaw-aligned grid.

    Args:
        xy_w: ``(B, ..., 2)`` world xy samples.
        grid_center_w: ``(B, 2)`` world xy of the grid center.
        grid_yaw: ``(B,)`` yaw of the grid frame in world.
        grid_shape: ``(H, W)`` where rows index y and columns index x.
        grid_resolution: cell size in metres.
    """
    if xy_w.shape[0] != grid_center_w.shape[0] or xy_w.shape[0] != grid_yaw.shape[0]:
        raise ValueError("xy_w, grid_center_w, and grid_yaw must share the same batch dimension.")

    center = grid_center_indices(grid_shape, device=xy_w.device, dtype=xy_w.dtype)
    center_row, center_col = center[0], center[1]

    rel = xy_w - _batch_view(grid_center_w, xy_w.ndim)
    yaw = _batch_view(grid_yaw, xy_w.ndim - 1)
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    # World -> grid-local uses R(-yaw). Local x maps to col; local y maps to row.
    local_x = c * rel[..., 0] + s * rel[..., 1]
    local_y = -s * rel[..., 0] + c * rel[..., 1]
    row = local_y / grid_resolution + center_row
    col = local_x / grid_resolution + center_col
    return torch.stack([row, col], dim=-1)


def grid_float_to_row_col(grid_float: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Round floating ``(..., 2)`` ``(row, col)`` coordinates to integer indices."""
    row = torch.round(grid_float[..., 0]).long()
    col = torch.round(grid_float[..., 1]).long()
    return row, col


def in_bounds_mask(row: torch.Tensor, col: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Return True where ``row``/``col`` are inside ``grid_shape``."""
    height, width = grid_shape
    return (row >= 0) & (row < height) & (col >= 0) & (col < width)


def clamp_row_col(
    row: torch.Tensor,
    col: torch.Tensor,
    grid_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp row/col indices into grid bounds for safe tensor gather."""
    height, width = grid_shape
    return row.clamp(0, height - 1), col.clamp(0, width - 1)


def row_col_to_flat(row: torch.Tensor, col: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Convert row-major grid indices to flat indices using ``flat = row * W + col``."""
    _, width = grid_shape
    return row * width + col


def gather_grid_by_row_col(
    ray_hits_w: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ``ray_hits_w`` by row/col, clamping out-of-bounds indices safely.

    Args:
        ray_hits_w: ``(B, H*W, C)`` flattened row-major grid values.
        row: ``(B, ...)`` row indices.
        col: ``(B, ...)`` column indices.
        grid_shape: ``(H, W)``.

    Returns:
        A tuple ``(gathered, mask)`` where ``gathered`` has shape
        ``(B, ..., C)`` and ``mask`` is True for originally in-bounds samples.
    """
    batch, cells, channels = ray_hits_w.shape
    height, width = grid_shape
    if cells != height * width:
        raise ValueError(f"grid_shape={grid_shape} implies {height * width} cells, got {cells}.")
    if row.shape != col.shape or row.shape[0] != batch:
        raise ValueError("row and col must have the same shape and match ray_hits_w batch.")

    valid = in_bounds_mask(row, col, grid_shape)
    row_safe, col_safe = clamp_row_col(row, col, grid_shape)
    flat = row_col_to_flat(row_safe, col_safe, grid_shape)

    b_idx = torch.arange(batch, device=ray_hits_w.device).view(batch, *([1] * (flat.ndim - 1)))
    b_idx = b_idx.expand_as(flat)
    gathered = ray_hits_w[b_idx, flat]
    if gathered.shape[-1] != channels:
        raise RuntimeError("Unexpected gather shape.")
    return gathered, valid


def terrain_z_to_foot_body_z(terrain_z_w: torch.Tensor, sole_z_offset: float) -> torch.Tensor:
    """Convert terrain/sole z to a foot-body target z."""
    return terrain_z_w - sole_z_offset


def foot_body_z_to_terrain_z(foot_body_z_w: torch.Tensor, sole_z_offset: float) -> torch.Tensor:
    """Convert a foot-body z target to terrain/sole z."""
    return foot_body_z_w + sole_z_offset


def terrain_xyz_to_foot_body_xyz(
    terrain_xyz_w: torch.Tensor,
    sole_z_offset: float,
) -> torch.Tensor:
    """Convert terrain xyz candidates to foot-body xyz targets."""
    out = terrain_xyz_w.clone()
    out[..., 2] = terrain_z_to_foot_body_z(out[..., 2], sole_z_offset)
    return out
