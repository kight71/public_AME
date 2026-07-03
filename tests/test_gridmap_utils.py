"""Tests for mdp/gridmap_utils.py center-based height-grid contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/gridmap_utils.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("gridmap_utils_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gridmap():
    return _load()


def _regular_rays(grid_shape=(21, 33), resolution=0.05):
    height, width = grid_shape
    x = torch.linspace(-resolution * (width - 1) / 2, resolution * (width - 1) / 2, width)
    y = torch.linspace(-resolution * (height - 1) / 2, resolution * (height - 1) / 2, height)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    z = grid_y + 10.0 * grid_x
    return torch.stack([grid_x, grid_y, z], dim=-1).reshape(1, height * width, 3)


def test_center_indices_for_default_scanner(gridmap):
    center = gridmap.grid_center_indices((21, 33), dtype=torch.float32)
    assert torch.allclose(center, torch.tensor([10.0, 16.0]))


def test_world_xy_to_grid_zero_yaw_centered(gridmap):
    xy_w = torch.tensor([[[0.0, 0.0], [0.20, 0.10], [-0.80, -0.50]]])
    grid_float = gridmap.world_xy_to_grid_float(
        xy_w,
        grid_center_w=torch.tensor([[0.0, 0.0]]),
        grid_yaw=torch.tensor([0.0]),
        grid_shape=(21, 33),
        grid_resolution=0.05,
    )
    row, col = gridmap.grid_float_to_row_col(grid_float)

    expected_row = torch.tensor([[10, 12, 0]])
    expected_col = torch.tensor([[16, 20, 0]])
    assert torch.equal(row, expected_row)
    assert torch.equal(col, expected_col)


def test_world_xy_to_grid_yaw_90_uses_grid_frame(gridmap):
    # This world point is local (x=0.20, y=0.10) rotated by +90 degrees.
    xy_w = torch.tensor([[[-0.10, 0.20]]])
    grid_float = gridmap.world_xy_to_grid_float(
        xy_w,
        grid_center_w=torch.tensor([[0.0, 0.0]]),
        grid_yaw=torch.tensor([torch.pi / 2]),
        grid_shape=(21, 33),
        grid_resolution=0.05,
    )
    row, col = gridmap.grid_float_to_row_col(grid_float)

    assert torch.equal(row, torch.tensor([[12]]))
    assert torch.equal(col, torch.tensor([[20]]))


def test_row_col_to_flat_is_row_major(gridmap):
    row = torch.tensor([[0, 1, 10]])
    col = torch.tensor([[0, 2, 16]])
    flat = gridmap.row_col_to_flat(row, col, (21, 33))
    assert torch.equal(flat, torch.tensor([[0, 35, 346]]))


def test_gather_grid_by_row_col_clamps_but_reports_mask(gridmap):
    rays = _regular_rays()
    row = torch.tensor([[10, -1, 21]])
    col = torch.tensor([[16, 3, 40]])

    gathered, mask = gridmap.gather_grid_by_row_col(rays, row, col, grid_shape=(21, 33))

    assert gathered.shape == (1, 3, 3)
    assert torch.equal(mask, torch.tensor([[True, False, False]]))
    # Center cell is exactly x=0, y=0, z=0.
    assert torch.allclose(gathered[0, 0], torch.tensor([0.0, 0.0, 0.0]), atol=1e-6)
    # Out-of-bounds entries are clamped for safe gather, not marked valid.
    assert torch.isfinite(gathered).all()


def test_terrain_foot_body_z_conversions(gridmap):
    terrain_z = torch.tensor([0.0, 0.20])
    sole_z_offset = -0.035409145057201385

    foot_body_z = gridmap.terrain_z_to_foot_body_z(terrain_z, sole_z_offset)
    restored = gridmap.foot_body_z_to_terrain_z(foot_body_z, sole_z_offset)

    assert torch.allclose(foot_body_z, torch.tensor([0.035409145057201385, 0.23540914]))
    assert torch.allclose(restored, terrain_z)
