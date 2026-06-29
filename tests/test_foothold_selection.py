"""Unit tests for mdp/foothold_selection.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/foothold_selection.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("foothold_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fs():
    return _load()


def _grid_rays(center_xy, half_size=0.5, res=0.05, z_fn=lambda x, y: 0.0):
    """Build a (1, K, 3) ray_hits_w tensor for one env over a (half_size*2)^2 patch."""
    n = int(2 * half_size / res) + 1
    xs = torch.linspace(center_xy[0] - half_size, center_xy[0] + half_size, n)
    ys = torch.linspace(center_xy[1] - half_size, center_xy[1] + half_size, n)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    z = z_fn(gx, gy)
    if not isinstance(z, torch.Tensor):
        z = torch.full_like(gx, float(z))
    return torch.stack([gx, gy, z], dim=-1).reshape(1, -1, 3)


def test_flat_terrain_returns_raibert_center(fs):
    raibert = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]]])  # (1, 2, 3)
    rays = _grid_rays((0.0, 0.1))
    sel, cost = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
    )
    # On flat terrain the cost minimum should sit on (delta only) → at the
    # cell closest to the Raibert xy.
    assert torch.allclose(sel[..., :2], raibert[..., :2], atol=0.05)


def test_step_edge_penalty_pushes_off_edge(fs):
    # Step edge at x=0.0: z=0 for x<0, z=0.15 for x>=0.
    def step(x, y):
        return (x >= 0.0).float() * 0.15
    raibert = torch.tensor([[[0.0, 0.0, 0.0]]])  # right on the edge
    rays = _grid_rays((0.0, 0.0), z_fn=step)
    sel, _ = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
    )
    # Roughness (max-min in window) is high on the edge → selector should
    # pick a cell away from the discontinuity.
    assert abs(sel[0, 0, 0].item()) > 0.02


def test_output_z_matches_grid_z(fs):
    def slope(x, y):
        return 0.1 * x
    raibert = torch.tensor([[[0.3, 0.0, 0.0]]])
    rays = _grid_rays((0.3, 0.0), z_fn=slope)
    sel, _ = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.05, alpha=0.0, beta=0.0, gamma=0.0, delta=1.0,
        obstacle_height_threshold=0.15,
    )
    # With only delta active and a tight window, selector picks the cell
    # closest to Raibert; its z must match slope(x≈0.3) ≈ 0.03.
    assert sel[0, 0, 2].item() == pytest.approx(0.03, abs=0.01)
