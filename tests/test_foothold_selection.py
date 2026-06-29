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


# ---------------------------------------------------------------------------
# Grid-path tests (max_pool2d fast path)
# ---------------------------------------------------------------------------

def _grid_rays_regular(B: int, H: int, W: int, res: float, z_fn=None):
    """Build a (B, H*W, 3) ray_hits_w tensor using row-major (H, W) layout.

    Matches IsaacLab grid_pattern ordering="xy":
      outer loop y (rows, H), inner loop x (cols, W).
    Grid is centred at origin. z defaults to 0.
    """
    x = torch.linspace(-res * (W - 1) / 2, res * (W - 1) / 2, W)
    y = torch.linspace(-res * (H - 1) / 2, res * (H - 1) / 2, H)
    # indexing="xy": grid_x.shape=(H, W), outer=y, inner=x
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    if z_fn is not None:
        z = z_fn(grid_x, grid_y)
        if not isinstance(z, torch.Tensor):
            z = torch.full_like(grid_x, float(z))
    else:
        z = torch.zeros_like(grid_x)
    flat = torch.stack([grid_x, grid_y, z], dim=-1).reshape(1, H * W, 3)
    return flat.expand(B, -1, -1).contiguous()


def test_grid_path_matches_brute_force_on_flat(fs):
    """Grid path and brute-force path must choose the same foothold on flat terrain."""
    B, H, W = 2, 11, 11
    res = 0.05
    rays = _grid_rays_regular(B, H, W, res)
    # Two arbitrary Raibert targets inside the grid
    raibert = torch.zeros(B, 2, 3)
    raibert[0, 0, :2] = torch.tensor([0.0, 0.0])
    raibert[0, 1, :2] = torch.tensor([0.1, 0.05])
    raibert[1, 0, :2] = torch.tensor([-0.05, 0.0])
    raibert[1, 1, :2] = torch.tensor([0.0, 0.1])

    common_kwargs = dict(
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
    )
    sel_brute, cost_brute = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays, **common_kwargs
    )
    sel_grid, cost_grid = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        grid_shape=(H, W), grid_resolution=res,
        **common_kwargs,
    )
    # On flat terrain roughness = 0 everywhere; both paths should pick cells at
    # the same (x, y) — the cell nearest the Raibert prior.
    assert torch.allclose(sel_brute[..., :2], sel_grid[..., :2], atol=1e-5), (
        f"brute={sel_brute[..., :2]}\ngrid={sel_grid[..., :2]}"
    )


def test_grid_path_step_edge(fs):
    """Grid path should avoid a step edge just like brute-force does."""
    B, H, W = 1, 11, 11
    res = 0.05

    def step(x, y):
        return (x >= 0.0).float() * 0.15

    rays = _grid_rays_regular(B, H, W, res, z_fn=step)
    raibert = torch.tensor([[[0.0, 0.0, 0.0]]])  # right on the edge

    sel_brute, _ = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
    )
    sel_grid, _ = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
        grid_shape=(H, W), grid_resolution=res,
    )
    # Both paths should push the selection away from the discontinuity.
    assert abs(sel_brute[0, 0, 0].item()) > 0.02, (
        f"brute-force failed to avoid edge: x={sel_brute[0, 0, 0].item()}"
    )
    assert abs(sel_grid[0, 0, 0].item()) > 0.02, (
        f"grid path failed to avoid edge: x={sel_grid[0, 0, 0].item()}"
    )


def test_grid_path_memory_smoke(fs):
    """Smoke test: real grid shape (B=512, H=21, W=33) completes without OOM."""
    B, H, W = 512, 21, 33
    res = 0.05
    rays = _grid_rays_regular(B, H, W, res)
    raibert = torch.zeros(B, 2, 3)

    sel, cost = fs.select_foothold_by_cost(
        raibert_xy_w=raibert, ray_hits_w=rays,
        window_m=0.10, alpha=2.0, beta=1.0, gamma=5.0, delta=1.0,
        obstacle_height_threshold=0.15,
        grid_shape=(H, W), grid_resolution=res,
    )
    assert sel.shape == (B, 2, 3), f"Unexpected shape: {sel.shape}"
    assert cost.shape == (B, 2), f"Unexpected cost shape: {cost.shape}"
    # All selected z should be 0 (flat terrain)
    assert torch.all(sel[..., 2] == 0.0)
