"""Unit tests for mdp/foothold_geometry.py."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

_MDP_DIR = (
    Path(__file__).resolve().parents[1]
    / "source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp"
)
_PKG = "foothold_geometry_test_mdp"


def _load_package_module(module_name: str):
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_MDP_DIR)]
        sys.modules[_PKG] = pkg

    full_name = f"{_PKG}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    path = _MDP_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fg():
    _load_package_module("gridmap_utils")
    return _load_package_module("foothold_geometry")


def _regular_rays(
    grid_shape=(21, 33),
    resolution=0.05,
    batch: int = 1,
    z_fn=None,
):
    height, width = grid_shape
    x = torch.linspace(-resolution * (width - 1) / 2, resolution * (width - 1) / 2, width)
    y = torch.linspace(-resolution * (height - 1) / 2, resolution * (height - 1) / 2, height)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    if z_fn is None:
        z = torch.zeros_like(grid_x)
    else:
        z = z_fn(grid_x, grid_y)
        if not isinstance(z, torch.Tensor):
            z = torch.full_like(grid_x, float(z))
    flat = torch.stack([grid_x, grid_y, z], dim=-1).reshape(1, height * width, 3)
    return flat.expand(batch, -1, -1).contiguous()


def _grid_kwargs(batch: int = 1, yaw: float = 0.0, grid_shape=(21, 33), resolution=0.05):
    return dict(
        grid_shape=grid_shape,
        grid_resolution=resolution,
        grid_center_w=torch.zeros(batch, 2),
        grid_yaw=torch.full((batch,), yaw),
    )


def test_flat_terrain(fg):
    rays = _regular_rays()
    centers = torch.zeros(1, 1, 3)
    yaws = torch.zeros(1, 1)
    out = fg.foot_patch_stats(
        centers, yaws, rays,
        foot_length=0.18, foot_width=0.065, n_long=4, n_lat=3,
        support_threshold=0.03,
        **_grid_kwargs(),
    )
    assert out["dz_omega"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["support_ratio"].item() == pytest.approx(1.0, abs=1e-6)
    assert out["overhang_ratio"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["slope_angle"].item() == pytest.approx(0.0, abs=1e-4)
    assert out["z_median"].item() == pytest.approx(0.0, abs=1e-6)


def test_step_edge_half_on_half_off(fg):
    def step(x, y):
        return (x >= 0.0).float() * 0.15

    rays = _regular_rays(z_fn=step)
    centers = torch.zeros(1, 1, 3)
    yaws = torch.zeros(1, 1)
    out = fg.foot_patch_stats(
        centers, yaws, rays,
        foot_length=0.18, foot_width=0.065, n_long=4, n_lat=3,
        support_threshold=0.03,
        **_grid_kwargs(),
    )
    assert out["dz_omega"].item() == pytest.approx(0.15, abs=0.02)
    assert out["support_ratio"].item() == pytest.approx(0.5, abs=0.15)


def test_pure_slope_10deg(fg):
    slope_rad = math.radians(10.0)

    def slope(x, y):
        return torch.tan(torch.tensor(slope_rad)) * x

    rays = _regular_rays(z_fn=slope)
    centers = torch.tensor([[[0.3, 0.0, 0.0]]])
    yaws = torch.zeros(1, 1)
    out = fg.foot_patch_stats(
        centers, yaws, rays,
        foot_length=0.18, foot_width=0.065, n_long=4, n_lat=3,
        support_threshold=0.03,
        **_grid_kwargs(),
    )
    assert math.degrees(out["slope_angle"].item()) == pytest.approx(10.0, abs=2.0)
    assert out["support_ratio"].item() > 0.5


def test_overhang_detection(fg):
    rays = _regular_rays().clone()
    # One footprint sample at body (-0.09, -0.0325) hits flat index 311 on the default grid.
    rays[0, 311, 2] = -0.10
    centers = torch.zeros(1, 1, 3)
    yaws = torch.zeros(1, 1)
    out = fg.foot_patch_stats(
        centers, yaws, rays,
        foot_length=0.18, foot_width=0.065, n_long=4, n_lat=3,
        support_threshold=0.03,
        **_grid_kwargs(),
    )
    assert out["overhang_ratio"].item() == pytest.approx(1.0 / 12.0, abs=0.05)
    assert out["support_ratio"].item() == pytest.approx(11.0 / 12.0, abs=0.05)


def test_yaw_rotation(fg):
    def slope(x, y):
        return 0.2 * x

    rays = _regular_rays(z_fn=slope)
    centers = torch.tensor([[[0.2, 0.0, 0.0]]])
    yaws0 = torch.zeros(1, 1)
    yaws90 = torch.tensor([[math.pi / 2]])

    out0 = fg.foot_patch_stats(
        centers, yaws0, rays,
        foot_length=0.18, foot_width=0.065, n_long=4, n_lat=3,
        support_threshold=0.03, return_debug=True,
        **_grid_kwargs(),
    )
    out90 = fg.foot_patch_stats(
        centers, yaws90, rays,
        foot_length=0.18, foot_width=0.065, n_long=4, n_lat=3,
        support_threshold=0.03, return_debug=True,
        **_grid_kwargs(),
    )
    assert out0["dz_omega"].item() == pytest.approx(out90["dz_omega"].item(), abs=0.02)
    assert not torch.allclose(out0["sample_xy_w"], out90["sample_xy_w"], atol=1e-4)


def test_shapes_batched(fg):
    B, M = 4, 7
    rays = _regular_rays(batch=B)
    centers = torch.randn(B, M, 3) * 0.05
    yaws = torch.randn(B, M) * 0.2
    out = fg.foot_patch_stats(
        centers, yaws, rays,
        foot_length=0.18, foot_width=0.065, n_long=4, n_lat=3,
        support_threshold=0.03,
        **_grid_kwargs(batch=B),
    )
    for key in (
        "z_median", "z_q70", "z_max", "z_min", "dz_omega", "support_ratio",
        "overhang_ratio", "footprint_in_bounds_ratio", "slope_angle",
    ):
        assert out[key].shape == (B, M), key
        assert out[key].dtype == torch.float32, key
        assert torch.isfinite(out[key]).all(), key
    assert out["slope_normal"].shape == (B, M, 3)
    assert out["sample_in_bounds"].shape == (B, M, 12)


def test_zero_offsets_matches_ray_hit(fg):
    def peak(x, y):
        return torch.exp(-((x - 0.2) ** 2 + y ** 2) / 0.01) * 0.5

    rays = _regular_rays(z_fn=peak)
    centers = torch.tensor([[[0.2, 0.0, 0.0]]])
    yaws = torch.zeros(1, 1)
    out = fg.foot_patch_stats(
        centers, yaws, rays,
        foot_length=0.0, foot_width=0.0, n_long=1, n_lat=1,
        support_threshold=0.03,
        **_grid_kwargs(),
    )
    # Nearest grid cell at (0.2, 0) on the peak map.
    ix = round(0.2 / 0.05 + 16)
    iy = round(0.0 / 0.05 + 10)
    flat = iy * 33 + ix
    expected_z = rays[0, flat, 2].item()
    assert out["z_median"].item() == pytest.approx(expected_z, abs=1e-5)


def test_grid_path_required_when_many_candidates(fg):
    rays = _regular_rays(batch=1)
    centers = torch.zeros(1, 3, 3)
    yaws = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="grid_shape"):
        fg.foot_patch_stats(
            centers, yaws, rays,
            foot_length=0.18, foot_width=0.065, n_long=4, n_lat=3,
            support_threshold=0.03,
        )
