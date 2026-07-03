"""Unit tests for mdp/foothold_candidates.py."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_MDP_DIR = (
    Path(__file__).resolve().parents[1]
    / "source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp"
)
_PKG = "foothold_candidates_test_mdp"


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
def fc():
    _load_package_module("gridmap_utils")
    return _load_package_module("foothold_candidates")


def _regular_rays(grid_shape=(21, 33), resolution=0.05, batch: int = 1):
    height, width = grid_shape
    x = torch.linspace(-resolution * (width - 1) / 2, resolution * (width - 1) / 2, width)
    y = torch.linspace(-resolution * (height - 1) / 2, resolution * (height - 1) / 2, height)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    z = torch.zeros_like(grid_x)
    flat = torch.stack([grid_x, grid_y, z], dim=-1).reshape(1, height * width, 3)
    return flat.expand(batch, -1, -1).contiguous()


def test_yaw_zero_matches_naive(fc):
    raibert = torch.tensor([[[0.20, 0.10]]])  # (1, 1, 2)
    rays = _regular_rays()
    candidates, mask = fc.generate_candidates(
        raibert,
        rays,
        grid_center_w=torch.tensor([[0.0, 0.0]]),
        grid_yaw=torch.tensor([0.0]),
        grid_shape=(21, 33),
        grid_resolution=0.05,
        half_width_cells=2,
    )
    assert candidates.shape == (1, 1, 25, 3)
    assert mask.shape == (1, 1, 25)
    assert mask.all()

    center_idx = 12  # middle of 5x5 row-major window
    assert torch.allclose(candidates[0, 0, center_idx, :2], torch.tensor([0.20, 0.10]), atol=1e-6)

    # Window spans row 10..14 and col 18..22 around center (12, 20).
    rows = torch.tensor([10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
                         13, 13, 13, 13, 13, 14, 14, 14, 14, 14])
    cols = torch.tensor([18, 19, 20, 21, 22] * 5)
    width = 33
    flat_idx = rows * width + cols
    assert torch.equal(candidates[0, 0, :, :2], rays[0, flat_idx, :2])


def test_yaw_90deg(fc):
    # Local Raibert (0.20, 0.10) rotated by +90 deg into world → same grid cell as yaw=0.
    raibert_yaw0 = torch.tensor([[[0.20, 0.10]]])
    raibert_yaw90 = torch.tensor([[[-0.10, 0.20]]])
    rays = _regular_rays()

    candidates0, mask0 = fc.generate_candidates(
        raibert_yaw0,
        rays,
        grid_center_w=torch.tensor([[0.0, 0.0]]),
        grid_yaw=torch.tensor([0.0]),
        grid_shape=(21, 33),
        grid_resolution=0.05,
        half_width_cells=2,
    )
    candidates90, mask90 = fc.generate_candidates(
        raibert_yaw90,
        rays,
        grid_center_w=torch.tensor([[0.0, 0.0]]),
        grid_yaw=torch.tensor([torch.pi / 2]),
        grid_shape=(21, 33),
        grid_resolution=0.05,
        half_width_cells=2,
    )
    assert torch.allclose(candidates0, candidates90, atol=1e-6)
    assert mask0.all() and mask90.all()


def test_raibert_near_edge(fc):
    # Place Raibert near the top-left corner so part of the 5x5 window falls OOB.
    raibert = torch.tensor([[[-0.80, -0.50]]])
    rays = _regular_rays()
    candidates, mask = fc.generate_candidates(
        raibert,
        rays,
        grid_center_w=torch.tensor([[0.0, 0.0]]),
        grid_yaw=torch.tensor([0.0]),
        grid_shape=(21, 33),
        grid_resolution=0.05,
        half_width_cells=2,
    )
    assert not mask.all()
    assert torch.isfinite(candidates).all()
    # Clamped OOB cells should copy nearest in-bounds neighbour, not NaN.
    assert candidates[..., 2].abs().max() == 0.0


def test_output_shape(fc):
    raibert = torch.zeros(4, 2, 2)
    rays = _regular_rays(batch=4)
    candidates, mask = fc.generate_candidates(
        raibert,
        rays,
        grid_center_w=torch.zeros(4, 2),
        grid_yaw=torch.zeros(4),
        grid_shape=(21, 33),
        grid_resolution=0.05,
        half_width_cells=3,
    )
    assert candidates.shape == (4, 2, 49, 3)
    assert mask.shape == (4, 2, 49)


def test_5x5_default(fc):
    raibert = torch.zeros(1, 2, 2)
    rays = _regular_rays()
    candidates, mask = fc.generate_candidates(
        raibert,
        rays,
        grid_center_w=torch.zeros(1, 2),
        grid_yaw=torch.zeros(1),
        grid_shape=(21, 33),
        grid_resolution=0.05,
    )
    assert candidates.shape[-2] == 25
    assert mask.shape[-1] == 25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VRAM smoke test")
def test_no_python_loops(fc):
    device = torch.device("cuda")
    B, F = 4096, 2
    raibert = torch.zeros(B, F, 2, device=device)
    rays = _regular_rays(batch=B).to(device)
    grid_center_w = torch.zeros(B, 2, device=device)
    grid_yaw = torch.zeros(B, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.max_memory_allocated(device)

    candidates, mask = fc.generate_candidates(
        raibert,
        rays,
        grid_center_w,
        grid_yaw,
        grid_shape=(21, 33),
        grid_resolution=0.05,
        half_width_cells=2,
    )

    peak = torch.cuda.max_memory_allocated(device)
    delta_mb = (peak - base) / (1024 * 1024)
    assert candidates.shape == (B, F, 25, 3)
    assert mask.shape == (B, F, 25)
    assert delta_mb < 100.0, f"peak allocation delta {delta_mb:.1f} MB exceeds 100 MB"
