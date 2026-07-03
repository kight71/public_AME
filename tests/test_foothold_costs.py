"""Unit tests for mdp/foothold_costs.py."""

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
_PKG = "foothold_costs_test_mdp"


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
    _load_package_module("planner")
    return _load_package_module("foothold_costs")


def test_cost_nominal_zero_at_raibert(fc):
    raibert = torch.tensor([[[0.2, 0.1]]])  # (1, 1, 2)
    at_prior = torch.tensor([[[[0.2, 0.1, 0.0]]]])
    offset = torch.tensor([[[[0.3, 0.1, 0.0]]]])

    c0 = fc.cost_nominal(at_prior, raibert, sigma_nominal=0.10)
    c1 = fc.cost_nominal(offset, raibert, sigma_nominal=0.10)
    assert torch.allclose(c0, torch.zeros_like(c0))
    assert c1.item() == pytest.approx(1.0, abs=1e-5)


def test_cost_reach_zero_at_comfort_center(fc):
    body_pos = torch.tensor([[0.0, 0.0, 0.80]])
    body_yaw = torch.tensor([0.0])
    comfort = ((0.0, 0.12, -0.78), (0.0, -0.12, -0.78))
    # Left foot at comfort center in body frame → world xyz
    cand = torch.zeros(1, 2, 1, 3)
    cand[0, 0, 0] = torch.tensor([0.0, 0.12, 0.80 - 0.78])
    c0 = fc.cost_reach(cand, body_pos, body_yaw, comfort_center_b=comfort, s_reach=0.15)
    assert c0[0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)

    off = cand.clone()
    off[0, 0, 0, 0] += 0.15
    c1 = fc.cost_reach(off, body_pos, body_yaw, comfort_center_b=comfort, s_reach=0.15)
    assert c1[0, 0, 0].item() == pytest.approx(1.0, abs=1e-4)


def test_cost_reach_body_frame_transform(fc):
    body_pos = torch.tensor([[0.0, 0.0, 0.80]])
    body_yaw = torch.tensor([math.pi / 2])
    comfort = ((0.0, 0.12, -0.78), (0.0, -0.12, -0.78))
    # yaw=90°: body +y (left) → world -x. Comfort left (+0.12 body y) → world (-0.12, 0).
    cand = torch.zeros(1, 1, 1, 3)
    cand[0, 0, 0] = torch.tensor([-0.12, 0.0, 0.02])
    c = fc.cost_reach(cand, body_pos, body_yaw, comfort_center_b=comfort, s_reach=0.15)
    assert c[0, 0, 0].item() == pytest.approx(0.0, abs=1e-4)


def test_cost_terrain_zero_at_flat(fc):
    dz = torch.tensor([[[0.0, 0.05]]])
    assert fc.cost_terrain(dz[..., :1], sigma_rough=0.05).item() == pytest.approx(0.0)
    assert fc.cost_terrain(dz[..., 1:], sigma_rough=0.05).item() == pytest.approx(1.0, abs=1e-5)


def test_cost_edge_zero_at_full_support(fc):
    sup = torch.tensor([[[1.0, 0.0]]])
    oh = torch.tensor([[[0.0, 1.0]]])
    c_good = fc.cost_edge(sup[..., :1], oh[..., :1])
    c_bad = fc.cost_edge(sup[..., 1:], oh[..., 1:])
    assert c_good.item() == pytest.approx(0.0)
    assert c_bad.item() == pytest.approx(3.0)


def test_cost_height_monotone(fc):
    fgc = _load_package_module("foot_geometry_constants")
    sole = fgc.G1_SOLE_Z_OFFSET
    stance = torch.zeros(1, 1)
    terrains = [0.0, 0.05, 0.10, 0.20]
    cands = torch.stack(
        [torch.tensor([0.0, 0.0, t - sole]) for t in terrains],
        dim=0,
    ).view(1, 1, 4, 3)
    costs = fc.cost_height(cands, stance, sigma_h=0.10, sole_z_offset=sole)[0, 0]
    assert costs[0].item() == pytest.approx(0.0)
    assert costs[1].item() == pytest.approx(0.25, abs=1e-5)
    assert costs[2].item() == pytest.approx(1.0, abs=1e-5)
    assert costs[3].item() > costs[2].item()


def test_cost_height_foot_body_at_stance(fc):
    fgc = _load_package_module("foot_geometry_constants")
    sole = fgc.G1_SOLE_Z_OFFSET
    stance = torch.zeros(1, 1)
    cands = torch.tensor([[[[0.0, 0.0, -sole]]]])
    assert fc.cost_height(cands, stance, sole_z_offset=sole)[0, 0, 0].item() == pytest.approx(0.0)


def test_cost_slope_zero_at_flat(fc):
    ang = torch.tensor([[[0.0, 0.2094]]])
    assert fc.cost_slope(ang[..., :1]).item() == pytest.approx(0.0)
    assert fc.cost_slope(ang[..., 1:]).item() == pytest.approx(1.0, abs=1e-4)


def test_cost_terrain_and_edge_orthogonal(fc):
    dz = torch.zeros(1, 1, 1)
    sup = torch.tensor([[[0.5]]])
    oh = torch.zeros(1, 1, 1)
    assert fc.cost_terrain(dz).item() == pytest.approx(0.0)
    assert fc.cost_edge(sup, oh).item() > 0.0


def test_cost_terrain_vs_edge_no_double_count(fc):
    dz = torch.tensor([[[0.05]]])
    sup = torch.tensor([[[1.0]]])
    oh = torch.zeros(1, 1, 1)
    assert fc.cost_terrain(dz, sigma_rough=0.05).item() == pytest.approx(1.0, abs=1e-5)
    assert fc.cost_edge(sup, oh).item() == pytest.approx(0.0)
