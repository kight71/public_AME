"""Unit tests for mdp/planner_lipm.py (3D-LIPM / ICP foothold math).

Loaded via importlib — no Isaac Sim dependency.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_LIPM_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/planner_lipm.py"
)


def _load_lipm():
    spec = importlib.util.spec_from_file_location("planner_lipm_under_test", _LIPM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lipm():
    return _load_lipm()


def test_step_length_at_step_start(lipm):
    v = torch.tensor([[0.75, 0.0]])
    sd, wd = lipm.lipm_step_length(
        v, remaining_delta_t=0.6, step_width=0.24, step_duration_ts=0.6
    )
    assert sd.item() == pytest.approx(0.45, abs=1e-5)
    assert wd.item() == pytest.approx(0.24, abs=1e-5)


def test_step_length_half_remaining(lipm):
    v = torch.tensor([[0.75, 0.0]])
    sd, _ = lipm.lipm_step_length(
        v, remaining_delta_t=0.3, step_width=0.24, step_duration_ts=0.6
    )
    assert sd.item() == pytest.approx(0.225, abs=1e-5)


def test_icp_propagation_hand_calc(lipm):
    z0 = 0.78
    omega0 = lipm.omega0_from_com_height(torch.tensor(z0))
    w0 = omega0.item()
    ts = 0.6
    xi0 = torch.tensor([[0.0, 0.0]])
    p = torch.tensor([[0.45, 0.10]])
    xi_f = lipm.predict_icp_end(xi0, p, omega0, step_duration_ts=ts)
    exp_ts = math.exp(w0 * ts)
    expected = exp_ts * xi0 + (1.0 - exp_ts) * p
    assert torch.allclose(xi_f, expected, atol=1e-5)


def test_straight_vs_turning_forward(lipm):
    com = torch.tensor([[0.0, 0.0]])
    vel = torch.tensor([[0.75, 0.0]])
    stance = torch.tensor([[0.0, 0.10]])
    swing = torch.tensor([[0.0, -0.10]])
    v_cmd = torch.tensor([[0.75, 0.0]])
    swing_side = torch.tensor([0], dtype=torch.long)

    straight = lipm.lipm_nominal_foothold_xy(
        com, vel, stance, swing, v_cmd, swing_side,
        remaining_delta_t=0.6, step_duration_ts=0.6, com_height=0.78, use_turning=False,
    )
    turning = lipm.lipm_nominal_foothold_xy(
        com, vel, stance, swing, v_cmd, swing_side,
        remaining_delta_t=0.6, step_duration_ts=0.6, com_height=0.78, use_turning=True,
    )
    assert torch.allclose(straight, turning, atol=1e-4)
    assert straight[0, 0].item() > 0.0


def test_zero_velocity_fallback_uses_swing_foot_not_stance(lipm):
    """Stall fallback must not collapse onto the stance foot."""
    com = torch.tensor([[1.0, 2.0]])
    vel = torch.tensor([[0.0, 0.0]])
    stance = torch.tensor([[0.9, 2.10]])   # left
    swing_last = torch.tensor([[0.9, 1.90]])  # right
    v_cmd = torch.tensor([[0.0, 0.0]])
    swing_side = torch.tensor([1], dtype=torch.long)

    out = lipm.lipm_nominal_foothold_xy(
        com, vel, stance, swing_last, v_cmd, swing_side,
        remaining_delta_t=0.6, step_duration_ts=0.6, com_height=0.78,
    )
    assert torch.allclose(out, swing_last, atol=1e-5)
    assert torch.linalg.vector_norm(out - stance, dim=-1).item() == pytest.approx(0.20, abs=1e-5)


def test_validate_lipm_rejects_multi_step(lipm):
    with pytest.raises(ValueError, match="n_future_steps=1"):
        lipm.validate_lipm_planner_config(use_lipm_prior=True, n_future_steps=2)


def test_validate_lipm_allows_single_step(lipm):
    lipm.validate_lipm_planner_config(use_lipm_prior=True, n_future_steps=1)


def test_validate_lipm_skipped_when_disabled(lipm):
    lipm.validate_lipm_planner_config(use_lipm_prior=False, n_future_steps=4)
