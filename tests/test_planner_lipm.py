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


def test_per_foot_effective_velocity_pure_yaw(lipm):
    v_cmd = torch.tensor([[0.0, 0.0]])
    wz = torch.tensor([1.0])
    hip_offset_w = torch.tensor([[[0.0, 0.12], [0.0, -0.12]]])

    v_eff = lipm.per_foot_effective_velocity(v_cmd, wz, hip_offset_w)

    expected = torch.tensor([[[-0.12, 0.0], [0.12, 0.0]]])
    assert torch.allclose(v_eff, expected, atol=1e-6)


def test_per_foot_effective_velocity_forward_matches_command(lipm):
    v_cmd = torch.tensor([[0.5, -0.1]])
    wz = torch.tensor([0.0])
    hip_offset_w = torch.tensor([[[0.0, 0.12], [0.0, -0.12]]])

    v_eff = lipm.per_foot_effective_velocity(v_cmd, wz, hip_offset_w)

    assert torch.allclose(v_eff[:, 0], v_cmd, atol=1e-6)
    assert torch.allclose(v_eff[:, 1], v_cmd, atol=1e-6)


def test_arc_nominal_pure_yaw_moves_off_neutral_hip(lipm):
    """Pure yaw should produce a tangential offset, not only a lateral width tweak."""
    landing_hip = torch.tensor([[0.0, 0.12]])
    v_eff = torch.tensor([[-0.12, 0.0]])
    swing_side = torch.tensor([0], dtype=torch.long)

    out = lipm.lipm_arc_nominal_foothold_xy(
        landing_hip,
        v_eff,
        swing_side,
        remaining_delta_t=0.6,
        step_duration_ts=0.6,
        com_height=0.78,
        step_width=0.24,
    )

    assert torch.isfinite(out).all()
    delta = out - landing_hip
    assert delta[0, 0].item() > 0.005
    assert delta[0, 1].item() < -0.005


def test_arc_nominal_zero_command_returns_neutral_landing_hip(lipm):
    """Zero effective speed has a neutral hip fallback, not an ICP/stall point."""
    landing_hip = torch.tensor([[1.0, 2.0]])
    v_eff = torch.tensor([[0.0, 0.0]])
    swing_side = torch.tensor([1], dtype=torch.long)

    out = lipm.lipm_arc_nominal_foothold_xy(
        landing_hip,
        v_eff,
        swing_side,
        remaining_delta_t=0.6,
        step_duration_ts=0.6,
        com_height=0.78,
        step_width=0.24,
    )

    assert torch.allclose(out, landing_hip, atol=1e-6)


def test_arc_effective_velocity_mixed_turn_inner_outer_lengths(lipm):
    """Forward + yaw: outer foot (right for CCW) gets higher effective speed."""
    v_cmd = torch.tensor([[0.5, 0.0]])
    wz = torch.tensor([0.8])
    hip_offset_w = torch.tensor([[[0.0, 0.12], [0.0, -0.12]]])

    v_eff = lipm.per_foot_effective_velocity(v_cmd, wz, hip_offset_w)
    speed = torch.linalg.vector_norm(v_eff, dim=-1)

    assert speed[0, 1].item() > speed[0, 0].item()
    assert speed[0, 1].item() / speed[0, 0].item() > 1.2


def test_arc_nominal_forward_only_produces_forward_target(lipm):
    """Pure forward (wz=0) should place target ahead of CoM."""
    landing_hip = torch.tensor([[0.3, 0.12]])
    v_eff = torch.tensor([[0.5, 0.0]])
    swing_side = torch.tensor([0], dtype=torch.long)

    out = lipm.lipm_arc_nominal_foothold_xy(
        landing_hip,
        v_eff,
        swing_side,
        remaining_delta_t=0.6,
        step_duration_ts=0.6,
        com_height=0.78,
        step_width=0.24,
    )

    assert out[0, 0].item() > 0.1
    assert out[0, 1].item() > 0.0


def test_arc_nominal_arc_walk_inner_outer_asymmetry(lipm):
    """Forward + yaw: inner vs outer foot has different effective velocity magnitudes.

    With hip_x=0 and wz>0, the tangent is purely in x for both feet, so
    single-step xy positions mirror symmetrically. The asymmetry manifests in
    effective speed (step length over a cycle). We verify via v_eff speeds.
    """
    v_cmd = torch.tensor([[0.5, 0.0]])
    wz = torch.tensor([0.8])
    hip_offset_w = torch.tensor([[[0.0, 0.12], [0.0, -0.12]]])

    v_eff = lipm.per_foot_effective_velocity(v_cmd, wz, hip_offset_w)
    speed_left = torch.linalg.vector_norm(v_eff[0, 0]).item()
    speed_right = torch.linalg.vector_norm(v_eff[0, 1]).item()

    assert speed_right > speed_left
    assert (speed_right - speed_left) > 0.15


def test_validate_lipm_rejects_multi_step(lipm):
    with pytest.raises(ValueError, match="n_future_steps=1"):
        lipm.validate_lipm_planner_config(use_lipm_prior=True, n_future_steps=2)


def test_validate_lipm_allows_single_step(lipm):
    lipm.validate_lipm_planner_config(use_lipm_prior=True, n_future_steps=1)


def test_validate_lipm_skipped_when_disabled(lipm):
    lipm.validate_lipm_planner_config(use_lipm_prior=False, n_future_steps=4)
