"""Unit tests for mdp/planner.py pure geometric primitives.

These tests must not import the ame_locomotion package itself, because its
__init__ chain pulls in Isaac Sim (omni.log) which is unavailable outside the
Isaac Lab runtime. We load planner.py directly via importlib so the math layer
stays decoupled from Isaac Sim.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_PLANNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/planner.py"
)


def _load_planner():
    spec = importlib.util.spec_from_file_location("planner_under_test", _PLANNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def planner():
    return _load_planner()


def _hip_offsets() -> torch.Tensor:
    """Body-frame hip offsets [left, right] for G1-ish geometry (y>0 = left)."""
    return torch.tensor([[0.0, 0.10, -0.80], [0.0, -0.10, -0.80]])


# ----- raibert_target -------------------------------------------------------


def test_raibert_target_zero_velocity(planner):
    root_pos = torch.tensor([[0.0, 0.0, 0.80]])
    root_vel = torch.tensor([[0.0, 0.0, 0.0]])
    yaw = torch.tensor([0.0])
    vel_cmd_b = torch.tensor([[0.0, 0.0, 0.0]])

    target = planner.raibert_target(
        root_pos_w=root_pos,
        root_lin_vel_w=root_vel,
        root_yaw=yaw,
        vel_cmd_b=vel_cmd_b,
        hip_offset_b=_hip_offsets(),
        t_swing=0.3,
        k_fb=0.05,
    )

    assert target.shape == (1, 2, 3)
    # zero velocity → target xy under hip (root_xy + hip_offset_b xy, yaw=0)
    assert torch.allclose(target[0, 0, :2], torch.tensor([0.0, 0.10]), atol=1e-6)
    assert torch.allclose(target[0, 1, :2], torch.tensor([0.0, -0.10]), atol=1e-6)
    # z is a placeholder (filled by projection); contract is z=0
    assert torch.allclose(target[..., 2], torch.zeros(1, 2), atol=1e-6)


def test_raibert_target_forward_velocity(planner):
    """Forward velocity command with matching actual velocity → no feedback term,
    target sits 0.5 * t_swing * v_des ahead of hip in body x."""
    root_pos = torch.tensor([[0.0, 0.0, 0.80]])
    vel_des_b = torch.tensor([[0.5, 0.0, 0.0]])
    root_vel = torch.tensor([[0.5, 0.0, 0.0]])
    yaw = torch.tensor([0.0])
    t_swing = 0.3

    target = planner.raibert_target(
        root_pos_w=root_pos,
        root_lin_vel_w=root_vel,
        root_yaw=yaw,
        vel_cmd_b=vel_des_b,
        hip_offset_b=_hip_offsets(),
        t_swing=t_swing,
        k_fb=0.05,
    )

    expected_x = 0.5 * t_swing * 0.5  # 0.075
    assert torch.allclose(target[0, 0, 0], torch.tensor(expected_x), atol=1e-6)
    assert torch.allclose(target[0, 1, 0], torch.tensor(expected_x), atol=1e-6)
    assert torch.allclose(target[0, 0, 1], torch.tensor(0.10), atol=1e-6)
    assert torch.allclose(target[0, 1, 1], torch.tensor(-0.10), atol=1e-6)


def test_raibert_target_yaw_rotation(planner):
    """Yaw=pi/2: body-x command should map to world-y."""
    root_pos = torch.tensor([[0.0, 0.0, 0.80]])
    vel_des_b = torch.tensor([[0.5, 0.0, 0.0]])
    root_vel = torch.tensor([[0.0, 0.5, 0.0]])  # world-y, matches body-x rotated
    yaw = torch.tensor([torch.pi / 2])
    t_swing = 0.3

    target = planner.raibert_target(
        root_pos_w=root_pos,
        root_lin_vel_w=root_vel,
        root_yaw=yaw,
        vel_cmd_b=vel_des_b,
        hip_offset_b=_hip_offsets(),
        t_swing=t_swing,
        k_fb=0.05,
    )

    # body-x = 0.075 ahead → world dx≈0, dy≈+0.075
    # hip body-y = +0.10 (left) → world dx = -0.10, dy = 0
    # left hip world = root + (-0.10, 0) = (-0.10, 0)
    # left target xy = (-0.10, 0.075)
    assert torch.allclose(target[0, 0, 0], torch.tensor(-0.10), atol=1e-5)
    assert torch.allclose(target[0, 0, 1], torch.tensor(0.075), atol=1e-5)
    # right hip body-y = -0.10 → world dx = +0.10
    assert torch.allclose(target[0, 1, 0], torch.tensor(0.10), atol=1e-5)
    assert torch.allclose(target[0, 1, 1], torch.tensor(0.075), atol=1e-5)


# ----- project_onto_elevation_map ------------------------------------------


def test_project_onto_elevation_map_flat(planner):
    target = torch.tensor([[[0.0, 0.0, 99.0], [0.1, 0.0, 99.0]]])
    height_map = torch.zeros(1, 21, 33)
    map_origin = torch.tensor([[-0.5, -0.5]])

    projected = planner.project_onto_elevation_map(
        target_w=target,
        height_map=height_map,
        map_origin_w=map_origin,
        resolution=0.05,
        safety_radius=0,
    )

    assert projected.shape == (1, 2, 3)
    # xy preserved
    assert torch.allclose(projected[..., :2], target[..., :2])
    # z snapped to map height (0)
    assert torch.allclose(projected[..., 2], torch.zeros(1, 2), atol=1e-6)


def test_project_onto_elevation_map_step(planner):
    """Map split at x=0: low (z=0) for x<0, high (z=0.2) for x>=0.
    With safety_radius=2, the 5x5 median window robustly picks the dominant
    side instead of jumping at the edge.
    """
    B, H, W = 1, 21, 21
    resolution = 0.05
    map_origin = torch.tensor([[-0.5, -0.5]])
    height_map = torch.zeros(B, H, W)
    for ix in range(W):
        x_w = -0.5 + ix * resolution
        if x_w >= 0.0:
            height_map[:, :, ix] = 0.20

    # Targets: one deep in the low patch, one just inside (1 cell from edge)
    target = torch.tensor([[[-0.20, 0.0, 99.0], [-0.05, 0.0, 99.0]]])
    projected = planner.project_onto_elevation_map(
        target_w=target,
        height_map=height_map,
        map_origin_w=map_origin,
        resolution=resolution,
        safety_radius=2,
    )

    # Deep in low patch → all 25 cells are 0 → median = 0
    assert torch.allclose(projected[0, 0, 2], torch.tensor(0.0), atol=1e-6)
    # 1 cell inside low: 5x5 window covers 3 low columns + 2 high columns
    # = 15 cells at 0 + 10 cells at 0.2 → median (13th sorted ascending) = 0
    assert torch.allclose(projected[0, 1, 2], torch.tensor(0.0), atol=1e-6)


def test_project_onto_elevation_map_safety_radius_zero(planner):
    """With safety_radius=0 we sample the nearest cell exactly — useful as
    an escape hatch for callers that have already done their own filtering."""
    B, H, W = 1, 5, 5
    resolution = 0.10
    map_origin = torch.tensor([[0.0, 0.0]])
    height_map = torch.zeros(B, H, W)
    height_map[0, 2, 3] = 0.50  # one cell with a tall spike

    target = torch.tensor([[[0.30, 0.20, 99.0], [0.00, 0.00, 99.0]]])  # ix=3, iy=2  and  ix=0, iy=0
    projected = planner.project_onto_elevation_map(
        target_w=target,
        height_map=height_map,
        map_origin_w=map_origin,
        resolution=resolution,
        safety_radius=0,
    )
    assert torch.allclose(projected[0, 0, 2], torch.tensor(0.50), atol=1e-6)
    assert torch.allclose(projected[0, 1, 2], torch.tensor(0.0), atol=1e-6)


# ----- swing_trajectory ----------------------------------------------------


def test_swing_trajectory_endpoints(planner):
    start = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    end = torch.tensor([[[0.3, 0.0, 0.1], [0.3, -0.2, 0.0]]])
    apex = 0.10

    p0 = torch.zeros(1, 2)
    out0 = planner.swing_trajectory(start, end, p0, apex=apex)
    assert torch.allclose(out0, start, atol=1e-6)

    p1 = torch.ones(1, 2)
    out1 = planner.swing_trajectory(start, end, p1, apex=apex)
    assert torch.allclose(out1, end, atol=1e-6)


def test_swing_trajectory_apex(planner):
    start = torch.tensor([[[0.0, 0.0, 0.0]]])
    end = torch.tensor([[[1.0, 0.0, 0.0]]])
    apex = 0.10

    p_mid = torch.tensor([[0.5]])
    out_mid = planner.swing_trajectory(start, end, p_mid, apex=apex)
    # xy at midpoint, z = baseline (0) + apex
    assert torch.allclose(out_mid[0, 0], torch.tensor([0.5, 0.0, apex]), atol=1e-6)

    # z monotonically increases on [0, 0.5] and decreases on [0.5, 1]
    phases = torch.linspace(0, 1, 11).unsqueeze(0)         # (1, 11)
    starts = start.expand(1, 11, 3)
    ends = end.expand(1, 11, 3)
    out = planner.swing_trajectory(starts, ends, phases, apex=apex)
    z = out[0, :, 2]
    assert torch.argmax(z).item() == 5  # peak at phase=0.5
    assert (z[1:6] > z[:5]).all()       # strictly increasing up to apex
    assert (z[6:] < z[5:-1]).all()      # strictly decreasing after apex


def test_swing_trajectory_zero_velocity_at_endpoints(planner):
    """C²-smooth: foot velocity at p=0 (takeoff) and p=1 (touchdown) must
    be near zero, both horizontally and vertically. Finite-difference
    approximation of d/dp of the trajectory at each endpoint."""
    start = torch.tensor([[[0.0, 0.0, 0.0]]])
    end = torch.tensor([[[0.6, 0.0, 0.0]]])         # 0.6m forward step
    apex = 0.10
    eps = 1e-3

    p0 = torch.tensor([[0.0]])
    p1 = torch.tensor([[eps]])
    vel_start = (planner.swing_trajectory(start, end, p1, apex=apex)
                 - planner.swing_trajectory(start, end, p0, apex=apex)) / eps
    assert torch.allclose(vel_start[0, 0], torch.zeros(3), atol=0.02), \
        f"foot velocity at takeoff should be ~0, got {vel_start[0, 0].tolist()}"

    p9 = torch.tensor([[1.0 - eps]])
    p10 = torch.tensor([[1.0]])
    vel_end = (planner.swing_trajectory(start, end, p10, apex=apex)
               - planner.swing_trajectory(start, end, p9, apex=apex)) / eps
    assert torch.allclose(vel_end[0, 0], torch.zeros(3), atol=0.02), \
        f"foot velocity at touchdown should be ~0, got {vel_end[0, 0].tolist()}"


def test_swing_trajectory_endpoint_height_difference(planner):
    """Step-down case: end is below start. Apex still sits above the higher
    endpoint and trajectory passes through both."""
    start = torch.tensor([[[0.0, 0.0, 0.20]]])
    end = torch.tensor([[[0.3, 0.0, 0.00]]])
    apex = 0.10

    p_mid = torch.tensor([[0.5]])
    out_mid = planner.swing_trajectory(start, end, p_mid, apex=apex)
    # linear midpoint z is 0.10; +apex bump = 0.20
    assert torch.allclose(out_mid[0, 0, 2], torch.tensor(0.20), atol=1e-6)

    # endpoints exact
    assert torch.allclose(
        planner.swing_trajectory(start, end, torch.zeros(1, 1), apex=apex), start, atol=1e-6
    )
    assert torch.allclose(
        planner.swing_trajectory(start, end, torch.ones(1, 1), apex=apex), end, atol=1e-6
    )
