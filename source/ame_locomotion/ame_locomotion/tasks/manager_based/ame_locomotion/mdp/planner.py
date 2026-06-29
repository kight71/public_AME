"""Pure-PyTorch geometric primitives for the footstep planner.

This module is intentionally Isaac-Sim-free so the math can be exercised by
unit tests without spinning up the simulator. Higher-level code in
``commands.py`` wraps these functions inside a ``CommandTerm`` that owns
per-environment state (stride phase, last touchdown, planned target).
"""

from __future__ import annotations

import torch

__all__ = ["raibert_target", "project_onto_elevation_map", "swing_trajectory"]


def _yaw_rotation(yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    zero = torch.zeros_like(yaw)
    one = torch.ones_like(yaw)
    row0 = torch.stack([c, -s, zero], dim=-1)
    row1 = torch.stack([s, c, zero], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def raibert_target(
    root_pos_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_yaw: torch.Tensor,
    vel_cmd_b: torch.Tensor,
    hip_offset_b: torch.Tensor,
    t_swing: float,
    k_fb: float = 0.0,
    raibert_factor: float = 0.5,
) -> torch.Tensor:
    """Raibert footstep heuristic.

    target_w = root_w + R(yaw) @ hip_offset_b
               + raibert_factor * t_swing * v_des_w
               + k_fb * (v_actual_w - v_des_w)

    Returns ``(B, F, 3)`` with ``z`` zeroed out; callers fill ``z`` via
    :func:`project_onto_elevation_map`.

    Args:
        root_pos_w: ``(B, 3)`` base position, world frame.
        root_lin_vel_w: ``(B, 3)`` base linear velocity, world frame.
        root_yaw: ``(B,)`` base yaw, radians.
        vel_cmd_b: ``(B, 3)`` commanded ``(vx, vy, wz)`` in body frame; ``wz``
            is currently unused (yaw rate handled by the env's command term).
        hip_offset_b: ``(F, 3)`` per-foot hip offsets in body frame.
        t_swing: single-support duration, seconds.
        k_fb: Raibert feedback gain on velocity tracking error.
        raibert_factor: scales the forward-push prediction. Classic Raibert is
            0.5 (foot lands at the predicted mid-swing hip position). Set
            lower to soften the prediction so the policy is not forced into
            aggressive push-off — empirically the pretrained AME policy walks
            with effective factor ≈ 0 (foot lands directly under the hip).
    """
    rot = _yaw_rotation(root_yaw)
    hip_offset_w = torch.einsum("bij,fj->bfi", rot, hip_offset_b)
    hip_w = root_pos_w.unsqueeze(1) + hip_offset_w

    vel_cmd_w = torch.einsum("bij,bj->bi", rot, vel_cmd_b)
    fb_w = root_lin_vel_w - vel_cmd_w
    delta = (raibert_factor * t_swing) * vel_cmd_w + k_fb * fb_w
    delta = delta.unsqueeze(1).expand_as(hip_w)

    target = hip_w + delta
    target[..., 2] = 0.0
    return target


def project_onto_elevation_map(
    target_w: torch.Tensor,
    height_map: torch.Tensor,
    map_origin_w: torch.Tensor,
    resolution: float,
    safety_radius: int = 0,
) -> torch.Tensor:
    """Snap ``target_w.z`` to the elevation map.

    Args:
        target_w: ``(B, F, 3)`` target positions, world frame. ``z`` is
            overwritten; ``xy`` is preserved.
        height_map: ``(B, H, W)`` z values, indexed as ``[b, iy, ix]``.
        map_origin_w: ``(B, 2)`` world ``xy`` of cell ``(ix=0, iy=0)``.
        resolution: cell size, metres.
        safety_radius: when ``<= 0`` use nearest-cell sample; otherwise take
            the median over a ``(2r+1)x(2r+1)`` window — robust at step edges.
    """
    B, F, _ = target_w.shape
    _, H, W = height_map.shape

    rel = target_w[..., :2] - map_origin_w.unsqueeze(1)
    idx = rel / resolution
    ix = idx[..., 0].round().long().clamp(0, W - 1)
    iy = idx[..., 1].round().long().clamp(0, H - 1)

    b_idx = torch.arange(B, device=height_map.device)
    if safety_radius <= 0:
        z = height_map[b_idx.unsqueeze(1).expand(B, F), iy, ix]
    else:
        r = safety_radius
        offsets = torch.arange(-r, r + 1, device=height_map.device)
        d_iy, d_ix = torch.meshgrid(offsets, offsets, indexing="ij")
        d_iy = d_iy.reshape(-1)
        d_ix = d_ix.reshape(-1)
        K = d_iy.numel()

        sample_iy = (iy.unsqueeze(-1) + d_iy.view(1, 1, K)).clamp(0, H - 1)
        sample_ix = (ix.unsqueeze(-1) + d_ix.view(1, 1, K)).clamp(0, W - 1)
        b_idx_expand = b_idx.view(B, 1, 1).expand(B, F, K)
        sampled = height_map[b_idx_expand, sample_iy, sample_ix]
        z = sampled.median(dim=-1).values

    out = target_w.clone()
    out[..., 2] = z
    return out


def swing_trajectory(
    start_w: torch.Tensor,
    end_w: torch.Tensor,
    phase: torch.Tensor,
    apex: float = 0.10,
) -> torch.Tensor:
    """C²-smooth swing curve with zero velocity and acceleration at endpoints.

    Uses **smootherstep** ``s(p) = 6p⁵ − 15p⁴ + 10p³`` for the xy baseline
    (and z baseline), plus ``apex · sin²(πp)`` for the vertical bump. Both
    components have

        f(0)=0, f(1)=hit endpoint    (positions exact)
        f'(0)=f'(1)=0                (zero takeoff / touchdown velocity)
        f''(0)=f''(1)=0              (zero takeoff / touchdown accel)

    A naïve linear-in-xy + parabolic-in-z trajectory (the previous version)
    had non-zero foot velocity at touchdown — telling the policy to slam its
    foot down at ~step_length / t_swing m/s. That is unphysical and was the
    likely cause of the policy refusing to track the reference precisely.

    Args:
        start_w: ``(B, F, 3)`` swing start (last touchdown).
        end_w: ``(B, F, 3)`` swing end (planned footstep).
        phase: ``(B, F)`` per-foot phase in ``[0, 1]``.
        apex: extra clearance over the linear-baseline z, metres.
    """
    p = phase.clamp(0.0, 1.0)
    s = 6.0 * p**5 - 15.0 * p**4 + 10.0 * p**3       # smootherstep
    s_xyz = s.unsqueeze(-1)                          # broadcast over xyz

    baseline = start_w + s_xyz * (end_w - start_w)   # C² interpolation
    z_bump = apex * torch.sin(torch.pi * p) ** 2      # apex bump, f'(0)=f'(1)=0

    out = baseline.clone()
    out[..., 2] = out[..., 2] + z_bump
    return out
