"""3D-LIPM / ICP nominal foothold generation (Chen Long thesis §4.3.2, flat ground).

Pure PyTorch — no Isaac Sim imports. Used by :class:`FootstepPlanCommand` when
``use_lipm_prior=True`` to replace Raibert + phantom + landing-horizon stacking.

**Scope:** Flat-ground eq. (4-5)–(4-11) only. Nominal xy is then projected onto
rough terrain by ``select_foothold_v2``. This is **not** the full stair-sequence
planner from thesis §4.3.2 (tread margins, per-level foothold count, eq. 4-12–4-17).

**Timing (paper):** Footholds are computed once per footstep **start** (``t=0``),
so ``remaining_delta_T = T_s - t = T_s``. ``predict_icp_end`` uses the full step
duration ``T_s``; offset ``b`` uses ``remaining_delta_T``.
"""

from __future__ import annotations

import torch

__all__ = [
    "omega0_from_com_height",
    "icp_xy",
    "predict_icp_end",
    "lipm_step_length",
    "lipm_offset_b",
    "lipm_nominal_foothold_xy",
    "validate_lipm_planner_config",
]


def validate_lipm_planner_config(*, use_lipm_prior: bool, n_future_steps: int) -> None:
    """Guard config: LIPM commit path plans one foothold per step-start only."""
    if use_lipm_prior and n_future_steps > 1:
        raise ValueError(
            "use_lipm_prior requires n_future_steps=1: the paper computes one foothold "
            "at each step start (t=0); multi-step recurrence is not implemented yet."
        )


def omega0_from_com_height(z0: torch.Tensor, g: float = 9.81) -> torch.Tensor:
    """Natural frequency ω₀ = √(g / z₀). ``z0`` may be scalar or ``(B,)``."""
    z = z0 if z0.dim() > 0 else z0.unsqueeze(0)
    return torch.sqrt(g / z.clamp_min(1e-3))


def icp_xy(com_xy: torch.Tensor, com_vel_xy: torch.Tensor, omega0: torch.Tensor) -> torch.Tensor:
    """Instantaneous capture point ξ = x + ẋ / ω₀. Shapes ``(B, 2)``."""
    w = omega0.unsqueeze(-1) if omega0.dim() == 1 else omega0
    return com_xy + com_vel_xy / w


def predict_icp_end(
    xi0: torch.Tensor,
    stance_foot_xy: torch.Tensor,
    omega0: torch.Tensor,
    step_duration_ts: float | torch.Tensor,
) -> torch.Tensor:
    """Predict ICP at **end of full step** — thesis eq. (4-7).

    Uses the full stride period ``T_s`` (not remaining ``δT``):

        ξ_f = e^{ω₀ T_s} ξ₀ + (1 − e^{ω₀ T_s}) p_stance
    """
    ts = (
        step_duration_ts
        if isinstance(step_duration_ts, torch.Tensor)
        else torch.tensor(step_duration_ts, device=xi0.device, dtype=xi0.dtype)
    )
    exp_ts = torch.exp(omega0 * ts)
    if exp_ts.dim() == 0:
        exp_ts = exp_ts.unsqueeze(0)
    e = exp_ts.unsqueeze(-1)
    return e * xi0 + (1.0 - e) * stance_foot_xy


def lipm_step_length(
    vel_cmd_xy: torch.Tensor,
    remaining_delta_t: float | torch.Tensor,
    step_width: float,
    step_duration_ts: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected step length and width — thesis eq. (4-5)(4-6).

    sd = |v| · δT        where δT = T_s − t (at step start δT = T_s)
    wd = |w| · δT / T_s
    """
    dt = (
        remaining_delta_t
        if isinstance(remaining_delta_t, torch.Tensor)
        else torch.tensor(remaining_delta_t, device=vel_cmd_xy.device, dtype=vel_cmd_xy.dtype)
    )
    sd = torch.linalg.vector_norm(vel_cmd_xy, dim=-1) * dt
    wd = abs(step_width) * dt / step_duration_ts
    return sd, wd


def lipm_offset_b(
    sd: torch.Tensor,
    wd: torch.Tensor,
    remaining_delta_t: float | torch.Tensor,
    omega0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ICP-to-foothold offset b — thesis eq. (4-9).

    b_x = sd / (e^{ω₀ δT} − 1)
    b_y = wd / (e^{ω₀ δT} + 1)
    """
    dt = (
        remaining_delta_t
        if isinstance(remaining_delta_t, torch.Tensor)
        else torch.tensor(remaining_delta_t, device=sd.device, dtype=sd.dtype)
    )
    exp_dt = torch.exp(omega0 * dt)
    denom_pos = (exp_dt - 1.0).clamp_min(1e-4)
    denom_neg = exp_dt + 1.0
    return sd / denom_pos, wd / denom_neg


def _side_sign(swing_foot: torch.Tensor) -> torch.Tensor:
    """``(-1)^n``: left (0) → +1, right (1) → −1."""
    return torch.where(swing_foot == 0, 1.0, -1.0)


def lipm_nominal_foothold_xy(
    com_xy: torch.Tensor,
    com_vel_xy: torch.Tensor,
    stance_foot_xy: torch.Tensor,
    swing_foot_xy: torch.Tensor,
    vel_cmd_xy: torch.Tensor,
    swing_foot: torch.Tensor,
    *,
    remaining_delta_t: float | torch.Tensor,
    step_duration_ts: float,
    com_height: float | torch.Tensor,
    step_width: float = 0.24,
    use_turning: bool = True,
    g: float = 9.81,
) -> torch.Tensor:
    """Nominal swing-foot xy in world frame — thesis eq. (4-10)/(4-11).

    Args:
        com_xy: CoM / root xy at step start, ``(B, 2)``.
        com_vel_xy: CoM / root linear velocity xy, ``(B, 2)``.
        stance_foot_xy: Opposite (stance) foot xy at last touchdown, ``(B, 2)``.
        swing_foot_xy: Swing foot xy at last touchdown (under-hip prior), ``(B, 2)``.
        vel_cmd_xy: Commanded velocity xy in **world** frame, ``(B, 2)``.
        swing_foot: ``(B,)`` long — 0=left, 1=right.
        remaining_delta_t: δT = T_s − t; at step start ``t=0`` → ``T_s``.
        step_duration_ts: Full stride period T_s (for ``ξ_f`` in eq. 4-7).
        com_height: Inverted-pendulum height z₀ (metres).
        step_width: Nominal step-width parameter w (metres).
        use_turning: If True, apply eq. (4-11) rotation; else eq. (4-10).

    Returns:
        ``(B, 2)`` nominal foothold xy (z handled by terrain selector).
    """
    B = com_xy.shape[0]
    z0 = (
        torch.full((B,), com_height, device=com_xy.device, dtype=com_xy.dtype)
        if isinstance(com_height, (int, float))
        else com_height
    )
    omega0 = omega0_from_com_height(z0)

    xi0 = icp_xy(com_xy, com_vel_xy, omega0)
    xi_f = predict_icp_end(xi0, stance_foot_xy, omega0, step_duration_ts)
    sd, wd = lipm_step_length(vel_cmd_xy, remaining_delta_t, step_width, step_duration_ts)
    bx, by = lipm_offset_b(sd, wd, remaining_delta_t, omega0)

    side = _side_sign(swing_foot)
    speed = torch.linalg.vector_norm(vel_cmd_xy, dim=-1)
    stall = speed < 1e-3

    if use_turning:
        theta = torch.atan2(vel_cmd_xy[:, 1], vel_cmd_xy[:, 0])
        off_x = -bx
        off_y = side * by
        c, s = torch.cos(theta), torch.sin(theta)
        rot_x = c * off_x - s * off_y
        rot_y = s * off_x + c * off_y
        p_hat = xi_f + torch.stack([rot_x, rot_y], dim=-1)
    else:
        p_hat = torch.stack([xi_f[:, 0] - bx, xi_f[:, 1] + side * by], dim=-1)

    # Zero-velocity degeneracy: keep swing foot at its last contact (lateral
    # separation preserved), not the stance foot (would collapse step width).
    p_hat = torch.where(stall.unsqueeze(-1), swing_foot_xy, p_hat)
    return p_hat
