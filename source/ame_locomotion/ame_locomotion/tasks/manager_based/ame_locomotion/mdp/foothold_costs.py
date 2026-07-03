"""Soft cost terms for Planner V2 foothold selection (Isaac-Sim-free)."""

from __future__ import annotations

import torch

from .foot_geometry_constants import G1_SOLE_Z_OFFSET

__all__ = [
    "cost_edge",
    "cost_height",
    "cost_nominal",
    "cost_reach",
    "cost_slope",
    "cost_terrain",
]


def _world_to_body(pos_w: torch.Tensor, body_pos_w: torch.Tensor, body_yaw: torch.Tensor) -> torch.Tensor:
    from .foothold_selection import _world_to_body as _wtb

    return _wtb(pos_w, body_pos_w, body_yaw)


def cost_nominal(
    candidates_w: torch.Tensor,
    raibert_xy_w: torch.Tensor,
    *,
    sigma_nominal: float = 0.10,
) -> torch.Tensor:
    """``||candidate_xy - raibert_xy||² / σ²``."""
    diff = candidates_w[..., :2] - raibert_xy_w.unsqueeze(-2)
    d2 = (diff * diff).sum(dim=-1)
    return d2 / (sigma_nominal * sigma_nominal)


def cost_reach(
    candidates_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_yaw: torch.Tensor,
    *,
    comfort_center_b: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, 0.12, -0.78),
        (0.0, -0.12, -0.78),
    ),
    s_reach: float = 0.15,
) -> torch.Tensor:
    """Quadratic distance from per-foot body-frame comfort pose."""
    pos_b = _world_to_body(candidates_w, body_pos_w, body_yaw)
    device, dtype = candidates_w.device, candidates_w.dtype
    left = torch.tensor(comfort_center_b[0], device=device, dtype=dtype)
    right = torch.tensor(comfort_center_b[1], device=device, dtype=dtype)
    centers = torch.stack([left, right], dim=0).view(1, 2, 1, 3)
    diff = pos_b - centers
    d2 = (diff * diff).sum(dim=-1)
    return d2 / (s_reach * s_reach)


def cost_terrain(
    dz_omega: torch.Tensor,
    *,
    sigma_rough: float = 0.05,
) -> torch.Tensor:
    """Patch roughness penalty: ``(dz_omega / σ)²``."""
    return (dz_omega / sigma_rough) ** 2


def cost_edge(
    support_ratio: torch.Tensor,
    overhang_ratio: torch.Tensor,
    *,
    lambda_support: float = 1.0,
    lambda_overhang: float = 2.0,
) -> torch.Tensor:
    """Poor support + overhang: ``λ_s(1-s) + λ_o·overhang``."""
    return lambda_support * (1.0 - support_ratio) + lambda_overhang * overhang_ratio


def cost_height(
    candidates_w: torch.Tensor,
    stance_terrain_z_w: torch.Tensor,
    *,
    sigma_h: float = 0.10,
    sole_z_offset: float = G1_SOLE_Z_OFFSET,
) -> torch.Tensor:
    """Quadratic step-height penalty on terrain/sole z.

    ``candidates_w[..., 2]`` is foot-body target z; terrain z is recovered via
    ``+ sole_z_offset`` (same convention as :func:`step_height_mask`).
    """
    candidate_terrain_z = candidates_w[..., 2] + sole_z_offset
    dz = (candidate_terrain_z - stance_terrain_z_w.unsqueeze(-1)).abs()
    return (dz / sigma_h) ** 2


def cost_slope(
    slope_angle: torch.Tensor,
    *,
    sigma_theta_rad: float = 0.2094,
) -> torch.Tensor:
    """Tilt penalty from plane-fit normal: ``(θ / σ)²``."""
    return (slope_angle / sigma_theta_rad) ** 2
