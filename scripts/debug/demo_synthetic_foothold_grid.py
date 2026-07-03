#!/usr/bin/env python3
"""Synthetic height-map demo for Planner V2 foothold pipeline (no Isaac Sim).

Builds a fake 0.05 m yaw-aligned scanner grid, generates 5×5 candidates around
a Raibert prior, runs ``foot_patch_stats`` + hard filters, and prints/plots the
best valid candidate (demo scorer until ``select_foothold_v2`` lands).

Usage::

    .AME/bin/python scripts/debug/demo_synthetic_foothold_grid.py
    .AME/bin/python scripts/debug/demo_synthetic_foothold_grid.py --scenario step --foot left
    .AME/bin/python scripts/debug/demo_synthetic_foothold_grid.py --scenario slope --yaw-deg 30 \\
        --out scripts/debug/output/synthetic_foothold.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Repo root on path for optional package import; we load mdp modules directly below.
_REPO = Path(__file__).resolve().parents[2]
_MDP = _REPO / "source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp"

if str(_REPO / "source/ame_locomotion") not in sys.path:
    sys.path.insert(0, str(_REPO / "source/ame_locomotion"))


def _import_mdp():
    """Import planner V2 mdp helpers (installed package or source tree)."""
    try:
        from ame_locomotion.tasks.manager_based.ame_locomotion.mdp import (
            foot_geometry_constants as fgc,
            foothold_candidates,
            foothold_geometry,
            foothold_selection,
            gridmap_utils,
        )
        return fgc, foothold_candidates, foothold_geometry, foothold_selection, gridmap_utils
    except ImportError:
        import importlib.util
        import types

        pkg_name = "demo_mdp"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(_MDP)]
            sys.modules[pkg_name] = pkg

        mods = {}
        for name in (
            "foot_geometry_constants",
            "gridmap_utils",
            "planner",
            "foothold_candidates",
            "foothold_geometry",
            "foothold_selection",
        ):
            full = f"{pkg_name}.{name}"
            path = _MDP / f"{name}.py"
            spec = importlib.util.spec_from_file_location(full, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
            mods[name] = mod
        return (
            mods["foot_geometry_constants"],
            mods["foothold_candidates"],
            mods["foothold_geometry"],
            mods["foothold_selection"],
            mods["gridmap_utils"],
        )


def build_height_grid(
    scenario: str,
    grid_shape: tuple[int, int],
    resolution: float,
) -> torch.Tensor:
    """Return ``(H, W)`` terrain z in metres."""
    h, w = grid_shape
    xs = torch.linspace(-resolution * (w - 1) / 2, resolution * (w - 1) / 2, w)
    ys = torch.linspace(-resolution * (h - 1) / 2, resolution * (h - 1) / 2, h)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")

    if scenario == "flat":
        z = torch.zeros_like(gx)
    elif scenario == "step":
        # Step riser at x = 0.10 m (world +x), 15 cm jump.
        z = torch.where(gx >= 0.10, torch.full_like(gx, 0.15), torch.zeros_like(gx))
    elif scenario == "slope":
        z = 0.12 * gx
    elif scenario == "hole":
        # 20 cm depression in a 0.25 m square ahead of the robot.
        in_hole = (gx > 0.05) & (gx < 0.30) & (gy.abs() < 0.125)
        z = torch.where(in_hole, torch.full_like(gx, -0.20), torch.zeros_like(gx))
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return z


def rays_from_grid(z_grid: torch.Tensor, resolution: float) -> torch.Tensor:
    """``(1, H*W, 3)`` ray hits, row-major (IsaacLab ordering)."""
    h, w = z_grid.shape
    xs = torch.linspace(-resolution * (w - 1) / 2, resolution * (w - 1) / 2, w)
    ys = torch.linspace(-resolution * (h - 1) / 2, resolution * (h - 1) / 2, h)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    flat = torch.stack([gx, gy, z_grid], dim=-1).reshape(1, h * w, 3)
    return flat


def terrain_to_foot_body(xyz_w: torch.Tensor, sole_z_offset: float) -> torch.Tensor:
    out = xyz_w.clone()
    out[..., 2] = out[..., 2] - sole_z_offset
    return out


def demo_pick(
    candidates_fb: torch.Tensor,
    raibert_xy: torch.Tensor,
    in_bounds: torch.Tensor,
    valid_hard: torch.Tensor,
    dz_omega: torch.Tensor,
    support_ratio: torch.Tensor,
    *,
    w_rough: float = 2.0,
    w_support: float = 1.0,
    w_nominal: float = 0.5,
) -> tuple[int, torch.Tensor]:
    """Demo scorer (lower is better). Returns ``(best_k, score_k)`` with shape ``(K,)``."""
    k = candidates_fb.shape[-2]
    cand_xy = candidates_fb.reshape(-1, k, 3)[0, :, :2]
    rb = raibert_xy.reshape(-1, 2)[0]
    d2 = ((cand_xy - rb) ** 2).sum(dim=-1)
    dz = dz_omega.reshape(-1, k)[0]
    sup = support_ratio.reshape(-1, k)[0]
    bounds_f = in_bounds.reshape(-1, k)[0]
    valid_f = valid_hard.reshape(-1, k)[0]
    score = w_rough * dz + w_support * (1.0 - sup) + w_nominal * d2
    score = torch.where(bounds_f & valid_f, score, torch.full_like(score, float("inf")))
    return int(score.argmin().item()), score


def run_demo(args: argparse.Namespace) -> int:
    fgc, fc, fg, fs, _ = _import_mdp()

    res = args.resolution
    grid_shape = (args.grid_h, args.grid_w)
    h, w = grid_shape

    z_grid = build_height_grid(args.scenario, grid_shape, res)
    ray_hits_w = rays_from_grid(z_grid, res)

    body_pos_w = torch.tensor([[0.0, 0.0, args.body_z]])
    body_yaw = torch.tensor([math.radians(args.yaw_deg)])
    grid_center_w = body_pos_w[:, :2]

    # Raibert priors: one step ahead in body +x, left/right hip y.
    yaw = body_yaw.item()
    c, s = math.cos(yaw), math.sin(yaw)
    fwd = args.step_x
    y_hip = args.hip_y
    raibert_xy = torch.tensor(
        [
            [
                [c * fwd - s * (+y_hip), s * fwd + c * (+y_hip)],
                [c * fwd - s * (-y_hip), s * fwd + c * (-y_hip)],
            ]
        ]
    )
    raibert_xyz = torch.zeros(1, 2, 3)
    raibert_xyz[..., :2] = raibert_xy

    candidates_w, in_bounds = fc.generate_candidates(
        raibert_xyz,
        ray_hits_w,
        grid_center_w,
        body_yaw,
        grid_shape=grid_shape,
        grid_resolution=res,
        half_width_cells=2,
    )
    candidates_fb = terrain_to_foot_body(candidates_w, fgc.G1_SOLE_Z_OFFSET)

    foot_idx = {"left": 0, "right": 1, "both": None}[args.foot]
    feet = [foot_idx] if foot_idx is not None else [0, 1]
    foot_names = ("left", "right")

    stance_terrain_z = torch.zeros(1, 2)  # both feet on z=0 terrain at start

    print(f"scenario={args.scenario}  grid={grid_shape} @ {res} m  yaw={args.yaw_deg:.1f}°")
    print(f"body_pos_w={body_pos_w[0].tolist()}  grid_center={grid_center_w[0].tolist()}")
    print()

    fig, axes = plt.subplots(1, len(feet), figsize=(5 * len(feet), 5), squeeze=False)

    for ax_i, f in enumerate(feet):
        cand = candidates_fb[:, f]  # (1, K, 3)
        cand_w = candidates_w[:, f]
        rb = raibert_xy[:, f]
        bounds = in_bounds[:, f]
        M = cand.shape[1]

        stats = fg.foot_patch_stats(
            cand,
            body_yaw.unsqueeze(0).expand(1, M),
            ray_hits_w,
            foot_length=fgc.G1_FOOT_LENGTH,
            foot_width=fgc.G1_FOOT_WIDTH,
            n_long=fgc.G1_FOOT_N_LONG,
            n_lat=fgc.G1_FOOT_N_LAT,
            support_threshold=0.03,
            grid_shape=grid_shape,
            grid_resolution=res,
            grid_center_w=grid_center_w,
            grid_yaw=body_yaw,
        )
        dz = stats["dz_omega"]
        support = stats["support_ratio"]

        reach = fs.reachability_mask(candidates_fb, body_pos_w, body_yaw)[:, f]
        step_m = fs.step_height_mask(candidates_fb, stance_terrain_z)[:, f]
        rough_m = fs.roughness_cap_mask(dz)
        valid = reach & step_m & rough_m & bounds

        bk, scores = demo_pick(
            candidates_fb[:, f : f + 1],
            raibert_xy[:, f : f + 1],
            bounds.unsqueeze(0),
            valid.unsqueeze(0),
            dz,
            support,
        )

        print(f"=== {foot_names[f]} foot  (Raibert xy = {rb[0].tolist()}) ===")
        print(
            f"{'k':>2}  {'xy_w':>22}  {'terr_z':>7}  "
            f"{'in_b':>4} {'reach':>5} {'step':>4} {'rough':>5}  "
            f"{'dz_Ω':>6} {'supp':>5}  {'score':>7}"
        )
        for k in range(M):
            xy = cand_w[0, k, :2].tolist()
            tz = cand_w[0, k, 2].item()
            sc = scores[k].item()
            sc_s = "inf" if math.isinf(sc) else f"{sc:7.3f}"
            mark = " <-- SELECTED" if k == bk and valid[0, k] else (" <-- fallback?" if k == bk else "")
            print(
                f"{k:2d}  ({xy[0]:+6.3f},{xy[1]:+6.3f})  {tz:7.3f}  "
                f"{int(bounds[0,k]):4d} {int(reach[0,k]):5d} {int(step_m[0,k]):4d} "
                f"{int(rough_m[0,k]):5d}  {dz[0,k]:6.3f} {support[0,k]:5.2f}  {sc_s}{mark}"
            )
        if not valid[0].any():
            print("  WARNING: no candidate passed all hard filters.")
        elif math.isinf(scores[bk].item()):
            print("  WARNING: selected index failed hard filters (check fallback logic).")
        else:
            sel = cand_w[0, bk]
            print(
                f"  → selected terrain xyz = ({sel[0]:+.3f}, {sel[1]:+.3f}, {sel[2]:+.3f})  "
                f"foot-body z = {candidates_fb[0, f, bk, 2].item():+.3f}"
            )
        print()

        ax = axes[0, ax_i]
        xs = ray_hits_w[0, :, 0].reshape(h, w).numpy()
        ys = ray_hits_w[0, :, 1].reshape(h, w).numpy()
        zz = z_grid.numpy()
        im = ax.pcolormesh(xs, ys, zz, shading="auto", cmap="terrain")
        plt.colorbar(im, ax=ax, label="terrain z (m)")

        xy_c = cand_w[0, :, :2].numpy()
        ok = valid[0].numpy()
        ax.scatter(
            xy_c[~ok, 0], xy_c[~ok, 1], c="red", s=40, marker="x", label="filtered out", zorder=4
        )
        ax.scatter(
            xy_c[ok, 0], xy_c[ok, 1], facecolors="none", edgecolors="lime", s=80, linewidths=1.5,
            label="valid", zorder=5,
        )
        ax.scatter(
            rb[0, 0].item(), rb[0, 1].item(), c="cyan", marker="+", s=120, linewidths=2,
            label="Raibert", zorder=6,
        )
        if valid[0].any() or bk < M:
            ax.scatter(
                xy_c[bk, 0], xy_c[bk, 1], c="gold", marker="*", s=200, edgecolors="k",
                label="selected", zorder=7,
            )
        ax.set_title(f"{foot_names[f]} — {args.scenario}")
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote plot to {out}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenario",
        choices=("flat", "step", "slope", "hole"),
        default="step",
        help="Synthetic terrain shape",
    )
    p.add_argument("--resolution", type=float, default=0.05, help="Grid cell size (m)")
    p.add_argument("--grid-h", type=int, default=21, help="Scanner grid rows (y)")
    p.add_argument("--grid-w", type=int, default=33, help="Scanner grid cols (x)")
    p.add_argument("--yaw-deg", type=float, default=0.0, help="Robot / grid yaw (degrees)")
    p.add_argument("--body-z", type=float, default=0.80, help="Pelvis height (m)")
    p.add_argument("--step-x", type=float, default=0.20, help="Raibert prior forward offset in body x (m)")
    p.add_argument("--hip-y", type=float, default=0.12, help="Hip lateral offset magnitude (m)")
    p.add_argument("--foot", choices=("left", "right", "both"), default="both")
    p.add_argument(
        "--out",
        type=str,
        default="scripts/debug/output/synthetic_foothold.png",
    )
    p.add_argument("--show", action="store_true", help="Open matplotlib window")
    return run_demo(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
