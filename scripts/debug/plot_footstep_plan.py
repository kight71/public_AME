#!/usr/bin/env python3
"""Plot footstep-plan dump from ``debug_footstep_plan.py`` and run sanity checks.

Usage::

    .AME/bin/python scripts/debug/plot_footstep_plan.py debug_plan.npz
    .AME/bin/python scripts/debug/plot_footstep_plan.py debug_plan.npz --out scripts/debug/output/footstep_xy.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _sanity_checks(data: dict[str, np.ndarray]) -> list[str]:
    """Return human-readable PASS/FAIL lines."""
    lines: list[str] = []
    plan = data["plan_buffer"]          # (T, B, N, 2, 3)
    target = data["target_w"]           # (T, B, 2, 3)
    foot = data["foot_pos_w"]           # (T, B, 2, 3)
    swing = data["swing_foot"]          # (T, B) or (T,)
    if swing.ndim == 1:
        swing = swing[:, None]
    phase = data["phase"]               # (T, B)
    vcmd = data["vel_cmd_b"]            # (T, B, 3)

    ok = np.isfinite(plan).all() and np.isfinite(target).all()
    lines.append(f"finite plan/target: {'PASS' if ok else 'FAIL'}")

    # GUI markers use target_w; after each commit target_w == plan_buffer[:, 0].
    marker_match = np.allclose(target, plan[:, :, 0], atol=1e-4, rtol=0.0)
    lines.append(f"target_w matches plan_buffer[:,0] (GUI marker source): {'PASS' if marker_match else 'FAIL'}")

    # Swing foot should flip multiple times over a long run.
    flips = np.sum(np.diff(swing[:, 0]) != 0)
    lines.append(f"swing_foot alternations (env0): {flips} -> {'PASS' if flips >= 3 else 'FAIL'}")

    # Phase wraps in [0, 1).
    phase_ok = (phase.min() >= -1e-4) and (phase.max() <= 1.0 + 1e-4)
    lines.append(f"phase in [0,1): {'PASS' if phase_ok else 'FAIL'}")

    # Zero-action dumps (Phase 0) do not move the robot even when vx_cmd=1.0.
    # Check planner places targets ahead of root instead of root displacement.
    root = data["root_pos"]
    dx = root[-1, 0, 0] - root[0, 0, 0]
    vx_cmd_mean = vcmd[:, 0, 0].mean()
    plan_center_x = plan[:, 0, 0, :, 0].mean(axis=-1)  # (T,) mean L/R plan x
    plan_lead_x = (plan_center_x - root[:, 0, 0]).mean()
    if abs(vx_cmd_mean) > 0.1:
        planner_ahead = plan_lead_x > 0.05
        lines.append(
            f"planner lead vs root (mean plan_x - root_x) = {plan_lead_x:.3f} m "
            f"(vx_cmd={vx_cmd_mean:.2f}): {'PASS' if planner_ahead else 'FAIL'}"
        )
    else:
        lines.append(f"planner lead vs root = {plan_lead_x:.3f} m (vx_cmd≈0, informational)")
    lines.append(
        f"root forward drift dx={dx:.3f} m over run (informational; zero-action expected ≈0)"
    )

    # Planned z should stay in a plausible band (not NaN / not sky).
    z_plan = plan[:, 0, 0, :, 2]
    z_ok = (z_plan.min() > -0.5) and (z_plan.max() < 2.0)
    lines.append(f"plan z in plausible band [{z_plan.min():.3f}, {z_plan.max():.3f}]: {'PASS' if z_ok else 'FAIL'}")

    # Feet vs plan: zero-action policy won't track, but plan should stay ahead of feet in x on average.
    plan_xy = plan[:, 0, 0, swing[:, 0], :2]
    foot_xy = foot[:, 0, swing[:, 0], :2]
    sep = np.linalg.norm(plan_xy - foot_xy, axis=-1).mean()
    lines.append(f"mean |plan_swing - foot_swing| xy = {sep:.3f} m (informational)")

    return lines


def _plot_xy(data: dict[str, np.ndarray], out: Path | None) -> None:
    root = data["root_pos"][:, 0, :2]
    plan = data["plan_buffer"][:, 0, 0]   # (T, 2, 3) first future step
    foot = data["foot_pos_w"][:, 0]       # (T, 2, 3)
    swing = data["swing_foot"]
    if swing.ndim == 1:
        swing = swing[:, None]
    swing = swing[:, 0]
    target = data["target_w"][:, 0]       # (T, 2, 3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(root[:, 0], root[:, 1], "k-", alpha=0.4, label="root xy")
    ax.plot(foot[:, 0, 0], foot[:, 0, 1], "b.", ms=2, alpha=0.5, label="left foot")
    ax.plot(foot[:, 1, 0], foot[:, 1, 1], "r.", ms=2, alpha=0.5, label="right foot")
    ax.plot(plan[:, 0, 0], plan[:, 0, 1], "c--", alpha=0.8, label="plan L (k=0)")
    ax.plot(plan[:, 1, 0], plan[:, 1, 1], "m--", alpha=0.8, label="plan R (k=0)")
    ax.plot(target[:, 0, 0], target[:, 0, 1], "b+", ms=4, alpha=0.3, label="target_w L (GUI)")
    ax.plot(target[:, 1, 0], target[:, 1, 1], "r+", ms=4, alpha=0.3, label="target_w R (GUI)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title("Top-down trajectories (env 0)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    t = np.arange(plan.shape[0])
    ax.plot(t, plan[:, 0, 2], "b-", alpha=0.8, label="plan L z")
    ax.plot(t, plan[:, 1, 2], "r-", alpha=0.8, label="plan R z")
    ax.plot(t, foot[:, 0, 2], "b:", alpha=0.5, label="foot L z")
    ax.plot(t, foot[:, 1, 2], "r:", alpha=0.5, label="foot R z")
    swing_mask = swing.astype(float)
    ax2 = ax.twinx()
    ax2.plot(t, swing_mask, "g-", alpha=0.25, label="swing_foot (0=L)")
    ax.set_xlabel("step")
    ax.set_ylabel("world z [m]")
    ax.set_title("Height + swing schedule")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Wrote plot to {out}")
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot and sanity-check footstep plan npz.")
    parser.add_argument("npz", type=str, help="Path to debug_plan.npz")
    parser.add_argument(
        "--out",
        type=str,
        default="scripts/debug/output/footstep_xy.png",
        help="Output PNG path (omit with --show for interactive window).",
    )
    parser.add_argument("--show", action="store_true", help="Open interactive matplotlib window.")
    args = parser.parse_args()

    path = Path(args.npz)
    if not path.is_file():
        print(f"[FAIL] Missing npz: {path}")
        return 1

    data = _load_npz(path)
    print(f"Loaded {path} keys={list(data.keys())} T={data['plan_buffer'].shape[0]}")

    print("\n--- sanity checks ---")
    checks = _sanity_checks(data)
    all_pass = True
    for line in checks:
        print(line)
        if line.rstrip().endswith("FAIL") and "informational" not in line:
            all_pass = False

    out = None if args.show else Path(args.out)
    _plot_xy(data, out)

    print("\n--- summary ---")
    if all_pass:
        print("offline plot checks: PASS")
        return 0
    print("offline plot checks: FAIL (see lines above)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
