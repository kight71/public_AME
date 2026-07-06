#!/usr/bin/env python3
"""Plot the ideal yaw-only LIPM foothold targets.

This is a lightweight offline view of the special branch in
``FootstepPlanCommand._lipm_targets_for_horizons`` for the case where
``vx = vy = 0`` and ``|wz|`` is large.  It does not run Isaac Sim or the policy;
it just visualizes the nominal foothold positions produced by the planner math.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rot2(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def _yaw_blend(
    *,
    vx: float,
    vy: float,
    wz: float,
    v_low: float,
    v_high: float,
    wz_deadband: float,
    wz_scale: float,
) -> float:
    speed_xy = math.hypot(vx, vy)
    yaw_strength = np.clip((abs(wz) - wz_deadband) / wz_scale, 0.0, 1.0)
    low_speed = np.clip((v_high - speed_xy) / max(v_high - v_low, 1e-6), 0.0, 1.0)
    return float(yaw_strength * low_speed)


def _simulate(args: argparse.Namespace) -> dict[str, np.ndarray | float]:
    hip = np.array(
        [
            [args.hip_x, +args.hip_y],
            [args.hip_x, -args.hip_y],
        ],
        dtype=float,
    )
    contacts = hip.copy()
    root_xy = np.array([0.0, 0.0], dtype=float)
    root_yaw = 0.0
    swing_foot = 0
    t_swing = args.t_step * args.t_swing_fraction
    blend = _yaw_blend(
        vx=args.vx,
        vy=args.vy,
        wz=args.wz,
        v_low=args.v_low,
        v_high=args.v_high,
        wz_deadband=args.wz_deadband,
        wz_scale=args.wz_scale,
    )

    root_xy_hist = [root_xy.copy()]
    root_yaw_hist = [root_yaw]
    left_hist = [contacts[0].copy()]
    right_hist = [contacts[1].copy()]
    target_hist: list[np.ndarray] = []
    ideal_landing_hist: list[np.ndarray] = []
    swing_hist: list[int] = []

    for _ in range(args.steps):
        horizon = t_swing
        turn_delta = np.clip(
            args.wz * horizon * args.yaw_gain,
            -args.max_delta,
            args.max_delta,
        )
        yaw_special = root_xy + _rot2(root_yaw + turn_delta) @ hip[swing_foot]
        ideal_landing = root_xy + _rot2(root_yaw + args.wz * horizon) @ hip[swing_foot]
        lipm_stall_fallback = contacts[swing_foot]
        target = (1.0 - blend) * lipm_stall_fallback + blend * yaw_special

        contacts[swing_foot] = target
        target_hist.append(target.copy())
        ideal_landing_hist.append(ideal_landing.copy())
        swing_hist.append(swing_foot)
        left_hist.append(contacts[0].copy())
        right_hist.append(contacts[1].copy())

        root_xy = root_xy + np.array([args.vx, args.vy], dtype=float) * horizon
        root_yaw = root_yaw + args.wz * horizon
        root_xy_hist.append(root_xy.copy())
        root_yaw_hist.append(root_yaw)
        swing_foot = 1 - swing_foot

    return {
        "root_xy": np.asarray(root_xy_hist),
        "root_yaw": np.asarray(root_yaw_hist),
        "left": np.asarray(left_hist),
        "right": np.asarray(right_hist),
        "target": np.asarray(target_hist),
        "ideal_landing": np.asarray(ideal_landing_hist),
        "swing": np.asarray(swing_hist),
        "blend": blend,
        "t_swing": t_swing,
    }


def _plot(data: dict[str, np.ndarray | float], args: argparse.Namespace) -> None:
    root_xy = data["root_xy"]
    root_yaw = data["root_yaw"]
    left = data["left"]
    right = data["right"]
    target = data["target"]
    ideal_landing = data["ideal_landing"]
    swing = data["swing"]
    blend = float(data["blend"])
    t_swing = float(data["t_swing"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(root_xy[:, 0], root_xy[:, 1], "k-", lw=1.5, alpha=0.6, label="root")
    ax.plot(left[:, 0], left[:, 1], "o-", color="#1f77b4", lw=1.5, ms=5, label="left contact")
    ax.plot(right[:, 0], right[:, 1], "o-", color="#d62728", lw=1.5, ms=5, label="right contact")

    left_targets = target[swing == 0]
    right_targets = target[swing == 1]
    if len(left_targets) > 0:
        ax.scatter(left_targets[:, 0], left_targets[:, 1], s=85, color="#1f77b4", marker="x", label="planned L")
    if len(right_targets) > 0:
        ax.scatter(right_targets[:, 0], right_targets[:, 1], s=85, color="#d62728", marker="x", label="planned R")
    ax.scatter(
        ideal_landing[:, 0],
        ideal_landing[:, 1],
        s=38,
        facecolors="none",
        edgecolors="#444444",
        alpha=0.65,
        label="full landing-yaw neutral",
    )

    stride = max(1, len(root_xy) // 12)
    arrow_len = args.hip_y * 0.9
    for i in range(0, len(root_xy), stride):
        yaw = root_yaw[i]
        start = root_xy[i]
        delta = np.array([math.cos(yaw), math.sin(yaw)]) * arrow_len
        ax.arrow(
            start[0],
            start[1],
            delta[0],
            delta[1],
            head_width=0.025,
            head_length=0.035,
            fc="black",
            ec="black",
            alpha=0.35,
            length_includes_head=True,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title(
        f"Yaw-only foothold plan: vx={args.vx:.2f}, vy={args.vy:.2f}, wz={args.wz:.2f} rad/s, "
        f"blend={blend:.2f}, swing_horizon={t_swing:.2f}s"
    )
    ax.legend(loc="best", fontsize=8)

    text = (
        f"yaw special target = R(root_yaw + clamp(wz*horizon*gain, +/-max_delta)) * hip_offset\n"
        f"gain={args.yaw_gain:.2f}, max_delta={args.max_delta:.2f} rad, "
        f"wz*horizon={args.wz * t_swing:.2f} rad"
    )
    fig.text(0.06, 0.02, text, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"Wrote {out}")
    print(f"blend={blend:.3f}, t_swing={t_swing:.3f}s, yaw_per_swing={args.wz * t_swing:.3f}rad")
    print("step swing target_x target_y ideal_full_yaw_x ideal_full_yaw_y")
    for i, foot in enumerate(swing):
        foot_name = "L" if foot == 0 else "R"
        print(
            f"{i:02d}   {foot_name}   "
            f"{target[i, 0]: .4f}  {target[i, 1]: .4f}   "
            f"{ideal_landing[i, 0]: .4f}  {ideal_landing[i, 1]: .4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vx", type=float, default=0.0, help="Body/world x velocity used by this flat ideal plot.")
    parser.add_argument("--vy", type=float, default=0.0, help="Body/world y velocity used by this flat ideal plot.")
    parser.add_argument("--wz", type=float, default=1.0, help="Yaw-rate command [rad/s].")
    parser.add_argument("--steps", type=int, default=16, help="Number of alternating swing-foot targets to draw.")
    parser.add_argument("--t-step", type=float, default=0.6, help="Planner t_step from FootstepPlanCommandCfg.")
    parser.add_argument("--t-swing-fraction", type=float, default=0.5)
    parser.add_argument("--hip-x", type=float, default=0.0)
    parser.add_argument("--hip-y", type=float, default=0.12)
    parser.add_argument("--yaw-gain", type=float, default=0.50)
    parser.add_argument("--max-delta", type=float, default=0.35)
    parser.add_argument("--v-low", type=float, default=0.05)
    parser.add_argument("--v-high", type=float, default=0.25)
    parser.add_argument("--wz-deadband", type=float, default=0.15)
    parser.add_argument("--wz-scale", type=float, default=0.60)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--out", type=str, default="scripts/debug/output/lipm_yaw_only_footholds.png")
    args = parser.parse_args()

    data = _simulate(args)
    _plot(data, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
