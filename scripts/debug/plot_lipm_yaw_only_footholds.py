#!/usr/bin/env python3
"""Plot LIPM arc-model foothold targets for yaw and mixed commands.

Uses the new ``lipm_arc_nominal_foothold_xy`` (per-foot effective velocity)
to visualize how the planner handles pure yaw, forward+yaw, and omni commands.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

_LIPM_PATH = (
    Path(__file__).resolve().parents[2]
    / "source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/planner_lipm.py"
)


def _load_lipm():
    spec = importlib.util.spec_from_file_location("planner_lipm", _LIPM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rot2(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def _advance_root(root_xy: np.ndarray, root_yaw: float, args: argparse.Namespace, dt: float) -> tuple[np.ndarray, float]:
    """Midpoint integration for command-following root motion over ``dt``."""
    mid_yaw = root_yaw + 0.5 * args.wz * dt
    vel_mid = _rot2(mid_yaw) @ np.array([args.vx, args.vy])
    return root_xy + vel_mid * dt, root_yaw + args.wz * dt


def _simulate(lipm, args: argparse.Namespace) -> dict:
    if args.t_double_support > 0.0 and 2.0 * args.t_double_support >= args.t_step:
        raise ValueError("Requires 0 <= 2 * t_double_support < t_step.")

    hip_b = np.array([[args.hip_x, +args.hip_y], [args.hip_x, -args.hip_y]])

    contacts = hip_b.copy()
    root_xy = np.array([0.0, 0.0])
    root_yaw = 0.0
    swing_foot = 0
    if args.t_double_support > 0.0:
        t_swing = 0.5 * (args.t_step - 2.0 * args.t_double_support)
        dt_to_next_swing = t_swing + args.t_double_support
        remaining_delta_t = t_swing
    else:
        t_swing = args.t_step * args.t_swing_fraction
        dt_to_next_swing = t_swing
        remaining_delta_t = args.t_step

    root_hist = [root_xy.copy()]
    yaw_hist = [root_yaw]
    left_hist = [contacts[0].copy()]
    right_hist = [contacts[1].copy()]
    target_hist = []
    swing_hist = []
    phase_segments = []
    time_s = 0.0

    for _ in range(args.steps):
        c, s = math.cos(root_yaw), math.sin(root_yaw)
        R = np.array([[c, -s], [s, c]])
        vel_cmd_w = R @ np.array([args.vx, args.vy])
        hip_w = (R @ hip_b.T).T

        landing_root, landing_yaw = _advance_root(root_xy, root_yaw, args, t_swing)
        landing_hip = landing_root + _rot2(landing_yaw) @ hip_b[swing_foot]

        v_cmd_t = torch.from_numpy(vel_cmd_w.astype(np.float32)).unsqueeze(0)
        wz_t = torch.tensor([args.wz], dtype=torch.float32)
        sw_t = torch.tensor([swing_foot], dtype=torch.long)
        hip_w_t = torch.from_numpy(hip_w.astype(np.float32)).unsqueeze(0)
        v_eff_t = lipm.per_foot_effective_velocity(v_cmd_t, wz_t, hip_w_t)[:, swing_foot]
        landing_hip_t = torch.from_numpy(landing_hip.astype(np.float32)).unsqueeze(0)

        target_t = lipm.lipm_arc_nominal_foothold_xy(
            landing_hip_t,
            v_eff_t,
            sw_t,
            remaining_delta_t=remaining_delta_t, step_duration_ts=args.t_step,
            com_height=args.com_height, step_width=args.step_width,
            landing_yaw=torch.tensor([landing_yaw], dtype=torch.float32),
        )
        target = target_t[0].numpy()

        swing_name = "L_swing" if swing_foot == 0 else "R_swing"
        phase_segments.append((time_s, time_s + t_swing, swing_name))
        if args.t_double_support > 0.0:
            phase_segments.append((time_s + t_swing, time_s + t_swing + args.t_double_support, "DS"))

        contacts[swing_foot] = target
        target_hist.append(target.copy())
        swing_hist.append(swing_foot)
        left_hist.append(contacts[0].copy())
        right_hist.append(contacts[1].copy())

        root_xy, root_yaw = _advance_root(root_xy, root_yaw, args, dt_to_next_swing)
        time_s += dt_to_next_swing
        root_hist.append(root_xy.copy())
        yaw_hist.append(root_yaw)
        swing_foot = 1 - swing_foot

    return {
        "root_xy": np.array(root_hist),
        "root_yaw": np.array(yaw_hist),
        "left": np.array(left_hist),
        "right": np.array(right_hist),
        "target": np.array(target_hist),
        "swing": np.array(swing_hist),
        "t_swing": t_swing,
        "t_double_support": args.t_double_support,
        "dt_to_next_swing": dt_to_next_swing,
        "phase_segments": phase_segments,
    }


def _plot_phase_timeline(ax, data: dict, args: argparse.Namespace) -> None:
    colors = {"L_swing": "#2196F3", "R_swing": "#F44336", "DS": "#43A047"}
    t_sw = data["t_swing"]
    t_ds = data["t_double_support"]
    if t_ds > 0.0:
        segments = [
            (0.0, t_sw, "L_swing", "L swing\nR stance"),
            (t_sw, t_ds, "DS", "DS\nboth feet"),
            (t_sw + t_ds, t_sw, "R_swing", "R swing\nL stance"),
            (2.0 * t_sw + t_ds, t_ds, "DS", "DS\nboth feet"),
        ]
    else:
        segments = [
            (0.0, t_sw, "L_swing", "L swing\nR stance"),
            (t_sw, args.t_step - t_sw, "R_swing", "R swing\nL stance"),
        ]

    for start, width, name, label in segments:
        ax.broken_barh([(start, width)], (0.32, 0.36), facecolors=colors[name], alpha=0.85)
        ax.text(
            start + 0.5 * width,
            0.50,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="white" if name != "DS" else "black",
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, args.t_step)
    ax.set_yticks([])
    ax.set_xlabel("time [s]")
    ax.set_title("One gait cycle contact phase")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor=colors["L_swing"], label="left swing"),
            Patch(facecolor=colors["DS"], label="double support"),
            Patch(facecolor=colors["R_swing"], label="right swing"),
        ],
        loc="upper right",
        fontsize=8,
        ncol=3,
    )


def _plot(data: dict, args: argparse.Namespace) -> None:
    root_xy = data["root_xy"]
    root_yaw = data["root_yaw"]
    left = data["left"]
    right = data["right"]
    target = data["target"]
    swing = data["swing"]

    fig = plt.figure(figsize=(8.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.0], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax_phase = fig.add_subplot(gs[1])
    ax.plot(root_xy[:, 0], root_xy[:, 1], "k-", lw=2, alpha=0.6, label="CoM path")
    ax.scatter(root_xy[0, 0], root_xy[0, 1], s=100, color="black", marker="s", zorder=6)
    ax.plot(left[:, 0], left[:, 1], "o-", color="#2196F3", lw=1.2, ms=6, label="left foot", alpha=0.85)
    ax.plot(right[:, 0], right[:, 1], "o-", color="#F44336", lw=1.2, ms=6, label="right foot", alpha=0.85)

    lt = target[swing == 0]
    rt = target[swing == 1]
    if len(lt) > 0:
        ax.scatter(lt[:, 0], lt[:, 1], s=100, color="#2196F3", marker="*", zorder=5, label="target L")
    if len(rt) > 0:
        ax.scatter(rt[:, 0], rt[:, 1], s=100, color="#F44336", marker="*", zorder=5, label="target R")

    stride = max(1, len(root_xy) // 10)
    for i in range(0, len(root_xy), stride):
        yaw = root_yaw[i]
        start = root_xy[i]
        delta = np.array([math.cos(yaw), math.sin(yaw)]) * args.hip_y * 0.8
        ax.arrow(
            start[0], start[1], delta[0], delta[1],
            head_width=0.015, head_length=0.012, fc="green", ec="green", alpha=0.6,
            length_includes_head=True,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title(
        f"LIPM Arc-Model: vx={args.vx:.2f}, vy={args.vy:.2f}, wz={args.wz:.2f} rad/s\n"
        f"L_swing {data['t_swing']:.2f}s | DS {data['t_double_support']:.2f}s | "
        f"R_swing {data['t_swing']:.2f}s | DS {data['t_double_support']:.2f}s"
    )
    ax.legend(loc="best", fontsize=8)

    v_eff_mag = math.hypot(args.vx + args.wz * args.hip_y, args.vy)
    text = (
        f"v_eff (left) = |v_cmd + wz*(-hip_y, 0)| = {v_eff_mag:.3f} m/s\n"
        f"t_step={args.t_step:.2f}s, t_swing={data['t_swing']:.2f}s, "
        f"DS={data['t_double_support']:.2f}s, yaw/swing={args.wz * data['t_swing']:.2f} rad"
    )
    fig.text(0.06, 0.02, text, fontsize=9)
    _plot_phase_timeline(ax_phase, data, args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"Wrote {out}")

    print(f"\nStep targets:")
    for i, foot in enumerate(swing):
        name = "L" if foot == 0 else "R"
        print(f"  {i:02d} ({name}): x={target[i, 0]:+.4f}  y={target[i, 1]:+.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--t-step", type=float, default=0.6)
    parser.add_argument("--t-double-support", type=float, default=0.05)
    parser.add_argument("--t-swing-fraction", type=float, default=0.5)
    parser.add_argument("--hip-x", type=float, default=0.0)
    parser.add_argument("--hip-y", type=float, default=0.12)
    parser.add_argument("--com-height", type=float, default=0.78)
    parser.add_argument("--step-width", type=float, default=0.24)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--out", type=str, default="scripts/debug/output/lipm_yaw_only_footholds.png")
    args = parser.parse_args()

    lipm = _load_lipm()
    data = _simulate(lipm, args)
    _plot(data, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
