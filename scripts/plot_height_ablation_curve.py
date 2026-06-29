#!/usr/bin/env python3
"""Plot HeightMLP elevation-input ablation curves from real logged data.

Expected CSV/NPZ fields:
    time, cmd_vx, cmd_vy, base_vx, base_vy, roll, pitch

If cmd_vy/base_vy are absent, the speed-tracking error falls back to
abs(cmd_vx - base_vx). The script never creates synthetic measurements.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


FIELD_ALIASES = {
    "time": ("time", "t", "time_s", "timestamp", "step_time"),
    "cmd_vx": ("cmd_vx", "command_vx", "vel_cmd_x", "vel_cmd_b_x", "vx_cmd"),
    "cmd_vy": ("cmd_vy", "command_vy", "vel_cmd_y", "vel_cmd_b_y", "vy_cmd"),
    "base_vx": ("base_vx", "root_vx", "root_vel_x", "base_lin_vel_x", "base_vel_x"),
    "base_vy": ("base_vy", "root_vy", "root_vel_y", "base_lin_vel_y", "base_vel_y"),
    "roll": ("roll", "base_roll", "root_roll"),
    "pitch": ("pitch", "base_pitch", "root_pitch"),
}

TEMPLATE_FIELDS = ["time", "cmd_vx", "cmd_vy", "base_vx", "base_vy", "roll", "pitch"]


def _write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TEMPLATE_FIELDS)
    print(f"Wrote empty data template: {path}")


def _load_table(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {str(k): np.asarray(data[k]).squeeze() for k in data.files}

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.dtype.names:
            return {name: np.asarray(arr[name]).squeeze() for name in arr.dtype.names}
        raise ValueError(f"{path} is a plain .npy array; use .npz or CSV with named fields.")

    if suffix == ".csv":
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
        if table.size == 0:
            raise ValueError(f"{path} contains a header but no measurement rows.")
        if table.dtype.names is None:
            raise ValueError(f"{path} must contain a header row.")
        return {name: np.asarray(table[name]).squeeze() for name in table.dtype.names}

    raise ValueError(f"Unsupported data file type: {path.suffix}. Use CSV, NPZ, or named NPY.")


def _lookup(table: dict[str, np.ndarray], canonical: str, *, required: bool = True) -> np.ndarray | None:
    keys = {k.lower(): k for k in table}
    for alias in FIELD_ALIASES[canonical]:
        source_key = keys.get(alias.lower())
        if source_key is not None:
            return np.asarray(table[source_key], dtype=float).reshape(-1)
    if required:
        aliases = ", ".join(FIELD_ALIASES[canonical])
        raise KeyError(f"Missing required field '{canonical}'. Accepted aliases: {aliases}")
    return None


def _prepare_series(path: Path) -> dict[str, np.ndarray]:
    table = _load_table(path)
    time = _lookup(table, "time")
    cmd_vx = _lookup(table, "cmd_vx")
    base_vx = _lookup(table, "base_vx")
    roll = _lookup(table, "roll")
    pitch = _lookup(table, "pitch")
    cmd_vy = _lookup(table, "cmd_vy", required=False)
    base_vy = _lookup(table, "base_vy", required=False)

    arrays = [time, cmd_vx, base_vx, roll, pitch]
    if cmd_vy is not None and base_vy is not None:
        arrays.extend([cmd_vy, base_vy])
    n = min(arr.size for arr in arrays)
    if n < 2:
        raise ValueError(f"{path} must contain at least two valid samples.")

    time = time[:n]
    cmd_vx = cmd_vx[:n]
    base_vx = base_vx[:n]
    roll = roll[:n]
    pitch = pitch[:n]

    if cmd_vy is not None and base_vy is not None:
        cmd_vy = cmd_vy[:n]
        base_vy = base_vy[:n]
        velocity_error = np.sqrt((cmd_vx - base_vx) ** 2 + (cmd_vy - base_vy) ** 2)
        speed_mode = "xy"
    else:
        velocity_error = np.abs(cmd_vx - base_vx)
        speed_mode = "x"

    attitude_error = np.sqrt(roll**2 + pitch**2)
    finite = np.isfinite(time) & np.isfinite(velocity_error) & np.isfinite(attitude_error)
    if finite.sum() < 2:
        raise ValueError(f"{path} has fewer than two finite samples after filtering.")

    time = time[finite]
    order = np.argsort(time)
    time = time[order]
    time = time - time[0]

    return {
        "time": time,
        "velocity_error": velocity_error[finite][order],
        "attitude_error": attitude_error[finite][order],
        "speed_mode": np.array(speed_mode),
    }


def _plot(real: dict[str, np.ndarray], flat: dict[str, np.ndarray], out_png: Path, out_pdf: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {"real": "#1f77b4", "flat": "#d62728"}
    labels = {"real": "Real elevation input", "flat": "Flat elevation prior"}

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True, constrained_layout=True)

    axes[0].plot(real["time"], real["velocity_error"], color=colors["real"], label=labels["real"])
    axes[0].plot(flat["time"], flat["velocity_error"], color=colors["flat"], label=labels["flat"])
    axes[0].set_ylabel(r"$e_v$ (m/s)")
    axes[0].set_title("Velocity Tracking Error")
    axes[0].grid(True, alpha=0.28, linewidth=0.8)
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].plot(real["time"], real["attitude_error"], color=colors["real"], label=labels["real"])
    axes[1].plot(flat["time"], flat["attitude_error"], color=colors["flat"], label=labels["flat"])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$e_{att}$ (rad)")
    axes[1].set_title("Body Attitude Fluctuation")
    axes[1].grid(True, alpha=0.28, linewidth=0.8)
    axes[1].legend(loc="upper right", frameon=True)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _summary(name: str, data: dict[str, np.ndarray]) -> str:
    ev = data["velocity_error"]
    eatt = data["attitude_error"]
    return (
        f"{name}: mean e_v={ev.mean():.4f} m/s, "
        f"rms e_v={np.sqrt(np.mean(ev**2)):.4f} m/s, "
        f"mean e_att={eatt.mean():.4f} rad, "
        f"rms e_att={np.sqrt(np.mean(eatt**2)):.4f} rad"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data", type=Path, help="CSV/NPZ for the real elevation input run.")
    parser.add_argument("--flat-data", type=Path, help="CSV/NPZ for the flat elevation prior run.")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"), help="Output directory for png/pdf.")
    parser.add_argument("--png-name", default="height_ablation_curve.png")
    parser.add_argument("--pdf-name", default="height_ablation_curve.pdf")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution. Use >=300 for reports.")
    parser.add_argument("--write-template", type=Path, help="Write an empty CSV template and exit.")
    args = parser.parse_args()

    if args.write_template:
        _write_template(args.write_template)
        return

    if args.real_data is None or args.flat_data is None:
        parser.error("--real-data and --flat-data are required unless --write-template is used.")
    if args.dpi < 300:
        parser.error("--dpi must be at least 300 for report-ready output.")

    real = _prepare_series(args.real_data)
    flat = _prepare_series(args.flat_data)
    out_png = args.out_dir / args.png_name
    out_pdf = args.out_dir / args.pdf_name
    _plot(real, flat, out_png, out_pdf, args.dpi)

    print(_summary("Real elevation input", real))
    print(_summary("Flat elevation prior", flat))
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
