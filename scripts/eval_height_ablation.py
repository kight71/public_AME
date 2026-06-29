#!/usr/bin/env python3
"""Run a HeightMLP elevation-input ablation and export real trajectory CSVs.

This script runs the same policy twice in the same Isaac Lab task:
    1. real_elevation_input: policy receives the environment height samples.
    2. flat_elevation_prior: policy receives a constant height-sample vector.

It records only measured rollout signals. It does not synthesize or smooth
results for plotting.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from isaaclab.app import AppLauncher


PROJ_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PACKAGE = PROJ_ROOT / "deploy_for_AME/robots/g1_29dof/config/policy/ame_height_mlp_stage2/v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="AME-G1-29DOF-HeightMLP-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=800)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out-dir", type=Path, default=Path("data/height_ablation"))
parser.add_argument("--policy-package", type=Path, default=DEFAULT_POLICY_PACKAGE)
parser.add_argument("--onnx", type=Path, help="Override ONNX path. Defaults to <policy-package>/exported/policy.onnx.")
parser.add_argument("--height-dim", type=int, default=96)
parser.add_argument("--flat-height-value", type=float, default=-0.8234)
parser.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import ame_locomotion.tasks  # noqa: F401

sys.path.insert(0, str(PROJ_ROOT / "deploy_for_AME/python"))
from ame_inference.onnx_runner import OnnxPolicyRunner  # noqa: E402


CSV_FIELDS = ["time", "cmd_vx", "cmd_vy", "base_vx", "base_vy", "roll", "pitch"]


def _policy_obs_tensor(obs, device: torch.device) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs
    if isinstance(obs, dict):
        for key in ("policy", "obs"):
            value = obs.get(key)
            if value is not None:
                return value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
    raise TypeError(f"Unable to extract policy observations from {type(obs)}")


def _quat_to_roll_pitch(quat: np.ndarray, order: str) -> tuple[np.ndarray, np.ndarray]:
    if order == "wxyz":
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    else:
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * (math.pi / 2.0), np.arcsin(sinp))
    return roll, pitch


def _get_robot(env):
    scene = env.unwrapped.scene
    try:
        return scene["robot"]
    except Exception:
        pass
    articulations = getattr(scene, "articulations", None)
    if isinstance(articulations, dict) and articulations:
        return next(iter(articulations.values()))
    raise RuntimeError("Unable to find robot articulation in env.unwrapped.scene")


def _get_base_velocity(robot) -> np.ndarray:
    data = robot.data
    if hasattr(data, "root_lin_vel_b"):
        return data.root_lin_vel_b[:, :2].detach().cpu().numpy().copy()
    return data.root_lin_vel_w[:, :2].detach().cpu().numpy().copy()


def _record_row(env, robot, t: float) -> dict[str, float]:
    cmd = env.unwrapped.command_manager.get_command("base_velocity")[:, :3].detach().cpu().numpy()
    base_vel = _get_base_velocity(robot)
    quat = robot.data.root_quat_w.detach().cpu().numpy()
    roll, pitch = _quat_to_roll_pitch(quat, args.quat_order)
    return {
        "time": float(t),
        "cmd_vx": float(cmd[0, 0]),
        "cmd_vy": float(cmd[0, 1]),
        "base_vx": float(base_vel[0, 0]),
        "base_vy": float(base_vel[0, 1]),
        "roll": float(roll[0]),
        "pitch": float(pitch[0]),
    }


def _run_condition(condition: str, flat: bool, out_path: Path) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device

    env = gym.make(args.task, cfg=env_cfg)
    robot = _get_robot(env)
    obs, _ = env.reset()

    onnx_path = args.onnx or (args.policy_package / "exported/policy.onnx")
    runner = OnnxPolicyRunner(onnx_path)
    dt = float(env.unwrapped.step_dt)

    rows: list[dict[str, float]] = []
    for step in range(args.num_steps):
        policy_obs = _policy_obs_tensor(obs, env.unwrapped.device).clone()
        if flat:
            if policy_obs.shape[-1] < args.height_dim:
                raise ValueError(
                    f"Policy obs dim {policy_obs.shape[-1]} is smaller than height_dim={args.height_dim}."
                )
            policy_obs[..., -args.height_dim :] = args.flat_height_value

        actions_np = runner.act(policy_obs.detach().cpu().numpy())
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=env.unwrapped.device)
        obs, *_ = env.step(actions)

        if step >= args.warmup_steps:
            rows.append(_record_row(env, robot, (step - args.warmup_steps) * dt))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{condition}: wrote {len(rows)} samples to {out_path}")
    env.close()


def main() -> None:
    real_path = args.out_dir / "height_ablation_real_elevation.csv"
    flat_path = args.out_dir / "height_ablation_flat_prior.csv"
    _run_condition("real_elevation_input", flat=False, out_path=real_path)
    _run_condition("flat_elevation_prior", flat=True, out_path=flat_path)
    print()
    print("Plot with:")
    print(
        "  python scripts/plot_height_ablation_curve.py "
        f"--real-data {real_path} --flat-data {flat_path} --out-dir figures"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
