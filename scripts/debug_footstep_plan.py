"""Dump footstep-plan trajectories for offline analysis.

Runs the chosen env with a zero-action policy for ``--num_steps`` env steps
and writes per-step ``(root_pos, root_vel, root_yaw, vel_cmd_b,
foot_pos_w, plan_buffer, last_contact_w, swing_foot, phase)`` to an npz.

Intended use: load the npz in a notebook, plot foot trajectory vs the
planned target and the elevation map to eyeball-check the planner.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump footstep-plan trajectories to npz.")
parser.add_argument("--task", type=str, default="AME-G1-29DOF-DTC-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--out", type=str, default="scripts/debug/output/debug_plan.npz")
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


def main():
    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric
    )
    # Play cfgs set num_envs=50 in __post_init__; force CLI value for lightweight dumps.
    env_cfg.scene.num_envs = args.num_envs
    if getattr(env_cfg.scene, "visualize_cam", None) is not None:
        env_cfg.scene.visualize_cam = None
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()

    footstep_term = env.unwrapped.command_manager.get_term("footstep_plan")
    foot_ids = footstep_term.foot_ids.cpu().numpy()
    robot = footstep_term.robot

    buffers = {
        "root_pos": [], "root_vel": [], "root_yaw": [], "vel_cmd_b": [],
        "foot_pos_w": [], "plan_buffer": [], "last_contact_w": [],
        "target_w": [], "swing_foot": [], "phase": [],
    }

    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    for _ in range(args.num_steps):
        env.step(action)
        buffers["root_pos"].append(robot.data.root_pos_w.cpu().numpy().copy())
        buffers["root_vel"].append(robot.data.root_lin_vel_w.cpu().numpy().copy())
        buffers["root_yaw"].append(robot.data.heading_w.cpu().numpy().copy())
        buffers["vel_cmd_b"].append(
            env.unwrapped.command_manager.get_command("base_velocity")[:, :3].cpu().numpy().copy()
        )
        buffers["foot_pos_w"].append(robot.data.body_pos_w[:, foot_ids].cpu().numpy().copy())
        buffers["plan_buffer"].append(footstep_term.plan_buffer.cpu().numpy().copy())
        buffers["last_contact_w"].append(footstep_term.last_contact_w.cpu().numpy().copy())
        buffers["target_w"].append(footstep_term.target_w.cpu().numpy().copy())
        buffers["swing_foot"].append(footstep_term.swing_foot.cpu().numpy().copy())
        buffers["phase"].append(footstep_term.phase.cpu().numpy().copy())

    np.savez(args.out, **{k: np.stack(v) for k, v in buffers.items()})
    print(f"Wrote {args.num_steps} steps to {args.out}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
