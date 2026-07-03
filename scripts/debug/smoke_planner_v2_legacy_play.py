"""PlannerV2-Legacy-Play smoke — matched A/B control, legacy selector + shared score.

Usage::

    .AME/bin/python scripts/debug/smoke_planner_v2_legacy_play.py --headless --enable_cameras --steps 30
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PlannerV2-Legacy-Play A/B control smoke.")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--heartbeat", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if getattr(args, "enable_cameras", None) is None:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import ame_locomotion.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_planner_common import format_cmd_buffers, run_smoke, step_loop

TASK = "AME-G1-29DOF-DTC-PlannerV2-Legacy-Play-v0"


def main() -> None:
    print(f"[smoke-v2-legacy] loading {TASK} ...", flush=True)
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env = gym.make(TASK, cfg=cfg)

    print(f"[smoke-v2-legacy] reset(seed={args.seed}) ...", flush=True)
    env.reset(seed=args.seed)
    print("[smoke-v2-legacy] reset done", flush=True)

    cmd = env.unwrapped.command_manager.get_term("footstep_plan")
    assert cmd.cfg.use_selector_v2 is False
    print(f"[smoke-v2-legacy] post-reset buffers: {format_cmd_buffers(cmd)}", flush=True)

    step_loop(env, args.steps, prefix="smoke-v2-legacy", cmd=cmd, heartbeat=args.heartbeat)

    scores = cmd.foothold_score_buffer[0].tolist()
    assert max(max(row) for row in scores) > 0.0, "legacy path must fill foothold_score via geometry scorer"

    print(f"[PASS] {TASK} steps={args.steps} use_selector_v2=False", flush=True)
    print(f"       foothold_score={scores}", flush=True)

    print("[smoke-v2-legacy] closing env ...", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    run_smoke("smoke-v2-legacy", main)
