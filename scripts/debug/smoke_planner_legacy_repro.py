"""Legacy DTC-Play reproducibility smoke — ONE env, two reset(seed) cycles.

Usage::

    .AME/bin/python scripts/debug/smoke_planner_legacy_repro.py --headless --enable_cameras --steps 30

Exit 0 prints ``[PASS] legacy bit-identical ...``.
"""

from __future__ import annotations

import argparse
import hashlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Legacy DTC-Play plan_buffer reproducibility smoke.")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--heartbeat", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if getattr(args, "enable_cameras", None) is None:
    args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import ame_locomotion.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_planner_common import run_smoke, step_loop

TASK = "AME-G1-29DOF-DTC-Play-v0"


def _digest(plan) -> str:
    return hashlib.sha256(plan.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def _run_once(env, cmd, steps: int, *, label: str) -> tuple[str, str]:
    plan0 = cmd.plan_buffer.clone()
    reset_digest = _digest(plan0)
    print(f"[smoke-legacy] {label} plan_buffer digest after reset: {reset_digest}", flush=True)
    step_loop(env, steps, prefix=f"smoke-legacy/{label}", cmd=None, heartbeat=args.heartbeat)
    final_digest = _digest(cmd.plan_buffer)
    print(f"[smoke-legacy] {label} plan_buffer digest after steps: {final_digest}", flush=True)
    return reset_digest, final_digest


def main() -> None:
    print(f"[smoke-legacy] loading {TASK} ...", flush=True)
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env = gym.make(TASK, cfg=cfg)
    cmd = env.unwrapped.command_manager.get_term("footstep_plan")

    print(f"[smoke-legacy] reset(seed={args.seed}) — cycle A", flush=True)
    env.reset(seed=args.seed)
    print("[smoke-legacy] reset done (cycle A)", flush=True)
    reset_a, final_a = _run_once(env, cmd, args.steps, label="cycle-A")

    print(f"[smoke-legacy] second reset(seed={args.seed}) — cycle B", flush=True)
    env.reset(seed=args.seed)
    print("[smoke-legacy] reset done (cycle B)", flush=True)
    reset_b, final_b = _run_once(env, cmd, args.steps, label="cycle-B")

    assert reset_a == reset_b, f"reset digest mismatch: {reset_a} vs {reset_b}"
    assert final_a == final_b, f"final digest mismatch: {final_a} vs {final_b}"
    print(f"[PASS] legacy bit-identical (seed={args.seed}, steps={args.steps}) digest={final_a}", flush=True)

    print("[smoke-legacy] closing env ...", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    run_smoke("smoke-legacy", main)
