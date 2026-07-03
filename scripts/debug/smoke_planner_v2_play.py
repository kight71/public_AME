"""PlannerV2-Play smoke — v2 selector + obs stack + optional mask diagnostics.

Usage::

    .AME/bin/python scripts/debug/smoke_planner_v2_play.py --headless --enable_cameras --steps 30
    .AME/bin/python scripts/debug/smoke_planner_v2_play.py --headless --debug-masks --strict

``--debug-masks`` prints per-layer mask valid counts on each plan commit (env 0).
``--strict`` fails if reset has any fallback or either foot has < ``--min-reset-valid`` candidates.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PlannerV2-Play foothold selector smoke.")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--heartbeat", type=int, default=5, help="Print buffer snapshot every N steps.")
parser.add_argument(
    "--debug-masks",
    action="store_true",
    help="Enable selector_v2_debug_masks (per-layer valid counts on each commit).",
)
parser.add_argument(
    "--strict",
    action="store_true",
    help="Exit non-zero if reset foothold health checks fail (see --min-reset-valid).",
)
parser.add_argument(
    "--min-reset-valid",
    type=int,
    default=8,
    help="Strict mode: each foot must have at least this many combined valid candidates at reset.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if getattr(args, "enable_cameras", None) is None:
    args.enable_cameras = True  # Play cfg ships visualize_cam

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

TASK = "AME-G1-29DOF-DTC-PlannerV2-Play-v0"
REQUIRED_POLICY_OBS = (
    "footstep_plan_xy",
    "footstep_phase_info",
    "footstep_local_heightscan",
    "footstep_foothold_score",
    "footstep_swing_side",
    "height_scan",
)


def _check_reset_health(cmd, *, min_valid: int) -> None:
    fb = cmd.used_fallback_buffer[0, 0].tolist()
    valid = cmd.valid_count_buffer[0, 0].tolist()
    scores = cmd.foothold_score_buffer[0, 0].tolist()
    if any(fb):
        raise AssertionError(f"reset fallback must be all False, got {fb}")
    if min(valid) < min_valid:
        raise AssertionError(
            f"reset valid_count per foot must be >= {min_valid}, got {valid} (scores={scores})"
        )
    ratio = max(valid) / max(min(valid), 1)
    if ratio > 2.5:
        print(
            f"[smoke-v2] WARN reset L/R valid asymmetry {valid[0]} vs {valid[1]} "
            f"(ratio={ratio:.2f}) — re-run with --debug-masks to locate mask layer",
            flush=True,
        )


def main() -> None:
    print(f"[smoke-v2] loading {TASK} ...", flush=True)
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    cfg.commands.footstep_plan.selector_v2_debug_masks = args.debug_masks
    env = gym.make(TASK, cfg=cfg)

    obs_mgr = env.unwrapped.observation_manager
    policy_terms = list(obs_mgr.active_terms["policy"])
    policy_dim = obs_mgr.group_obs_dim["policy"][0]
    print(f"[smoke-v2] policy obs terms ({policy_dim} dim): {policy_terms}", flush=True)
    for name in REQUIRED_POLICY_OBS:
        assert name in policy_terms, f"missing policy obs term: {name}"
    assert policy_terms[-1] == "height_scan", "height_scan must be last (AME tail-slice)"

    swing_tracking_w = env.unwrapped.reward_manager.get_term_cfg("footstep_swing_tracking").weight
    assert swing_tracking_w == 0.0, f"expected swing_tracking off, got {swing_tracking_w}"

    print(f"[smoke-v2] reset(seed={args.seed}) ...", flush=True)
    env.reset(seed=args.seed)
    print("[smoke-v2] reset done", flush=True)

    cmd = env.unwrapped.command_manager.get_term("footstep_plan")
    assert cmd.cfg.use_selector_v2 is True
    print(f"[smoke-v2] post-reset buffers: {format_cmd_buffers(cmd)}", flush=True)
    if cmd.cfg.use_selector_v2:
        print(
            "[smoke-v2] post-reset mask attribution (k=0): "
            f"reach={cmd.mask_reach_valid_buffer[0, 0].tolist()} "
            f"step_h={cmd.mask_step_height_valid_buffer[0, 0].tolist()} "
            f"rough={cmd.mask_roughness_valid_buffer[0, 0].tolist()} "
            f"bounds={cmd.mask_in_bounds_valid_buffer[0, 0].tolist()}",
            flush=True,
        )

    if args.strict:
        _check_reset_health(cmd, min_valid=args.min_reset_valid)

    step_loop(env, args.steps, prefix="smoke-v2", cmd=cmd, heartbeat=args.heartbeat)

    scores = cmd.foothold_score_buffer[0].tolist()
    fallback = cmd.used_fallback_buffer[0].tolist()
    valid = cmd.valid_count_buffer[0].tolist()

    print(f"[PASS] {TASK} steps={args.steps} use_selector_v2=True", flush=True)
    print(f"       foothold_score={scores} fallback={fallback} valid_count={valid}", flush=True)
    print(f"       swing_tracking.weight=0.0  policy_dim={policy_dim}", flush=True)
    if not args.strict:
        print(
            "[smoke-v2] NOTE: structural PASS only — use --strict for reset health gate, "
            "--debug-masks for per-layer mask breakdown",
            flush=True,
        )

    print("[smoke-v2] closing env ...", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    run_smoke("smoke-v2", main)
