"""Quantify how well a pretrained policy lands on the heuristic planner's
predicted footstep targets.

For each touchdown event during inference, record:
    actual_landing_xy   — robot foot pos in world at the moment its phase
                          crosses from swing to stance
    predicted_landing_xy — the planner's committed target for that foot
                           immediately before the new commit overwrites it
    residual            = actual - predicted

Aggregated stats tell us whether the AME-trained policy implicitly steps the
way Raibert+terrain projection would predict. Small residuals ⇒ the planner
is a useful "weak teacher" and Phase 1 reward can lock onto it without
breaking the existing gait. Large residuals ⇒ the policy uses a different
strategy and we need either a learned planner or a curriculum.

Usage:
    .AME/bin/python scripts/eval_footstep_residual.py \\
        --task AME-G1-29DOF-Play-v0 \\
        --checkpoint pretrained/ame1.pt \\
        --num_envs 16 --num_steps 1500
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="AME-G1-29DOF-Play-v0")
parser.add_argument("--checkpoint", type=str, default="pretrained/ame1.pt")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=1500)
parser.add_argument("--warmup_steps", type=int, default=100,
                    help="Discard touchdowns inside the first N steps (lets the policy stabilize).")
parser.add_argument("--out", type=str, default="footstep_residual.npz")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
# The PLAY env configs include a recording camera; AppLauncher needs the
# rendering experience to allow cameras even in headless mode.
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- now safe to import torch / Isaac Lab modules ---------------------------

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

from rsl_rl.runners import OnPolicyRunner

import ame_locomotion.tasks  # noqa: F401
from ame_locomotion.tasks.manager_based.ame_locomotion.agents.ame_rsl_rl_ppo_cfg import (
    G1AMEPPORunnerCfg,
    G1TerrainMlpPPORunnerCfg,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rsl_rl"))
from vec_env_adapter import IsaacLabTensorDictAdapter  # noqa: E402


def _resolve_agent_cfg(task: str):
    if "HeightMLP" in task:
        return G1TerrainMlpPPORunnerCfg()
    return G1AMEPPORunnerCfg()


def _log(msg):
    print(f"[eval_residual] {msg}", flush=True)


def main():
    _log("parsing env_cfg ...")
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    _log(f"resolving agent_cfg for task={args.task} ...")
    agent_cfg = _resolve_agent_cfg(args.task)
    agent_cfg.device = args.device
    # Match play.py: seed must be set on env_cfg, otherwise some downstream
    # randomization paths can stall.
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args.device

    _log("gym.make ...")
    env = gym.make(args.task, cfg=env_cfg)
    _log("RslRlVecEnvWrapper ...")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    _log("IsaacLabTensorDictAdapter ...")
    env = IsaacLabTensorDictAdapter(env)

    _log("OnPolicyRunner ...")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    _log(f"runner.load({args.checkpoint}) ...")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    _log("policy ready.")

    footstep_term = env.unwrapped.command_manager.get_term("footstep_plan")

    obs = env.get_observations()
    # initial baselines (just after reset)
    prev_swing = footstep_term.swing_foot.clone()
    prev_target = footstep_term.target_w.clone()
    prev_vel_cmd = env.unwrapped.command_manager.get_command("base_velocity")[:, :3].clone()

    # records — keep flat lists, stack at the end
    rec_step = []
    rec_env = []
    rec_foot = []
    rec_actual = []           # (3,) world xyz
    rec_predicted = []        # (3,) world xyz
    rec_vel_cmd_b = []        # (3,) at moment of commit
    rec_root_yaw = []         # (,)

    for step in range(args.num_steps):
        with torch.inference_mode():
            result = policy(obs)
            actions = result[0] if isinstance(result, (tuple, list)) else result
            obs, *_ = env.step(actions)

        # crossover detection: swing_foot just toggled
        cur_swing = footstep_term.swing_foot
        crossover = cur_swing != prev_swing

        if step >= args.warmup_steps and crossover.any():
            ids = crossover.nonzero(as_tuple=False).squeeze(-1)
            # the foot that *just* landed is the one stored in prev_swing
            foots_landed = prev_swing[ids]                        # (E_event,)
            row = torch.arange(ids.numel(), device=ids.device)
            # actual landing position was just latched into last_contact_w[foot_landed]
            actual_w = footstep_term.last_contact_w[ids][row, foots_landed]   # (E_event, 3)
            # predicted position is the prior target for that foot, before the new commit
            predicted_w = prev_target[ids][row, foots_landed]                 # (E_event, 3)
            yaws = footstep_term.robot.data.heading_w[ids]                    # (E_event,)
            vels_cmd_b = prev_vel_cmd[ids]                                    # (E_event, 3)

            for i in range(ids.numel()):
                rec_step.append(step)
                rec_env.append(int(ids[i].item()))
                rec_foot.append(int(foots_landed[i].item()))
                rec_actual.append(actual_w[i].cpu().numpy())
                rec_predicted.append(predicted_w[i].cpu().numpy())
                rec_root_yaw.append(float(yaws[i].item()))
                rec_vel_cmd_b.append(vels_cmd_b[i].cpu().numpy())

        prev_swing = cur_swing.clone()
        prev_target = footstep_term.target_w.clone()
        prev_vel_cmd = env.unwrapped.command_manager.get_command("base_velocity")[:, :3].clone()

    if not rec_step:
        print("[WARN] No touchdown events recorded. Try increasing --num_steps.")
        env.close()
        return

    actual = np.stack(rec_actual)           # (T, 3)
    predicted = np.stack(rec_predicted)     # (T, 3)
    residual = actual - predicted           # (T, 3) world-frame

    # also express residual in body frame (forward = body x = root yaw direction)
    yaws = np.array(rec_root_yaw)
    c = np.cos(yaws)
    s = np.sin(yaws)
    rx = residual[:, 0]
    ry = residual[:, 1]
    residual_body = np.stack([c * rx + s * ry, -s * rx + c * ry, residual[:, 2]], axis=-1)

    # summary stats
    def _stats(arr, name):
        return (f"  {name:<24s}  mean={arr.mean():+.4f}  std={arr.std():.4f}  "
                f"median={np.median(arr):+.4f}  p95={np.percentile(np.abs(arr), 95):.4f}  "
                f"|max|={np.max(np.abs(arr)):.4f}")

    rxy = np.linalg.norm(residual[:, :2], axis=1)
    rxy_body = np.linalg.norm(residual_body[:, :2], axis=1)
    print()
    print("=" * 78)
    print(f"Footstep landing residual — {len(rec_step)} touchdowns from "
          f"{args.num_envs} envs × {args.num_steps} steps "
          f"(warmup {args.warmup_steps})")
    print("=" * 78)
    print(f"  residual_xy norm (m, world frame)")
    print(f"    mean={rxy.mean():.4f}  median={np.median(rxy):.4f}  "
          f"p95={np.percentile(rxy, 95):.4f}  max={rxy.max():.4f}")
    print(f"  residual_xy norm (m, body frame)")
    print(f"    mean={rxy_body.mean():.4f}  median={np.median(rxy_body):.4f}  "
          f"p95={np.percentile(rxy_body, 95):.4f}  max={rxy_body.max():.4f}")
    print()
    print("  per-axis residuals (body frame, m):")
    print(_stats(residual_body[:, 0], "forward (body x)"))
    print(_stats(residual_body[:, 1], "lateral (body y)"))
    print(_stats(residual[:, 2], "z (world)"))
    print()
    # split by foot
    foots = np.array(rec_foot)
    for f, name in [(0, "left"), (1, "right")]:
        mask = foots == f
        if mask.any():
            sub = rxy_body[mask]
            print(f"  {name:<5s}  count={mask.sum():4d}  mean={sub.mean():.4f}  "
                  f"median={np.median(sub):.4f}  p95={np.percentile(sub, 95):.4f}")
    print("=" * 78)

    np.savez(
        args.out,
        step=np.array(rec_step),
        env=np.array(rec_env),
        foot=np.array(rec_foot),
        actual=actual,
        predicted=predicted,
        residual=residual,
        residual_body=residual_body,
        root_yaw=np.array(rec_root_yaw),
        vel_cmd_b=np.stack(rec_vel_cmd_b),
    )
    print(f"\nRaw events saved to {args.out}")
    env.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException as e:
        print(f"[eval_residual] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
