"""One-shot sanity check for height_scanner ``ray_alignment='yaw'`` (Planner V2 Task 0.5).

Loads an env, steps once, and verifies that consecutive ``ray_hits_w`` cells
step in the scanner/body-yaw frame — not in world-axis-aligned coordinates.

Usage::

    .AME/bin/python scripts/debug/inspect_scanner_frame.py --headless
    .AME/bin/python scripts/debug/inspect_scanner_frame.py --headless --test_yaw_deg 45

Expected PASS:
  flat[i+1] - flat[i] (same row, col+1) ≈ resolution * (cos(yaw), sin(yaw))
  flat[W] - flat[0] (row+1, same col)   ≈ resolution * (-sin(yaw), cos(yaw))
  where W is the grid width in cells (33 for the default scanner).

Delete this script after confirming; it is intentionally not part of CI.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect height_scanner yaw-aligned grid frame.")
parser.add_argument("--task", type=str, default="AME-G1-29DOF-DTC-Play-v0")
parser.add_argument("--num_envs", type=int, default=1, help="Envs to spawn (1 is enough for this check).")
parser.add_argument(
    "--test_yaw_deg",
    type=float,
    default=None,
    help="If set, rotate env 0 to this yaw (degrees) before stepping — strong yaw-align test.",
)
parser.add_argument("--env_id", type=int, default=0, help="Which env index to inspect.")
parser.add_argument("--atol", type=float, default=1e-3, help="Tolerance on xy diffs (metres).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not hasattr(args, "headless") or args.headless is None:
    args.headless = True
# Play cfgs attach visualize_cam; headless sanity check does not need it.
if getattr(args, "enable_cameras", None) is None:
    args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.utils import parse_env_cfg

import ame_locomotion.tasks  # noqa: F401


def _log(msg: str) -> None:
    print(msg, flush=True)


def _grid_shape_from_scanner(height_scanner) -> tuple[int, int, float]:
    pcfg = height_scanner.cfg.pattern_cfg
    if not isinstance(pcfg, GridPatternCfg):
        raise RuntimeError(f"Expected GridPatternCfg, got {type(pcfg)}")
    width = round(pcfg.size[0] / pcfg.resolution) + 1
    height = round(pcfg.size[1] / pcfg.resolution) + 1
    return height, width, pcfg.resolution


def _maybe_set_env_yaw(robot, env_id: int, yaw_deg: float) -> None:
    device = robot.device
    env_ids = torch.tensor([env_id], device=device, dtype=torch.long)
    root_pose = robot.data.root_link_pose_w[env_ids].clone()
    yaw_rad = math.radians(yaw_deg)
    quat = quat_from_euler_xyz(
        torch.zeros(1, device=device),
        torch.zeros(1, device=device),
        torch.tensor([yaw_rad], device=device),
    )
    root_pose[:, 3:7] = quat
    robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    _log(f"[inspect] Set env {env_id} root yaw to {yaw_deg:.1f} deg")


def inspect_env(
    *,
    ray_hits_w: torch.Tensor,
    grid_yaw: float,
    grid_center_xy: torch.Tensor,
    height: int,
    width: int,
    resolution: float,
    env_id: int,
    atol: float,
) -> bool:
    xy = ray_hits_w[:, :2].cpu()
    yaw = float(grid_yaw)
    _log(f"\n=== env {env_id} ===")
    _log(f"grid_yaw (heading_w): {math.degrees(yaw):.2f} deg")
    _log(f"grid_center_w xy: {grid_center_xy.cpu().tolist()}")
    _log(f"grid_shape (H, W): ({height}, {width}), resolution: {resolution} m")
    _log(f"ray_hits_w shape: {tuple(ray_hits_w.shape)}")
    _log("first 8 xy (flat index order):")
    for i in range(min(8, xy.shape[0])):
        _log(f"  [{i:3d}] x={xy[i, 0].item():+.4f}  y={xy[i, 1].item():+.4f}")

    if xy.shape[0] < width + 1:
        _log("[FAIL] Not enough rays to check row step.")
        return False

    d_col = xy[1] - xy[0]
    # Row-major flat = row * W + col. flat[W] is (row=1, col=0), not flat[W-1]->flat[W]
    # (that wraps from end-of-row to start-of-next-row diagonally across the patch).
    d_row = xy[width] - xy[0]
    expected_col = resolution * torch.tensor([math.cos(yaw), math.sin(yaw)])
    expected_row = resolution * torch.tensor([-math.sin(yaw), math.cos(yaw)])

    col_ok = torch.allclose(d_col, expected_col, atol=atol)
    row_ok = torch.allclose(d_row, expected_row, atol=atol)

    _log(f"\ncol+1 diff flat[1]-flat[0]:     {d_col.tolist()}")
    _log(f"expected resolution*(cos,sin): {expected_col.tolist()}  -> {'PASS' if col_ok else 'FAIL'}")
    _log(f"row+1 diff flat[{width}]-flat[0]: {d_row.tolist()}")
    _log(f"expected resolution*(-sin,cos): {expected_row.tolist()}  -> {'PASS' if row_ok else 'FAIL'}")

    if abs(yaw) > 0.05:
        axis_aligned_col = abs(d_col[1].item()) < atol and abs(abs(d_col[0].item()) - resolution) < atol
        axis_aligned_row = abs(d_row[0].item()) < atol and abs(abs(d_row[1].item()) - resolution) < atol
        if axis_aligned_col and axis_aligned_row:
            _log(
                "[FAIL] Robot yaw is non-zero but diffs look world-axis-aligned. "
                "ray_alignment='yaw' may not be active."
            )
            return False

    overall = col_ok and row_ok
    _log(f"\nenv {env_id} overall: {'PASS' if overall else 'FAIL'}")
    return overall


def main() -> int:
    _log(f"[inspect] task={args.task} num_envs={args.num_envs} test_yaw_deg={args.test_yaw_deg}")
    env_cfg = parse_env_cfg(
        args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric
    )
    # Play tasks add visualize_cam; strip it so headless runs without --enable_cameras.
    if getattr(env_cfg.scene, "visualize_cam", None) is not None:
        env_cfg.scene.visualize_cam = None
        _log("[inspect] Disabled visualize_cam for headless geometry check.")
    _log("[inspect] Creating env...")
    env = gym.make(args.task, cfg=env_cfg)
    _log("[inspect] Resetting env...")
    env.reset()
    _log("[inspect] Reset done.")

    robot = env.unwrapped.scene["robot"]
    height_scanner = env.unwrapped.scene["height_scanner"]
    height, width, resolution = _grid_shape_from_scanner(height_scanner)

    if args.test_yaw_deg is not None:
        _maybe_set_env_yaw(robot, args.env_id, args.test_yaw_deg)

    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    _log("[inspect] Stepping once...")
    env.step(action)
    _log("[inspect] Step done.")

    env_id = args.env_id
    if env_id >= args.num_envs:
        _log(f"[FAIL] env_id={env_id} >= num_envs={args.num_envs}")
        env.close()
        return 1

    ray_hits = height_scanner.data.ray_hits_w[env_id]
    valid = torch.isfinite(ray_hits).all(dim=-1)
    if not valid.all():
        bad = (~valid).sum().item()
        _log(f"[WARN] {bad} invalid ray hits (inf/nan); using finite subset for display only.")

    grid_yaw = robot.data.heading_w[env_id].item()
    grid_center = height_scanner.data.pos_w[env_id, :2]

    ok = inspect_env(
        ray_hits_w=ray_hits,
        grid_yaw=grid_yaw,
        grid_center_xy=grid_center,
        height=height,
        width=width,
        resolution=resolution,
        env_id=env_id,
        atol=args.atol,
    )

    if ok:
        _log("\n[inspect] Scanner frame contract: PASS")
    else:
        _log("\n[inspect] Scanner frame contract: FAIL — fix grid_yaw / gridmap_utils before Task 1+")

    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        _log("[inspect] Unhandled exception:")
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    sys.exit(exit_code)
