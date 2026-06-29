#!/usr/bin/env python3
"""Lightweight ONNX-side observation sensitivity checks for AME HeightMLP."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEPLOY_PYTHON = os.path.join(PROJECT_ROOT, "deploy", "python")
if DEPLOY_PYTHON not in sys.path:
    sys.path.insert(0, DEPLOY_PYTHON)

from ame_inference import HeightMlpObservationBuilder, OnnxPolicyRunner, load_policy_package


def summarize_case(name: str, obs: np.ndarray, runner: OnnxPolicyRunner, package) -> np.ndarray:
    raw_action = runner.act(obs)[0]
    q_target = package.action.process(raw_action)
    delta = q_target - package.default_joint_pos
    top = np.argsort(np.abs(raw_action))[-3:][::-1]
    print(name)
    print(f"  max_raw={float(np.max(np.abs(raw_action))):.4f}")
    print(f"  mean_abs_raw={float(np.mean(np.abs(raw_action))):.4f}")
    print(f"  max_abs_dq={float(np.max(np.abs(delta))):.4f}")
    print(f"  q_range=[{float(np.min(q_target)):.4f}, {float(np.max(q_target)):.4f}]")
    print(f"  top3={[(int(i), float(raw_action[i])) for i in top]}")
    return raw_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep AME HeightMLP observations against an exported ONNX policy.")
    parser.add_argument(
        "policy_dir",
        nargs="?",
        default="deploy/robots/g1_29dof/config/policy/ame_height_mlp/v0",
        help="Policy version directory containing params/deploy.yaml and exported/policy.onnx.",
    )
    parser.add_argument(
        "--height-values",
        type=float,
        nargs="*",
        default=[-1.2, -0.854, -0.6, -0.4, 0.0],
        help="Height values to sweep for all 96 height samples.",
    )
    parser.add_argument("--feedback-steps", type=int, default=8, help="Number of last_action feedback rollout steps.")
    args = parser.parse_args()

    package = load_policy_package(args.policy_dir)
    runner = OnnxPolicyRunner(package.onnx_path)
    height_dim = next(term.dim for term in package.observation_terms if term.name == "height_samples")

    print(f"policy_dir: {package.root}")
    print(f"obs_dim: {package.obs_dim}, action_dim: {package.action_dim}")
    print(f"default_root_pos_w: {package.root_pos_w.tolist()}")
    print(f"height_cfg: {package.height_samples_cfg}")

    for height in args.height_values:
        builder = HeightMlpObservationBuilder(package)
        obs = builder.pack(
            projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            velocity_command=np.zeros(3, dtype=np.float32),
            joint_pos=package.default_joint_pos,
            joint_vel=np.zeros(package.joint_dim, dtype=np.float32),
            last_action=np.zeros(package.action_dim, dtype=np.float32),
            height_samples=np.full(height_dim, height, dtype=np.float32),
            root_pos_w=package.root_pos_w,
        )
        summarize_case(f"height={height:.3f}", obs, runner, package)

    builder = HeightMlpObservationBuilder(package)
    obs = builder.pack(
        base_ang_vel=np.array([0.0, 2.0, -2.0], dtype=np.float32),
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        velocity_command=np.zeros(3, dtype=np.float32),
        joint_pos=package.default_joint_pos,
        joint_vel=np.zeros(package.joint_dim, dtype=np.float32),
        last_action=np.zeros(package.action_dim, dtype=np.float32),
        height_samples=np.full(height_dim, -0.854, dtype=np.float32),
        root_pos_w=package.root_pos_w,
    )
    summarize_case("gyro=(0,2,-2)", obs, runner, package)

    builder = HeightMlpObservationBuilder(package)
    obs = builder.pack(
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        velocity_command=np.zeros(3, dtype=np.float32),
        joint_pos=package.default_joint_pos,
        joint_vel=np.full(package.joint_dim, 2.0, dtype=np.float32),
        last_action=np.zeros(package.action_dim, dtype=np.float32),
        height_samples=np.full(height_dim, -0.854, dtype=np.float32),
        root_pos_w=package.root_pos_w,
    )
    summarize_case("joint_vel=2.0", obs, runner, package)

    print(f"last_action feedback rollout ({args.feedback_steps} steps)")
    builder = HeightMlpObservationBuilder(package)
    last_action = np.zeros(package.action_dim, dtype=np.float32)
    for step in range(args.feedback_steps):
        obs = builder.pack(
            projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            velocity_command=np.zeros(3, dtype=np.float32),
            joint_pos=package.default_joint_pos,
            joint_vel=np.zeros(package.joint_dim, dtype=np.float32),
            last_action=last_action,
            height_samples=np.full(height_dim, -0.854, dtype=np.float32),
            root_pos_w=package.root_pos_w,
        )
        last_action = summarize_case(f"feedback_step={step}", obs, runner, package)


if __name__ == "__main__":
    main()
