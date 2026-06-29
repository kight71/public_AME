#!/usr/bin/env python3
"""Offline smoke test for an AME HeightMLP policy package.

This is the first holosoma-style deploy entry point: it validates a policy
package, builds observations in deploy.yaml order, runs ONNX, and converts the
raw action into target joint positions.
"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Check an AME HeightMLP deploy package.")
    parser.add_argument(
        "policy_dir",
        nargs="?",
        default="deploy/robots/g1_29dof/config/policy/ame_height_mlp/v0",
        help="Policy version directory containing params/deploy.yaml and exported/policy.onnx.",
    )
    parser.add_argument("--skip-onnx", action="store_true", help="Only validate package and observation packing.")
    args = parser.parse_args()

    package = load_policy_package(args.policy_dir)
    print(f"policy_dir: {package.root}")
    print(f"deploy_yaml: {package.deploy_yaml}")
    print(f"onnx_path: {package.onnx_path}")
    print(f"step_dt: {package.step_dt:.4f}")
    print(f"joint_dim: {package.joint_dim}")
    print(f"action_dim: {package.action_dim}")
    print(f"obs_dim: {package.obs_dim}")
    print("obs_terms:")
    for term in package.observation_terms:
        print(f"  - {term.name}: dim={term.dim}, hist={term.history_length}")

    builder = HeightMlpObservationBuilder(package)
    obs = builder.zeros()
    print(f"packed_obs_shape: {tuple(obs.shape)}")
    if obs.shape != (1, package.obs_dim):
        raise RuntimeError(f"Packed obs shape {obs.shape} does not match expected (1, {package.obs_dim}).")

    if args.skip_onnx:
        print("onnx: skipped")
        return

    runner = OnnxPolicyRunner(package.onnx_path)
    print(f"onnx_inputs: {runner.input_names}")
    print(f"onnx_outputs: {runner.output_names}")
    raw_action = runner.act(obs)
    print(f"raw_action_shape: {tuple(raw_action.shape)}")
    if raw_action.shape[-1] != package.action_dim:
        raise RuntimeError(f"ONNX action dim {raw_action.shape[-1]} does not match expected {package.action_dim}.")

    q_target = package.action.process(raw_action)
    print(f"q_target_shape: {tuple(q_target.shape)}")
    print(f"q_target_min: {float(np.min(q_target)):.4f}")
    print(f"q_target_max: {float(np.max(q_target)):.4f}")
    print("check: passed")


if __name__ == "__main__":
    main()
