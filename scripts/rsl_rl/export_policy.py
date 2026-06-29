# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Export a trained RSL RL policy to ONNX/JIT format and generate deploy.yaml.

Usage:
    python scripts/rsl_rl/export_policy.py \
        --task AME-G1-29DOF-HeightMLP-Play-v0 \
        --checkpoint logs/rsl_rl/g1_height_mlp/2026-05-22_20-47-07/model_14999.pt \
        --output_dir deploy/robots/g1_29dof/config/policy/ame_height_mlp/v0

The script:
    1. Loads the trained policy from checkpoint
    2. Exports to ONNX (and optionally Torch JIT)
    3. Generates deploy.yaml with obs/action config
    4. Validates PyTorch vs ONNX output alignment (max_abs_error < 1e-4)
"""

import argparse
import os
import sys

PROJ_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(PROJ_ROOT_DIR, "rsl_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Export RSL-RL policy to deployable format.")
parser.add_argument("--task", type=str, default="AME-G1-29DOF-HeightMLP-Play-v0",
                    help="Name of the task (used to load config).")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Output directory for exported policy and config.")
parser.add_argument("--onnx_only", action="store_true", default=False,
                    help="Export ONNX only (skip JIT export).")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments for validation.")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import yaml

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from exporter import export_policy_as_onnx, export_policy_as_jit, _OnnxPolicyExporter

import ame_locomotion.tasks  # noqa: F401


def _build_height_mlp_obs_config() -> dict:
    """Build the observation config for the HeightMLP policy's deploy.yaml.

    This is the known observation structure for the G1 29DOF HeightMLP policy:
      1. base_ang_vel (3)     - scale 0.2
      2. projected_gravity (3) - scale 1.0
      3. velocity_commands (3) - scale 1.0
      4. joint_pos_rel (29)    - scale 1.0
      5. joint_vel_rel (29)    - scale 0.05
      6. last_action (29)      - scale 1.0
      7. height_samples (96)   - scale 1.0

    All history_length=1 (no history stacking for HeightMLP training).
    """
    def term(scale, params=None, history_length=1):
        return {
            "params": {} if params is None else params,
            "clip": None,
            "scale": scale,
            "history_length": history_length,
        }

    return {
        "base_ang_vel": term([0.2, 0.2, 0.2]),
        "projected_gravity": term([1.0, 1.0, 1.0]),
        "keyboard_velocity_commands": term(
            [1.0, 1.0, 1.0],
            {
                "command_name": "base_velocity",
                "default": [0.0, 0.0, 0.0],
                "forward": 1.0,
                "stop": 0.0,
                "lateral": 0.0,
                "yaw": 0.5,
            },
        ),
        "joint_pos_rel": term([1.0] * 29),
        "joint_vel_rel": term([0.05] * 29),
        "last_action": term([1.0] * 29),
        "height_samples": term([1.0] * 96),
    }


# G1 29DOF joint order in USD (matching SDK names / Isaac Lab articulation order)
# This is the canonical joint_ids_map from the existing velocity deploy.yaml.
# Maps from SDK-name order (Isaac Lab internal joint indices) to controller joint order.
_G1_29DOF_JOINT_IDS_MAP = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]

# G1 29DOF deploy gains copied from the existing velocity deploy.yaml. Keep these
# in controller joint order until deploy config generation is fully automated.
_G1_29DOF_JOINT_STIFFNESS = [
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
    200.0, 200.0, 200.0,
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
]
_G1_29DOF_JOINT_DAMPING = [
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
    5.0, 5.0, 5.0,
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
]

# Default joint positions in controller joint order (matching joint_ids_map).
# Values from the existing velocity deploy.yaml for G1 29DOF.
_G1_29DOF_DEFAULT_JOINT_POS = [
    -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.3, 0.3, 0.3, 0.3, -0.2, -0.2,
    0.25, -0.25, 0.0, 0.0, 0.0, 0.0,
    0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0,
]


def _generate_deploy_yaml(
    env,
    env_cfg: ManagerBasedRLEnvCfg,
    policy_nn: object,
    output_dir: str,
):
    """Generate deploy.yaml for the HeightMLP policy.

    Uses known constants for G1 29DOF robot parameters and constructs
    the observation/action config from the policy type.

    Args:
        env: The wrapped gym environment (after reset).
        env_cfg: The environment configuration.
        policy_nn: The policy neural network module.
        output_dir: Path to the policy version directory (e.g., .../v0/).
    """
    params_dir = os.path.join(output_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    num_joints = 29
    terrain_obs_dim = getattr(policy_nn, "terrain_obs_dim", 96)

    # ---- Action config ----
    action_scale = 0.25
    if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "joint_pos"):
        action_scale = float(env_cfg.actions.joint_pos.scale)

    # ---- Command config ----
    # Use the training command range, not the Play env range. Play configs clamp
    # lin_vel_x to [1.0, 1.0], which would make deploy always command forward motion.
    command_ranges = {
        "lin_vel_x": [0.0, 1.5],
        "lin_vel_y": [0.0, 0.0],
        "ang_vel_z": [-1.0, 1.0],
        "heading": None,
    }

    # ---- Observation config ----
    obs_config = _build_height_mlp_obs_config()

    # ---- Height MLP specific config ----
    height_samples_cfg = {
        "mode": "flat",
        "resolution": 0.15,
        "size": [1.65, 1.05],
        "num_samples": terrain_obs_dim,
        "value_bias": 0.0,
        "clip": [-1.2, 0.0],
        "override_enabled": False,
        "override_value": -0.854,
    }

    # ---- Assemble deploy config ----
    deploy_cfg = {
        "step_dt": env.unwrapped.step_dt,
        "terrain_obs_dim": terrain_obs_dim,
        "joint_ids_map": _G1_29DOF_JOINT_IDS_MAP,
        "stiffness": _G1_29DOF_JOINT_STIFFNESS,
        "damping": _G1_29DOF_JOINT_DAMPING,
        "default_joint_pos": _G1_29DOF_DEFAULT_JOINT_POS,
        "commands": {
            "base_velocity": {
                "ranges": command_ranges,
            },
        },
        "actions": {
            "JointPositionAction": {
                "clip": None,
                "joint_names": [".*"],
                "scale": [action_scale] * num_joints,
                "offset": _G1_29DOF_DEFAULT_JOINT_POS,
                "joint_ids": None,
            },
        },
        "observations": obs_config,
        "height_samples": height_samples_cfg,
        "ablation": {
            "zero_last_action": False,
            "base_ang_vel": {
                "lowpass_alpha": 1.0,
                "clip": None,
            },
            "joint_vel_rel": {
                "lowpass_alpha": 1.0,
                "clip": None,
            },
        },
    }

    # Write deploy.yaml
    yaml_path = os.path.join(params_dir, "deploy.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(deploy_cfg, f, default_flow_style=None, sort_keys=False)
    print(f"[INFO] deploy.yaml saved to: {yaml_path}")

    return deploy_cfg


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    """Main export routine."""

    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required. Provide a path to a trained model checkpoint (.pt file).")

    # Override configurations with CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = agent_cfg.device if agent_cfg.device is not None else "cuda:0"

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Output dir: {args_cli.output_dir}")

    # Load checkpoint path
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading checkpoint from: {resume_path}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # Convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner and load checkpoint
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Export only supports OnPolicyRunner, got: {agent_cfg.class_name}")

    runner.load(resume_path)

    # Get inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Extract normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    print(f"[INFO] Policy type: {type(policy_nn).__name__}")

    # ---- Export ----
    exported_dir = os.path.join(args_cli.output_dir, "exported")
    os.makedirs(exported_dir, exist_ok=True)

    # Export to ONNX
    onnx_path = os.path.join(exported_dir, "policy.onnx")
    print(f"[INFO] Exporting ONNX to: {onnx_path}")
    export_policy_as_onnx(
        policy_nn,
        normalizer=normalizer,
        path=exported_dir,
        filename="policy.onnx",
        verbose=False,
    )
    print(f"[INFO] ONNX export complete.")

    # Optionally export to Torch JIT
    if not args_cli.onnx_only:
        jit_path = os.path.join(exported_dir, "policy.pt")
        print(f"[INFO] Exporting Torch JIT to: {jit_path}")
        export_policy_as_jit(
            policy_nn,
            normalizer=normalizer,
            path=exported_dir,
            filename="policy.pt",
        )
        print(f"[INFO] JIT export complete.")

    # ---- Generate deploy.yaml ----
    _generate_deploy_yaml(env, env_cfg, policy_nn, args_cli.output_dir)

    # ---- Validate PyTorch vs ONNX alignment ----
    _validate_export(policy_nn, normalizer, onnx_path, env)

    print("[INFO] Export complete!")
    print(f"[INFO] Output structure:")
    print(f"  {args_cli.output_dir}/")
    print(f"  ├── exported/")
    print(f"  │   └── policy.onnx")
    if not args_cli.onnx_only:
        print(f"  │   └── policy.pt")
    print(f"  └── params/")
    print(f"      └── deploy.yaml")

    # Close the simulator
    env.close()


def _validate_export(policy_nn, normalizer, onnx_path, env):
    """Validate PyTorch vs ONNX output alignment.

    Creates the exporter wrapper, runs PyTorch inference,
    runs ONNX Runtime inference, and compares outputs.
    """
    # Compute input obs dimension
    if hasattr(policy_nn, "actor_proprio_dim") and hasattr(policy_nn, "terrain_obs_dim"):
        obs_dim = policy_nn.actor_proprio_dim + policy_nn.terrain_obs_dim
        action_dim = policy_nn.actor[-1].out_features if hasattr(policy_nn.actor[-1], "out_features") else -1
    elif hasattr(policy_nn, "actor"):
        obs_dim = policy_nn.actor[0].in_features
        action_dim = policy_nn.actor[-1].out_features if hasattr(policy_nn.actor[-1], "out_features") else -1
    else:
        raise RuntimeError("Cannot determine obs/action dims from policy; refusing to skip export validation.")
    if obs_dim <= 0 or action_dim <= 0:
        raise RuntimeError(f"Invalid export dimensions: obs_dim={obs_dim}, action_dim={action_dim}.")

    print(f"[INFO] Validating export: obs_dim={obs_dim}, action_dim={action_dim}")

    # Create exporter wrapper for PyTorch inference
    exporter = _OnnxPolicyExporter(policy_nn, normalizer)
    exporter.to("cpu")
    exporter.eval()

    # Generate random test obs
    torch.manual_seed(42)
    test_obs = torch.randn(1, obs_dim, device="cpu")

    # Run PyTorch inference
    with torch.inference_mode():
        actions_pt = exporter(test_obs)

    # Run ONNX Runtime inference
    try:
        import onnxruntime as ort

        # Allow CPU fallback if ONNX Runtime GPU not available
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )

        ort_inputs = {ort_session.get_inputs()[0].name: test_obs.numpy()}
        actions_onnx = ort_session.run(None, ort_inputs)[0]
        actions_onnx = torch.from_numpy(actions_onnx)

        # Compute max absolute error
        max_abs_error = (actions_pt - actions_onnx).abs().max().item()

        print(f"  obs_dim: {obs_dim}")
        print(f"  action_dim: {action_dim}")
        print(f"  max_abs_error: {max_abs_error:.2e}")

        if max_abs_error < 1e-4:
            print("[PASS] ONNX export validated: PyTorch and ONNX outputs match (max_abs_error < 1e-4).")
        else:
            raise RuntimeError(f"ONNX validation failed: max_abs_error={max_abs_error:.2e} >= 1e-4.")

    except ImportError:
        raise RuntimeError("onnxruntime is required for export validation but is not installed.")
    except Exception as e:
        raise RuntimeError(f"ONNX validation failed: {e}") from e


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
