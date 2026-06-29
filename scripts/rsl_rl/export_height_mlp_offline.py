# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export the G1 29DOF HeightMLP policy without launching Isaac Sim.

This script is intended for low-VRAM machines. It does not create an Isaac app,
does not instantiate an environment, and does not run Play. It reconstructs the
ActorCriticTerrainMlp policy from the saved agent.yaml/checkpoint on CPU.
"""

from __future__ import annotations

import argparse
import os
import sys

PROJ_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(PROJ_ROOT_DIR, "rsl_rl"))

import torch
import yaml

from exporter import _OnnxPolicyExporter, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.modules.actor_critic_terrain_mlp import ActorCriticTerrainMlp


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


_G1_29DOF_JOINT_IDS_MAP = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
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
_G1_29DOF_DEFAULT_JOINT_POS = [
    -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.3, 0.3, 0.3, 0.3, -0.2, -0.2,
    0.25, -0.25, 0.0, 0.0, 0.0, 0.0,
    0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0,
]


def _obs_term(scale: list[float], params: dict | None = None, history_length: int = 1) -> dict:
    return {
        "params": {} if params is None else params,
        "clip": None,
        "scale": scale,
        "history_length": history_length,
    }


def _build_deploy_yaml(policy: ActorCriticTerrainMlp, output_dir: str):
    params_dir = os.path.join(output_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    terrain_obs_dim = int(policy.terrain_obs_dim)
    deploy_cfg = {
        "step_dt": 0.02,
        "terrain_obs_dim": terrain_obs_dim,
        "root_pos_w": [0.0, 0.0, 0.8],
        "joint_ids_map": _G1_29DOF_JOINT_IDS_MAP,
        "stiffness": _G1_29DOF_JOINT_STIFFNESS,
        "damping": _G1_29DOF_JOINT_DAMPING,
        "default_joint_pos": _G1_29DOF_DEFAULT_JOINT_POS,
        "commands": {
            "base_velocity": {
                "ranges": {
                    "lin_vel_x": [0.0, 1.5],
                    "lin_vel_y": [0.0, 0.0],
                    "ang_vel_z": [-1.0, 1.0],
                    "heading": None,
                },
            },
        },
        "actions": {
            "JointPositionAction": {
                "clip": None,
                "joint_names": [".*"],
                "scale": [0.25] * 29,
                "offset": _G1_29DOF_DEFAULT_JOINT_POS,
                "joint_ids": None,
            },
        },
        "observations": {
            "base_ang_vel": _obs_term([0.2, 0.2, 0.2]),
            "projected_gravity": _obs_term([1.0, 1.0, 1.0]),
            "keyboard_velocity_commands": _obs_term(
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
            "joint_pos_rel": _obs_term([1.0] * 29),
            "joint_vel_rel": _obs_term([0.05] * 29),
            "last_action": _obs_term([1.0] * 29),
            "height_samples": _obs_term([1.0] * terrain_obs_dim),
        },
        "height_samples": {
            "mode": "flat",
            "resolution": 0.15,
            "size": [1.65, 1.05],
            "num_samples": terrain_obs_dim,
            "value_bias": 0.0,
            "clip": [-1.2, 0.0],
            "override_enabled": False,
            "override_value": -0.854,
        },
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

    yaml_path = os.path.join(params_dir, "deploy.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(deploy_cfg, f, Dumper=_NoAliasDumper, default_flow_style=None, sort_keys=False)
    print(f"[INFO] deploy.yaml saved to: {yaml_path}")


def _load_policy(agent_yaml: str, checkpoint: str, obs_dim: int, num_actions: int) -> ActorCriticTerrainMlp:
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)

    policy_cfg = dict(agent_cfg["policy"])
    class_name = policy_cfg.pop("class_name")
    if class_name != "ActorCriticTerrainMlp":
        raise ValueError(f"Offline exporter only supports ActorCriticTerrainMlp, got {class_name}.")

    # Infer the correct critic obs dim from the checkpoint state dict.
    # The critic typically sees more observations than the actor (e.g., base_lin_vel).
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = loaded["model_state_dict"]

    terrain_obs_dim = int(policy_cfg.get("terrain_obs_dim", 96))
    terrain_embedding_dim = int(policy_cfg.get("terrain_embedding_dim", 64))

    # critic.0.weight: [hidden, critic_input_dim]
    # critic_input_dim = critic_proprio_dim + terrain_embedding_dim
    # critic_obs_dim = critic_proprio_dim + terrain_obs_dim
    critic_input_dim = state_dict["critic.0.weight"].shape[1]
    critic_proprio_dim = critic_input_dim - terrain_embedding_dim
    critic_obs_dim = critic_proprio_dim + terrain_obs_dim

    print(f"[DEBUG] actor_obs_dim={obs_dim}, critic_obs_dim={critic_obs_dim}, "
          f"terrain_obs_dim={terrain_obs_dim}, terrain_embedding_dim={terrain_embedding_dim}")

    fake_obs = {
        "policy": torch.zeros(1, obs_dim),
        "critic": torch.zeros(1, critic_obs_dim),
    }
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = ActorCriticTerrainMlp(fake_obs, obs_groups, num_actions, **policy_cfg)

    policy.load_state_dict(state_dict, strict=True)
    policy.to("cpu")
    policy.eval()
    return policy


def _validate(policy: ActorCriticTerrainMlp, normalizer, onnx_path: str, obs_dim: int):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for export validation.") from exc

    exporter = _OnnxPolicyExporter(policy, normalizer)
    exporter.to("cpu")
    exporter.eval()

    torch.manual_seed(42)
    test_obs = torch.randn(1, obs_dim)
    with torch.inference_mode():
        actions_pt = exporter(test_obs)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actions_onnx = session.run(None, {session.get_inputs()[0].name: test_obs.numpy()})[0]
    actions_onnx = torch.from_numpy(actions_onnx)
    max_abs_error = (actions_pt - actions_onnx).abs().max().item()

    print(f"  obs_dim: {obs_dim}")
    print(f"  action_dim: {actions_pt.shape[-1]}")
    print(f"  max_abs_error: {max_abs_error:.2e}")
    if max_abs_error >= 1e-4:
        raise RuntimeError(f"ONNX validation failed: max_abs_error={max_abs_error:.2e} >= 1e-4.")
    print("[PASS] ONNX export validated.")


def main():
    parser = argparse.ArgumentParser(description="Offline export HeightMLP policy to deploy package.")
    parser.add_argument("--checkpoint", required=True, help="Path to model_*.pt checkpoint.")
    parser.add_argument("--agent_yaml", default=None, help="Path to agent.yaml. Defaults to checkpoint run params/agent.yaml.")
    parser.add_argument("--output_dir", required=True, help="Output policy version directory.")
    parser.add_argument("--obs_dim", type=int, default=192, help="Full HeightMLP observation dimension.")
    parser.add_argument("--num_actions", type=int, default=29, help="Action dimension.")
    parser.add_argument("--onnx_only", action="store_true", help="Skip TorchScript export.")
    args = parser.parse_args()

    agent_yaml = args.agent_yaml
    if agent_yaml is None:
        agent_yaml = os.path.join(os.path.dirname(args.checkpoint), "params", "agent.yaml")

    print(f"[INFO] Loading policy from: {args.checkpoint}")
    print(f"[INFO] Loading agent config from: {agent_yaml}")
    policy = _load_policy(agent_yaml, args.checkpoint, args.obs_dim, args.num_actions)
    normalizer = policy.actor_obs_normalizer if hasattr(policy, "actor_obs_normalizer") else None

    exported_dir = os.path.join(args.output_dir, "exported")
    os.makedirs(exported_dir, exist_ok=True)

    print(f"[INFO] Exporting ONNX to: {os.path.join(exported_dir, 'policy.onnx')}")
    export_policy_as_onnx(policy, normalizer=normalizer, path=exported_dir, filename="policy.onnx")
    if not args.onnx_only:
        print(f"[INFO] Exporting TorchScript to: {os.path.join(exported_dir, 'policy.pt')}")
        export_policy_as_jit(policy, normalizer=normalizer, path=exported_dir, filename="policy.pt")

    _build_deploy_yaml(policy, args.output_dir)
    _validate(policy, normalizer, os.path.join(exported_dir, "policy.onnx"), args.obs_dim)

    print("[INFO] Offline export complete.")


if __name__ == "__main__":
    main()
