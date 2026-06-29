# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AME Locomotion is a reproduction of the paper "Attention-Based Map Encoding for Learning Generalized Legged Locomotion" (AME). It trains a Unitree G1 29-DoF humanoid robot to walk over rough terrain using a CNN + Multi-Head Attention terrain encoder and PPO.

- **Simulation:** NVIDIA Isaac Sim 5.0.0 + Isaac Lab 2.3.0
- **Robot:** Unitree G1 29-DoF
- **RL:** PPO via custom RSL-RL fork
- **Observation:** 3D elevation map (33x21 grid) + proprioception

## Project Structure

```
source/ame_locomotion/      -- Isaac Lab extension (tasks, MDP, terrains, robot assets)
rsl_rl/                     -- Custom RSL-RL with AME network modules
scripts/rsl_rl/             -- Training, play, export scripts
scripts/sim2sim/            -- Sim2sim analysis helpers (HeightMLP obs inspection, package checks)
deploy_for_AME/             -- C++ sim2sim skeleton (WIP, see AME_SIM2SIM_NOTES.md)
unitree_model/              -- Robot asset files (URDF/USD/meshes) referenced by env configs
pretrained/                 -- Pre-trained checkpoints (ame1.pt, ame2.pt)
run_train.sh / run_play.sh  -- One-shot launchers used by README quick-start
```

> **Note on gym registration:** Env IDs are registered in `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/__init__.py`, **not** the parent `ame_locomotion/__init__.py` (which only contains commented-out template code).

### Key Architecture Points

- **AME Encoder:** `rsl_rl/rsl_rl/modules/actor_critic_encoder.py` — CNN (Conv2d 3→16→64) extracts terrain features from 3D elevation map, then MultiheadAttention (16 heads, dim=64) fuses proprioception query with terrain key/value.
- **HeightMLP Baseline:** `rsl_rl/rsl_rl/modules/actor_critic_terrain_mlp.py` — MLP-based terrain encoder (flat height samples instead of AME attention).
- **Env Config:** `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py` — Central config with `FINETUNE = True/False` toggle for two-stage training.
- **Observations:** `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/observations.py` — `elevation_map()` (3D xyz for AME), `height_samples()` (z-only for HeightMLP).
- **Terrains:** `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/terrains/` — Stage 1 rough terrain, Stage 2 finetune terrain (harder).
- **PPO Configs:** `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/agents/ame_rsl_rl_ppo_cfg.py` — PPO hyperparameters and runner config.

### Registered Gym Environments

Variants exist to ablate the **terrain encoder** (AME vs. HeightMLP), the **robot asset loader** (default USD vs. URDF), and the **foot collision geometry** (capsule vs. STL mesh). All `-Play-v0` variants are evaluation configs (fewer envs, no domain randomization).

| ID | Description |
|---|---|
| `AME-G1-29DOF-v0` / `-Play-v0` | AME encoder, default USD asset |
| `AME-G1-29DOF-HeightMLP-v0` / `-Play-v0` | HeightMLP baseline, default USD asset |
| `AME-G1-29DOF-HeightMLP-ZeroCmd-Play-v0` | HeightMLP play with zero command (gait stress test) |
| `AME-G1-29DOF-USD-FootSTL-HeightMLP-v0` / `-Play-v0` | USD asset with STL foot mesh collision, HeightMLP |
| `AME-G1-29DOF-USD-FootSTL-Forward-HeightMLP-v0` / `-Play-v0` | Same as above, forward-only command range |
| `AME-G1-29DOF-URDF-v0` / `-Play-v0` | URDF-loaded robot (improved foot collision), AME encoder |
| `AME-G1-29DOF-URDF-HeightMLP-v0` / `-Play-v0` | URDF-loaded robot, HeightMLP baseline |

## Commands

### Installation

```bash
# Install the Isaac Lab extension
python -m pip install -e source/ame_locomotion

# Install the custom RSL-RL
python -m pip install -e rsl_rl
```

### Training

```bash
# Stage 1 (rough terrain pretraining)
python scripts/rsl_rl/train.py --task AME-G1-29DOF-v0 --max_iterations 15000 --headless

# Stage 2 (finetuning) — set FINETUNE=True in velocity_env_cfg_29dof.py first
python scripts/rsl_rl/train.py --task AME-G1-29DOF-v0 --max_iterations 15000 --headless --resume

# HeightMLP baseline
python scripts/rsl_rl/train.py --task AME-G1-29DOF-HeightMLP-v0 --max_iterations 15000 --headless

# With logger
python scripts/rsl_rl/train.py --task AME-G1-29DOF-v0 --headless --logger wandb --log_project_name ame_locomotion
```

### Evaluation

```bash
python scripts/rsl_rl/play.py --task AME-G1-29DOF-Play-v0 --checkpoint pretrained/ame1.pt --num_envs 1 --video
python scripts/rsl_rl/play.py --task AME-G1-29DOF-Play-v0 --checkpoint pretrained/ame2.pt --num_envs 1 --vis_attention --save_attention_weights
```

### Policy Export

```bash
python scripts/rsl_rl/export_policy.py --checkpoint <path/to/model.pt> --output_dir <dir>
```

This exports ONNX + JIT + deploy.yaml and validates PyTorch vs ONNX alignment (< 1e-4 max error).

### Validation & Testing

```bash
# Validate export logic (no Isaac Sim needed, unit-test style)
python scripts/rsl_rl/validate_export_logic.py

# List registered environments
python scripts/list_envs.py

# Test with random/zero agent (no Isaac Sim needed)
python scripts/random_agent.py --task AME-G1-29DOF-HeightMLP-v0 --num_envs 1
python scripts/zero_agent.py --task AME-G1-29DOF-HeightMLP-v0 --num_envs 1

# Offline HeightMLP export (no Isaac Sim needed)
python scripts/rsl_rl/export_height_mlp_offline.py --checkpoint <path>
```

### Linting

```bash
pre-commit run --all-files
```

Config: `.pre-commit-config.yaml` — black (line-length 120), flake8, isort, pyupgrade, codespell, license headers.

### Training CLI Arguments

Key args (see `scripts/rsl_rl/cli_args.py`): `--resume`, `--load_run`, `--checkpoint`, `--run_name`, `--logger {wandb,tensorboard,neptune}`, `--log_project_name`.

## Two-Stage Training

1. **Stage 1 (pretraining):** Train with `ROUGH_TERRAINS_CFG` (easier: step height 0.05-0.20m).
2. **Stage 2 (finetuning):** Set `FINETUNE = True` in `velocity_env_cfg_29dof.py` to switch to `FINETUNE_ROUGH_TERRAINS_CFG` (harder: step height 0.05-0.25m), then resume training.

## AME Encoder Config

Key parameters in `ActorCriticEncoder` (default):
- `map_scan_dim = (33, 21, 3)` — terrain grid dimensions
- `mha_dim = 64` — attention feature dimension
- `num_heads = 16` — attention heads
- `cnn_downsample = True` — stride-2 in first conv to reduce sequence length
- `attach_global = False` — AME2-style global context (set True for ame2.pt)
- `num_proprio = 66` — proprioceptive state dimension

### Pretrained checkpoints

| File | Config | Notes |
|---|---|---|
| `pretrained/ame1.pt` | `attach_global=False` | Default AME setup. Use plain `--vis_attention` for clean attention maps. |
| `pretrained/ame2.pt` | `attach_global=True` | AME2-style global context (MLP + max-pool). Higher performance but attention is less interpretable and training is heavier. |

When evaluating, the checkpoint's `attach_global` flag must match the env config's encoder construction — loading `ame2.pt` against a default `attach_global=False` model will fail with a shape mismatch.

## Sim2sim Deployment Status

`deploy_for_AME/` is a **work-in-progress** C++ skeleton copied from `unitree_rl_lab/deploy`. The shared framework (FSM, observation/action managers, ONNX Runtime runner, Unitree articulation adapter) is imported as-is. The AME-specific work tracked in `deploy_for_AME/AME_SIM2SIM_NOTES.md` is still open:

1. Export full AME inference graph to ONNX (CNN/MHA terrain path for AME; `terrain_encoder + actor` for HeightMLP).
2. Generate AME `deploy.yaml` from Isaac Lab env config (obs order, scales, action scale/offset, PD gains, joint order).
3. Add AME height observations to C++ obs registry (flat zeros → MuJoCo raycast).
4. Replace copied Unitree policy/config examples with AME packages.
5. Validate PyTorch vs ONNX outputs before MuJoCo sim2sim.

Treat the deploy path as a scaffold, not a finished pipeline.
