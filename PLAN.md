# Footstep-Planning Locomotion — Project Plan

> Living roadmap. Update this file as phases land or scope changes.
> Last updated: 2026-06-18

## Motivation

The current AME setup gives the policy strong terrain perception but still asks PPO to *discover* a gait from velocity-tracking reward alone. AME's terrain encoder cannot help until the policy can already walk, so the early-training exploration cost is identical to a non-AME policy. The fix we are pursuing: **replace the implicit "walk anywhere that matches velocity cmd" task signal with an explicit "land your feet at these xy z targets" task signal**. This compresses the exploration manifold from "all possible motions" to "all possible ways to reach the planned footsteps", and makes AME's elevation map a first-class input to the planning side, not just a reactive feature.

Reference paradigm: **DTC (Deep Tracking Control, Jenelten et al. 2024, ETH RSL)** — model-based footstep planner + RL tracker on ANYmal. We adapt the architecture to a 29-DoF humanoid (G1).

## Architecture (target)

```
                       ┌─────────────────────────────┐
                       │   AME terrain encoder       │
height_scanner ───────►│   (existing)                │
                       └──────────┬──────────────────┘
                                  │ terrain feature
                                  ▼
velocity_cmd ────► ┌──────────────────────────────┐
root_state ──────► │  Footstep Planner            │──► next N footsteps (xy, z, t_contact)
                   │  Phase 0: Raibert heuristic  │    + swing phase, contact mask
                   │  Phase 1: learned net        │
                   │  Phase 2: LIP / MPC          │
                   └──────────────────────────────┘
                                  │ footstep plan
                                  ▼
                   ┌──────────────────────────────┐
                   │  RL Tracker (policy)         │──► joint actions
                   │  reward: footstep tracking + │
                   │          contact phase +     │
                   │          regularizers        │
                   └──────────────────────────────┘
```

Key design choice: the planner lives as an Isaac Lab **CommandTerm**. This is idiomatic (mirrors `base_velocity` command), gives the planner a place to hold per-env state (stride phase, latched target), is naturally observable via `mdp.generated_commands`, and is decoupled from reward terms so we can iterate on each side independently.

## Phase Plan

| Phase | Deliverable | Status |
|---|---|---|
| **Phase 0** | Heuristic Raibert footstep planner as `CommandTerm`, no reward changes — verify that observable plan does not break current PPO and visualization is correct | **CODE DONE**, GUI eyeball check pending |
| **Phase 1** | Reward redesign: footstep swing-trajectory tracking + contact-phase consistency, on top of existing 21 baseline rewards. raibert_factor=0.0 to align with pretrained AME's natural gait. | **CODE DONE**, A/B training pending |
| **Phase 1.5** | Base-reward fix: replace sparse `termination_penalty` with dense `alive`, add `foot_clearance` shaping, lower `joint_deviation_arms`. See `G1RoughEnvCfg_Unitree` / `AME-G1-29DOF-Unitree-v0`. Diagnosed root cause of shuffle local-optimum: missing ~1.65/step positive shaping vs. unitree_rl_lab. | **CODE DONE 2026-06-19**, training in progress |
| **Phase 2** | Learned planner net (BC warmstart from Phase 0 heuristic, then PPO co-training with tracker) | TODO |
| **Phase 3** | Model-based planner (LIP / capture-point) with elevation-map projection — DTC humanoid analog | TODO |
| **Phase 4** | Sim2sim ONNX export of planner+tracker stack; integrate into `deploy_for_AME/` | TODO |

We do **Phase 0 first** to validate the integration plumbing and visualization before touching reward dynamics.

---

# Phase 0 — Heuristic Footstep Planner (current focus)

**Goal:** Add a Raibert-style footstep planner that runs as a `CommandTerm`, exposes its plan as an observation, and visualizes the planned footsteps in the GUI. **No reward changes in this phase** — the planner output is unused by training. We only verify it (a) does not break the existing pipeline, (b) produces sensible footsteps that we can visually verify on rough terrain, and (c) projects onto the elevation map correctly.

## File map

| File | Action | Responsibility |
|---|---|---|
| `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/commands.py` | Modify | Add `FootstepPlanCommand` class + `FootstepPlanCommandCfg`. Holds per-env stride phase, last contact pos, current swing target. |
| `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/planner.py` | Create | Pure-function geometric primitives: `raibert_target()`, `project_onto_elevation_map()`, `swing_trajectory()`. No Isaac Sim deps so it is unit-testable. |
| `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/observations.py` | Modify | Add `footstep_plan(env, command_name)` observation that flattens the plan into a fixed-shape tensor for the policy. |
| `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/__init__.py` | Modify | Re-export new symbols. |
| `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py` | Modify | Wire `FootstepPlanCommandCfg` into `CommandsCfg`, add observation term. **No reward changes.** |
| `tests/test_planner_geometry.py` | Create | Unit tests for `mdp/planner.py` pure functions (no Isaac Sim needed). |
| `scripts/debug_footstep_plan.py` | Create | Standalone script: load env, run zero policy, dump plan trajectories to npz for offline plotting. |

## Interfaces produced by Phase 0

Later phases will rely on these:

```python
# mdp/planner.py
def raibert_target(
    root_pos_w: torch.Tensor,        # (B, 3)
    root_lin_vel_w: torch.Tensor,    # (B, 3)
    root_yaw: torch.Tensor,          # (B,)
    vel_cmd_b: torch.Tensor,         # (B, 3) — vx, vy, wz in body frame
    hip_offset_b: torch.Tensor,      # (2, 3) — left/right hip offset in body frame
    t_swing: float,
    k_fb: float,                     # Raibert feedback gain
) -> torch.Tensor:                   # (B, 2, 3) — world-frame xyz target per foot
    ...

def project_onto_elevation_map(
    target_w: torch.Tensor,          # (B, 2, 3)
    height_map: torch.Tensor,        # (B, H, W) — z values, world frame
    map_origin_w: torch.Tensor,      # (B, 2) — xy of map[0,0]
    resolution: float,
    safety_radius: int = 2,
) -> torch.Tensor:                   # (B, 2, 3) — z replaced with safe local min within radius
    ...

# mdp/commands.py
class FootstepPlanCommand(CommandTerm):
    # per-env state
    phase: torch.Tensor              # (B,) in [0, 1)
    swing_foot: torch.Tensor         # (B,) int in {0=left, 1=right}
    target_w: torch.Tensor           # (B, 2, 3) — current planned target for each foot
    last_contact_w: torch.Tensor     # (B, 2, 3) — last touchdown pos per foot

    @property
    def command(self) -> torch.Tensor:  # (B, K) — flattened plan exposed to policy
        ...
```

## Bite-sized tasks

### Task 0.1: Pure geometry primitives (test-first)

**Files:**
- Create: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/planner.py`
- Create: `tests/test_planner_geometry.py`

**Why first:** Geometry is the only part we can test without spinning up Isaac Sim. Get the math right in isolation.

- [x] **Step 1:** Write `tests/test_planner_geometry.py` with three tests:
  - `test_raibert_target_zero_velocity`: zero velocity cmd → target lands directly under hip offset
  - `test_raibert_target_forward_velocity`: `vel_cmd_b = [0.5, 0, 0]`, `t_swing = 0.3` → target is ~0.075m ahead of hip in body x
  - `test_project_onto_elevation_map_flat`: flat map → projected z equals map height at target xy
  - `test_project_onto_elevation_map_step`: map with a 0.2m step → projected z picks the safe lower patch within `safety_radius`

- [x] **Step 2:** Run tests — expect ImportError on `planner` module.

- [x] **Step 3:** Implement `planner.py` with `raibert_target` and `project_onto_elevation_map`. Body→world rotation via yaw only (humanoid base z-axis ≈ world z most of the time). Use `torch.scatter` / `unfold` for the safety-radius local min.

- [x] **Step 4:** Run tests until all pass.

- [x] **Step 5:** Verify — `python -m pytest tests/test_planner_geometry.py -v`. Sanity check shapes and dtypes match the interface block above.

### Task 0.2: Swing trajectory primitive

**Files:**
- Modify: `mdp/planner.py` (append `swing_trajectory()`)
- Modify: `tests/test_planner_geometry.py` (append swing tests)

- [x] **Step 1:** Add test `test_swing_trajectory_endpoints`: at `phase=0` returns last contact, at `phase=1` returns target. At `phase=0.5` z is highest (apex). Use cubic / sinusoidal apex of 0.1m.

- [x] **Step 2:** Implement:

```python
def swing_trajectory(
    start_w: torch.Tensor,           # (B, 2, 3)
    end_w: torch.Tensor,             # (B, 2, 3)
    phase: torch.Tensor,             # (B, 2) in [0,1]
    apex: float = 0.10,
) -> torch.Tensor:                   # (B, 2, 3)
    """xy linear interp, z parabolic with given apex above max(start.z, end.z)."""
    ...
```

- [x] **Step 3:** Run tests.

### Task 0.3: `FootstepPlanCommand` term (Isaac Sim required)

**Files:**
- Modify: `mdp/commands.py`

- [x] **Step 1:** Add `FootstepPlanCommandCfg` dataclass with fields:

```python
@configclass
class FootstepPlanCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
    height_scanner_name: str = "height_scanner"
    velocity_command_name: str = "base_velocity"
    t_step: float = 0.6                   # full stride period
    t_swing_fraction: float = 0.5         # fraction of stride that is single-support
    raibert_k: float = 0.05
    apex: float = 0.10
    n_future_steps: int = 2               # plan horizon exposed to policy
    safety_radius: int = 2
    class_type: type = MISSING            # filled by subclass
```

- [x] **Step 2:** Implement `FootstepPlanCommand(CommandTerm)`:
  - In `__init__`: cache foot body ids, hip offsets (in body frame), grab height_scanner and velocity_command from `env.scene` / `env.command_manager`.
  - Per-env buffers: `phase (B,)`, `swing_foot (B,)`, `target_w (B,2,3)`, `last_contact_w (B,2,3)`, `plan_buffer (B, n_future_steps, 2, 3)`.
  - `_update_command()`:
    1. Advance phase by `env.step_dt / t_step`, wrap to [0,1).
    2. On phase crossing 0.5 → swap swing foot; latch current foot xy z as `last_contact_w`.
    3. Use `planner.raibert_target` for the swing leg target, `project_onto_elevation_map` to snap z.
    4. Roll forward N future steps for `plan_buffer`.
  - `_resample_command()`: reset phase to 0, set both feet `last_contact_w` to current foot positions.
  - `command` property: return `plan_buffer.flatten(start_dim=1)`.

- [x] **Step 3:** Add `class_type = FootstepPlanCommand` to the Cfg.

### Task 0.4: Observation term

**Files:**
- Modify: `mdp/observations.py`
- Modify: `mdp/__init__.py`

- [x] **Step 1:** Add at the bottom of `observations.py`:

```python
def footstep_plan(env: ManagerBasedRLEnv, command_name: str = "footstep_plan") -> torch.Tensor:
    """Returns flattened footstep plan in *body* frame (yaw-only) for the policy.
    Shape: (B, n_future_steps * 2 * 3).
    """
    cmd = env.command_manager.get_term(command_name)
    plan_w = cmd.plan_buffer                                # (B, N, 2, 3)
    # transform to body frame using root yaw — implement helper
    return _world_to_body_yaw(plan_w, env.scene["robot"]).flatten(start_dim=1)
```

- [x] **Step 2:** Implement `_world_to_body_yaw` helper (private).

- [x] **Step 3:** Re-export `footstep_plan` from `mdp/__init__.py`. *(Already wildcard-exported.)*

### Task 0.5: Wire into env cfg (no reward change)

**Files:**
- Modify: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py`

- [x] **Step 1:** In `CommandsCfg`, add:

```python
footstep_plan = mdp.FootstepPlanCommandCfg(
    foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    t_step=0.6, t_swing_fraction=0.5, raibert_k=0.05, apex=0.10,
    n_future_steps=2, safety_radius=2,
    resampling_time_range=(10.0, 10.0),
    debug_vis=True,
)
```

- [x] **Step 2:** In `ObservationsCfg.PolicyCfg`, add at the end (so we can A/B by toggling). *(Plumbed-in but commented out — see note in env cfg. Enabling it grows policy obs by 12 dims, which breaks `pretrained/ame1.pt` loading. Re-enable in Phase 1 when we retrain.)*

```python
footstep_plan = ObsTerm(
    func=mdp.footstep_plan,
    params={"command_name": "footstep_plan"},
)
```

- [x] **Step 3:** **Do not touch rewards.** Verify the env still constructs:

```bash
python scripts/list_envs.py
python scripts/zero_agent.py --task AME-G1-29DOF-HeightMLP-v0 --num_envs 1
```

Expected: no exceptions; observation shape grew by `n_future_steps * 2 * 3 = 12` floats.

### Task 0.6: Debug visualization

**Files:**
- Modify: `mdp/commands.py` (`FootstepPlanCommand._set_debug_vis_impl`, `_debug_vis_callback`)

- [x] **Step 1:** Implement debug viz: two small spheres per env at `target_w[:, 0]` (left, blue) and `target_w[:, 1]` (right, red); one bigger sphere at `last_contact_w` (semi-transparent). Use `isaaclab.markers.VisualizationMarkers` like other commands. *(Done inline with 0.3.)*

- [ ] **Step 2:** Verify in GUI:

```bash
python scripts/rsl_rl/play.py --task AME-G1-29DOF-Play-v0 \
    --checkpoint pretrained/ame1.pt --num_envs 1
```

Watch the robot walk; planned footsteps should track ahead of the swing foot and visibly snap onto step edges in rough terrain. **This is the eyeball acceptance criterion for Phase 0.**

### Task 0.7: Offline plan dump for analysis

**Files:**
- Create: `scripts/debug_footstep_plan.py`

- [x] **Step 1:** Script loads `AME-G1-29DOF-HeightMLP-Play-v0`, runs zero-action for 500 steps, records `(t, root_pos, root_vel, plan_buffer, foot_pos)` to `debug_plan.npz`.

- [ ] **Step 2:** Verify by loading the npz in a notebook and plotting xy trajectory of foot vs planned target — they should diverge while velocity command exists (zero-action policy can't track) but the **planner output itself** should be smooth and on the elevation map.

## Phase 0 acceptance criteria

1. All unit tests in `tests/test_planner_geometry.py` pass.
2. Existing training and play scripts still work; no regression on `AME-G1-29DOF-Play-v0` with `pretrained/ame1.pt`.
3. Footstep markers visible in GUI, planted on terrain, alternating left/right at ~0.6s period.
4. `scripts/debug_footstep_plan.py` produces a clean trajectory dump.
5. Brief writeup added below summarizing observed behavior on rough terrain (any surprises, planner failure modes — e.g. cliff projection, stride-length saturation).

---

# Phase 1 — Reward redesign

## Design decisions (locked in)

- **raibert_factor = 0.0**: residual eval showed pretrained AME walks with effective Raibert factor ≈ 0 (foot lands directly under hip, not Raibert-style push-off). Setting the planner to match this minimizes the initial bias between planner prediction and policy behavior. After verifying utility we can try 0.3 or 0.5 to push the policy toward more dynamic gaits.
- **Dense reward over sparse**: sparse touchdown reward is gameable on humanoids (foot drag, slide, etc.). Instead we use per-step swing-trajectory tracking + contact-phase consistency.
- **No obs change**: `footstep_plan` observation is wired but commented out so `pretrained/ame1.pt` remains loadable. The policy is shaped only by the new reward signals.
- **Additive, not replacing**: the 21 baseline reward terms (velocity tracking, regularizers, etc.) keep their original weights. Footstep rewards are added on top.

## Implemented terms

- `mdp.footstep_swing_tracking(std=0.08, apex=0.10)` — per-step dense reward using `planner.swing_trajectory(last_contact, target, swing_phase)` as reference. Masked to current swing foot only. Output ∈ [0, 2].
- `mdp.footstep_contact_phase(force_threshold=1.0)` — per-step ground-truth contact vs planned schedule. Output ∈ [0, 1].

## Registered task

- `AME-G1-29DOF-Footstep-v0` — same env as `AME-G1-29DOF-v0` but with `footstep_swing_tracking.weight=1.0` and `footstep_contact_phase.weight=0.5`.

## A/B training plan

For a clean sample-efficiency comparison, train both from scratch (no warm-start) for ~2-3k iters:

```bash
# Baseline (21 reward terms, no footstep reward)
.AME/bin/python scripts/rsl_rl/train.py --task AME-G1-29DOF-v0 \
    --max_iterations 3000 --headless --run_name baseline_v0 \
    --logger wandb --log_project_name ame_footstep_exp

# Treatment (+ footstep rewards)
.AME/bin/python scripts/rsl_rl/train.py --task AME-G1-29DOF-Footstep-v0 \
    --max_iterations 3000 --headless --run_name footstep_v0 \
    --logger wandb --log_project_name ame_footstep_exp
```

Compare metrics:
- `Train/mean_reward` curve slope (sample efficiency)
- `Train/mean_episode_length` (how quickly the policy avoids termination)
- `Episode_Reward/track_lin_vel_xy_exp` at matching iters (does footstep reward hurt velocity tracking?)
- `Episode_Reward/footstep_swing_tracking` in treatment (does the policy actually optimize this?)

**Acceptance criteria for Phase 1**:
- Treatment hits the same `mean_episode_length` as baseline in **fewer iterations** (≥20% reduction = positive)
- Treatment does not regress on `track_lin_vel_xy_exp` by more than 10%
- If yes → escalate to Phase 1b: raibert_factor=0.3 to test "soft Raibert pull"
- If no → revisit reward shape (std, weight, dense vs sparse, add obs)

---

# Phase 2 — Learned planner (later)

- Net inputs: AME terrain feature (reuse encoder) + root state + velocity command + last contact.
- Net outputs: `(B, N, 2, 3)` next footsteps + per-step contact duration.
- Training: BC warmstart from Phase 0 heuristic → joint PPO with the tracker.
- Open question: shared backbone with policy actor, or separate net?

---

# Phase 3 — MPC / LIP planner (research target)

- Linear inverted pendulum capture-point planner constrained by elevation-map safe-patch mask.
- Closest to DTC humanoid extension. Compare against Phase 2 learned planner.

---

# Phase 4 — Sim2sim deployment

- Export planner + tracker as ONNX (planner runs CPU-side, tracker GPU-side acceptable).
- Update `deploy_for_AME/AME_SIM2SIM_NOTES.md` plumbing for the new observation channel.

---

## Risks / open issues

- **Bipedal feasibility:** Raibert was designed for hopping/quadrupeds. On G1, naive Raibert step targets may be unreachable during single-support phase → tracker diverges. Phase 0's eyeball check will tell us how bad.
- **Phase clock vs. learned timing:** Hardcoded `t_step=0.6` may fight the natural-period of the existing PPO policy when we eventually train on this. Be prepared to add `t_step` to the planner's resampled command space.
- **Elevation map noise:** Existing `elevation_map(noise=True)` adds noise to the policy observation. Planner uses the **clean** height_scanner output (which is fine, planner is privileged), but be careful not to use the noisy obs by mistake.
- **AME attention interpretability:** After we condition on the plan, attention maps may collapse. Acceptable cost, but worth recording before/after for the paper.
- **Yaw rate (wz) ignored in planner extrapolation** *(2026-06-18)*: ~~`_commit_plan` extrapolates future root pose by linear motion at the *current* yaw~~ **FIXED 2026-06-18**: now uses mid-path yaw for velocity rotation and end-of-horizon yaw for hip-offset/raibert rotation. Constant-wz approximation, accurate to 2nd order for moderate wz (chord error ~6% at wz=1 rad/s × 0.6s horizon).

- **`foot_clearance` is Stage 1 training wheels — NOT terrain-aware** *(2026-06-19)*: `foot_clearance_reward` uses world-frame `body_pos_w[..., 2]` against a fixed `target_height=0.1`. On flat ground (terrain_level 0) this is fine and aligns with planner apex=0.10m. But once `Curriculum/terrain_levels > 0` and step heights start exceeding 10cm, the term punishes correct stair-climbing motion. The terrain-relative fix (subtract ray-cast local ground from foot_z) would be functionally identical to `footstep_swing_tracking`'s z component (planner uses `start_w.z + apex`, where `start_w` is the last contact world coord — naturally terrain-tracking). Per consistency principle, do NOT add a second terrain-aware height shaping; instead: **plan to fade out `foot_clearance` as `footstep_swing_tracking` takes over** when stacking Phase 1 footstep rewards on top of `G1RoughEnvCfg_Unitree`. Monitor for the conflict: when `terrain_levels` first ramps past 0, watch `Episode_Reward/foot_clearance` — if it starts decreasing while feet_air_time grows, that's the conflict firing and a signal to zero out `foot_clearance` weight.
