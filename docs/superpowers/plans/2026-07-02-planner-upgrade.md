# Local-Optimization Footstep Planner (Phase A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the current `select_foothold_by_cost` (roughness + slope + obstacle + distance-to-prior) into a **local-optimization foothold selector** that (a) evaluates candidates using the same foot-polygon geometry as the BeamDojo reward side, (b) enforces reachability / cross-leg / step-height hard constraints, (c) adds soft costs for edge/overhang, reach comfort, and stance-height continuity, and (d) has a fallback for empty candidate sets. First-version scope is **Phase A only** (geometric optimization). DCM and future-step feasibility (Phase B / C) are explicitly out of scope for this plan and will be tracked in a follow-up.

**Motivation:** The DTC / Footstep and BeamDojo routes have complementary strengths but their geometry is currently siloed — planner evaluates candidates via single-point ray-hit z, while the reward side already samples a full 4×3 foot polygon for `foothold_sampling`. This plan unifies the geometry so both sides speak the same language, and gives the planner enough physical realism to stop selecting overhanging / unreachable / cross-leg candidates that the tracker cannot execute.

**Non-goal:** This plan does **not** switch the default env's planner behaviour. All changes are additive: existing `AME-G1-29DOF-*` and `DTC-*` envs keep working with their current configs. A new env variant (`DTC-Planner-v2`) opts into the upgraded selector.

**Architecture (target):**

```
Raibert prior  ─────────────────────────────────┐
                                                ▼
height_scanner (privileged, 21×33 @ 0.05m)  ─► candidate window
                                                │
                       ┌─────────── foot_patch_stats ─────────┐
                       │  z_median/z_q70, Δz_Ω, ρ_support,   │
                       │  slope_normal — per candidate       │
                       └─────────────┬───────────────────────┘
                                     ▼
                       ┌─────── hard filter mask ────────┐
                       │  reachability box, cross-leg,   │
                       │  |Δz − z_stance|, roughness cap │
                       └─────────────┬───────────────────┘
                                     ▼
                       ┌─────── weighted soft cost ──────┐
                       │  J_terrain + J_nominal +        │
                       │  J_reach + J_edge + J_height    │
                       │  (+ J_slope from patch normal)  │
                       └─────────────┬───────────────────┘
                                     ▼
                        argmin over valid candidates
                                     ▼
                        (with fallback if empty set)
                                     ▼
              FootstepPlanCommand.plan_buffer + foothold_score_buffer
                                     ▼
              policy obs: footstep_plan / phase_info / local_heightscan
                          + NEW: foothold_score, swing_side
```

**Tech Stack:** Isaac Lab 2.3.0 / Isaac Sim 5.0.0, custom RSL-RL fork (`rsl_rl/`), PyTorch, pytest for unit tests (use `.AME/bin/python`).

## Global Constraints

- Python runtime: `.AME/bin/python`. Never the system Python.
- All new `mdp/` math primitives must be Isaac-Sim-free (import only `torch`) so they can be unit-tested via `importlib` loading. Follow the pattern in `tests/test_planner_geometry.py` and `tests/test_foothold_selection.py`.
- **Existing envs bit-identical.** Do NOT modify existing gym IDs, env configs, reward functions, encoder configs, or the default behaviour of `select_foothold_by_cost` when the new kwargs are absent. All existing training / play / export commands must still work bit-identically. Regressions here block the plan.
- **New env variants may reweight rewards as documented.** PlannerV2 intentionally sets `footstep_swing_tracking.weight = 0.0` (see Task 7 rationale). This is not "additive" in the strict sense, but it is scoped to the new variant only and is necessary to let the planner's foothold-quality signal reach the reward (swing tracking rewards following the Raibert arc, which would mask any planner improvement). Existing envs are untouched.
- New env variants are registered under new IDs: `AME-G1-29DOF-DTC-PlannerV2-v0` / `-Play-v0` (v2 selector) and `AME-G1-29DOF-DTC-PlannerV2-Legacy-v0` / `-Play-v0` (same env cfg, `use_selector_v2=False` — the A/B control). The existing `AME-G1-29DOF-DTC-v0` keeps its current planner.
- Foot geometry constants (`foot_length=0.18`, `foot_width=0.065`, `sole_z_offset=-0.03540914...`, `n_long=4`, `n_lat=3`) must come from the shared pure module `mdp/foot_geometry_constants.py` so planner-side and reward-side geometry cannot drift.
- Grid-map, yaw-role, row/column, z-frame, and footprint out-of-bounds semantics are locked in `docs/superpowers/plans/2026-07-02-planner-upgrade-contract.md`. If this plan and the contract disagree, the contract wins.
- All new tensors follow the `(B, N, F, ...)` layout used by `FootstepPlanCommand` (B=envs, N=`n_future_steps`, F=2 feet). No Python loops over B or N in hot paths.
- Fallback path is **required** for every hard filter — an empty candidate set must never produce NaN / inf / out-of-bounds output. The fallback returns a conservative under-hip target with `foothold_score=0`.
- Any candidate-selection change is validated by (a) unit tests on synthetic terrain and (b) a smoke training run of ≥ 200 iterations at ≥ 256 envs before committing to it as default in any env config.

## Task DAG

```
Task 0 (coord/z contract + candidate-gen primitive)
        │
        ▼
Task 1 (foothold_geometry) ─┬─► Task 2 (hard filters)  ─┐
                            │                            │
                            └─► Task 3 (soft costs)    ──┼─► Task 4 (selector v2 wiring) ─► Task 5 (CommandTerm + fallback)
                                                         │                                          │
                                                         │                                          ▼
                                                         │                        Task 6 (obs: score + swing_side)
                                                         │                                          │
                                                         │                                          ▼
                                                         └───────────────► Task 7 (env cfg + gym registration + diagnostics)
                                                                                                    │
                                                                                                    ▼
                                                                              Task 8a (smoke A/B 200 iter, 256 envs)
                                                                                                    │
                                                                                                    ▼
                                                                              Task 8b (full A/B ≥3000 iter, 4096 envs)
                                                                                                    │
                                                                                                    ▼
                                                                                       Task 9 (decision gate: default or roll back)
```

Task 0 is a **prerequisite** for everything — it locks the coordinate / z / candidate-generation contract that Tasks 1-5 all depend on. Skipping it will make the implementation subtly wrong (see risk table). Tasks 2 and 3 are independent of each other (both depend on Task 1). Tasks 1-3 can be executed in parallel by subagents. Tasks 4-9 are sequential.

---

## Task 0: Coordinate / z / candidate-generation contract (Pre-flight)

**Purpose:** Lock the shared conventions that Tasks 1-5 depend on. Skipping this task will produce subtle correctness bugs that only surface as "training doesn't converge and A/B looks noisy" — see the risk table for a concrete failure mode inventory.

**Files:**
- Create/update: `docs/superpowers/plans/2026-07-02-planner-upgrade-contract.md` (single-page contract reference doc — Tasks 1-5 point to it)
- Create: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/foot_geometry_constants.py`
- Create: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/gridmap_utils.py`
- Create: `tests/test_gridmap_utils.py`
- Create: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/foothold_candidates.py`
- Create: `tests/test_foothold_candidates.py`

### 0.1 Contract: height_scanner frame

**Verified from `velocity_env_cfg_29dof.py:72-80`:**
```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",                                          # <-- key
    pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0]),
)
```

`ray_alignment="yaw"` means the ray pattern **rotates with the robot's yaw**. Therefore:

- `ray_hits_w` is a regular grid in **scanner (root-yaw-aligned) frame**, NOT in world-axis-aligned frame.
- The grid is defined as robot/scanner-centered: local `(0, 0)` is the center cell, not cell `(0, 0)`.
- The map `sample_xy_w → (row, col)` requires transforming into the grid frame first:
  ```
  xy_local = R(-grid_yaw) · (xy_w - grid_center_w)
  col      = round(xy_local.x / grid_resolution + (W - 1) / 2)
  row      = round(xy_local.y / grid_resolution + (H - 1) / 2)
  flat     = row * W + col
  ```
- Since the scanner is real-robot yaw-aligned, `grid_yaw` comes from the real scanner/body yaw. This is distinct from phantom/planner yaw when `use_phantom=True`.

**Contract for Tasks 1-5:** Any function that does grid-indexed gather must accept `grid_center_w: (B, 2)` and `grid_yaw: (B,)`, or call shared helpers in `mdp/gridmap_utils.py`. Do not derive row/col by hand in multiple modules.

### 0.2 Contract: z semantics

**Frame conventions (declared once, referenced everywhere):**

| Symbol | Meaning | Frame |
|---|---|---|
| `root_pos_w[..., 2]` | Base link (torso) height above world origin | world, ~0.75-0.85m nominal |
| `foot_body_z_w` | Foot body link position (ankle-ish) | world |
| `sole_z_w` | Foot sole contact z (physical touchdown height) | world |
| `sole_z_offset` | `-0.03540914...` — signed offset such that `sole_z_w = foot_body_z_w + sole_z_offset` | body frame constant, must match `_BEAMDOJO_FOOTHOLD_PARAMS` |
| `terrain_z_w` | Elevation of ground surface at (x, y) | world |
| Body-frame foot position | `R(yaw)^T · (foot_body_pos_w - root_pos_w)` | body-frame, z ≈ -0.78 nominal |

**`plan_buffer` and `selected_xyz_w` semantics (LOCKED):** Both store **foot body frame targets** in world frame (i.e., the position the ankle should reach). This is because the existing DTC tracker reward reads foot body links, not sole points. Therefore:

```
selected_z_w = terrain_z_w - sole_z_offset      # -sole_z_offset is positive ~3.5cm
```

Any primitive that produces `selected_xyz_w` **MUST** apply this offset before writing to the plan buffer. Tests must cover this: given a flat terrain at `z=0`, a chosen foothold's `selected_z_w` should be ~`+0.0354`, not `0.0`.

Candidate generation returns terrain xyz from `ray_hits_w`. Before applying reachability or comfort costs, convert to foot-body target xyz:

```
candidate_foot_body_z_w = candidate_terrain_z_w - sole_z_offset
```

Step-height continuity uses terrain/sole z, not foot-body z. For current contacts:

```
stance_terrain_z_w = last_contact_foot_body_z_w + sole_z_offset
```

### 0.3 Contract: candidate generation primitive

Extract candidate generation from Task 4 into a separate pure function so it is unit-testable in isolation and reusable.

**Interface:**

```python
def generate_candidates(
    raibert_xy_w: torch.Tensor,      # (B, F, 2)  — Raibert prior xy (world)
    ray_hits_w: torch.Tensor,        # (B, K, 3)  — scanner hits (world)
    grid_center_w: torch.Tensor,     # (B, 2)     — world xy of grid center
    grid_yaw: torch.Tensor,          # (B,)       — yaw of the real scanner grid
    *,
    grid_shape: tuple[int, int],     # (H, W)
    grid_resolution: float,
    half_width_cells: int = 2,       # 2 → 5×5=25 candidates. Use 3 for 7×7=49.
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns:
      candidate_xyz_w: (B, F, K_c, 3)  — K_c = (2*half_width_cells + 1)^2, fixed
      in_bounds_mask:  (B, F, K_c)     — False if the cell falls outside the grid
                                         (near-edge cases). Downstream filters treat
                                         out-of-bounds cells as invalid.
    """
```

Semantics:
1. Transform `raibert_xy_w` into scanner grid frame using `grid_yaw` and `grid_center_w`.
2. Round to nearest grid cell `(row_c, col_c)`.
3. Generate a fixed odd window of grid indices `(row_c + drow, col_c + dcol)` for `drow, dcol ∈ [-half, +half]`, row-major.
4. Gather `ray_hits_w` at those indices (this returns world-frame xyz — no need to transform back).
5. Mark out-of-bounds cells in `in_bounds_mask=False` and clamp their xy/z to the nearest in-bounds neighbour so downstream code never sees NaN.

**Why fixed odd window and not "hits within `window_m`":** Fixed shape `(B, F, K_c, 3)` keeps all downstream ops vectorized. A variable candidate count would require ragged tensors or a large boolean mask threaded through Tasks 1-4. The reviewer's suggestion; adopted.

- [ ] **Step 1: Write failing tests (`tests/test_foothold_candidates.py`).**

  1. `test_yaw_zero_matches_naive`: `grid_yaw=0`, `grid_center_w=(0,0)`, Raibert at `(0.20, 0.10)`, `resolution=0.05`, `grid_shape=(21,33)`, `half=2` → returned candidates form a 5×5 grid centered on `(row=12, col=20)` in row-major layout; center candidate xy is exactly `(0.20, 0.10)`.
  2. `test_yaw_90deg`: `grid_yaw=π/2`, local Raibert `(0.20, 0.10)` rotated into world → grid cell remains `(row=12, col=20)`.
  3. `test_raibert_near_edge`: Raibert xy near grid corner → `in_bounds_mask` has False entries; xyz for out-of-bounds cells is clamped to the nearest in-bounds cell (no NaN).
  4. `test_output_shape`: `(B=4, F=2)`, `half=3` → output `(4, 2, 49, 3)`, `in_bounds_mask (4, 2, 49)`.
  5. `test_5x5_default`: default kwargs → `K_c = 25`.
  6. `test_no_python_loops`: run with `B=4096, F=2, half=2` under `torch.cuda.max_memory_allocated` monitoring → peak allocation delta < 100 MB.

- [ ] **Step 2: Run tests, expect ImportError.**

- [ ] **Step 3: Implement `foothold_candidates.py`.**

  Use `gridmap_utils.py` for world→row/col conversion, row-major flattening, and safe gather. Do not duplicate this math.

- [ ] **Step 4: Iterate until tests pass.**

- [ ] **Step 5: One-shot yaw sanity script** (5 minutes, no commit needed).

  Write a throwaway `scripts/debug/inspect_scanner_frame.py` that:
  1. Instantiates `AME-G1-29DOF-DTC-Play-v0` with `num_envs=4`.
  2. Steps once, reads `ray_hits_w[0]` and the real scanner/body yaw as `grid_yaw[0]`.
  3. Prints the first 8 xy positions and the grid yaw.
  4. Sanity: if `grid_yaw ≠ 0`, the xy positions should NOT be axis-aligned; the local first-order diff along the flat-index axis should equal `resolution * (cos(yaw), sin(yaw))` (or `(-sin, cos)` for the orthogonal axis).

  This confirms the assumption in 0.1 empirically before all downstream tasks depend on it. Delete the script after confirming.

- [ ] **Verify:**
  ```bash
  .AME/bin/python -m pytest tests/test_foothold_candidates.py -v
  ```
  All 6 tests pass. Step 5 script prints yaw-aligned first-order diff matching the theoretical value.

### 0.4 Reachability z_range default

**Bug found in draft:** the draft's Task 2 default `z_range=(-0.20, 0.20)` would filter out **all** candidates because body-frame foot z is nominally around `-0.78m` (see comfort_center in Task 3). Task 2 must use `z_range=(-1.05, -0.55)` as default — a reachable range around the nominal leg length, allowing ±25 cm from the neutral standing pose.

Locked in this contract, actually applied in Task 2.

### 0.5 `foothold_score` formula (LOCKED)

**Formula for the obs signal:**
```
raw_quality  = support_ratio * exp(-0.5 * (slope_angle / 0.2094)^2)     # σ_slope = 12°
foothold_score = where(used_fallback, 0.0, raw_quality.clamp(0, 1))
```

**Rationale:** `foothold_score` is exposed to the policy as observation, so it must reflect **geometric footability of this point** (support + flatness) — not planner cost residuals like `J_reach` or `J_nominal` (which encode operator preferences, not physics). Using `exp(-J_total)` would leak reach/nominal preferences into what the policy sees, making the signal harder to interpret and coupling downstream training to soft-cost weight choices.

Locked here; applied in Task 4.

---

---

## Task 1: Shared foot-patch geometry primitive

**Files:**
- Create: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/foothold_geometry.py`
- Create: `tests/test_foothold_geometry.py`

**Interface (single pure function, no Isaac Sim import):**

```python
def foot_patch_stats(
    centers_w: torch.Tensor,        # (B, M, 3)   xyz of candidate centers (M = candidates per env, or F for per-foot)
    yaws: torch.Tensor,             # (B, M)      per-candidate yaw (world frame)
    ray_hits_w: torch.Tensor,       # (B, K, 3)   privileged elevation samples (K = H*W on a regular grid)
    *,
    foot_length: float,             # 0.18
    foot_width: float,              # 0.065
    n_long: int,                    # 4
    n_lat: int,                     # 3
    support_threshold: float,       # 0.03 (m) — how close sample z must be to patch median to count as "supported"
    # Grid fast-path (MANDATORY when M > 2; raises ValueError if absent in that case).
    # See Task 0.1 and the contract doc for the coord-frame contract: grid is
    # robot/scanner-centered and yaw-aligned.
    grid_shape: tuple[int, int] | None = None,    # (H, W) of ray_hits_w; required for planner path
    grid_resolution: float | None = None,         # cell size in metres (e.g. 0.05)
    grid_center_w: torch.Tensor | None = None,    # (B, 2) world xy of scanner grid center
    grid_yaw: torch.Tensor | None = None,         # (B,)  real scanner/grid yaw. Required with grid_center_w.
    return_debug: bool = False,                   # if True, also return sample_xy_w / sample_grid_row_col for tests
) -> dict[str, torch.Tensor]:
    """Returns per-candidate stats. All outputs shape (B, M) unless noted.

    z lookup strategy:
      - If grid_shape/grid_resolution/grid_center_w/grid_yaw are all provided
        → grid-indexed gather (O(B*M*S), no large intermediate tensor). Required
        for planner path (M > 2). Grid frame is root-yaw-aligned (see Task 0.1);
        the transform sample_xy_w → row/col uses gridmap_utils.world_xy_to_grid_float.
      - Else (M ≤ 2 only) → brute-force nearest-ray argmin (legacy, mirrors
        _foothold_sample_geometry). For reward-side actual-foot use and unit tests only.

    Keys:
      z_median      : median of foot-sample terrain z
      z_q70         : 0.70 quantile of foot-sample terrain z (upper support surface;
                      used to avoid being pulled low by patch samples that fall
                      off a small step edge inside the foot polygon)
      z_max, z_min  : per-patch extrema
      dz_omega      : z_max - z_min (patch roughness)
      support_ratio : fraction of samples within support_threshold of z_median
      overhang_ratio: fraction of samples with z < z_median - support_threshold (deep-drop count)
      sample_in_bounds: (B, M, S) bool footprint sample in grid bounds
      footprint_in_bounds_ratio: fraction of footprint samples inside the grid
      slope_normal  : (B, M, 3) unit normal from least-squares plane fit z = ax + by + c
      slope_angle   : arccos(dot(slope_normal, e_z))

    Debug keys (only if return_debug=True):
      sample_xy_w   : (B, M, S, 2)  per-sample world xy
      sample_grid_row_col: (B, M, S, 2)  per-sample grid indices (long)
    """
```

Note: `foot_patch_stats` operates on **arbitrary candidate centers**, not on the robot's actual foot bodies. This lets both the planner (M=candidate count) and — as an optional refactor later — the reward side (M=F=2) share the same code.

**Parity contract with reward side:** With `foot_length=0.18, foot_width=0.065, n_long=4, n_lat=3, sole_z_offset=-0.03540914`, calling `foot_patch_stats(centers_w=foot_body_xyz_w, yaws=foot_yaw_w, ...)` **must** produce `z_median` that matches the reward side's `_foothold_sample_geometry` output within `1e-5 m` (numerical noise). This is tested in `test_planner_reward_geometry_parity` (Task 4 verify) — a regression here would silently split the training signal between planner and reward.

- [ ] **Step 1: Write failing tests (`tests/test_foothold_geometry.py`)**

  Test cases:
  1. `test_flat_terrain`: 5×5 flat patch → `dz_omega ≈ 0`, `support_ratio == 1.0`, `overhang_ratio == 0`, `slope_angle ≈ 0`, `z_median == 0`.
  2. `test_step_edge_half_on_half_off`: step at x=0, z=0.15; candidate centered on edge → `support_ratio ≈ 0.5`, `dz_omega ≈ 0.15`.
  3. `test_pure_slope_10deg`: `z = tan(10°) * x` → `slope_angle` within ±1° of 10°, `support_ratio > 0.5` (samples spread around median).
  4. `test_overhang_detection`: flat top with a single dropped-out sample cell → `overhang_ratio == 1/12`, `support_ratio == 11/12`.
  5. `test_yaw_rotation`: patch on a slope where rotating candidate yaw 90° changes which samples fall on the high side → `dz_omega` invariant, but sample xy layout changes. Call with `return_debug=True` and assert `sample_xy_w` shifts correctly under the 90° rotation.
  6. `test_shapes_batched`: `(B=4, M=7)` random input → all outputs correct shape, dtype float32, no NaN.
  7. `test_zero_offsets_matches_ray_hit`: `foot_length=foot_width=0`, `n_long=n_lat=1` → `z_median == z_ray_hit_at(x, y)` for isolated peaks in the map.

- [ ] **Step 2: Run tests, expect ImportError.**
  ```bash
  .AME/bin/python -m pytest tests/test_foothold_geometry.py -v
  ```

- [ ] **Step 3: Implement `foothold_geometry.py`.**
  - Rotate `(n_long × n_lat)` body-frame grid by per-candidate yaw (broadcast over M).
  - **z lookup is grid-indexed, not brute-force nearest-ray.** The height_scanner's ray hits lie on a root-yaw-aligned regular grid (21×33 @ 0.05m; see Task 0.1). Grid-cell lookup is closed-form **in scanner frame**:
    ```python
    row_col = world_xy_to_grid_float(
        sample_xy_w,
        grid_center_w=grid_center_w,
        grid_yaw=grid_yaw,
        grid_shape=grid_shape,
        grid_resolution=grid_resolution,
    ).round().long()
    ```
    then `gather`. This is O(B·M·S) with no large intermediate tensor. **Do NOT** use `round((sample_xy_w - grid_center_w) / resolution + center_index)` directly — that skips the yaw transform and silently mis-indexes at any non-zero yaw.
  - **Grid path is MANDATORY when M > 2** (planner path, up to M=49 candidates). Callers must pass `grid_shape` and `grid_resolution` kwargs; if absent and M > 2, `raise ValueError`. The brute-force nearest-ray path (mirroring `_foothold_sample_geometry`) is **only** permitted when M ≤ 2 (reward-side actual-foot case, or unit tests with tiny M). Rationale: at B=4096, M=49, S=12, K=693 the brute-force diff tensor is ~13 GB → OOM. Do not leave this as an accidental fallback.
  - Plane fit: batched least-squares `[[Σx², Σxy, Σx],[Σxy, Σy², Σy],[Σx, Σy, S]] · [a,b,c]ᵀ = [Σxz, Σyz, Σz]ᵀ` via `torch.linalg.solve` with a singular fallback (`det < 1e-8` → `slope_angle = 0`, `slope_normal = e_z`).

- [ ] **Step 4: Iterate until all tests pass.**

- [ ] **Verify:**
  ```bash
  .AME/bin/python -m pytest tests/test_foothold_geometry.py -v
  ```
  All 7 tests pass. Peak VRAM under `nvidia-smi` during `test_shapes_batched` with `B=64, M=49` should be < 500 MB.

---

## Task 2: Hard-filter masks (reachability, cross-leg, step-height, roughness cap)

**Files:**
- Modify: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/foothold_selection.py`
- Modify: `tests/test_foothold_selection.py`

**Interface (append to `foothold_selection.py`):**

```python
def reachability_mask(
    candidates_w: torch.Tensor,     # (B, F, K, 3) — K candidates per foot
    body_pos_w: torch.Tensor,       # (B, 3)
    body_yaw: torch.Tensor,         # (B,)
    *,
    x_range: tuple[float, float] = (-0.12, 0.35),
    y_range_left: tuple[float, float] = (0.06, 0.28),
    y_range_right: tuple[float, float] = (-0.28, -0.06),
    z_range: tuple[float, float] = (-1.05, -0.55),   # body-frame foot z (see NOTE)
) -> torch.Tensor:                  # (B, F, K) bool — True means valid
    ...
# NOTE on z_range: body-frame foot z is nominally ~-0.78 (see Task 3
# cost_reach comfort_center_b = (0.0, ±0.12, -0.78)). A range of ±25 cm
# around that neutral leg length is (-1.05, -0.55). An earlier draft
# used (-0.20, 0.20), which would filter out **every** candidate at the
# nominal standing pose. See Task 0.4.

def step_height_mask(
    candidates_w: torch.Tensor,     # (B, F, K, 3)
    stance_terrain_z_w: torch.Tensor,  # (B, F) — terrain/sole z of the currently-planted foot
    *,
    max_dz: float = 0.20,
) -> torch.Tensor:                  # (B, F, K)
    ...

def roughness_cap_mask(
    dz_omega: torch.Tensor,         # (B, F, K)
    *,
    max_dz_omega: float = 0.10,
) -> torch.Tensor:                  # (B, F, K)
    ...
```

The cross-leg constraint is baked into `reachability_mask` via the asymmetric `y_range_left` vs `y_range_right`.

- [ ] **Step 1: Write failing tests.**

  Append to `tests/test_foothold_selection.py`:
  1. `test_reachability_zero_yaw`: robot at `body_pos_w=(0, 0, 0.80)`, yaw=0; terrain candidate at world `(0.2, 0.15, 0.0)` first converts to foot-body z `0.0354`, so body-frame z ≈ `-0.7646`, inside default z_range `(-1.05, -0.55)`; left foot valid, right foot invalid (crosses to left side).
  2. `test_reachability_yawed_robot`: robot at `body_pos_w=(0, 0, 0.80)`, yaw=90°; candidate with matching rotated body-frame xy respects the body-frame boxes.
  3. `test_reachability_z_default_at_nominal_stance`: robot at `body_pos_w=(0, 0, 0.80)`, terrain candidate on flat ground at world z=0 converted to foot-body z should be valid. (Regression against the old `z_range=(-0.20, 0.20)` bug.)
  4. `test_step_height_flat_stance`: `stance_z=0`, candidates at z ∈ {−0.3, −0.1, 0, 0.1, 0.3} → only middle three valid with `max_dz=0.20`.
  5. `test_roughness_cap`: dz_omega=[0.02, 0.08, 0.15] with cap 0.10 → mask=[T, T, F].
  6. `test_masks_composable_shape`: `(B=2, F=2, K=13)` random input → all three masks produce `(2, 2, 13)` bool tensors with no shape errors.

- [ ] **Step 2: Run tests, expect AttributeError.**

- [ ] **Step 3: Implement the three masks.**

  Use `_yaw_rotation` from `planner.py` (import at module top — `planner.py` is already Isaac-Sim-free). For `reachability_mask`, compute body-frame candidates as `R(body_yaw)ᵀ · (candidate_foot_body_xyz_w - body_pos_w)` then apply per-foot y-range slices.

- [ ] **Step 4: Iterate until tests pass.**

- [ ] **Verify:**
  ```bash
  .AME/bin/python -m pytest tests/test_foothold_selection.py -v
  ```
  All previous tests + 5 new tests pass. Existing 8 tests must not regress.

---

## Task 3: New soft cost functions (J_reach, J_edge, J_height, J_slope)

**Files:**
- Create: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/foothold_costs.py`
- Create: `tests/test_foothold_costs.py`

Split from `foothold_selection.py` so each cost is single-purpose and unit-testable. `select_foothold_by_cost` will import from here.

**Interface (all pure functions, (B, F, K) → (B, F, K)):**

```python
def cost_nominal(
    candidates_w: torch.Tensor,     # (B, F, K, 3)
    raibert_xy_w: torch.Tensor,     # (B, F, 2)   Raibert prior xy
    *,
    sigma_nominal: float = 0.10,    # metres
) -> torch.Tensor:
    """||candidate_xy - raibert_xy||² / σ². Stays close to the Raibert prior."""

def cost_reach(
    candidates_w: torch.Tensor,     # (B, F, K, 3)
    body_pos_w: torch.Tensor,       # (B, 3)
    body_yaw: torch.Tensor,         # (B,)
    *,
    comfort_center_b: tuple[tuple[float, float, float], tuple[float, float, float]]
        = ((0.0, 0.12, -0.78), (0.0, -0.12, -0.78)),  # (left, right) in body frame
    s_reach: float = 0.15,
) -> torch.Tensor:
    """Quadratic distance from body-frame comfort center. Encourages neutral stance geometry.
    Note: this is DIFFERENT from cost_nominal — cost_nominal pulls toward Raibert (which
    is command-driven and can be anywhere ahead of the robot), cost_reach pulls toward
    a fixed body-relative comfort pose (neutral standing). Both are needed.
    """

def cost_terrain(
    dz_omega: torch.Tensor,         # (B, F, K)   patch roughness from foot_patch_stats
    *,
    sigma_rough: float = 0.05,      # metres
) -> torch.Tensor:
    """Patch roughness penalty. Pure z_max - z_min signal — orthogonal to support/overhang.
    Kept separate from cost_edge so weights are transparent (see NOTE below).
    """

def cost_edge(
    support_ratio: torch.Tensor,    # (B, F, K)  from foot_patch_stats
    overhang_ratio: torch.Tensor,   # (B, F, K)
    *,
    lambda_support: float = 1.0,
    lambda_overhang: float = 2.0,
) -> torch.Tensor:
    """Penalises poor foot polygon support and per-sample overhang drops.
    cost = lambda_support * (1 - support_ratio) + lambda_overhang * overhang_ratio
    """

def cost_height(
    candidate_z_w: torch.Tensor,    # (B, F, K)
    stance_terrain_z_w: torch.Tensor,  # (B, F)
    *,
    sigma_h: float = 0.10,
) -> torch.Tensor:
    """Quadratic |Δz|² / σ² penalty. Weight must be low if training up-stairs skill."""

def cost_slope(
    slope_angle: torch.Tensor,      # (B, F, K)  radians
    *,
    sigma_theta_rad: float = 0.2094,  # 12°
) -> torch.Tensor:
    """θ² / σ² penalty from plane-fit normal."""
```

**NOTE on cost_terrain vs cost_edge (no double-penalty on support_ratio):** An earlier
draft mixed `1 - support_ratio` into both `w_terrain` and `cost_edge`, which double-counted
poor support. The final decomposition is:

| Cost | Signal | What it penalises |
|---|---|---|
| `cost_terrain` | `dz_omega` (patch max-min z) | Overall roughness of the patch |
| `cost_edge`    | `(1 - support_ratio) + 2*overhang_ratio` | Poor foot support & deep drops |
| `cost_slope`   | `slope_angle` | Steeply-tilted supporting surface |

These three consume orthogonal quantities from `foot_patch_stats` and can be weighted
independently.

- [ ] **Step 1: Write failing tests (`tests/test_foothold_costs.py`).**

  One test per cost function:
  1. `test_cost_nominal_zero_at_raibert`: candidate xy = Raibert xy → cost=0; candidate 0.10m offset → cost=1.0.
  2. `test_cost_reach_zero_at_comfort_center`: candidate at comfort center → cost=0; candidate 0.15 m offset in any direction → cost≈1.0 (one σ_reach).
  3. `test_cost_reach_body_frame_transform`: yawed robot → cost computed in body frame, not world.
  4. `test_cost_terrain_zero_at_flat`: `dz_omega=0` → cost=0; `dz_omega=σ_rough` → cost=1.0.
  5. `test_cost_edge_zero_at_full_support`: `support_ratio=1, overhang_ratio=0` → cost=0; `support_ratio=0, overhang_ratio=1` → cost=3.0 (1.0 + 2.0).
  6. `test_cost_height_monotone`: `|Δz|` increases → cost increases monotonically; `Δz=σ_h` → cost=1.0.
  7. `test_cost_slope_zero_at_flat`: `slope_angle=0` → cost=0; `slope_angle=σ_theta` → cost=1.0.
  8. `test_cost_terrain_and_edge_orthogonal`: `dz_omega=0, support_ratio=0.5` → `cost_terrain=0`, `cost_edge>0` (no bleed).

- [ ] **Step 2: Run tests, expect ImportError.**

- [ ] **Step 3: Implement `foothold_costs.py`.**

- [ ] **Step 4: Iterate until tests pass.**

- [ ] **Verify:**
  ```bash
  .AME/bin/python -m pytest tests/test_foothold_costs.py -v
  ```

---

## Task 4: Selector v2 — integrate patch stats, hard filters, new soft costs

**Files:**
- Modify: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/foothold_selection.py`
- Modify: `tests/test_foothold_selection.py`

**Design decision:** Keep `select_foothold_by_cost` **bit-identical** when the new kwargs are absent. Add a new function `select_foothold_v2` that composes patch stats + hard filters + new costs. This preserves the guarantee that existing envs are unaffected.

**Interface:**

```python
def select_foothold_v2(
    raibert_xy_w: torch.Tensor,     # (B, F, 3)  Raibert / phantom prior — provides candidate window center
    ray_hits_w: torch.Tensor,       # (B, K, 3)  privileged elevation (root-yaw-aligned grid, see Task 0.1)
    body_pos_w: torch.Tensor,       # (B, 3)     body frame for reachability
    body_yaw: torch.Tensor,         # (B,)       body yaw for reachability
    target_yaw: torch.Tensor,       # (B, F)     candidate foot-polygon yaw
    stance_terrain_z_w: torch.Tensor,  # (B, F)  terrain/sole z of the currently-planted foot per side
    *,
    # Grid contract (all four required; see Task 0.1)
    grid_shape: tuple[int, int],    # (H, W) of ray_hits_w — required (no legacy path here)
    grid_resolution: float,
    grid_center_w: torch.Tensor,    # (B, 2)  world xy of scanner/grid center
    grid_yaw: torch.Tensor,         # (B,)    yaw of the real scanner grid
    # Candidate window
    half_width_cells: int = 2,      # 2 → 5×5=25 candidates at res=0.05 (window ≈ ±0.10m).
                                    # Use 3 → 7×7=49 for ≈ ±0.15m if the argmin shows aliasing.
    # Foot / sole geometry (must match _BEAMDOJO_FOOTHOLD_PARAMS)
    foot_length: float = 0.18,
    foot_width: float = 0.065,
    n_long: int = 4,
    n_lat: int = 3,
    support_threshold: float = 0.03,
    sole_z_offset: float = -0.03540914,  # foot_body_z = sole_z + sole_z_offset
    # Hard-filter thresholds
    reach_x_range: tuple = (-0.12, 0.35),
    reach_y_range_left: tuple = (0.06, 0.28),
    reach_y_range_right: tuple = (-0.28, -0.06),
    reach_z_range: tuple = (-1.05, -0.55),      # body-frame foot z; see Task 0.4
    max_step_dz: float = 0.20,
    max_dz_omega: float = 0.10,
    # Soft-cost weights
    w_terrain: float = 1.0,
    w_nominal: float = 0.5,
    w_reach: float = 1.0,
    w_height: float = 0.5,
    w_edge: float = 1.0,
    w_slope: float = 0.3,
    # Fallback
    enable_fallback: bool = True,
) -> dict[str, torch.Tensor]:
    """Returns:
      selected_xyz_w      : (B, F, 3)  chosen FOOT BODY target (world). Task 0.2 contract:
                                       z = terrain_z - sole_z_offset (≈ +3.5cm above sole).
      foothold_score      : (B, F)     in [0, 1]. See Task 0.5 for the locked formula:
                                         support_ratio * exp(-0.5 * (slope/σ_slope)²), σ_slope=12°.
                                         0 iff used_fallback=True.
      valid_count         : (B, F)     number of candidates that survived hard filters
      used_fallback       : (B, F)     bool; True = candidate set was empty, used under-hip fallback
      per_cost_debug      : (B, F, 6)  stacked [J_terrain, J_nominal, J_reach, J_height, J_edge, J_slope]
                                       at chosen cell (for wandb logging)
    """
```

**Candidate generation:** Uses `generate_candidates(...)` from Task 0.3 with `half_width_cells` (default 2 → 5×5=25). This is a **fixed** window in scanner grid frame — not a variable "hits within window_m" mask. Fixed shape keeps all downstream ops vectorized. Dense re-sampling (Nx=13, Ny=9) is deferred to Phase B.

**Note on window size:** The draft claimed "window_m=0.10 → 7×7=49" but the arithmetic is `−0.10, −0.05, 0, 0.05, 0.10` = 5 cells per axis → **5×5=25**. The existing `select_foothold_by_cost` uses `ksize = 2*radius+1 = 5` under the same window/resolution, so this correction keeps parity with the legacy path. Use `half_width_cells=3` (7×7=49) only if profiling shows the 5×5 argmin is aliasing.

**Chosen z (Task 0.2 contract):** The winning candidate's terrain z is the patch `z_q70` (upper support surface), and `selected_z_w = z_q70 - sole_z_offset` before writing to `plan_buffer`. Downstream reward code (which reads foot-body positions) will see a target that lands the sole exactly on `z_q70` when the foot is placed accurately.

**Fallback:** When `valid_count == 0` for a (b, f) slot, return the ray-hit cell directly under the hip (body-frame `(0.0, ±hip_y, -leg_length)` converted through `body_pos_w/body_yaw`), snapped to the nearest ray, then apply the same `-sole_z_offset` correction, with `foothold_score=0` and `used_fallback=True`. Downstream code (Task 5) is required to check `used_fallback` and skip any tracking-quality metric on that step.

- [ ] **Step 1: Write failing tests.**

  Append to `tests/test_foothold_selection.py`:
  1. `test_v2_matches_v1_on_flat_when_hard_filters_inactive`: extreme wide reach ranges + max_dz=inf → v2 selection matches `select_foothold_by_cost` on flat terrain **after inverting the sole_z_offset** (v1 returns ray z, v2 returns foot-body z), soft-cost weights matched.
  2. `test_v2_hard_filter_rejects_cross_leg`: single candidate at body-frame `(0.0, -0.15, -0.78)` for **left** foot → v2 fires fallback (candidate set empty after cross-leg filter).
  3. `test_v2_fallback_returns_under_hip`: force empty candidate set via extreme thresholds → returned xyz is under body-frame `(0, ±hip_y, -leg_length)`, `used_fallback=True`, `foothold_score=0`.
  4. `test_v2_prefers_flat_over_edge`: two candidates, one on flat one on 20cm step edge, both reachable → v2 picks flat.
  5. `test_v2_score_saturates`: perfect flat candidate at Raibert center → `foothold_score > 0.9`.
     Score formula regression: `foothold_score = support_ratio * exp(-0.5 * (slope/0.2094)²)`.
  6. `test_v2_score_zero_on_fallback`: force fallback → `foothold_score == 0.0` exactly.
  7. `test_v2_selected_z_applies_sole_offset`: flat terrain at z=0, pick a candidate → `selected_z_w ≈ +0.0354` (i.e. `-sole_z_offset`), NOT `0.0`.
  8. `test_v2_grid_contract_required`: call without `grid_center_w` or `grid_yaw` → clear ValueError.
  9. `test_v2_output_shapes_and_dtypes`: `(B=4)` random input → all output tensors correct shape/dtype/device.
  10. `test_v2_yaw_rotation_invariance`: same terrain with root at yaw=0 vs yaw=π/2 (with Raibert prior rotated consistently) → selected candidate's body-frame position matches. This is the regression against the Task 0.1 coord-frame bug.
  11. `test_planner_reward_geometry_parity`: given foot body pose `foot_body_xyz_w`, `foot_yaw_w`, call `foot_patch_stats(centers_w=foot_body_xyz_w, yaws=foot_yaw_w, foot_length=0.18, foot_width=0.065, n_long=4, n_lat=3, ...)` and compare shared terrain samples / `z_median` / `dz_omega` against the reward-side `_foothold_sample_geometry` (imported from the BeamDojo reward module) on the same terrain. Do not compare support ratios unless the test explicitly aligns the sole plane with the planner support reference. This guards the parity contract from Task 1.

- [ ] **Step 2: Run tests.**

- [ ] **Step 3: Implement `select_foothold_v2`.**

- [ ] **Step 4: Iterate until tests pass.**

- [ ] **Verify (regression + new):**
  ```bash
  .AME/bin/python -m pytest tests/test_foothold_selection.py tests/test_foothold_costs.py tests/test_foothold_geometry.py tests/test_planner_geometry.py -v
  ```
  All prior tests still pass. New tests pass. Existing `test_grid_path_matches_brute_force_on_flat` etc. unchanged (proof `select_foothold_by_cost` is untouched).

---

## Task 5: Wire v2 into `FootstepPlanCommand` behind a flag

**Files:**
- Modify: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/commands.py`

**Additions to `FootstepPlanCommandCfg`:**

```python
use_selector_v2: bool = False   # opt-in; default keeps legacy behaviour
# All the v2 thresholds and weights, exposed for env-cfg overrides.
selector_v2_max_step_dz: float = 0.20
selector_v2_max_dz_omega: float = 0.10
selector_v2_w_terrain: float = 1.0
selector_v2_w_nominal: float = 0.5
selector_v2_w_reach: float = 1.0
selector_v2_w_height: float = 0.5
selector_v2_w_edge: float = 1.0
selector_v2_w_slope: float = 0.3
```

**Additions to `FootstepPlanCommand`:**

New buffers (allocated in `__init__`, updated in `_commit_plan`):
```python
self.foothold_score_buffer = torch.zeros(B, N, 2, device=self.device)
self.used_fallback_buffer  = torch.zeros(B, N, 2, dtype=torch.bool, device=self.device)
self.valid_count_buffer    = torch.zeros(B, N, 2, dtype=torch.long, device=self.device)
self.per_cost_debug_buffer = torch.zeros(B, N, 2, 6, device=self.device)
```

In `_commit_plan`, when `self.cfg.use_selector_v2` is True:

1. Initialise a **planned-stance-z accumulator** at the current real contact heights:
   ```python
   # (B, F) — terrain/sole z of the currently-planted foot per side, updated as we plan future steps.
   planned_terrain_z = self.last_contact_w[..., 2] + sole_z_offset
   ```
2. For each future step `k = 0 .. N-1` (sequential in `k`, vectorized over B and F):
   1. Determine `swing_idx` (the foot being planned this step) and `stance_idx = 1 - swing_idx`.
   2. `stance_terrain_z_w = planned_terrain_z[:, stance_idx]` — this is the **terrain/sole z the planted foot will be at when this future step lands**, which for `k=0` equals current contact foot-body z plus `sole_z_offset`, and for `k>0` equals the previous iteration's planned landing terrain z. Without this accumulator, `step_height_mask` would evaluate step-height continuity against the wrong stance height for `k>=1`, letting the planner chain up impossibly-tall future steps.
   3. Compute Raibert target for `(k, swing_idx)` as before.
   4. Call `select_foothold_v2(...)`.
   5. Write results into `plan_buffer[:, k, swing_idx]`, `foothold_score_buffer`, `used_fallback_buffer`, `valid_count_buffer`, `per_cost_debug_buffer`.
   6. Update the accumulator: `planned_terrain_z[:, swing_idx] = selected_xyz_w[:, swing_idx, 2] + sole_z_offset` (convert foot-body target back to sole/terrain z for the next iteration's step-height check).

**Vectorization note:** The loop over `k` is sequential (each step's stance depends on the previous step's swing target), so we accept an N-step Python loop here. B and F are still fully vectorized inside `select_foothold_v2`. For the current config (`n_future_steps ≤ 4`) this is a negligible overhead.

**Metrics (published to `self.metrics` dict, wandb-visible):**
```python
self.metrics["selector_v2_mean_score"]      = foothold_score_buffer[:, 0].mean(dim=-1)  # per env
self.metrics["selector_v2_fallback_rate"]   = used_fallback_buffer[:, 0].float().mean(dim=-1)
self.metrics["selector_v2_valid_count"]     = valid_count_buffer[:, 0].float().mean(dim=-1)
```

- [ ] **Step 1: Add cfg fields and buffers.** No behaviour change yet (guard everything behind `use_selector_v2=False` default).

- [ ] **Step 2: Implement the v2 branch in `_commit_plan`.**

- [ ] **Step 3: Wire metrics into `self.metrics` in `_update_metrics`.**

- [ ] **Step 4: Sanity check with a no-op env** — run the existing zero-agent smoke to confirm `use_selector_v2=False` path is bit-identical:
  ```bash
  .AME/bin/python scripts/zero_agent.py --task AME-G1-29DOF-DTC-Play-v0 --num_envs 1
  ```
  No new print output, no exceptions. Compare `plan_buffer[0, 0, 0]` against a pre-change baseline (if available) to confirm no drift.

- [ ] **Verify:** Existing tests all pass. `use_selector_v2=False` produces identical `plan_buffer` values to before this task (checked by running the same scripted env for 100 steps with a fixed seed and comparing tensor equality).

---

## Task 6: New observation terms — foothold_score + swing_side

**Files:**
- Modify: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/observations.py`
- Modify: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/__init__.py` (re-export if not already wildcarded)

**New obs terms:**

```python
def footstep_foothold_score(
    env: ManagerBasedRLEnv, command_name: str = "footstep_plan"
) -> torch.Tensor:
    """Per planned foothold quality score. Shape (B, N*2). Values in [0, 1].
    0 indicates fallback was used (planner had no valid candidate)."""
    cmd = env.command_manager.get_term(command_name)
    return cmd.foothold_score_buffer.flatten(start_dim=1)


def footstep_swing_side(
    env: ManagerBasedRLEnv, command_name: str = "footstep_plan"
) -> torch.Tensor:
    """One-hot encoding of current swing foot. Shape (B, 2). [1,0]=left swinging, [0,1]=right."""
    cmd = env.command_manager.get_term(command_name)
    side = cmd.swing_foot  # (B,) in {0, 1}
    return torch.nn.functional.one_hot(side, num_classes=2).float()
```

- [ ] **Step 1: Add the two obs functions.** No `__all__` update needed if `mdp/__init__.py` uses `from .observations import *` with no explicit list; verify by grep.

- [ ] **Step 2:** Sanity check via a tiny script that instantiates the env with these obs added and confirms shapes.

- [ ] **Verify:** Import test — `.AME/bin/python -c "from ame_locomotion.tasks.manager_based.ame_locomotion.mdp import footstep_foothold_score, footstep_swing_side; print('ok')"`.

---

## Task 7: New env variant + gym registration + diagnostics

**Files:**
- Modify: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py`
- Modify: `source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/__init__.py`

**New env class:** `G1RoughEnvCfg_DTC_PlannerV2` — **inherits from `G1RoughEnvCfg_DTC_FORWARD`** (locked in 2026-07-02).

Rationale for the FORWARD baseline: forward-only command range gives the cleanest A/B signal for planner-level changes (no yaw / lateral confound), and the FORWARD config already went through one round of DTC tuning (`raibert_factor=0.3`, `use_phantom=True`, `max_leash=0.5`) so we inherit a working starting point.

In `__post_init__`:
```python
super().__post_init__()

# ---- Planner: switch to v2 selector -------------------------------------
self.commands.footstep_plan.use_selector_v2 = True
# Copy the exact v2 defaults from the plan doc — env cfg is the source of truth for training runs.
self.commands.footstep_plan.selector_v2_max_step_dz = 0.20
self.commands.footstep_plan.selector_v2_max_dz_omega = 0.10
self.commands.footstep_plan.selector_v2_w_terrain = 1.0
self.commands.footstep_plan.selector_v2_w_nominal = 0.5
self.commands.footstep_plan.selector_v2_w_reach = 1.0
self.commands.footstep_plan.selector_v2_w_height = 0.5
self.commands.footstep_plan.selector_v2_w_edge = 1.0
self.commands.footstep_plan.selector_v2_w_slope = 0.3

# ---- Rewards: disable legacy swing tracking (see docstring) -------------
# `footstep_swing_tracking` was the dense reference-trajectory follower from
# Phase 1 (weight 2.0 in DTC_FORWARD). We *intentionally* zero it out here:
#
#   1. Phase A's goal is to isolate the effect of planner geometry quality.
#      swing_tracking couples reward magnitude to *reference-vs-actual* foot
#      trajectory error, which mixes planner errors and policy errors into
#      one scalar and confounds the A/B comparison.
#   2. Historical failure mode (docs/superpowers/plans/2026-07-01-beamdojo.md):
#      swing_tracking creates a saddle against `track_lin_vel_xy_exp` — the
#      policy trades forward progress for cleaner in-place foot arcs. A
#      higher-quality planner does not fix this; removing the term does.
#   3. When Phase A ships, we will re-introduce a *touchdown-only* reward
#      (r_td on first contact) in a follow-up plan — that is the correct
#      shape for a planner-tracking task. Dense swing tracking stays off.
#
# We keep the code path intact (weight=0, not deleted) so ablations can
# re-enable it via one line if needed.
self.rewards.footstep_swing_tracking.weight = 0.0

# `footstep_contact_phase` (AND-product on left/right contact schedule) is
# *kept* at the FORWARD baseline's weight (1.0). It rewards correct
# single-stance timing without referencing any trajectory, so it does not
# suffer from the swing_tracking saddle and it gives the tracker a cheap
# gait-shape signal while the planner takes over foothold placement.

# ---- Observations: expose planner quality signal ------------------------
# Add new obs to both policy and critic groups (do NOT remove existing terms).
self.observations.policy.footstep_foothold_score = ObsTerm(
    func=mdp.footstep_foothold_score, params={"command_name": "footstep_plan"}
)
self.observations.policy.footstep_swing_side = ObsTerm(
    func=mdp.footstep_swing_side, params={"command_name": "footstep_plan"}
)
self.observations.critic.footstep_foothold_score = ObsTerm(...)
self.observations.critic.footstep_swing_side = ObsTerm(...)
```

**Play variant:** `G1RoughEnvCfg_DTC_PlannerV2_PLAY` — inherits from `G1RoughEnvCfg_DTC_FORWARD_PLAY` and applies the same planner + reward overrides.

**A/B control variant (MUST ship with the v2 variant):** `G1RoughEnvCfg_DTC_PlannerV2_LEGACY` — identical to `G1RoughEnvCfg_DTC_PlannerV2` except `use_selector_v2 = False`. This is the matched-regime baseline for Task 8: same forward command range, same swing-tracking-off, same new obs terms — only the planner selector differs. Without this, the A/B would confound planner geometry with reward/command changes.

```python
class G1RoughEnvCfg_DTC_PlannerV2_LEGACY(G1RoughEnvCfg_DTC_PlannerV2):
    def __post_init__(self):
        super().__post_init__()
        self.commands.footstep_plan.use_selector_v2 = False
```

Play variant `G1RoughEnvCfg_DTC_PlannerV2_LEGACY_PLAY` mirrors the same relationship.

**Gym IDs:**
- `AME-G1-29DOF-DTC-PlannerV2-v0` → `G1RoughEnvCfg_DTC_PlannerV2`
- `AME-G1-29DOF-DTC-PlannerV2-Play-v0` → `G1RoughEnvCfg_DTC_PlannerV2_PLAY`
- `AME-G1-29DOF-DTC-PlannerV2-Legacy-v0` → `G1RoughEnvCfg_DTC_PlannerV2_LEGACY` (A/B control)
- `AME-G1-29DOF-DTC-PlannerV2-Legacy-Play-v0` → `G1RoughEnvCfg_DTC_PlannerV2_LEGACY_PLAY`

- [ ] **Step 1: Add the env cfg classes (v2 + legacy + their play variants).**

- [ ] **Step 2: Register the four gym IDs.**

- [ ] **Step 3: Verify env constructs.**
  ```bash
  .AME/bin/python scripts/list_envs.py | grep PlannerV2
  .AME/bin/python scripts/zero_agent.py --task AME-G1-29DOF-DTC-PlannerV2-Play-v0 --num_envs 1
  .AME/bin/python scripts/zero_agent.py --task AME-G1-29DOF-DTC-PlannerV2-Legacy-Play-v0 --num_envs 1
  ```
  Expected: four new IDs listed; both zero-agents run 100 steps without exception; obs vector length grew by `N*2 + 2 = 6` from the DTC baseline and is **identical** between v2 and Legacy (same obs terms, only planner flag differs).

- [ ] **Step 4: Confirm existing envs unaffected.**
  ```bash
  .AME/bin/python scripts/zero_agent.py --task AME-G1-29DOF-DTC-Play-v0 --num_envs 1
  ```
  No changes vs. baseline.

- [ ] **Verify:** All four commands above succeed. `scripts/list_envs.py` shows PlannerV2 IDs. No changes to any existing env's obs shape, action shape, or reward composition.

---

## Task 8a: Smoke A/B (200 iter, 256 envs) — Gate 1

**Files:** None (training runs, wandb logs).

**Purpose:** Cheap sanity gate. Answers only: *does the v2 selector run without exploding, and is `selector_v2_fallback_rate` in a healthy range (<5%)?* This gate does NOT decide whether v2 becomes default — PPO at 200 iter / 256 envs has too much variance for that. The A/B is `PlannerV2` vs `PlannerV2-Legacy` — **both share forward command range, swing-tracking-off, and the new obs terms**. The only difference is `use_selector_v2`.

**Why not `AME-G1-29DOF-DTC-v0` as baseline:** that env has omni command range and swing tracking on, so it differs from PlannerV2 in two extra ways (command range + reward term). Any gap would confound planner geometry with those changes and would not be attributable to the selector. The matched-regime `Legacy` variant removes this confound.

- [ ] **Step 1: Baseline run (matched control — legacy selector, otherwise identical env).**
  ```bash
  .AME/bin/python scripts/rsl_rl/train.py \
      --task AME-G1-29DOF-DTC-PlannerV2-Legacy-v0 \
      --num_envs 256 --max_iterations 200 --headless \
      --run_name plannerv2_legacy_ctrl --logger wandb \
      --log_project_name ame_planner_v2
  ```

- [ ] **Step 2: Treatment run (v2 selector).**
  ```bash
  .AME/bin/python scripts/rsl_rl/train.py \
      --task AME-G1-29DOF-DTC-PlannerV2-v0 \
      --num_envs 256 --max_iterations 200 --headless \
      --run_name plannerv2_treatment --logger wandb \
      --log_project_name ame_planner_v2
  ```

- [ ] **Step 3: Collect metrics side-by-side.**
  From wandb, extract (both runs):
  - `Train/mean_reward` at iter 100 and 200
  - `Train/mean_episode_length` at iter 100 and 200
  - `Episode_Reward/track_lin_vel_xy_exp` at iter 200 — primary task metric
  - `Episode_Reward/footstep_contact_phase` at iter 200 — gait timing health
  - `Episode_Reward/foothold_sampling` (if enabled in the inherited env) at iter 200 — this is where the planner's better foothold choice should show up, since the reward evaluates actual landing support
  Treatment-only metrics (v2):
  - `Metrics/footstep_plan/selector_v2_mean_score` — should trend up over iters
  - `Metrics/footstep_plan/selector_v2_fallback_rate` — must be < 5%; if higher, the hard filters are too tight (loop back to Task 9 Option B)
  - `Metrics/footstep_plan/selector_v2_valid_count` — sanity: should be > 0 on most steps

- [ ] **Gate 1 pass criteria (ALL must hold):**
  - Both runs completed 200 iter without exceptions.
  - `selector_v2_fallback_rate < 5%` mean over the last 50 iter on the v2 run. If ≥5%, hard filters are too tight — go to Task 9 Option B, tune thresholds, re-run Task 8a.
  - `selector_v2_valid_count > 0` on ≥99% of steps.
  - v2 `mean_reward` is not catastrophically worse than Legacy (within ~2× the run-to-run PPO noise band). "Clearly worse at 200 iter" is not conclusive — it only rules v2 out if the fallback / valid-count diagnostics are also unhealthy.
  - No NaN in `foothold_score_buffer`, `plan_buffer`, `per_cost_debug_buffer` over the whole run.

- [ ] **Verify:** Write a brief comparison table in a new file `docs/superpowers/plans/2026-07-02-planner-upgrade-results.md` with these numbers and one paragraph of interpretation. Explicitly state the matched-regime pairing (v2 vs Legacy, identical except selector flag) so the reader does not mistake this for a v2-vs-DTC-v0 comparison. Do NOT declare v2 the winner from this gate — that is Task 8b's job.

---

## Task 8b: Full A/B (≥3000 iter, 4096 envs) — Gate 2

**Purpose:** The decision-quality A/B. Runs long enough for PPO variance to average out and for terrain curriculum to make progress. Only run this if Task 8a Gate 1 passes.

- [ ] **Step 1: Baseline run (matched control).**
  ```bash
  .AME/bin/python scripts/rsl_rl/train.py \
      --task AME-G1-29DOF-DTC-PlannerV2-Legacy-v0 \
      --num_envs 4096 --max_iterations 3000 --headless \
      --run_name plannerv2_legacy_full --logger wandb \
      --log_project_name ame_planner_v2
  ```

- [ ] **Step 2: Treatment run (v2 selector).**
  ```bash
  .AME/bin/python scripts/rsl_rl/train.py \
      --task AME-G1-29DOF-DTC-PlannerV2-v0 \
      --num_envs 4096 --max_iterations 3000 --headless \
      --run_name plannerv2_full --logger wandb \
      --log_project_name ame_planner_v2
  ```

  If wall-clock budget allows, run with 2-3 seeds per condition and report mean ± std. Otherwise document the single-seed limitation explicitly in the results doc.

- [ ] **Step 3: Collect metrics.** Same list as Task 8a, plus:
  - Terrain curriculum: `Curriculum/terrain_levels` trajectory over training. This is the primary success metric — a better planner should let the curriculum climb faster/higher.
  - `Episode_Termination/base_contact` rate. Should be substantially lower for v2 if the planner is genuinely improving foothold quality.

- [ ] **Verify:** Append full-run comparison to `2026-07-02-planner-upgrade-results.md`. Explicitly compare the final 200-iter window means; do NOT read off single-iter values.

---

## Task 9: Decision gate

Based on **Task 8b (full-run)** results — NOT Task 8a alone — choose one:

- [ ] **Option A — v2 clearly helps or ties vs Legacy on the full run:** Update `G1RoughEnvCfg_DTC_PlannerV2_LEGACY` to set `use_selector_v2=True` (i.e. fold the v2 selector into the matched-regime baseline). Whether to also flip it on in the omni-range `G1RoughEnvCfg_DTC` (with swing tracking on) is a **separate** decision that requires a follow-up A/B in the swing-on regime — do not assume the Phase A win transfers. Move to Phase B planning (DCM cost, first-touchdown reward, foot patch stats sharing with reward side).

- [ ] **Option B — v2 hurts a specific metric but the diagnostic reveals a fixable cause** (e.g., fallback rate too high → loosen a threshold; slope cost overrides progress → drop `w_slope`): tune, re-run Task 8b, re-enter this gate. Skip Task 8a on the retune (already validated).

- [ ] **Option C — v2 provides no benefit at 3000 iter and diagnostics look clean:** Do not make PlannerV2 the default. Document the negative result. Reconsider whether the DTC route is worth continued investment vs. consolidating with the BeamDojo route (see `2026-07-01-beamdojo.md`).

Write the decision + supporting numbers into `docs/superpowers/plans/2026-07-02-planner-upgrade-results.md`.

---

## Out of scope (explicit deferrals)

The following items from the brainstorming were considered and pushed to a follow-up plan:

- **J_dcm** (LIP capture-point cost). Depends on CoM state plumbing and risks re-entering the swing-tracking vs velocity-tracking saddle documented in `2026-07-01-beamdojo.md`. Add only after Phase A shows stable ≥ 1000 episode length.
- **J_progress / J_yaw** as candidate costs. Progress is redundant with `track_lin_vel_xy_exp` on the reward side; yaw candidate sampling requires expanding candidate count. Defer.
- **J_future** (two-step feasibility). Requires evaluating patch stats for hypothetical next-next targets. Costly and only useful once single-step selection is proven.
- **Dense candidate re-sampling (Nx=13, Ny=9)**. Current scanner resolution (0.05 m) gives 7×7 native cells in a 10 cm window — sufficient for MVP. Re-sample only if the argmin distribution shows aliasing.
- **Yaw candidate sampling**. `ψ_j = ψ_root` for Phase A.
- **Neural foothold proposer** (Phase D of the original tree). Only meaningful if the geometric planner ships and stabilises first.
- **Sim2real deployment** of the v2 selector. Separate plan.
- **first-touchdown reward** and **stable-contact reward**. These need `first_contact_this_swing` event detection in `FootstepPlanCommand`, which is a self-contained change but not required for the planner upgrade itself. Track as a Phase A' sub-plan.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hard filters too tight → fallback fires >20% → policy sees garbage targets → training collapses | Medium | Task 8 explicitly monitors `selector_v2_fallback_rate`. Task 9 Option B loops back to threshold tuning. Fallback returns a **conservative** under-hip target (not NaN), so worst case is degradation to Raibert-under-hip, not divergence. |
| VRAM OOM in `foot_patch_stats` at B=4096 with brute-force z lookup | High if grid path not taken | Task 1 Step 3 mandates the grid-fast-path. Task 1 Verify includes a VRAM smoke check. |
| Changes to `select_foothold_by_cost` accidentally break the existing DTC env | Low (explicit non-goal) | Task 4 creates `select_foothold_v2` as a **new** function. All existing tests must pass. Task 7 Step 4 explicitly tests the unchanged baseline. |
| Foot geometry constants drift out of sync between planner and reward side | Medium | Global Constraints call out that the foot-length / width / offset must exactly match `_BEAMDOJO_FOOTHOLD_PARAMS`. Task 4 `test_planner_reward_geometry_parity` catches drift at CI time. Consider following up with a shared constants module (out of scope for this plan). |
| Plane fit unstable when foot polygon is small (3-sample degenerate case) | Low | Task 1 Step 3 uses `torch.linalg.solve` with a fallback to `slope_angle=0` when the normal equation is singular (det < 1e-8). Add this to the test suite in Step 1 case 3. |
| **Grid-frame yaw misalignment: primitives use `(sample_xy_w - grid_center_w) / resolution` as cell index, silently mis-indexing at any non-zero yaw** | **High if Task 0 skipped** | Task 0.1 locks the scanner-frame contract. Task 1/4 interfaces mandate `grid_center_w` and `grid_yaw`. `test_v2_yaw_rotation_invariance` (Task 4) is the direct regression test. |
| **Reachability z_range defaults filter out all candidates at nominal stance** | **High if Task 2 shipped with draft defaults** | Task 0.4 + Task 2 default `z_range=(-1.05, -0.55)` (body-frame nominal foot z ≈ -0.78, ±0.25m). `test_reachability_z_default_at_nominal_stance` regression-tests this. |
| **`selected_xyz_w` z-frame ambiguity (foot-body vs sole target) causes silent 3.5cm offset between planner and reward** | **High if Task 0.2 skipped** | Task 0.2 locks `plan_buffer` as foot-body target with `selected_z_w = terrain_z - sole_z_offset`. `test_v2_selected_z_applies_sole_offset` (Task 4) is the regression. |
| **`foothold_score` obs signal has undefined semantics — different implementers give different formulas** | **High if Task 0.5 skipped** | Task 0.5 locks the formula: `support_ratio * exp(-slope²/σ²)`. `test_v2_score_saturates` and `test_v2_score_zero_on_fallback` regression-test it. |
| Chained future-step planning uses stale stance terrain z, letting the planner chain up impossibly-tall future steps | Medium | Task 5 introduces the `planned_terrain_z` accumulator. Without it, `step_height_mask` for `k>=1` compares against the current-real stance terrain height, not the previously-planned one. |

---

## Verification checklist (before marking plan complete)

- [ ] Task 0 contract doc `2026-07-02-planner-upgrade-contract.md` merged; Task 0.5 yaw sanity script confirmed the scanner is root-yaw-aligned.
- [ ] All new unit tests pass: `tests/test_foothold_candidates.py`, `tests/test_foothold_geometry.py`, `tests/test_foothold_costs.py`, extended `tests/test_foothold_selection.py` (including `test_planner_reward_geometry_parity`, `test_v2_yaw_rotation_invariance`, `test_v2_selected_z_applies_sole_offset`).
- [ ] All prior tests still pass: `tests/test_planner_geometry.py`, unmodified `tests/test_foothold_selection.py` cases.
- [ ] Existing envs unchanged: `zero_agent.py` output on `AME-G1-29DOF-DTC-Play-v0` and `AME-G1-29DOF-Play-v0` produces same obs shape and first-step obs values as pre-change.
- [ ] New envs constructible: `AME-G1-29DOF-DTC-PlannerV2-v0` and `-Play-v0` listed by `list_envs.py`, run 100 zero-agent steps without error.
- [ ] Task 8a Gate 1 passes (fallback rate < 5%, no NaN, no exceptions).
- [ ] Task 8b full-run completed and results written to `2026-07-02-planner-upgrade-results.md`.
- [ ] Task 9 decision documented.
- [ ] `pre-commit run --all-files` clean on all touched files.

---

## Time estimate

- Task 0: 0.5 day (contract doc + `foothold_candidates.py` + 5 min yaw sanity script)
- Task 1: 0.5 day (well-scoped; tests-first)
- Task 2: 0.25 day
- Task 3: 0.25 day
- Task 4: 0.75 day (integration + test carefully; parity test is important)
- Task 5: 0.5 day (planned_terrain_z accumulator adds a small loop)
- Task 6: 0.1 day
- Task 7: 0.35 day (4 env IDs: v2 + Legacy + 2 play variants)
- Task 8a: 0.25 day (200 iter × 256 envs, mostly wait)
- Task 8b: 1.0 day (3000 iter × 4096 envs × ideally 2-3 seeds, mostly wait)
- Task 9: 0.1 day (decision only)

**Total code: ~3.0 days. Total wall-clock incl. full-run training: ~4.5–5.5 days.**
