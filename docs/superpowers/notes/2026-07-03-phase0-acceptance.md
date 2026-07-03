# Phase 0 Acceptance Notes (2026-07-03)

Scope: PLAN.md Phase 0 — heuristic `FootstepPlanCommand`, **not** Planner V2 Task 0.

## Automated / offline (Task 0.7)

Scripts added under `scripts/debug/`:

| Script | Purpose |
|---|---|
| `run_phase0_acceptance.sh` | One-shot dump + plot + prints GUI checklist |
| `plot_footstep_plan.py` | Loads `debug_plan.npz`, plots xy/z trajectories, runs sanity checks |
| `inspect_scanner_frame.py` | Verifies height grid rotates with yaw (prerequisite for z snap) |

Run on machine with Isaac Sim:

```bash
chmod +x scripts/debug/run_phase0_acceptance.sh
./scripts/debug/run_phase0_acceptance.sh
```

**Offline sanity checks** (in `plot_footstep_plan.py`):

- `target_w == plan_buffer[:,0]` — GUI markers read `target_w`; must match first planned step.
- `swing_foot` alternates multiple times over 500 steps.
- `phase ∈ [0,1)`.
- Plan z finite and in plausible band.
- Forward root drift when `vx_cmd > 0`.

Output plot: `scripts/debug/output/footstep_xy.png`

## GUI eyeball (Task 0.6) — manual

```bash
.AME/bin/python scripts/rsl_rl/play.py \
    --task AME-G1-29DOF-Play-v0 \
    --checkpoint pretrained/ame1.pt \
    --num_envs 1
```

**Marker legend** (`commands.py` `_debug_vis_callback`):

- Index 0 / blue: `target_w[:, 0]` left plan
- Index 1 / red: `target_w[:, 1]` right plan
- Index 2 / gray: `last_contact_w` (both feet)
- Index 3 / yellow: phantom pelvis (DTC phantom mode only)

**Expected behaviour:**

1. Targets are world-fixed between crossover events; recomputed only when swing foot switches.
2. Z comes from height scanner (privileged); markers should sit on visible terrain.
3. With `raibert_factor=0` (default Play inherits 0 from base cfg), targets stay near hips + velocity lead.
4. DTC-Play adds cost-based foothold selection — markers may shift sideways on step edges to lower roughness.

**Known benign warnings:**

- `FabricManager::initializePointInstancer mismatched prototypes` on `/Visuals/Command/footstep_plan` — Isaac Fabric + point instancer prototype mismatch; markers usually still render. Confirm visually that colors match legend above.

## Scanner frame (cross-check)

`inspect_scanner_frame.py --test_yaw_deg 45` → **PASS** on 2026-07-03.

Confirms `ray_alignment="yaw"`: grid indexing for z snap / Planner V2 geometry is valid.

## Phase 0 checklist status

| Item | Status |
|---|---|
| Unit tests `test_planner_geometry.py` | ✅ (pre-existing) |
| Scanner yaw contract (`inspect_scanner_frame.py`) | ✅ 2026-07-03 |
| Offline dump + plot (`debug_footstep_plan.py` + `plot_footstep_plan.py`) | ✅ scripts ready; run locally for PNG |
| GUI marker eyeball on rough terrain | ⏳ manual — use commands above |
| This writeup | ✅ |

## Observations / failure modes to watch in GUI

- **Floating markers:** height scanner miss / invalid ray hit → check `ray_hits_w` validity near grid edge.
- **Markers slide in world frame mid-stride:** would indicate replanning every step (bug); should only update on crossover.
- **Left/right swapped colors:** marker index mismatch — compare against `target_w` in offline dump.
- **Phantom yellow sphere far from robot:** expected under velocity command + leash; not a frame bug.
