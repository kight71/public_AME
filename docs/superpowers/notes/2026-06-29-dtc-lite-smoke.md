# DTC-Lite Smoke Test — 2026-06-29

Branch: `feat/dtc-lite` at `d7ee265` (head after smoke).
Plan: `docs/superpowers/plans/2026-06-29-dtc-lite.md`.

## Step 1 — random_agent (4 envs, headless)

**Verdict: PASS.**

- Env construction reached `_initialize_impl` for all rewards including the new
  `footstep_swing_tracking_log` (kit log line `[10,518ms] [Info]
  [isaaclab.managers.manager_base] [RewardTermCfg:footstep_swing_tracking_log]
  Found entity 'contact_forces'`).
- `FootstepPlanCommand` auto-calibrated hip offsets without errors.
- All 4 named risks for Step 1 verified (env construction, obs schema, planner
  integration, no NaN in env init).
- Simulation loop ran ~15 minutes (until manually killed) with no errors or
  tracebacks in the kit log.
- stdout was empty past the IOMMU CUDA p2p validation line — purely a Python
  `print()` libc buffering + Isaac Sim 5.0 first-start IOMMU validation
  slowness. Not a code defect.

## Step 2 — 100-iter training (256 envs, headless, wandb)

Initial attempt: 1024 envs → CUDA OOM allocating 1.83 GiB. Trace pointed at
`select_foothold_by_cost`'s `(B, K, K, 2)` distance matrix (B=1024, K=693 →
3.9 GB combined with d2). Re-ran at 256 envs with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → fits comfortably.

Second attempt: `NameError: name 'ActorCriticDTC' is not defined` from
`on_policy_runner.py:430`'s `eval(class_name)`. The new class was exported
from `rsl_rl.modules` but not imported in the runner module's scope, so
`eval` could not resolve the string. Fixed at commit `d7ee265` (one-line
import added). Re-ran: success.

**Verdict: PASS (pipeline) / FAIL (training quality, as expected for 100
iter on a fresh policy at 1/4 the normal batch size).**

Pipeline confirmations:

- ActorCriticDTC built with actor `input_dim=216`, critic `input_dim=2298`
  (proprio 96 + 3 planner-derived terms 120 + critic privileged extras).
- Wandb run `smoke_dtc_lite_256` synced to `ame_locomotion` project.
- 100 iterations completed in ~135s wall (after Isaac Sim startup).
- All 43 tensorboard scalars present, including new
  `Episode_Reward/footstep_swing_tracking_log` and
  `Episode_Reward/planner_consistency`.
- No NaN / Inf anywhere in the run.

Final reward sample (iter 99):

| Scalar | iter 0 | iter 50 | iter 99 |
|---|---|---|---|
| `Train/mean_reward` | -45737 | -7.6 | -7.0 |
| `Episode_Reward/footstep_swing_tracking_log` | -0.023 | -0.004 | -0.005 |
| `Episode_Reward/planner_consistency` | **-1747.8** | -0.17 | -0.13 |
| `Episode_Reward/track_lin_vel_xy_exp` | 0.001 | 0.000 | 0.000 |
| `Episode_Reward/termination_penalty` | -0.149 | -0.200 | -0.200 |
| `Curriculum/terrain_levels` | 3.40 | **0.00** | **0.00** |
| `Episode_Termination/base_contact` | 2.75 | 92.5 | **126.75** |

## Three follow-up issues to fix before real training

### 1. `planner_consistency` initialization artifact (HIGH priority)

`prev_plan_buffer` is initialized to zeros in `FootstepPlanCommand.__init__`.
On the first `_commit_plan`, the diff from zeros to the actual world-frame
plan (which contains foothold positions a meter or so from the origin) is
**huge** — squared and summed produces ~70+ m² per env on the first frame,
times weight -20, times batch averaging → mean_reward of **-45,737 at iter 0**.

The artifact converges away after ~10 iterations as all envs commit at least
once. But the initial spike dominates the value function fit and wastes
~10 iterations on a non-learning task signal.

**Fix:** in `_resample_command`, snapshot `prev_plan_buffer[env_ids] =
plan_buffer[env_ids].clone()` AFTER the first `_commit_plan(env_ids)` call.
That makes the first consistency diff for a newly-reset env equal zero.
One-line change.

### 2. `select_foothold_by_cost` (K, K) memory (HIGH priority)

Brute-force pairwise distance matrix `(B, K, K, 2)` is 3.9 GB at B=1024,
K=693 — forces smoke to use only 256 envs on a 16 GB GPU. Real training
typically uses 4096 envs.

**Fix:** the scanner is a regular grid (`patterns.GridPatternCfg`); replace
the K×K NN search with an O(1) grid-bucket lookup using known cell
resolution. Same fix should be applied to `local_heightscan_around`'s
similar pattern (`(B, F, n*n, K, 2)`, 1.1 GB at B=4096).

Both helpers will need rewriting. Expected ~50 LoC each plus tests.

### 3. `Curriculum/terrain_levels` collapse to 0 (pre-existing, plan Task 11 fix specified)

Identical to prior AME runs — the `terrain_levels_vel` demotion threshold
(50% of commanded distance over the episode) is too aggressive for
20s episodes at 1 m/s commands. Robots get demoted constantly even when
walking fine.

**Fix already specified in plan Task 11**: replace with `_terrain_levels_lenient`
(promotion at 70% terrain length, demotion at 30% commanded distance). Apply
when DTC-lite enters real training.

## Other observations

- `Episode_Termination/base_contact = 126.75` at iter 99 means robots are
  falling almost immediately. Expected for early random policy on rough
  terrain at level 0. Combined with the curriculum collapse, episodes are
  reset frequently.
- `track_lin_vel_xy_exp`, `alive`, `track_ang_vel_z_exp` all settled near
  zero — policy hasn't found a "stand up" basin yet. Normal for 100 iter
  on this batch size; previous AME / DTC runs needed 1000–3000 iter for
  visible gait emergence.
- Iteration time settled to ~1.35 s at 256 envs, projecting ~3.5 hr for a
  full 10000-iter training. Real-batch (4096 envs) projected ~14 hr after
  fixing follow-up #2.

## Next steps

1. **Apply fixes #1 and #2** as a fresh plan (or as direct fixes on `feat/dtc-lite`).
2. Apply fix #3 (Task 11 from the plan) only when ready to bake a real
   training run.
3. Then run a real 3000–10000 iter training and compare against the existing
   DTC baseline at the same iter count.

## Post-fix retest (commit `e0409ec`)

Applied the one-line `prev_plan_buffer` re-sync in `_resample_command` and
re-ran the same 100-iter smoke (256 envs, wandb run
`smoke_dtc_lite_256_post_fix`). Comparison vs. pre-fix run:

| Scalar | iter 0 before → after | iter 99 before → after |
|---|---|---|
| `Train/mean_reward` | **-45737 → -6.2** | -7.0 → **-4.3** |
| `Episode_Reward/planner_consistency` | -1747.8 → **-0.049** | -0.131 → **-0.001** |
| `Episode_Reward/footstep_swing_tracking_log` | -0.023 → -0.023 | -0.005 → -0.001 |
| `Episode_Termination/base_contact` | 2.75 → 2.75 | **126.75 → 28.5** |
| `Curriculum/terrain_levels` | 3.40 → 3.40 | 0.00 → 0.00 (unchanged — followup #3) |

**Unexpected finding:** the iter-0 spike was actively poisoning the value
function, not just wasting compute. With the spike removed, end-of-run
fall counts (`base_contact`) drop 4.4× even though the policy still hasn't
learned anything useful. The corrupted value targets at iter 0 were
propagating downstream and degrading exploration.

This bumps the priority of the analogous follow-ups (K² memory rewrite,
curriculum) — early signal quality matters more than the per-iter cost
on these short runs.
