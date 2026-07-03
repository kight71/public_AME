#!/usr/bin/env bash
# Task 8a: matched-regime A/B smoke (200 iter × 256 envs).
# Per-mask attribution logs to wandb under Metrics/footstep_plan/selector_v2_mask_* .
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.AME/bin/python}"
PROJ="${LOG_PROJECT:-ame_planner_v2}"
EXTRA=("$@")

run_train() {
  echo ""
  echo "=== $1 ==="
  "$PY" "$ROOT/scripts/rsl_rl/train.py" \
    --task "$2" \
    --num_envs 256 \
    --max_iterations 200 \
    --headless \
    --run_name "$3" \
    --logger wandb \
    --log_project_name "$PROJ" \
    "${EXTRA[@]}"
}

echo "Task 8a Gate 1 — run Legacy first, then PlannerV2 (separate Isaac processes)."
run_train "Legacy control" "AME-G1-29DOF-DTC-PlannerV2-Legacy-v0" "plannerv2_legacy_ctrl"
run_train "V2 treatment" "AME-G1-29DOF-DTC-PlannerV2-v0" "plannerv2_treatment"

cat <<'EOF'

Done. In wandb (V2 run), check last ~50 iter means:

  Metrics/footstep_plan/selector_v2_fallback_rate          (< 10% gate; plan said 5%)
  Metrics/footstep_plan/selector_v2_mask_reach_valid       (expect ~15 both feet avg)
  Metrics/footstep_plan/selector_v2_mask_roughness_valid   (bottleneck if << reach)
  Metrics/footstep_plan/selector_v2_mask_roughness_valid_R (long-horizon foot)
  Metrics/footstep_plan/selector_v2_mask_step_height_valid
  Metrics/footstep_plan/selector_v2_mask_in_bounds_valid
  Episode_Reward/track_lin_vel_xy_exp

If fallback > 10%: tune the mask layer that is lowest vs reach (usually roughness → max_dz_omega).

EOF
