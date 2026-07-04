#!/usr/bin/env bash
# Placement-reward A/B: PlannerV2 baseline vs PlannerV2+PlaceRew.
#
# Matched regime — same v2 selector, obs, phantom (Option B), forward cmd.
# Only difference: foothold_penalty (-0.1) + footstep_placement_overlap (1.0).
#
# Usage:
#   ./scripts/debug/run_placerev_ab.sh              # 3000 iter A/B (both arms)
#   ITER=15000 NUM_ENVS=1024 ./scripts/debug/run_placerev_ab.sh treatment
#   ./scripts/debug/run_placerev_ab.sh baseline --resume
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.AME/bin/python}"
PROJ="${LOG_PROJECT:-ame_planner_v2}"
ITER="${ITER:-3000}"
NUM_ENVS="${NUM_ENVS:-1024}"

MODE="both"
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    both|baseline|control|ctrl|treatment|placerew) MODE="$arg" ;;
    *) EXTRA+=("$arg") ;;
  esac
done

run_train() {
  echo ""
  echo "=== $1 ==="
  "$PY" "$ROOT/scripts/rsl_rl/train.py" \
    --task "$2" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$ITER" \
    --headless \
    --run_name "$3" \
    --logger wandb \
    --log_project_name "$PROJ" \
    "${EXTRA[@]}"
}

case "$MODE" in
  both)
    echo "Placement-reward A/B — $ITER iter × $NUM_ENVS envs (wandb: $PROJ)"
    run_train "Baseline (no placement rewards)" \
      "AME-G1-29DOF-DTC-PlannerV2-v0" "plannerv2_placerev_ctrl"
    run_train "Treatment (+foothold_penalty +overlap)" \
      "AME-G1-29DOF-DTC-PlannerV2-PlaceRew-v0" "plannerv2_placerew"
    ;;
  baseline|control|ctrl)
    run_train "Baseline" "AME-G1-29DOF-DTC-PlannerV2-v0" "plannerv2_placerev_ctrl"
    ;;
  treatment|placerew)
    run_train "Treatment" "AME-G1-29DOF-DTC-PlannerV2-PlaceRew-v0" "plannerv2_placerew"
    ;;
  *)
    echo "Unknown mode: $MODE (use both|baseline|treatment)" >&2
    exit 1
    ;;
esac

cat <<'EOF'

Done. Compare last ~500 iter means in wandb:

  Episode_Reward/track_lin_vel_xy_exp
  Episode_Reward/track_ang_vel_z_exp
  Episode_Reward/foothold_penalty          (treatment only; should trend toward 0)
  Episode_Reward/footstep_placement_overlap (treatment only; should rise toward 2)
  Metrics/footstep_plan/selector_v2_fallback_rate
  Episode/episode_length                  (higher = fewer falls)

If treatment wins on lin_vel + overlap without higher fallback/falls:
  → Phase 2: set phantom_yaw_track_robot=False on PlaceRew and re-run.

EOF
