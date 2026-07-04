#!/usr/bin/env bash
# Monitor Task 8b long training — prints key metrics at ~3000 iter intervals.
# Usage: bash scripts/debug/monitor_task8b.sh [wandb_run_dir]
#   If no arg, auto-detects the latest wandb run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -n "${1:-}" ]; then
  LOG="$1/files/output.log"
else
  LATEST=$(ls -td "$ROOT"/wandb/run-* 2>/dev/null | head -1)
  if [ -z "$LATEST" ]; then
    echo "No wandb runs found." && exit 1
  fi
  LOG="$LATEST/files/output.log"
  echo "Auto-detected: $LATEST"
fi

echo "Monitoring: $LOG"
echo "Will report at iter 3000, 6000, 9000, 12000, 15000."
echo "Press Ctrl+C to stop."
echo ""

CHECKPOINTS=(3000 6000 9000 12000 15000)
REPORTED=()

report_metrics() {
  local iter=$1
  echo ""
  echo "============================================================"
  echo "  CHECKPOINT: iter $iter"
  echo "============================================================"
  # Get the block around this iteration
  grep -A 40 "Learning iteration $iter/" "$LOG" | grep -E \
    "track_lin_vel_xy_exp|mean_reward|episode_length|fallback_rate|valid_count|mask_reach_valid|mask_step_height_valid|mask_roughness_valid|mask_in_bounds_valid|roughness_valid_L|roughness_valid_R" \
    | head -12
  echo "------------------------------------------------------------"
}

while true; do
  for cp in "${CHECKPOINTS[@]}"; do
    # Skip if already reported
    if printf '%s\n' "${REPORTED[@]}" 2>/dev/null | grep -qx "$cp"; then
      continue
    fi
    # Check if this iteration exists in log
    if grep -q "Learning iteration $cp/" "$LOG" 2>/dev/null; then
      report_metrics "$cp"
      REPORTED+=("$cp")
    fi
  done

  # Check if training is done
  if [ "${#REPORTED[@]}" -eq "${#CHECKPOINTS[@]}" ]; then
    echo ""
    echo "All checkpoints reported. Training complete."
    exit 0
  fi

  sleep 60
done
