#!/usr/bin/env bash
# Task 8b: V2 long training (15000 iter × 256 envs).
# Checkpoints saved every 3000 iterations for staged review.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.AME/bin/python}"
PROJ="${LOG_PROJECT:-ame_planner_v2}"
EXTRA=("$@")

echo "Task 8b — V2 long training: 15000 iter, checkpoint every 3000."
echo "wandb project: $PROJ"
echo ""

"$PY" "$ROOT/scripts/rsl_rl/train.py" \
  --task "AME-G1-29DOF-DTC-PlannerV2-v0" \
  --num_envs 1024 \
  --max_iterations 15000 \
  --headless \
  --run_name "plannerv2_8b_long" \
  --logger wandb \
  --log_project_name "$PROJ" \
  "${EXTRA[@]}"

echo ""
echo "Done. Checkpoints saved every 100 iter (default save_interval)."
echo "Use scripts/debug/monitor_task8b.sh to check metrics at 3k intervals."
