#!/usr/bin/env bash
# Low-step terrain-window A/B training: no alive reward + wider forward velocity std.
#
# Use this when the baseline terrain-window run learns stability/yaw quickly
# but keeps a high forward velocity error.  The env keeps the same LIPM,
# terrain-window candidates, low-step terrain, and DS gait as
# run_lipm_terrain_window_train.sh; it only removes alive and sets
# track_lin_vel_xy_exp std from 0.25 to 0.5.
#
# Usage:
#   ./scripts/debug/run_lipm_terrain_window_noalive_velstd05_train.sh
#   ITER=3000 NUM_ENVS=8192 LOGGER=wandb ./scripts/debug/run_lipm_terrain_window_noalive_velstd05_train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.AME/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

TASK="${TASK:-AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-TerrainWindow-NoAliveVelStd05-v0}"
ITER="${ITER:-3000}"
NUM_ENVS="${NUM_ENVS:-1024}"
RUN_NAME="${RUN_NAME:-terrain_window_noalive_velstd05}"
LOGGER="${LOGGER:-tensorboard}"
LOG_PROJECT="${LOG_PROJECT:-ame_lipm_arc}"

echo "LIPM arc-model low-step terrain-window no-alive vel-std0.5 training"
echo "  root       : $ROOT"
echo "  python     : $PY"
echo "  task       : $TASK"
echo "  iterations : $ITER"
echo "  num_envs   : $NUM_ENVS"
echo "  run_name   : $RUN_NAME"
echo "  logger     : $LOGGER"
if [[ "$LOGGER" == "wandb" ]]; then
  echo "  wandb proj : $LOG_PROJECT"
fi
echo ""

CMD=(
  "$PY" "$ROOT/scripts/rsl_rl/train.py"
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --max_iterations "$ITER"
  --headless
  --run_name "$RUN_NAME"
  --logger "$LOGGER"
)

if [[ "$LOGGER" == "wandb" ]]; then
  CMD+=(--log_project_name "$LOG_PROJECT")
fi

CMD+=("$@")

echo "Command:"
printf '  %q' "${CMD[@]}"
echo ""
echo ""

cd "$ROOT"
exec "${CMD[@]}"
