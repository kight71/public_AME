#!/usr/bin/env bash
# Low-step terrain-window training for the LIPM arc-model PaperMLP pipeline.
#
# This script is the terrain counterpart of run_lipm_arc_model_train.sh.  It
# deliberately defaults to the low-step TerrainWindow task, so long terrain
# runs do not accidentally fall back to the flat-omni task.
#
# Usage:
#   ./scripts/debug/run_lipm_terrain_window_train.sh
#   ITER=6000 NUM_ENVS=2048 ./scripts/debug/run_lipm_terrain_window_train.sh
#
# Continue from the 500-iter low-step smoke checkpoint:
#   ./scripts/debug/run_lipm_terrain_window_train.sh \
#     --resume \
#     --load_run 2026-07-07_16-56-56_terrain_window_lowstep_500 \
#     --checkpoint model_499.pt
#
# Switch to wandb logging:
#   LOGGER=wandb LOG_PROJECT=ame_lipm_arc ./scripts/debug/run_lipm_terrain_window_train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.AME/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

TASK="${TASK:-AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-TerrainWindow-v0}"
ITER="${ITER:-3000}"
NUM_ENVS="${NUM_ENVS:-1024}"
RUN_NAME="${RUN_NAME:-terrain_window_lowstep_long}"
LOGGER="${LOGGER:-tensorboard}"
LOG_PROJECT="${LOG_PROJECT:-ame_lipm_arc}"

echo "LIPM arc-model low-step terrain-window training"
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
