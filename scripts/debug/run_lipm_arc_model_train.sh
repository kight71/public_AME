#!/usr/bin/env bash
# Full flat-ground omni training for the LIPM arc-model PaperMLP pipeline.
#
# Run after arc smokes pass (see run_lipm_arc_model_smokes.sh).
#
# Usage:
#   ./scripts/debug/run_lipm_arc_model_train.sh
#   ITER=15000 NUM_ENVS=1024 ./scripts/debug/run_lipm_arc_model_train.sh
#   LOGGER=wandb LOG_PROJECT=ame_lipm ./scripts/debug/run_lipm_arc_model_train.sh --resume
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.AME/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

TASK="${TASK:-AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-FlatOmni-v0}"
ITER="${ITER:-15000}"
NUM_ENVS="${NUM_ENVS:-1024}"
RUN_NAME="${RUN_NAME:-lipm_arc_flat_omni}"
LOGGER="${LOGGER:-tensorboard}"
LOG_PROJECT="${LOG_PROJECT:-ame_lipm_arc}"

echo "LIPM arc-model flat omni training"
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
