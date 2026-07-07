#!/usr/bin/env bash
# Flat-ground yaw-only smoke training for the LIPM PaperMLP pipeline.
#
# Usage:
#   ./scripts/debug/run_lipm_paper_mlp_flat_yaw_smoke.sh
#   ITER=100 NUM_ENVS=256 ./scripts/debug/run_lipm_paper_mlp_flat_yaw_smoke.sh
#   LOGGER=wandb LOG_PROJECT=ame_lipm ./scripts/debug/run_lipm_paper_mlp_flat_yaw_smoke.sh --resume
#
# Command range: vx 0, vy 0, wz [-1.0, 1.0].
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Override: PY=/path/to/python  or  CONDA_ENV=/path/to/env
CONDA_ENV="${CONDA_ENV:-/root/autodl-tmp/conda_envs/AME}"
PY="${PY:-$CONDA_ENV/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$ROOT/.AME/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  PY=python
fi

TASK="${TASK:-AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-FlatYaw-v0}"
ITER="${ITER:-500}"
NUM_ENVS="${NUM_ENVS:-1024}"
RUN_NAME="${RUN_NAME:-lipm_paper_mlp_flat_yaw500}"
LOGGER="${LOGGER:-tensorboard}"
LOG_PROJECT="${LOG_PROJECT:-ame_lipm_paper_mlp}"

echo "LIPM PaperMLP flat-ground yaw-only smoke"
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
