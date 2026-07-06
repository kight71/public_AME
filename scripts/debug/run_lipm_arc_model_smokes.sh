#!/usr/bin/env bash
# LIPM arc-model validation smokes (per-foot v_eff + landing-yaw hip anchor).
#
# Run in separate Isaac processes to avoid gym.make hangs. Recommended order:
#   1. FlatYaw  — pure yaw arc footholds
#   2. FlatXY   — forward/lateral regression (wz=0)
#   3. FlatOmni — combined xy + yaw
#
# Usage:
#   ./scripts/debug/run_lipm_arc_model_smokes.sh
#   ITER=100 NUM_ENVS=256 ./scripts/debug/run_lipm_arc_model_smokes.sh
#   SKIP_XY=1 ./scripts/debug/run_lipm_arc_model_smokes.sh   # yaw + omni only
#   LOGGER=wandb LOG_PROJECT=ame_lipm ./scripts/debug/run_lipm_arc_model_smokes.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.AME/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

ITER="${ITER:-100}"
NUM_ENVS="${NUM_ENVS:-1024}"
LOGGER="${LOGGER:-tensorboard}"
LOG_PROJECT="${LOG_PROJECT:-ame_lipm_arc}"
SKIP_XY="${SKIP_XY:-0}"
EXTRA=("$@")

run_smoke() {
  local script="$1"
  local label="$2"
  echo ""
  echo "=== LIPM arc smoke: $label ==="
  ITER="$ITER" NUM_ENVS="$NUM_ENVS" LOGGER="$LOGGER" LOG_PROJECT="$LOG_PROJECT" \
    "$ROOT/scripts/debug/$script" "${EXTRA[@]}"
}

echo "LIPM arc-model smoke suite"
echo "  root     : $ROOT"
echo "  python   : $PY"
echo "  iter     : $ITER"
echo "  num_envs : $NUM_ENVS"
echo "  logger   : $LOGGER"
if [[ "$LOGGER" == "wandb" ]]; then
  echo "  wandb    : $LOG_PROJECT"
fi
echo ""

run_smoke run_lipm_paper_mlp_flat_yaw_smoke.sh "FlatYaw (wz only)"
if [[ "$SKIP_XY" != "1" ]]; then
  run_smoke run_lipm_paper_mlp_flat_xy_smoke.sh "FlatXY (xy only, wz=0)"
fi
run_smoke run_lipm_paper_mlp_flat_omni_smoke.sh "FlatOmni (xy + yaw)"

echo ""
echo "LIPM arc-model smokes finished OK."
