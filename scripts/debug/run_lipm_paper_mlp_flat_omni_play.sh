#!/usr/bin/env bash
# Visualize the flat-ground omni LIPM PaperMLP planner.
#
# Usage:
#   ./scripts/debug/run_lipm_paper_mlp_flat_omni_play.sh --load_run <run> --checkpoint <ckpt>
#   NUM_ENVS=4 ./scripts/debug/run_lipm_paper_mlp_flat_omni_play.sh --load_run <run> --checkpoint <ckpt>
#
# The env enables footstep_plan debug markers:
#   blue/red large spheres  : planned left/right footholds
#   small gray spheres      : last-contact foot positions
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.AME/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

TASK="${TASK:-AME-G1-29DOF-DTC-PlannerV2-LIPM-PaperMLP-FlatOmni-Play-v0}"
NUM_ENVS="${NUM_ENVS:-16}"

echo "LIPM PaperMLP flat-ground omni planner visualization"
echo "  root     : $ROOT"
echo "  python   : $PY"
echo "  task     : $TASK"
echo "  num_envs : $NUM_ENVS"
echo ""

CMD=(
  "$PY" "$ROOT/scripts/rsl_rl/play.py"
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --real-time
)

CMD+=("$@")

echo "Command:"
printf '  %q' "${CMD[@]}"
echo ""
echo ""

cd "$ROOT"
exec "${CMD[@]}"
