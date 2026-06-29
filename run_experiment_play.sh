#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-up10}"
shift || true

CHECKPOINT="${CHECKPOINT:-pretrained/ame1.pt}"

case "${SCENE}" in
  up10)
    TASK="AME-G1-29DOF-Exp-StepUp10-Play-v0"
    ;;
  up20)
    TASK="AME-G1-29DOF-Exp-StepUp20-Play-v0"
    ;;
  down10)
    TASK="AME-G1-29DOF-Exp-StepDown10-Play-v0"
    ;;
  down20)
    TASK="AME-G1-29DOF-Exp-StepDown20-Play-v0"
    ;;
  blocks)
    TASK="AME-G1-29DOF-Exp-RandomBlocks-Play-v0"
    ;;
  pyramid_up)
    TASK="AME-G1-29DOF-Exp-PyramidUp-Play-v0"
    ;;
  pyramid_down)
    TASK="AME-G1-29DOF-Exp-PyramidDown-Play-v0"
    ;;
  *)
    echo "Unknown scene: ${SCENE}" >&2
    echo "Usage: bash run_experiment_play.sh [up10|up20|down10|down20|blocks|pyramid_up|pyramid_down] [extra play.py args...]" >&2
    exit 2
    ;;
esac

python scripts/rsl_rl/play.py \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  --num_envs 1 \
  --real-time \
  "$@"
