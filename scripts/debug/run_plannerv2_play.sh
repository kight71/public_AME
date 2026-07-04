#!/usr/bin/env bash
# Visual eval for PlannerV2 long training (iter 12500 checkpoint).
# GUI mode (no --headless): Isaac Sim window stays open until you close it.
# Video mode: pass --video to also save an mp4 under logs/.../videos/play/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.AME/bin/python}"
RUN_DIR="2026-07-03_21-55-37_plannerv2_8b_long"
CKPT="${CKPT:-$ROOT/logs/rsl_rl/g1_ame/$RUN_DIR/model_12500.pt}"

echo "PlannerV2 play — checkpoint: $CKPT"
echo "Task: AME-G1-29DOF-DTC-PlannerV2-Play-v0"
echo ""
echo "Modes:"
echo "  GUI (default): Isaac Sim window — close window to exit"
echo "  Video:         bash $0 --headless --video --video_length 600"
echo ""

exec "$PY" "$ROOT/scripts/rsl_rl/play.py" \
  --task "AME-G1-29DOF-DTC-PlannerV2-Play-v0" \
  --checkpoint "$CKPT" \
  --num_envs 1 \
  --real-time \
  "$@"
