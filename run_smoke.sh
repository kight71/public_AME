#!/bin/bash
# Smoke test: random agent on BeamDojo env, then 100-iter training.
# Usage: bash run_smoke.sh
set -e
set -o pipefail

PY=/home/tan/pure_AME/.AME/bin/python
LOG=/tmp/ame_smoke
mkdir -p "$LOG"

echo "=== [1/2] random_agent (1 env, ~300s timeout) ==="
timeout 300 $PY scripts/random_agent.py \
    --task AME-G1-29DOF-BeamDojo-Smoke-v0 \
    --num_envs 1 \
    --headless \
    2>&1 | tee "$LOG/random4.log"

echo ""
echo "=== [2/2] training smoke (64 envs, 100 iters) ==="
timeout 1800 $PY scripts/rsl_rl/train.py \
    --task AME-G1-29DOF-BeamDojo-Smoke-v0 \
    --num_envs 64 \
    --max_iterations 100 \
    --headless \
    2>&1 | tee "$LOG/train_smoke.log"

echo ""
echo "=== done. logs: $LOG/random4.log, $LOG/train_smoke.log ==="
