#!/bin/bash
# BeamDojo training entry point: model-free foothold policy + sampling reward.
# Usage: bash train_beamdojo.sh
set -e

PY=/home/tan/pure_AME/.AME/bin/python

nohup "$PY" scripts/rsl_rl/train.py \
    --task AME-G1-29DOF-BeamDojo-v0 \
    --max_iterations 15000 \
    --headless \
    --logger wandb \
    --log_project_name AME_locomotion \
    --run_name beamdojo_single_critic \
    > train_beamdojo.log 2>&1 &
