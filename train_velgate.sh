nohup python scripts/rsl_rl/train.py \
--task AME-G1-29DOF-MLP-BeamDojo-Flat-v2 \
--max_iterations 15000 \
--headless \
--logger wandb \
--log_project_name AME_locomotion \
--run_name mlp_beamdojo_flat_v2_step_positive \
> train_velgate.log 2>&1 &
