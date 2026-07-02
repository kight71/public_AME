nohup .AME/bin/python scripts/rsl_rl/train.py \
--task AME-G1-29DOF-MLP-BeamDojo-Flat-Omni-v4 \
--resume \
--load_run 2026-07-02_16-57-39_mlp_beamdojo_flat_omni_v3_style_ft \
--checkpoint model_3500.pt \
--max_iterations 4500 \
--headless \
--logger wandb \
--log_project_name AME_locomotion \
--run_name mlp_beamdojo_flat_omni_v4_foot_orientation_ft \
> train_velgate.log 2>&1 &
