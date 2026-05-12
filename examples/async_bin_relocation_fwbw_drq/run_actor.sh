export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_bin_fwbw_resnet_096 \
    --seed 0 \
    --random_steps 200 \
    --encoder_type resnet-pretrained \
    --replay_buffer_type memory_efficient_replay_buffer \
    --branch_method "fractal" \
    --split_method "time" \
    --branching_factor 3 \
    --max_depth 3 \
    --alpha 0.2 \
    --max_traj_length 100 \
    --workspace_width 0.3 \
    --fw_ckpt_path /home/student/code/serl/examples/async_bin_relocation_fwbw_drq/checkpoints/fw \
    --bw_ckpt_path /home/student/code/serl/examples/async_bin_relocation_fwbw_drq/checkpoints/bw \
    --fw_reward_classifier_ckpt_path "/home/student/code/serl/examples/async_bin_relocation_fwbw_drq/classifier/fw_classifier_trained" \
    --bw_reward_classifier_ckpt_path "/home/student/code/serl/examples/async_bin_relocation_fwbw_drq/classifier/bw_classifier_trained" \
    # --eval_checkpoint_step 31000 \
    # --eval_checkpoint_step 1000
