export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.15 && \
python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_drq_rlpd_20demos_objRel_fwbw \
    --seed 1 \
    --max_steps 30_000 \
    --random_steps 0 \
    --encoder_type resnet-pretrained \
    --replay_buffer_type fractal_symmetry_replay_buffer \
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