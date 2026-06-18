# --replay_buffer_capacity 200_000 \ # automatically handled by replay buffer logic
export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/fw-$TIMESTAMP" && \

python async_drq_randomized.py "$@" \
    --seed 1 \
    --replay_buffer_type memory_efficient_replay_buffer \
    --demo_path ./demos/fw_demos/baseline_01.pkl \
    --exp_name=serl_dev_drq_rlpd20demos_bin_fwbw_resnet_096_fw \
    --learner \
    --env FrankaBinRelocation-Vision-v0 \
    --max_steps 30_000 \
    --random_steps 200 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --starting_branch_count 27 \
    --branch_method "constant" \
    --split_method "never" \
    --alpha 0.2 \
    --max_depth 3 \
    --branching_factor 3 \
    --workspace_width 0.3 \
    --fwbw fw \
    --checkpoint_period 500 \
    --checkpoint_path $CHECKPOINT_DIR
