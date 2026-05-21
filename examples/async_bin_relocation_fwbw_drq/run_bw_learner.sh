export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/bw-$TIMESTAMP" && \

python async_drq_randomized.py "$@" \
    --learner \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_bin_fwbw_resnet_096_bw \
    --seed 1 \
    --max_steps 30_000 \
    --random_steps 200 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --replay_buffer_type fractal_symmetry_replay_buffer \
    --starting_branch_count 27 \
    --branch_method "constant" \
    --split_method "never" \
    --alpha 0.2 \
    --max_depth 3 \
    --branching_factor 3 \
    --workspace_width 0.3 \
    --fwbw bw \
    --demo_path ./demos/bw_demos/baseline_01.pkl \
    --checkpoint_period 500 \
    --checkpoint_path $CHECKPOINT_DIR
