#!/usr/bin/env bash

# --replay_buffer_capacity 200_000 \ # automatically handled by replay buffer logic

# Homography augmentation is opt-in. Example:
# FRONT_HOMOGRAPHY_PATH=/path/to/front_homography.json ./run_bw_learner.sh
FRONT_HOMOGRAPHY_PATH="${FRONT_HOMOGRAPHY_PATH:-}"
HOMOGRAPHY_ARGS=()
if [[ -n "$FRONT_HOMOGRAPHY_PATH" ]]; then
    REPLAY_BUFFER_TYPE="${REPLAY_BUFFER_TYPE:-fractal_symmetry_replay_buffer}"
    if [[ "$REPLAY_BUFFER_TYPE" != "fractal_symmetry_replay_buffer" ]]; then
        echo "FRONT_HOMOGRAPHY_PATH requires fractal_symmetry_replay_buffer." >&2
        exit 2
    fi
    HOMOGRAPHY_ARGS=(--front_homography_path "$FRONT_HOMOGRAPHY_PATH")
else
    REPLAY_BUFFER_TYPE="${REPLAY_BUFFER_TYPE:-memory_efficient_replay_buffer}"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/bw-$TIMESTAMP" && \

python async_drq_randomized.py "$@" \
    --seed 1 \
    --replay_buffer_type "$REPLAY_BUFFER_TYPE" \
    "${HOMOGRAPHY_ARGS[@]}" \
    --demo_path ./demos/bw_demos/baseline_01.pkl \
    --exp_name=serl_dev_drq_rlpd20demos_bin_fwbw_resnet_096_bw \
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
    --fwbw bw \
    --checkpoint_period 500 \
    --checkpoint_path $CHECKPOINT_DIR
