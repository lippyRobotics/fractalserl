
# All export statements end with && \ to chain them together
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# XLA memory fraction with learner+action <0.8. Learner needs more.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
# Use malloc_async to reduce fragmentation, overlap memory allocation with compute, lower stalls and improve worklads. Requires cuda11.2+
export TF_GPU_ALLOCATOR=cuda_malloc_async && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export ENV_NAME="FrankaPegInsert-Vision-v0" && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/checkpoints-$TIMESTAMP" && \

# Create checkpoint directory if it doesn't exist. Used to saved learn policy.
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Creating checkpoint directory: $CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR" || {
        echo "Failed to create checkpoint directory!" >&2
        exit 1
    }
fi

python async_drq_randomized.py "$@" \
    --learner \
    --render \
    --env $ENV_NAME \
    --exp_name="PegInsert-march_2026" \
    --seed 5 \
    --training_starts 1 \
    --max_steps 3501 \
    --save_model \
    --batch_size 256 \
    --critic_actor_ratio 8 \
    --replay_buffer_capacity 5_000 \
    --random_steps 1_000 \
    --encoder_type resnet-pretrained \
    --demo_path peg_insert_20_demos_2026-03-19_16-45-22.pkl \
    --save_model \
    --replay_buffer_type "memory_efficient_replay_buffer" \
    --branch_method "constant" \
    --starting_branch_count 27 \
    --workspace_width 0.3 \
    --checkpoint_period 500 \
    --checkpoint_path "$CHECKPOINT_DIR" \
