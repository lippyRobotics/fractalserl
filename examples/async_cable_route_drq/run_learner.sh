# All export statements end with && \ to chain them together
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# XLA memory fraction with learner+action <0.8. Learner needs more.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
# Use malloc_async to reduce fragmentation, overlap memory allocation with compute, lower stalls and improve worklads. Requires cuda11.2+
export TF_GPU_ALLOCATOR=cuda_malloc_async && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/checkpoints-$TIMESTAMP" && \

python async_drq_randomized.py "$@" \
    --learner \
    --env FrankaCableRoute-Vision-v0 \
    --exp_name="Franka-CableRoute-V2" \
    --seed 1 \
    --random_steps 600 \
    --training_starts 1 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --max_steps 8501 \
    --replay_buffer_type fractal_symmetry_replay_buffer \
    --replay_buffer_capacity 3_600_000 \
    --starting_branch_count 27 \
    --branch_method "constant" \
    --split_method "never" \
    --alpha 0.2 \
    --max_depth 3 \
    --branching_factor 3 \
    --workspace_width 0.3 \
    --encoder_type resnet-pretrained \
    --demo_path /home/student/code/serl/examples/async_cable_route_drq/demos/fractal_01.pkl \
    --checkpoint_period 1000 \
    --checkpoint_path $CHECKPOINT_DIR
