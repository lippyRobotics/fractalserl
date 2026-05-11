# All export statements end with && \ to chain them together
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# XLA memory fraction with learner+action <0.8. Learner needs more.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
# Use malloc_async to reduce fragmentation, overlap memory allocation with compute, lower stalls and improve worklads. Requires cuda11.2+
export TF_GPU_ALLOCATOR=cuda_malloc_async && \

export CHECKPOINT_EVAL="/home/student/code/serl/examples/async_cable_route_drq/checkpoints" && \
export STEP=${STEP}

for STEP in 0 4000 2000 1000 0; do
    echo "Testing $STEP"
    python async_drq_randomized.py "$@" \
        --actor \
        --render \
        --env FrankaCableRoute-Vision-v0 \
        --eval_checkpoint_step $STEP \
        --reward_classifier_ckpt_path /home/student/code/serl/examples/async_cable_route_drq/classifier/checkpoints/ \
        --eval_n_trajs 50 \
        --checkpoint_path "$CHECKPOINT_EVAL/baseline_01" 
done
