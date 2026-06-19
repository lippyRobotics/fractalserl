# All export statements end with && \ to chain them together
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# XLA memory fraction with learner+action <0.8. Learner needs more.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
# Use malloc_async to reduce fragmentation, overlap memory allocation with compute, lower stalls and improve worklads. Requires cuda11.2+
export TF_GPU_ALLOCATOR=cuda_malloc_async && \

export CHECKPOINT_EVAL="/home/student/code/serl/examples/async_bin_relocation_fwbw_drq/checkpoints"
export CLASSIFIER_DIR="/home/student/code/serl/examples/async_bin_relocation_fwbw_drq/classifier"
export STEP=15000

python async_drq_randomized.py \
    --actor \
    --render \
    --env FrankaBinRelocation-Vision-v0 \
    --bw_reward_classifier_ckpt_path "$CLASSIFIER_DIR/bw_classifier_trained/" \
    --fw_reward_classifier_ckpt_path "$CLASSIFIER_DIR/fw_classifier_trained/" \
    --eval_checkpoint_step $STEP \
    --eval_n_trajs 50 \
    --bw_ckpt_path "$CHECKPOINT_EVAL/baseline_01_bw" \
    --fw_ckpt_path "$CHECKPOINT_EVAL/baseline_01_fw" \
    "$@"
