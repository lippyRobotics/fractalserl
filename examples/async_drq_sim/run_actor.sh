#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
# export MUJOCO_GL=egl && \
export MUJOCO_GL=${MUJOCO_GL:-glfw}
export TF_GPU_ALLOCATOR=cuda_malloc_async && \
PYTHONNOUSERSITE=1

python "$SCRIPT_DIR/async_drq_sim.py" "$@"\
    --actor \
    --max_traj_length 200 \
    --random_steps 0 \
    --env PandaPickCubeVision-v0 \
    --ip localhost \
    --use_classifier_reward=True \
    --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --reward_classifier_threshold=0.5 \
    --classifier_image_keys wrist \
    --classifier_use_proprio=False \
    --zero_env_reward=True \
    --terminate_on_classifier_success=True \
