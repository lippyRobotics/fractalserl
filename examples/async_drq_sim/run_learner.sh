#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4
export MUJOCO_GL=${MUJOCO_GL:-glfw}
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Example classifier/RLPD flags:
#   --demo_path examples/async_drq_sim/demos/pickcube_20_demos.pkl \
#   --use_classifier_reward=True \
#   --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
#   --zero_env_reward=True

python "$SCRIPT_DIR/async_drq_sim.py" --learner "$@"
