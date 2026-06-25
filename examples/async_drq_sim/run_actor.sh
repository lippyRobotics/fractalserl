#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
export MUJOCO_GL=${MUJOCO_GL:-glfw}
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Example classifier flags:
#   --use_classifier_reward=True \
#   --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
#   --zero_env_reward=True

python "$SCRIPT_DIR/async_drq_sim.py" --actor "$@"

# Note: `"$@"` expands to all arguments passed to the shell script, preserving each argument exactly.
# bash run_actor.sh --env PandaPickCubeVision-v0 --ip localhost
# becomes:
# python async_drq_sim.py \
#     --actor \
#     --env PandaPickCubeVision-v0 \
#     --ip localhost