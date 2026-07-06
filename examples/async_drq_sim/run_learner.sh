#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7 && \
# export MUJOCO_GL=egl && \
export MUJOCO_GL=${MUJOCO_GL:-glfw}
export TF_GPU_ALLOCATOR=cuda_malloc_async && \

python "$SCRIPT_DIR/async_drq_sim.py" "$@"\
    --learner \
    --alpha 0.2 \
    --batch_size 256 \
    --branch_method constant \
    --checkpoint_path examples/async_drq_sim/checkpoints/pickcube-rlpd-classifier-sparse-demos \
    --checkpoint_period 1000 \
    --classifier_image_keys wrist \
    --critic_actor_ratio 4 \
    --demo_path examples/async_drq_sim/demos/pickcube_20_demos_classifier_sparse.pkl \
    --env PandaPickCubeVision-v0 \
    --exp_name SymmetryImplementationSimulation \
    --max_steps 20_000 \
    --random_steps 1000 \
    --replay_buffer_capacity 3_600_000 \
    --replay_buffer_type memory_efficient_replay_buffer \
    --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --run_name Training... \
    --starting_branch_count 9 \
    --training_starts 1000 \
    --use_classifier_reward=True \
    --workspace_width 0.15 \
