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
    --env PandaPickCubeVision-v0 \
    --exp_name SymmetryImplementationSimulation \
    --run_name Training... \
    --replay_buffer_type fractal_symmetry_replay_buffer \
    --replay_buffer_capacity 30_000 \
    --max_steps 20_000 \
    --training_starts 1000 \
    --random_steps 1000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --starting_branch_count 9 \
    --branch_method constant \
    --workspace_width 0.15 \
    --alpha 0.2 \
    --env PandaPickCubeVision-v0 \
    --demo_path examples/async_drq_sim/demos/pickcube_20_demos_classifier_sparse.pkl \
    --batch_size 256 \
    --checkpoint_period 1000 \
    --checkpoint_path examples/async_drq_sim/checkpoints/pickcube-rlpd-classifier-sparse-demos \
    --use_classifier_reward=True \
    --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --reward_classifier_threshold=0.5 \
    --classifier_image_keys wrist \
    --classifier_use_proprio=False \
    --zero_env_reward=True \
    --terminate_on_classifier_success=True \
