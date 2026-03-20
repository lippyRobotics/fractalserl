# All export statements end with && \ to chain them together
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# XLA memory fraction with learner+action <0.8. Learner needs more.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
# Use malloc_async to reduce fragmentation, overlap memory allocation with compute, lower stalls and improve worklads. Requires cuda11.2+
export TF_GPU_ALLOCATOR=cuda_malloc_async && \

python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaCableRoute-Vision-v0 \
    --seed 0 \
    --random_steps 0 \
    --encoder_type resnet-pretrained \
    --reward_classifier_ckpt_path /home/student/code/serl/examples/async_cable_route_drq/classifier/checkpoints/ \
    --max_traj_length 100 \
    # --eval_checkpoint_step 20000 \
    # --eval_n_trajs 20 \
    

