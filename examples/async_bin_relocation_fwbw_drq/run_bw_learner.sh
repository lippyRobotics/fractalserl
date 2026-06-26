export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export SCRIPT_DIR=$(dirname "$(realpath "$0")") && \
export TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S") && \
export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/bw-$TIMESTAMP" && \

# Capacity must satisfy ceil(capacity / expected_branches) >= real env steps so the
# image ring never wraps and decorrelates stored states from their frames. With
# expected_branches = starting_branch_count^2 = 27^2 = 729 and --max_steps 30_000:
#   capacity >= 30_000 * 729 ~= 21_870_000  ->  22_000_000.
# --front_plane_homography points to a .npy holding the 3x3 state-(x,y)->pixel
# matrix calibrated per camera placement. Drop the flag to disable the
# front-camera warp.
python async_drq_randomized.py "$@" \
    --seed 1 \
    --replay_buffer_type fractal_symmetry_replay_buffer \
    --replay_buffer_capacity 22_000_000 \
    --front_plane_homography ./front_plane_homography.npy \
    --world_fixed_img_keys front \
    --demo_path ./demos/bw_demos/baseline_01.pkl \
    --exp_name=serl_dev_drq_rlpd20demos_bin_fwbw_resnet_096_bw \
    --learner \
    --env FrankaBinRelocation-Vision-v0 \
    --max_steps 30_000 \
    --random_steps 200 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --starting_branch_count 27 \
    --branch_method "constant" \
    --split_method "never" \
    --alpha 0.2 \
    --max_depth 3 \
    --branching_factor 3 \
    --workspace_width 0.3 \
    --fwbw bw \
    --checkpoint_period 500 \
    --checkpoint_path $CHECKPOINT_DIR
