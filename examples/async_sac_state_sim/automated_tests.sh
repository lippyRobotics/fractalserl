#!/bin/bash

SEEDS=$1
WANDB_OUTPUT_DIR=~/wandb_logs
TEST="async_sac_state_sim.py"
CONDA_ENV="serl"
ENV="PandaReachCube-v0"
MAX_STEPS=10000
TRAINING_STARTS=1000
RANDOM_STEPS=1000
CRITIC_ACTOR_RATIO=8
EXP_NAME="DUMMY_TEST"
REPLAY_BUFFER_TYPE="fractal_symmetry_replay_buffer"
BATCH_SIZE=256

BASE_ARGS="--env $ENV --exp_name $EXP_NAME --wandb_output_dir $WANDB_OUTPUT_DIR --training_starts $TRAINING_STARTS --random_steps $RANDOM_STEPS --batch_size $BATCH_SIZE --critic_actor_ratio $CRITIC_ACTOR_RATIO"
ARGS=""

function run_test {

    for seed in $(seq 1 1 $SEEDS)
    do
        # OPEN_PORTS=$( comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 2 )
        # PORTS=( $OPEN_PORTS )
        # PORT_NUMBER=${PORTS[0]}
        # BROADCAST_PORT=${PORTS[1]}

        # ARGS+=" --port_number $PORT_NUMBER --broadcast_port $BROADCAST_PORT"

        echo "Running constant with args: $ARGS"
        tmux respawn-pane -k -t serl_session:0.1
        tmux respawn-pane -k -t serl_session:0.2
        tmux send-keys -t serl_session:0.1 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --actor --max_steps 2000000000 --seed $seed $BASE_ARGS $ARGS" C-m
        tmux send-keys -t serl_session:0.2 "conda activate $CONDA_ENV && bash automated_tests_helper.sh --learner --max_steps $MAX_STEPS --seed $seed $BASE_ARGS $ARGS" C-m "exit" C-m

        # Wait for learner to finish
        while ! tmux capture-pane -t serl_session:0.2 -p | grep "logout" > /dev/null;
        do 
            sleep 1
        done
        echo "Finished!"
    done
}

# BASELINE TESTING


# for replay_buffer_capacity in 10000
# do
#     ARGS="--run_name baseline --replay_buffer_type replay_buffer  --replay_buffer_capacity $replay_buffer_capacity"
#     run_test
# done



# CONSTANT TESTING

for starting_branch_count in 9
do
    for workspace_width in .5
    do
        for replay_buffer_capacity in $((10000 * $starting_branch_count * $starting_branch_count * $starting_branch_count))
        do
            ARGS="--run_name new_fancy_thing --replay_buffer_type $REPLAY_BUFFER_TYPE --replay_buffer_capacity $replay_buffer_capacity --workspace_width $workspace_width --branch_method 'constant' --starting_branch_count $starting_branch_count"
            run_test
        done
    done
done
  

# # FRACTAL TESTING

# for replay_buffer_capacity in 10000
# do
#     for workspace_width in 0.5
#     do
#         for alpha in 0.9
#         do
#             for branching_factor in 3 9
#             do
#                 for max_depth in 2 4
#                 do
#                     # Fractal Expansion
#                     ARGS="--run_name fractal_expansion-$branching_factor^$max_depth-alpha-$alpha-workspace_width-$workspace_width-batch-size-$batch_size-capacity-$replay_buffer_capacity --replay_buffer_type $REPLAY_BUFFER_TYPE --batch_size $batch_size --replay_buffer_capacity $replay_buffer_capacity --workspace_width $workspace_width --branch_method 'fractal' --alpha $alpha --branching_factor $branching_factor --max_depth $max_depth"
#                     run_test

#                     # Fractal Contraction
#                     ARGS="--run_name fractal_contraction-$branching_factor^$max_depth-alpha-$alpha-workspace_width-$workspace_width-batch-size-$batch_size-capacity-$replay_buffer_capacity --replay_buffer_type $REPLAY_BUFFER_TYPE --batch_size $batch_size --replay_buffer_capacity $replay_buffer_capacity --workspace_width $workspace_width --branch_method 'contraction' --alpha $alpha --branching_factor $branching_factor --max_depth $max_depth"
#                     run_test
#                 done
#             done
#         done
#     done
# done


# # DISASSOCIATIVE TESTING

# for replay_buffer_capacity in 10000
# do
#     for workspace_width in 0.5
#     do
#         for alpha in 0.9
#         do
#             for min_branch_count in 1 3 9
#             do
#                 for max_branch_count in 3 9 27
#                 do
#                     # Disassociative (Hourglass)
#                     ARGS="--run_name disassociative-hourglass-$min_branch_count:$max_branch_count-alpha-$alpha-workspace_width-$workspace_width-batch-size-$batch_size-capacity-$replay_buffer_capacity --replay_buffer_type $REPLAY_BUFFER_TYPE --batch_size $batch_size --replay_buffer_capacity $replay_buffer_capacity --workspace_width $workspace_width --branch_method 'disassociated' --min_branch_count $min_branch_count --max_branch_count $max_branch_count --disassociated_type 'hourglass' --alpha $alpha"
#                     run_test

#                     # Disassociative (Octahedron)
#                     ARGS="--run_name disassociative-hourglass-$min_branch_count:$max_branch_count-alpha-$alpha-workspace_width-$workspace_width-batch-size-$batch_size-capacity-$replay_buffer_capacity --replay_buffer_type $REPLAY_BUFFER_TYPE --batch_size $batch_size --replay_buffer_capacity $replay_buffer_capacity --workspace_width $workspace_width --branch_method 'disassociated' --min_branch_count $min_branch_count --max_branch_count $max_branch_count --disassociated_type 'octahedron' --alpha $alpha"

#                 done
#             done
#         done
#     done
# done


tmux kill-window -t serl_session:$SEED
