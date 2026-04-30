export CHECKPOINT_EVAL="/home/student/code/serl/examples/async_peg_insert_drq/checkpoints" && \


for STEP in 3500 3000 2500 2000 1500 1000 500; do
    echo "Testing $STEP"
    bash run_eval.sh \
        --eval_checkpoint_step $STEP \
        --checkpoint_path "$CHECKPOINT_EVAL/fractal27_05" \
    2>&1 | grep "success rate"
done
