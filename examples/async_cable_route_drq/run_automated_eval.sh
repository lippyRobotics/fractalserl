export CHECKPOINT_EVAL="/home/student/code/serl/examples/async_cable_route_drq/checkpoints" && \

for STEP in 8000 6000 4000 2000 1000; do
	echo "Testing $STEP"
	bash run_eval.sh \
		--eval_checkpoint_step $STEP \
		--checkpoint_path "CHECKPOINT_EVAL/baseline_05" \
		--eval_n_trajs 50 \
	2<&1 | grep "success rate"
done
