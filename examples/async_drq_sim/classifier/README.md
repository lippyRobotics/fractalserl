# Simulation Reward Classifier

This workflow records diverse visual success/failure examples, trains a binary
reward classifier, and validates it before the classifier is used for RL.
Commands assume the SERL repository root and an activated `serl` environment.

## Prerequisites

```bash
export MUJOCO_GL=${MUJOCO_GL:-glfw}
unset PYOPENGL_PLATFORM

mkdir -p ~/.serl
wget -O ~/.serl/resnet10_params.pkl \
    https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
```

The classifier searches both the current directory and
`~/.serl/resnet10_params.pkl`. Without pretrained parameters it warns and uses
randomly initialized encoder weights.

The camera renderer was corrected to produce distinct, full-frame `front` and
`wrist` images. Delete classifier datasets and checkpoints recorded before
this correction; the trainer rejects datasets whose camera keys are duplicates.

## Record Positive Examples

The scripted controller collects real lifted-cube states. Synthetic forced
positives are disabled by default.

```bash
python examples/async_drq_sim/classifier/record_demo.py \
    --env PandaPickCubeVision-v0 \
    --label positive \
    --policy scripted \
    --num_samples 1000 \
    --num_trajectories 200 \
    --sample_every_n 2 \
    --max_samples_per_trajectory 10 \
    --force_positive_state_fallback=False \
    --output_path examples/async_drq_sim/classifier/demos/success.pkl
```

## Record Negative Examples

Use both random negatives and scripted hard negatives. Scripted negatives
include approach, near-grasp, grasp, and insufficient-lift states.

```bash
python examples/async_drq_sim/classifier/record_demo.py \
    --env PandaPickCubeVision-v0 \
    --label negative \
    --policy random \
    --num_samples 500 \
    --sample_every_n 2 \
    --max_samples_per_trajectory 10 \
    --output_path examples/async_drq_sim/classifier/demos/failure_random.pkl

python examples/async_drq_sim/classifier/record_demo.py \
    --env PandaPickCubeVision-v0 \
    --label negative \
    --policy scripted \
    --num_samples 500 \
    --sample_every_n 2 \
    --max_samples_per_trajectory 10 \
    --output_path examples/async_drq_sim/classifier/demos/failure_scripted.pkl
```

Samples are distributed across each trajectory rather than taking adjacent
frames. Saved trajectory boundaries are retained for leakage-resistant
train/evaluation splitting.

## Train

```bash
python examples/async_drq_sim/classifier/train_reward_classifier.py \
    --env PandaPickCubeVision-v0 \
    --positive_demo_paths examples/async_drq_sim/classifier/demos/success.pkl \
    --negative_demo_paths examples/async_drq_sim/classifier/demos/failure_random.pkl \
    --negative_demo_paths examples/async_drq_sim/classifier/demos/failure_scripted.pkl \
    --classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --image_keys front \
    --image_keys wrist \
    --batch_size 64 \
    --num_epochs 100 \
    --steps_per_epoch 2 \
    --train_split 0.9 \
    --eval_every 10
```

The trainer prints:

- Unique-frame counts and positive/negative mean-image difference.
- Train/evaluation trajectory and sample counts.
- Train and held-out loss/accuracy.
- Held-out positive recall and negative accuracy.
- Average positive and negative probabilities.

The command above trains a two-camera classifier. To train and test using only
the forward camera, remove `--image_keys wrist` from both commands and retain
`--image_keys front`. Use the same image-key selection whenever that checkpoint
is loaded. A front-only classifier uses less GPU memory; two image streams at
batch size 256 can require multi-gigabyte temporary allocations.

## Acceptance Criteria

Do not deploy a classifier that remains near BCE `0.693` or balanced accuracy
`0.5`. As a practical initial gate, require:

- Held-out accuracy at least 90%.
- High positive recall, preferably at least 95%.
- High negative accuracy, preferably at least 95%.
- Clearly separated positive and negative probabilities.
- Live rollout probabilities that rise only after a genuine lift.

Increase trajectory diversity and add hard negatives before increasing model
size or training duration.

## Test

```bash
python examples/async_drq_sim/classifier/test_classifier.py \
    --env PandaPickCubeVision-v0 \
    --classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --image_keys front \
    --image_keys wrist \
    --positive_demo_path examples/async_drq_sim/classifier/demos/success.pkl \
    --negative_demo_path examples/async_drq_sim/classifier/demos/failure_scripted.pkl \
    --num_eval_steps 200 \
    --output_report examples/async_drq_sim/classifier/classifier_report.json
```

## CPU Validation

Use CPU-only JAX for smoke tests when CUDA is unavailable:

```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

python examples/async_drq_sim/classifier/train_reward_classifier.py \
    --env PandaPickCubeVision-v0 \
    --positive_demo_paths /tmp/serl_classifier_positive.pkl \
    --negative_demo_paths /tmp/serl_classifier_negative.pkl \
    --classifier_ckpt_path /tmp/serl_classifier \
    --batch_size 16 \
    --num_epochs 10 \
    --steps_per_epoch 1
```

When `JAX_PLATFORMS=cpu` exactly, simulation scripts skip initialization of an
installed but unused CUDA plugin. GPU behavior is unchanged otherwise.

## GPU Training

```bash
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES
export MUJOCO_GL=glfw
unset PYOPENGL_PLATFORM

python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

Confirm JAX reports a GPU before starting long training. Reduce batch size if
XLA reports large allocation failures.

## Use With RL

Record RL demonstrations:

```bash
python examples/async_drq_sim/record_demo.py \
    --env PandaPickCubeVision-v0 \
    --num_trajectories 20 \
    --policy scripted \
    --success_only=True \
    --output_path examples/async_drq_sim/demos/pickcube_20_demos.pkl
```

Start learner and actor with matching classifier settings:

```bash
bash examples/async_drq_sim/run_learner.sh \
    --env PandaPickCubeVision-v0 \
    --demo_path examples/async_drq_sim/demos/pickcube_20_demos.pkl \
    --use_classifier_reward=True \
    --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --zero_env_reward=True

bash examples/async_drq_sim/run_actor.sh \
    --env PandaPickCubeVision-v0 \
    --use_classifier_reward=True \
    --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --zero_env_reward=True
```

Without classifier flags, `async_drq_sim.py` retains ordinary image DrQ
behavior. With demonstrations, the learner samples separate online and demo
buffers in a 50/50 ratio.
