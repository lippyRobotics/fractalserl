# Async Image DrQ Simulation

This example runs asynchronous image-based DrQ/RLPD training in `franka_sim`.
The default task is `PandaPickCubeVision-v0`, with SERL observations containing
`state`, `front`, and `wrist`.

Supported workflows:

- Online image DrQ with an online replay buffer.
- RLPD with separate online and demonstration replay buffers and 50/50 batches.
- Optional binary visual classifier reward.
- Local or networked learner/actor processes.

Run all commands from the SERL repository root unless stated otherwise.

## Setup

```bash
conda activate serl

export MUJOCO_GL=${MUJOCO_GL:-glfw}
unset PYOPENGL_PLATFORM

mkdir -p ~/.serl
wget -O ~/.serl/resnet10_params.pkl \
    https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
```

`glfw` is the default for a local X11 desktop. Set `MUJOCO_GL=egl` or
`MUJOCO_GL=osmesa` only on a headless machine configured for that backend.

## Validate Installation

Compile the relevant Python packages:

```bash
python -m compileall examples/async_drq_sim franka_sim serl_launcher
```

Verify simulation images and observation shapes:

```bash
python - <<'PY'
import gym
import franka_sim
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

env = gym.make("PandaPickCubeVision-v0", disable_env_checker=True)
env = SERLObsWrapper(env)
env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
obs, _ = env.reset()
print(env.observation_space)
print(obs.keys())
print(env.action_space)
env.close()
PY
```

Verify the asynchronous script without starting networking or training:

```bash
bash examples/async_drq_sim/run_actor.sh \
    --env PandaPickCubeVision-v0 \
    --dry_run=True
```

## Select JAX Backend

For normal GPU training, leave CPU forcing disabled and verify JAX first:

```bash
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY
```

Do not start a long run unless this reports a GPU. The simulation code does
not disable or patch CUDA during normal GPU use.

For CPU-only smoke tests:

```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
```

When `JAX_PLATFORMS=cpu` exactly, the simulation scripts skip initialization
of an installed but unused CUDA plugin. CPU is suitable for validation and
small classifier tests, but full image RL training is expected to use a GPU.

## Training a Vision-Based Binary Classifier

The reward classifier is a binary visual success detector. With
`--zero_env_reward=True`, RL receives only the classifier's binary reward.
Otherwise, classifier success reward is added to the environment reward.

The complete classifier data, training, and evaluation workflow is documented
in [classifier/README.md](classifier/README.md).

Minimal sequence:

```bash
python examples/async_drq_sim/classifier/record_demo.py \
    --env PandaPickCubeVision-v0 \
    --label positive \
    --policy scripted \
    --num_samples 1000 \
    --force_positive_state_fallback=False \
    --output_path examples/async_drq_sim/classifier/demos/success.pkl

python examples/async_drq_sim/classifier/record_demo.py \
    --env PandaPickCubeVision-v0 \
    --label negative \
    --policy random \
    --num_samples 500 \
    --output_path examples/async_drq_sim/classifier/demos/failure_random.pkl

python examples/async_drq_sim/classifier/record_demo.py \
    --env PandaPickCubeVision-v0 \
    --label negative \
    --policy scripted \
    --num_samples 500 \
    --output_path examples/async_drq_sim/classifier/demos/failure_scripted.pkl

python examples/async_drq_sim/classifier/train_reward_classifier.py \
    --env PandaPickCubeVision-v0 \
    --positive_demo_paths examples/async_drq_sim/classifier/demos/success.pkl \
    --negative_demo_paths examples/async_drq_sim/classifier/demos/failure_random.pkl \
    --negative_demo_paths examples/async_drq_sim/classifier/demos/failure_scripted.pkl \
    --classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --image_keys front \
    --image_keys wrist \
    --batch_size 64 \
    --steps_per_epoch 2

python examples/async_drq_sim/classifier/test_classifier.py \
    --env PandaPickCubeVision-v0 \
    --classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --image_keys front \
    --image_keys wrist \
    --positive_demo_path examples/async_drq_sim/classifier/demos/success.pkl \
    --negative_demo_path examples/async_drq_sim/classifier/demos/failure_scripted.pkl \
    --output_report examples/async_drq_sim/classifier/classifier_report.json
```

The camera flags select which image observations the classifier uses.
`--image_keys front --image_keys wrist` trains a two-camera classifier.
For a front-only classifier, remove `--image_keys wrist` from both the
training and testing commands.

The same camera choice must be used downstream wherever the classifier
checkpoint is loaded. If the checkpoint was trained with both cameras, pass
`--classifier_image_keys front --classifier_image_keys wrist` when recording
sparse-reward RL demos and when running both the DRL learner and actor. If the
checkpoint was trained front-only, pass only `--classifier_image_keys front`.
The RL policy can still observe both cameras; these flags only control the
visual reward classifier.

## Record RL Demonstrations

Record image transitions compatible with the SERL replay buffer. The recorder
can save either the simulator's dense environment reward or a sparse binary
classifier reward. Dense demos can be recorded before the classifier exists.
Sparse classifier demos should be recorded only after the classifier checkpoint
has been trained and tested.

### Option A: Dense-Reward RL Demos

Use this when training with the normal simulator reward, or when you want RLPD
demos to retain the dense pick-and-lift shaping reward.

```bash
python examples/async_drq_sim/record_demo.py \
    --env PandaPickCubeVision-v0 \
    --policy scripted \
    --num_trajectories 20 \
    --max_traj_length 200 \
    --success_only=True \
    --demo_reward_mode dense \
    --output_path examples/async_drq_sim/demos/pickcube_20_demos_dense.pkl
```

Train with those demos and dense environment rewards:

```bash
bash examples/async_drq_sim/run_learner.sh \
    --env PandaPickCubeVision-v0 \
    --demo_path examples/async_drq_sim/demos/pickcube_20_demos_dense.pkl \
    --batch_size 256 \
    --exp_name sim-pickcube-drq-w-vision \
    --run_name learner-rlpd-dense-demos \
    --checkpoint_period 1000 \
    --checkpoint_path examples/async_drq_sim/checkpoints/pickcube-rlpd-dense-demos

bash examples/async_drq_sim/run_actor.sh \
    --env PandaPickCubeVision-v0 \
    --ip localhost
```

### Option B: Sparse Classifier-Reward RL Demos

Use this when training with `--zero_env_reward=True` and you want both online
and demonstration replay to use the same binary visual success reward. The
classifier checkpoint must already exist, and the image keys must match the
classifier training command.

```bash
python examples/async_drq_sim/record_demo.py \
    --env PandaPickCubeVision-v0 \
    --policy scripted \
    --num_trajectories 20 \
    --max_traj_length 200 \
    --success_only=True \
    --demo_reward_mode classifier_sparse \
    --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --reward_classifier_threshold=0.5 \
    --classifier_image_keys front \
    --classifier_image_keys wrist \
    --classifier_use_proprio=False \
    --terminate_on_classifier_success=True \
    --output_path examples/async_drq_sim/demos/pickcube_20_demos_classifier_sparse.pkl
```

Train with those demos and classifier-only sparse rewards:

```bash
bash examples/async_drq_sim/run_learner.sh \
    --env PandaPickCubeVision-v0 \
    --demo_path examples/async_drq_sim/demos/pickcube_20_demos_classifier_sparse.pkl \
    --batch_size 256 \
    --exp_name sim-pickcube-drq-w-vision \
    --run_name learner-rlpd-classifier-sparse-demos \
    --checkpoint_period 1000 \
    --checkpoint_path examples/async_drq_sim/checkpoints/pickcube-rlpd-classifier-sparse-demos \
    --use_classifier_reward=True \
    --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --reward_classifier_threshold=0.5 \
    --classifier_image_keys front \
    --classifier_image_keys wrist \
    --classifier_use_proprio=False \
    --zero_env_reward=True \
    --terminate_on_classifier_success=True

bash examples/async_drq_sim/run_actor.sh \
    --env PandaPickCubeVision-v0 \
    --ip localhost \
    --use_classifier_reward=True \
    --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --reward_classifier_threshold=0.5 \
    --classifier_image_keys front \
    --classifier_image_keys wrist \
    --classifier_use_proprio=False \
    --zero_env_reward=True \
    --terminate_on_classifier_success=True
```

Use `--policy random` for a recorder smoke test. Manual control is not
available in this simulator, so `manual` falls back to random actions.
The scripted policy uses measured TCP/cube position feedback with a slow
approach, alignment hold, controlled descent, gradual grasp, and fixed-height
lift. Keep `--max_traj_length` at 200 or higher for this conservative sequence.
For front-only classifier demos, remove both `--classifier_image_keys wrist`
flags and use a front-only classifier checkpoint.

## Training in Sim with Vision, No Demos, No Classifier (Dense)

The learner owns model updates and replay buffers. The actor interacts with
the environment, sends transitions to the learner, and receives updated
network parameters.

Start the learner in terminal 1:

```bash
conda activate serl
export MUJOCO_GL=glfw

bash examples/async_drq_sim/run_learner.sh \
    --env PandaPickCubeVision-v0 \
    --exp_name sim-pickcube-drq-w-vision \
    --run_name learner-01 \
    --checkpoint_period 1000 \
    --checkpoint_path examples/async_drq_sim/checkpoints/pickcube
```

Start the actor in terminal 2:

```bash
conda activate serl
export MUJOCO_GL=glfw

bash examples/async_drq_sim/run_actor.sh \
    --env PandaPickCubeVision-v0 \
    --ip localhost
```

The actor waits for the learner server. The learner waits for online replay to
reach `--training_starts`, so both processes must be running.

## Training in Sim with Vision, RLPD Demos, No Classifier (Dense)

Pass demonstrations only to the learner. When demonstrations are present,
`--batch_size` must be even. Each learner batch contains half online samples
and half demonstration samples. Checkpoint flags are optional, but recommended
for full training runs. Use a dedicated checkpoint directory to keep RLPD
checkpoints separate from baseline training.

Terminal 1:

```bash
bash examples/async_drq_sim/run_learner.sh \
    --env PandaPickCubeVision-v0 \
    --demo_path examples/async_drq_sim/demos/pickcube_20_demos_dense.pkl \
    --batch_size 256 \
    --exp_name sim-pickcube-drq-w-vision \
    --run_name learner-w-demos \
    --checkpoint_period 1000 \
    --checkpoint_path examples/async_drq_sim/checkpoints/pickcube-rlpd
```

Terminal 2:

```bash
bash examples/async_drq_sim/run_actor.sh \
    --env PandaPickCubeVision-v0 \
    --ip localhost
```

## Training in Sim with Vision, RLPD Demos and Classifier

Provide matching classifier settings to learner and actor. The classifier
checkpoint must be accessible to both processes. The learner separately saves
RL policy checkpoints to `--checkpoint_path`.

Terminal 1:

```bash
bash examples/async_drq_sim/run_learner.sh \
    --env PandaPickCubeVision-v0 \
    --demo_path examples/async_drq_sim/demos/pickcube_20_demos_classifier_sparse.pkl \
    --batch_size 256 \
    --exp_name sim-pickcube-drq-w-vision \
    --run_name learner-rlpd-classifier \
    --checkpoint_period 1000 \
    --checkpoint_path examples/async_drq_sim/checkpoints/pickcube-rlpd-classifier \
    --use_classifier_reward=True \
    --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --reward_classifier_threshold=0.5 \
    --classifier_image_keys front \
    --classifier_image_keys wrist \
    --classifier_use_proprio=False \
    --zero_env_reward=True \
    --terminate_on_classifier_success=True
```

Terminal 2:

```bash
bash examples/async_drq_sim/run_actor.sh \
    --env PandaPickCubeVision-v0 \
    --ip localhost \
    --use_classifier_reward=True \
    --reward_classifier_ckpt_path examples/async_drq_sim/classifier/checkpoints/pickcube \
    --reward_classifier_threshold=0.5 \
    --classifier_image_keys front \
    --classifier_image_keys wrist \
    --classifier_use_proprio=False \
    --zero_env_reward=True \
    --terminate_on_classifier_success=True
```

Use repeated `--classifier_image_keys` flags to select classifier cameras, for
example `--classifier_image_keys front --classifier_image_keys wrist`. If not
provided, all non-state image keys are inferred. Set
`--classifier_use_proprio=True` only if the checkpoint was trained with
`--use_proprio=True`.

## Deploy Across Two Machines

Run the learner on a machine reachable by the actor. The default Agentlace
ports are TCP `5488` and `5489`; allow them through the host firewall.

On the learner machine:

```bash
bash examples/async_drq_sim/run_learner.sh \
    --env PandaPickCubeVision-v0 \
    --exp_name sim-pickcube-remote
```

On the actor machine:

```bash
bash examples/async_drq_sim/run_actor.sh \
    --env PandaPickCubeVision-v0 \
    --ip LEARNER_HOST_OR_IP
```

Use the same environment, encoder, image keys, seed-sensitive model settings,
and classifier configuration on both machines. Classifier checkpoint paths may
differ between hosts, but they must contain equivalent checkpoints.

## Tmux Launchers

For a basic local run:

```bash
bash examples/async_drq_sim/tmux_launch.sh
```

For RLPD with demonstrations:

```bash
DEMO_PATH=demos/pickcube_20_demos.pkl \
bash examples/async_drq_sim/tmux_rlpd_launch.sh
```

Additional actor and learner flags can be passed through `EXTRA_ARGS`:

```bash
EXTRA_ARGS="--env PandaPickCubeVision-v0 --debug=True" \
bash examples/async_drq_sim/tmux_launch.sh
```

## Operational Checks

- Use `--debug=True` to disable online Weights & Biases logging during tests.
- Set `--checkpoint_period` and `--checkpoint_path` on the learner to save
  model checkpoints.
- Check that actor transitions increase the learner replay-buffer progress.
- Check classifier `prob`, accuracy, and confusion matrix before enabling it as
  the only reward.
- Keep classifier image keys and proprioception settings identical between
  training, learner, and actor.
- Use an even batch size whenever demonstrations or preloaded RLDS data are
  supplied.

## Common Failures

`X11: Failed to open display`
: Confirm `DISPLAY` is set and use `MUJOCO_GL=glfw` from an X11 session.

JAX CUDA initialization fails
: Run `nvidia-smi`, then verify `jax.default_backend()` and `jax.devices()`.
  Use the CPU-only environment variables for smoke testing while repairing the
  host NVIDIA driver/JAX CUDA stack.

No image keys found
: Use a vision environment such as `PandaPickCubeVision-v0` and confirm the
  wrapped observation contains `front` and/or `wrist`.

Learner remains at `Filling up replay buffer`
: Start the actor, verify `--ip`, and confirm TCP ports `5488` and `5489` are
  reachable.

Classifier checkpoint shape mismatch
: Use the same image keys and `use_proprio` setting used during classifier
  training.
