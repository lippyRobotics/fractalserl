"""Beginner tutorial for training a vision-based reward classifier.

This script teaches a classifier to answer a binary question:
"Does this camera observation look like success?"

The training data comes from demonstration trajectories saved on disk:

1. Positive demonstrations: trajectories that end in success.
2. Negative demonstrations: trajectories that show failure or non-success.

At a high level, the script does the following:

1. Build the robot environment only to recover the observation/action spaces.
2. Load positive and negative trajectories into replay buffers.
3. Repeatedly draw a half-batch from each buffer.
4. Convert those samples into images plus binary labels.
5. Train a classifier with binary cross-entropy.
6. Save the trained model as a checkpoint for later evaluation or reward shaping.

Reading tips for new students:

- `observations` are the states at time step t.
- `next_observations` are the states at time step t + 1.
- The classifier never predicts an action. It only predicts whether an
  observation should be labeled positive or negative.
- JAX/Flax code often separates "define the computation" from "run the
  computation". The `train_step` function below is the compiled update rule.

Typical usage:

```bash
python examples/async_bin_relocation_fwbw_drq/train_reward_classifier.py \
  --positive_demo_paths=/path/to/success.pkl \
  --negative_demo_paths=/path/to/failure.pkl \
  --classifier_ckpt_path=/tmp/reward_classifier_ckpt
```
"""

import os
# Set JAX's XLA backend not to grab most GPU memory up front when the program starts. 
# Otherwise the XLA will preallocate a large chunk of GPU memmory and starve other processes.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Set JAX's XLA backend to cap GPU memory use to up to 20%. 
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import pickle as pkl
import jax
from jax import numpy as jnp
import flax
import flax.linen as nn
from flax.training import checkpoints
import optax
from tqdm import tqdm
import gym
from absl import app, flags

from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper
from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    populate_data_store,
)
from serl_launcher.networks.reward_classifier import create_classifier

import franka_env
from franka_env.envs.wrappers import Quat2EulerWrapper
from franka_env.envs.relative_env import RelativeFrame

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("positive_demo_paths", None, "paths to positive demos")
flags.DEFINE_multi_string("negative_demo_paths", None, "paths to negative demos")
flags.DEFINE_string("classifier_ckpt_path", ".", "Path to classifier checkpoint")
flags.DEFINE_integer("batch_size", 256, "Batch size for training")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs for training")


def main(_):
    """Construct the wrapped environment and launch training.

    The environment is used here as a convenient source of the observation and
    action spaces expected by the classifier and replay buffers. Training itself
    happens purely from the saved demonstrations, not from fresh online rollouts.
    """
    env = gym.make("FrankaBinRelocation-Vision-v0", save_video=False)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraWrapper(env)

    # This example keeps only the front camera stream for classification.
    train_reward_classifier(env.front_observation_space, env.action_space)


def train_reward_classifier(observation_space, action_space):
    """Train a binary reward classifier from offline demonstrations.

    Tutorial overview:

    1. Identify which observation entries are image-like inputs.
    2. Load success and failure trajectories into separate replay buffers.
    3. Sample equally from both buffers to keep the dataset balanced.
    4. Apply random crop augmentation to improve visual robustness.
    5. Optimize a classifier that outputs one logit per example.
    6. Save the final classifier state for future use.

    Args:
        observation_space: Observation structure expected by the classifier.
        action_space: Action structure needed by the replay buffer API.

    NOTE: this function is duplicated and used in both
    `async_bin_relocation_fwbw_drq` and `async_cable_route_drq` examples.
    """
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    # In this codebase, low-dimensional proprioceptive features usually contain
    # "state" in the key name, so the remaining keys are camera observations.
    image_keys = [k for k in observation_space.keys() if "state" not in k]

    # Positive buffer: successful demonstrations.
    pos_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=20000,
        image_keys=image_keys,
    )
    pos_buffer = populate_data_store(pos_buffer, FLAGS.positive_demo_paths)

    # Negative buffer: failed or non-successful demonstrations.
    neg_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=20000,
        image_keys=image_keys,
    )
    neg_buffer = populate_data_store(neg_buffer, FLAGS.negative_demo_paths)

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")

    # Each iterator returns mini-batches directly on the JAX device layout.
    # We use half of the final batch from positives and half from negatives so
    # that the binary labels remain balanced.
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    # The classifier is initialized from a real example batch so Flax can infer
    # the expected input shapes for every camera stream.
    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, sample["next_observations"], image_keys)

    def data_augmentation_fn(rng, observations):
        """Apply the same style of random crop augmentation to each image key."""
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    # Define the training step
    @jax.jit
    def train_step(state, batch, key):
        """Run one compiled gradient step and report loss/accuracy."""
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["data"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["data"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    # Training Loop
    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels.
        #
        # POTENTIAL BUG: Positive examples use `next_observations` while
        # negative examples use `observations`. If the intended supervision is
        # "classify final success-like states versus final failure-like states",
        # this mismatch may let the model learn temporal offset cues instead of
        # the reward concept itself.
        sample = concat_batches(
            pos_sample["next_observations"], neg_sample["observations"], axis=0
        )
        # Random crops are a standard vision trick: the label stays the same,
        # but the model sees slightly different versions of the same image.
        rng, key = jax.random.split(rng)
        sample = data_augmentation_fn(key, sample)
        labels = jnp.concatenate(
            [
                jnp.ones((FLAGS.batch_size // 2, 1)),
                jnp.zeros((FLAGS.batch_size // 2, 1)),
            ],
            axis=0,
        )
        batch = {"data": sample, "labels": labels}

        # One optimizer step over the balanced batch.
        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    # Save a plain Flax checkpoint so downstream scripts can load the model
    # without depending on Orbax checkpointing behavior.
    flax.config.update("flax_use_orbax_checkpointing", False)
    checkpoints.save_checkpoint(
        FLAGS.classifier_ckpt_path,
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )


if __name__ == "__main__":
    app.run(main)
