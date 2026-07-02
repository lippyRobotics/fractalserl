#!/usr/bin/env python3

import math
import os
import pickle as pkl

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".5")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from flax.training import checkpoints
from tqdm import tqdm

import franka_sim  # noqa: F401

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.vision.data_augmentations import batched_random_crop

from sim_classifier_utils import (
    configure_jax_cpu_only,
    infer_image_keys,
    make_serl_vision_env,
    validate_image_keys,
)


FLAGS = flags.FLAGS
flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Environment name.")
flags.DEFINE_multi_string("positive_demo_paths", None, "Positive classifier demo files.")
flags.DEFINE_multi_string("negative_demo_paths", None, "Negative classifier demo files.")
flags.DEFINE_string("classifier_ckpt_path", "classifier_ckpts", "Output checkpoint path.")
flags.DEFINE_integer("batch_size", 64, "Balanced batch size.")
flags.DEFINE_integer("num_epochs", 100, "Training epochs.")
flags.DEFINE_integer("steps_per_epoch", 1, "Optimizer steps per epoch.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_multi_string("image_keys", None, "Image keys to use. If None, infer all image keys.")
flags.DEFINE_bool("use_proprio", False, "Whether classifier uses proprioceptive state.")
flags.DEFINE_integer("eval_every", 10, "Evaluate every N epochs.")
flags.DEFINE_float("train_split", 0.9, "Fraction of trajectories used for training.")


def concat_batches(left, right, axis=0):
    return jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=axis), left, right)


def _check_paths(paths, name):
    if not paths:
        raise ValueError(f"--{name} must include at least one pickle file.")
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing {name}: {missing}")


def _load_trajectories(paths):
    trajectories = []
    for path in paths:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
        current = []
        for transition in transitions:
            current.append(transition)
            if bool(transition["dones"]):
                trajectories.append(current)
                current = []
        if current:
            trajectories.append(current)
        print(f"Loaded {len(transitions)} transitions from {path}.")
    return trajectories


def _split_trajectories(trajectories, rng):
    if len(trajectories) < 2:
        raise ValueError(
            "At least two recorded trajectories per class are required for a "
            "held-out split. Regenerate classifier demonstrations with the updated recorder."
        )
    order = rng.permutation(len(trajectories))
    train_count = int(round(len(trajectories) * FLAGS.train_split))
    train_count = min(max(train_count, 1), len(trajectories) - 1)
    train = [trajectories[index] for index in order[:train_count]]
    evaluation = [trajectories[index] for index in order[train_count:]]
    return train, evaluation


def _flatten(trajectories):
    return [transition for trajectory in trajectories for transition in trajectory]


def _make_buffer(observation_space, action_space, image_keys, transitions):
    buffer = MemoryEfficientReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=max(100, len(transitions) * 2),
        image_keys=image_keys,
    )
    for transition in transitions:
        buffer.insert(transition)
    return buffer


def _validate_visual_data(pos_transitions, neg_transitions, image_keys):
    sample_count = min(256, len(pos_transitions), len(neg_transitions))
    if sample_count == 0:
        raise ValueError("Classifier datasets must contain both positive and negative samples.")

    pos_indices = np.linspace(0, len(pos_transitions) - 1, sample_count, dtype=np.int32)
    neg_indices = np.linspace(0, len(neg_transitions) - 1, sample_count, dtype=np.int32)
    pos_images = {}
    neg_images = {}
    for key in image_keys:
        pos_images[key] = np.stack(
            [pos_transitions[index]["next_observations"][key] for index in pos_indices]
        )
        neg_images[key] = np.stack(
            [neg_transitions[index]["observations"][key] for index in neg_indices]
        )
        unique_pos = len({image.tobytes() for image in pos_images[key]})
        unique_neg = len({image.tobytes() for image in neg_images[key]})
        class_difference = float(
            np.abs(pos_images[key].astype(np.float32).mean(0) - neg_images[key].astype(np.float32).mean(0)).mean()
        )
        print(
            f"{key}: unique positive={unique_pos}/{sample_count}, "
            f"unique negative={unique_neg}/{sample_count}, "
            f"mean class difference={class_difference:.3f}/255"
        )

    for index, left in enumerate(image_keys):
        for right in image_keys[index + 1 :]:
            identical_fraction = float(
                np.mean(
                    [
                        np.array_equal(pos_images[left][i], pos_images[right][i])
                        and np.array_equal(neg_images[left][i], neg_images[right][i])
                        for i in range(sample_count)
                    ]
                )
            )
            if identical_fraction > 0.95:
                raise ValueError(
                    f"Image keys {left!r} and {right!r} are identical in "
                    f"{identical_fraction:.1%} of checked samples. Delete the old "
                    "datasets and record them again with the corrected camera renderer."
                )


def train_reward_classifier(observation_space, action_space):
    if FLAGS.batch_size % 2 != 0:
        raise ValueError("--batch_size must be even for balanced positive/negative training.")
    if FLAGS.steps_per_epoch <= 0:
        raise ValueError("--steps_per_epoch must be positive.")
    if not 0.0 < FLAGS.train_split < 1.0:
        raise ValueError("--train_split must be strictly between 0 and 1.")
    _check_paths(FLAGS.positive_demo_paths, "positive_demo_paths")
    _check_paths(FLAGS.negative_demo_paths, "negative_demo_paths")

    image_keys = list(FLAGS.image_keys or infer_image_keys(observation_space))
    validate_image_keys(observation_space, image_keys)
    print(f"image keys: {image_keys}")
    print(f"use proprio: {FLAGS.use_proprio}")

    pos_trajectories = _load_trajectories(FLAGS.positive_demo_paths)
    neg_trajectories = _load_trajectories(FLAGS.negative_demo_paths)
    rng_np = np.random.default_rng(FLAGS.seed)
    pos_train_trajs, pos_eval_trajs = _split_trajectories(pos_trajectories, rng_np)
    neg_train_trajs, neg_eval_trajs = _split_trajectories(neg_trajectories, rng_np)

    pos_train = _flatten(pos_train_trajs)
    neg_train = _flatten(neg_train_trajs)
    pos_eval = _flatten(pos_eval_trajs)
    neg_eval = _flatten(neg_eval_trajs)
    _validate_visual_data(pos_train, neg_train, image_keys)

    pos_buffer = _make_buffer(observation_space, action_space, image_keys, pos_train)
    neg_buffer = _make_buffer(observation_space, action_space, image_keys, neg_train)
    pos_eval_buffer = _make_buffer(observation_space, action_space, image_keys, pos_eval)
    neg_eval_buffer = _make_buffer(observation_space, action_space, image_keys, neg_eval)

    print(
        f"trajectory split: positive={len(pos_train_trajs)} train/{len(pos_eval_trajs)} eval, "
        f"negative={len(neg_train_trajs)} train/{len(neg_eval_trajs)} eval"
    )
    print(
        f"sample split: positive={len(pos_train)} train/{len(pos_eval)} eval, "
        f"negative={len(neg_train)} train/{len(neg_eval)} eval"
    )

    half_batch = FLAGS.batch_size // 2
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, init_key = jax.random.split(rng)
    pos_sample = pos_buffer.sample(half_batch)
    neg_sample = neg_buffer.sample(half_batch)
    sample = concat_batches(pos_sample, neg_sample, axis=0)
    classifier = create_classifier(
        init_key,
        sample["next_observations"],
        image_keys,
        use_proprio=FLAGS.use_proprio,
    )

    def data_augmentation_fn(key, observations):
        crop_keys = jax.random.split(key, len(image_keys))
        for pixel_key, crop_key in zip(image_keys, crop_keys):
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], crop_key, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    def make_batch(pos_source, neg_source, augment_key=None):
        pos_sample = pos_source.sample(half_batch)
        neg_sample = neg_source.sample(half_batch)
        observations = concat_batches(
            pos_sample["next_observations"], neg_sample["observations"], axis=0
        )
        if augment_key is not None:
            observations = data_augmentation_fn(augment_key, observations)
        labels = jnp.concatenate(
            [jnp.ones((half_batch, 1)), jnp.zeros((half_batch, 1))], axis=0
        )
        return {"data": observations, "labels": labels}

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["data"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        logits = state.apply_fn({"params": state.params}, batch["data"], train=False)
        accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])
        return state, loss, accuracy

    @jax.jit
    def eval_step(state, batch):
        logits = state.apply_fn({"params": state.params}, batch["data"], train=False)
        probabilities = nn.sigmoid(logits)
        predictions = probabilities >= 0.5
        loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()
        return {
            "loss": loss,
            "accuracy": jnp.mean(predictions == batch["labels"]),
            "positive_recall": jnp.mean(predictions[:half_batch]),
            "negative_accuracy": jnp.mean(~predictions[half_batch:]),
            "positive_probability": jnp.mean(probabilities[:half_batch]),
            "negative_probability": jnp.mean(probabilities[half_batch:]),
        }

    eval_steps = max(1, math.ceil(max(len(pos_eval), len(neg_eval)) / half_batch))
    for epoch in tqdm(range(FLAGS.num_epochs)):
        train_losses = []
        train_accuracies = []
        for _ in range(FLAGS.steps_per_epoch):
            rng, aug_key, train_key = jax.random.split(rng, 3)
            batch = make_batch(pos_buffer, neg_buffer, augment_key=aug_key)
            classifier, train_loss, train_accuracy = train_step(
                classifier, batch, train_key
            )
            train_losses.append(float(train_loss))
            train_accuracies.append(float(train_accuracy))

        should_eval = (
            epoch == 0
            or (epoch + 1) % FLAGS.eval_every == 0
            or epoch == FLAGS.num_epochs - 1
        )
        if should_eval:
            eval_metrics = []
            for _ in range(eval_steps):
                eval_metrics.append(
                    eval_step(classifier, make_batch(pos_eval_buffer, neg_eval_buffer))
                )
            averaged = {
                key: float(np.mean([float(metrics[key]) for metrics in eval_metrics]))
                for key in eval_metrics[0]
            }
            print(
                f"Epoch {epoch + 1}: train_loss={np.mean(train_losses):.4f}, "
                f"train_accuracy={np.mean(train_accuracies):.4f}, "
                f"eval_loss={averaged['loss']:.4f}, "
                f"eval_accuracy={averaged['accuracy']:.4f}, "
                f"positive_recall={averaged['positive_recall']:.4f}, "
                f"negative_accuracy={averaged['negative_accuracy']:.4f}, "
                f"positive_prob={averaged['positive_probability']:.4f}, "
                f"negative_prob={averaged['negative_probability']:.4f}"
            )

    flax.config.update("flax_use_orbax_checkpointing", False)
    checkpoints.save_checkpoint(
        FLAGS.classifier_ckpt_path,
        classifier,
        step=FLAGS.num_epochs * FLAGS.steps_per_epoch,
        overwrite=True,
    )
    print(f"saved classifier checkpoint to {FLAGS.classifier_ckpt_path}")


def main(_):
    configure_jax_cpu_only()
    env = make_serl_vision_env(FLAGS.env)
    train_reward_classifier(env.observation_space, env.action_space)


if __name__ == "__main__":
    app.run(main)
