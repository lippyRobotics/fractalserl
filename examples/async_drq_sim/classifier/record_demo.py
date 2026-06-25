#!/usr/bin/env python3

import pickle as pkl

import numpy as np
from absl import app, flags

import franka_sim  # noqa: F401

from sim_classifier_utils import (
    ensure_parent,
    force_pickcube_success_state,
    make_serl_vision_env,
    pickcube_success,
    random_action,
    scripted_pickcube_action,
)


FLAGS = flags.FLAGS
flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Environment name.")
flags.DEFINE_string("output_path", "examples/async_drq_sim/classifier/demos/samples.pkl", "Output pickle path.")
flags.DEFINE_enum("label", "positive", ["positive", "negative", "mixed"], "Examples to record.")
flags.DEFINE_integer("num_samples", 1000, "Number of transitions to save.")
flags.DEFINE_integer("num_trajectories", 200, "Maximum trajectories to try.")
flags.DEFINE_integer("max_traj_length", 200, "Maximum trajectory length.")
flags.DEFINE_bool("render", False, "Render the environment.")
flags.DEFINE_enum("policy", "scripted", ["scripted", "random", "manual"], "Data policy.")
flags.DEFINE_integer("save_every_n", 50, "Write a partial dataset every N saved samples.")
flags.DEFINE_integer("sample_every_n", 2, "Keep at most one candidate every N environment steps.")
flags.DEFINE_integer(
    "max_samples_per_trajectory",
    10,
    "Maximum saved samples per trajectory; evenly subsample candidates when needed.",
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_bool(
    "force_positive_state_fallback",
    False,
    "For positive scripted data, force a lifted cube state if rollout positives are scarce.",
)


def _action(env, rng, step):
    if FLAGS.policy == "manual":
        print("Manual control is not available for franka_sim; using random actions.")
        return random_action(env, rng)
    if FLAGS.policy == "scripted":
        return scripted_pickcube_action(env, step, FLAGS.max_traj_length)
    return random_action(env, rng)


def _save(samples):
    ensure_parent(FLAGS.output_path)
    with open(FLAGS.output_path, "wb") as f:
        pkl.dump(samples, f)


def main(_):
    if FLAGS.sample_every_n <= 0:
        raise ValueError("--sample_every_n must be positive.")
    if FLAGS.max_samples_per_trajectory <= 0:
        raise ValueError("--max_samples_per_trajectory must be positive.")

    rng = np.random.default_rng(FLAGS.seed)
    env = make_serl_vision_env(FLAGS.env, render=FLAGS.render)
    samples = []
    last_saved_count = 0

    for traj_idx in range(FLAGS.num_trajectories):
        if len(samples) >= FLAGS.num_samples:
            break
        obs, _ = env.reset(seed=FLAGS.seed + traj_idx)
        trajectory_samples = []
        if (
            FLAGS.label == "positive"
            and FLAGS.policy == "scripted"
            and FLAGS.force_positive_state_fallback
            and traj_idx > max(2, FLAGS.num_trajectories // 4)
        ):
            force_pickcube_success_state(env)

        for step in range(FLAGS.max_traj_length):
            action = _action(env, rng, step)
            next_obs, reward, done, truncated, info = env.step(action)
            success = bool(info.get("success", False)) or pickcube_success(env)

            wants_positive = FLAGS.label in {"positive", "mixed"} and success
            wants_negative = FLAGS.label in {"negative", "mixed"} and not success
            if (wants_positive or wants_negative) and step % FLAGS.sample_every_n == 0:
                trajectory_samples.append(
                    {
                        "observations": obs,
                        "actions": action,
                        "next_observations": next_obs,
                        "rewards": np.asarray(float(success), dtype=np.float32),
                        "masks": np.asarray(1.0 - float(done), dtype=np.float32),
                        "dones": bool(done or truncated),
                    }
                )

            obs = next_obs
            if done or truncated:
                break

        if len(trajectory_samples) > FLAGS.max_samples_per_trajectory:
            indices = np.linspace(
                0,
                len(trajectory_samples) - 1,
                FLAGS.max_samples_per_trajectory,
                dtype=np.int32,
            )
            trajectory_samples = [trajectory_samples[index] for index in indices]

        remaining = FLAGS.num_samples - len(samples)
        trajectory_samples = trajectory_samples[:remaining]
        if trajectory_samples:
            # Saved classifier samples are sparse, so explicitly delimit each
            # trajectory for memory-efficient replay reconstruction.
            trajectory_samples[-1]["dones"] = True
            trajectory_samples[-1]["masks"] = np.asarray(0.0, dtype=np.float32)
            samples.extend(trajectory_samples)

        if (
            FLAGS.save_every_n > 0
            and len(samples) // FLAGS.save_every_n > last_saved_count // FLAGS.save_every_n
        ):
            _save(samples)
            last_saved_count = len(samples)
            print(f"saved {len(samples)} samples to {FLAGS.output_path}")

    if not samples:
        raise RuntimeError(
            "No classifier samples were collected. For positives, try "
            "--policy scripted and increase --max_traj_length."
        )
    if len(samples) < FLAGS.num_samples:
        print(f"Warning: collected {len(samples)}/{FLAGS.num_samples} requested samples.")
    _save(samples)
    print(f"wrote {len(samples)} samples to {FLAGS.output_path}")


if __name__ == "__main__":
    app.run(main)
