#!/usr/bin/env python3

import os
import pickle as pkl
import sys

import numpy as np
from absl import app, flags

import franka_sim  # noqa: F401

_CLASSIFIER_DIR = os.path.join(os.path.dirname(__file__), "classifier")
if _CLASSIFIER_DIR not in sys.path:
    sys.path.append(_CLASSIFIER_DIR)

from sim_classifier_utils import (  # noqa: E402
    classifier_probability,
    configure_jax_cpu_only,
    ensure_parent,
    infer_image_keys,
    load_classifier_func,
    make_serl_vision_env,
    pickcube_success,
    random_action,
    scripted_pickcube_action,
)


FLAGS = flags.FLAGS
flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Environment name.")
flags.DEFINE_string(
    "output_path", "examples/async_drq_sim/demos/sim_pickcube_image_demos.pkl", "Output pickle path."
)
flags.DEFINE_integer("num_trajectories", 20, "Number of trajectories to record.")
flags.DEFINE_integer("max_traj_length", 200, "Maximum trajectory length.")
flags.DEFINE_bool("render", False, "Render the environment.")
flags.DEFINE_enum("policy", "scripted", ["scripted", "random", "manual"], "Demo policy.")
flags.DEFINE_bool("success_only", False, "Only save trajectories that reach success.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_enum(
    "demo_reward_mode",
    "dense",
    ["dense", "classifier_sparse"],
    "Reward saved into the demo file. 'dense' stores the environment reward. "
    "'classifier_sparse' stores binary classifier success rewards.",
)
flags.DEFINE_string("reward_classifier_ckpt_path", None, "Optional success classifier checkpoint.")
flags.DEFINE_integer("reward_classifier_ckpt_step", None, "Checkpoint step to restore.")
flags.DEFINE_float("reward_classifier_threshold", 0.5, "Classifier success threshold.")
flags.DEFINE_multi_string("classifier_image_keys", None, "Classifier image keys.")
flags.DEFINE_bool("classifier_use_proprio", False, "Whether classifier uses state.")
flags.DEFINE_bool(
    "terminate_on_classifier_success",
    True,
    "When saving classifier_sparse rewards, mark classifier-success transitions done.",
)


def _action(env, rng, step):
    if FLAGS.policy == "manual":
        print("Manual control is not available for franka_sim; using random actions.")
        return random_action(env, rng)
    if FLAGS.policy == "scripted":
        return scripted_pickcube_action(env, step, FLAGS.max_traj_length)
    return random_action(env, rng)


def main(_):
    configure_jax_cpu_only()
    rng = np.random.default_rng(FLAGS.seed)
    env = make_serl_vision_env(FLAGS.env, render=FLAGS.render)
    classifier_fn = None
    if FLAGS.demo_reward_mode == "classifier_sparse" and not FLAGS.reward_classifier_ckpt_path:
        raise ValueError(
            "--reward_classifier_ckpt_path is required when "
            "--demo_reward_mode=classifier_sparse."
        )
    if FLAGS.reward_classifier_ckpt_path:
        image_keys = FLAGS.classifier_image_keys or infer_image_keys(env.observation_space)
        classifier_fn = load_classifier_func(
            FLAGS.reward_classifier_ckpt_path,
            env.observation_space.sample(),
            image_keys,
            checkpoint_step=FLAGS.reward_classifier_ckpt_step,
            threshold=FLAGS.reward_classifier_threshold,
            use_proprio=FLAGS.classifier_use_proprio,
        )

    transitions = []
    saved_trajs = 0
    attempts = 0
    max_attempts = max(FLAGS.num_trajectories * 20, FLAGS.num_trajectories)
    while saved_trajs < FLAGS.num_trajectories and attempts < max_attempts:
        attempts += 1
        obs, _ = env.reset(seed=FLAGS.seed + attempts)
        traj = []
        reached_success = False

        for step in range(FLAGS.max_traj_length):
            action = _action(env, rng, step)
            next_obs, reward, done, truncated, info = env.step(action)

            classifier_success = False
            if classifier_fn is not None:
                _, prob = classifier_probability(classifier_fn, next_obs)
                classifier_success = prob >= FLAGS.reward_classifier_threshold
                info = dict(info)
                info["classifier_prob"] = prob
                info["classifier_success"] = bool(classifier_success)

            env_success = bool(info.get("success", False)) if isinstance(info, dict) else False
            reached_success = (
                reached_success
                or env_success
                or classifier_success
                or pickcube_success(env)
            )

            terminal = bool(done or truncated)
            transition_reward = reward
            transition_done = terminal
            if FLAGS.demo_reward_mode == "classifier_sparse":
                transition_reward = 1.0 if classifier_success else 0.0
                transition_done = bool(
                    terminal
                    or (
                        classifier_success
                        and FLAGS.terminate_on_classifier_success
                    )
                )
            traj.append(
                {
                    "observations": obs,
                    "actions": action,
                    "next_observations": next_obs,
                    "rewards": np.asarray(transition_reward, dtype=np.float32),
                    "masks": np.asarray(1.0 - float(transition_done), dtype=np.float32),
                    "dones": transition_done,
                }
            )
            obs = next_obs
            if terminal or reached_success:
                break

        if (not FLAGS.success_only) or reached_success:
            transitions.extend(traj)
            saved_trajs += 1
            print(f"saved trajectory {saved_trajs}/{FLAGS.num_trajectories} length={len(traj)}")
        else:
            print(f"discarded unsuccessful trajectory length={len(traj)}")

    if saved_trajs < FLAGS.num_trajectories:
        raise RuntimeError(
            f"Only saved {saved_trajs}/{FLAGS.num_trajectories} trajectories. "
            "Try --success_only=False, --policy scripted, or a longer max_traj_length."
        )

    ensure_parent(FLAGS.output_path)
    with open(FLAGS.output_path, "wb") as f:
        pkl.dump(transitions, f)
    print(f"wrote {len(transitions)} transitions to {FLAGS.output_path}")


if __name__ == "__main__":
    app.run(main)
