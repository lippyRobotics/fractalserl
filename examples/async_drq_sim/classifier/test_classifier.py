#!/usr/bin/env python3

import json
import os
import pickle as pkl

import numpy as np
from absl import app, flags

import franka_sim  # noqa: F401

from sim_classifier_utils import (
    classifier_probability,
    configure_jax_cpu_only,
    ensure_parent,
    infer_image_keys,
    load_classifier_func,
    make_serl_vision_env,
    pickcube_success,
    random_action,
    scripted_pickcube_action,
    validate_image_keys,
)


FLAGS = flags.FLAGS
flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Environment name.")
flags.DEFINE_string("classifier_ckpt_path", None, "Classifier checkpoint path.")
flags.DEFINE_integer("classifier_ckpt_step", None, "Checkpoint step to restore.")
flags.DEFINE_float("threshold", 0.5, "Success threshold.")
flags.DEFINE_multi_string("image_keys", None, "Image keys to use.")
flags.DEFINE_bool("use_proprio", False, "Whether classifier uses state.")
flags.DEFINE_integer("num_eval_steps", 200, "Number of live evaluation steps.")
flags.DEFINE_bool("render", False, "Render environment.")
flags.DEFINE_string("positive_demo_path", None, "Optional positive demo pickle.")
flags.DEFINE_string("negative_demo_path", None, "Optional negative demo pickle.")
flags.DEFINE_string("output_report", None, "Optional JSON report path.")
flags.DEFINE_integer("seed", 0, "Random seed.")


def _eval_obs(classifier_fn, obs, label, rows):
    logit, prob = classifier_probability(classifier_fn, obs)
    pred = prob >= FLAGS.threshold
    rows.append({"label": int(label), "pred": int(pred), "prob": prob, "logit": logit})


def _eval_demo(path, label, classifier_fn, rows):
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        transitions = pkl.load(f)
    for transition in transitions:
        obs = transition["next_observations"] if label == 1 else transition["observations"]
        _eval_obs(classifier_fn, obs, label, rows)


def _summary(rows):
    labels = np.asarray([r["label"] for r in rows], dtype=np.int32)
    preds = np.asarray([r["pred"] for r in rows], dtype=np.int32)
    probs = np.asarray([r["prob"] for r in rows], dtype=np.float32)
    pos = labels == 1
    neg = labels == 0
    tp = int(np.sum((preds == 1) & pos))
    tn = int(np.sum((preds == 0) & neg))
    fp = int(np.sum((preds == 1) & neg))
    fn = int(np.sum((preds == 0) & pos))
    return {
        "num_examples": int(len(rows)),
        "positive_accuracy": float(np.mean(preds[pos] == 1)) if np.any(pos) else None,
        "negative_accuracy": float(np.mean(preds[neg] == 0)) if np.any(neg) else None,
        "average_positive_probability": float(np.mean(probs[pos])) if np.any(pos) else None,
        "average_negative_probability": float(np.mean(probs[neg])) if np.any(neg) else None,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def main(_):
    configure_jax_cpu_only()
    if not FLAGS.classifier_ckpt_path:
        raise ValueError("--classifier_ckpt_path is required.")
    env = make_serl_vision_env(FLAGS.env, render=FLAGS.render)
    image_keys = list(FLAGS.image_keys or infer_image_keys(env.observation_space))
    validate_image_keys(env.observation_space, image_keys)
    classifier_fn = load_classifier_func(
        FLAGS.classifier_ckpt_path,
        env.observation_space.sample(),
        image_keys,
        checkpoint_step=FLAGS.classifier_ckpt_step,
        threshold=FLAGS.threshold,
        use_proprio=FLAGS.use_proprio,
    )

    rows = []
    rng = np.random.default_rng(FLAGS.seed)
    obs, _ = env.reset(seed=FLAGS.seed)
    _eval_obs(classifier_fn, obs, 0, rows)
    for step in range(FLAGS.num_eval_steps):
        action = scripted_pickcube_action(env, step, FLAGS.num_eval_steps)
        next_obs, _, done, truncated, _ = env.step(action)
        _eval_obs(classifier_fn, next_obs, int(pickcube_success(env)), rows)
        if FLAGS.render:
            print(f"scripted step={step} prob={rows[-1]['prob']:.3f} label={rows[-1]['label']}")
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset(seed=FLAGS.seed + step + 1)

    obs, _ = env.reset(seed=FLAGS.seed + 10000)
    for step in range(max(1, FLAGS.num_eval_steps // 2)):
        next_obs, _, done, truncated, _ = env.step(random_action(env, rng))
        _eval_obs(classifier_fn, next_obs, 0, rows)
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset(seed=FLAGS.seed + 20000 + step)

    _eval_demo(FLAGS.positive_demo_path, 1, classifier_fn, rows)
    _eval_demo(FLAGS.negative_demo_path, 0, classifier_fn, rows)

    report = _summary(rows)
    print(json.dumps(report, indent=2))
    if FLAGS.output_report:
        ensure_parent(FLAGS.output_report)
        with open(FLAGS.output_report, "w", encoding="utf-8") as f:
            json.dump({"summary": report, "examples": rows}, f, indent=2)
        print(f"wrote report to {FLAGS.output_report}")


if __name__ == "__main__":
    app.run(main)
