#!/usr/bin/env python3

import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import cv2
import os
import sys

from typing import Any, Dict, Optional
import pickle as pkl
import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

import franka_sim

_CLASSIFIER_DIR = os.path.join(os.path.dirname(__file__), "classifier")
if _CLASSIFIER_DIR not in sys.path:
    sys.path.append(_CLASSIFIER_DIR)
from sim_classifier_utils import (
    ClassifierRewardWrapper,
    configure_jax_cpu_only,
    infer_image_keys,
    load_classifier_func,
    validate_image_keys,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_string("run_name", None, "Name of run for wandb logging")
flags.DEFINE_integer("max_traj_length", 1000, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

# flags for replay buffer
flags.DEFINE_string("replay_buffer_type", "memory_efficient_replay_buffer", "Which replay buffer to use")
flags.DEFINE_string("branch_method", None, "Method for how many branches to generate")
flags.DEFINE_string("split_method", None, "Method for when to change number of branches generated")
flags.DEFINE_float("workspace_width", 0.5, "Workspace width in meters")
flags.DEFINE_integer("max_depth",None,"Maximum layers of depth")
flags.DEFINE_integer("starting_branch_count", None, "Initial number of branches")
flags.DEFINE_integer("branching_factor", None, "Rate of change of branches per dimension (x,y)") # For fractal_branch and fractal_contraction
flags.DEFINE_float("alpha",None,"alpha value")
flags.DEFINE_enum("disassociated_type", None, ["octahedron", "hourglass"], 
                  "Type of disassociated fracal rollout. Octahedron: expand from min to max then contract to min,"
                   + " Hourglass: Contract from max to min then expand to max")
flags.DEFINE_integer("min_branch_count", None, "Minimum number of branches for disassociated fractal rollout")
flags.DEFINE_integer("max_branch_count", None, "Maximum number of branches for disassociated fractal rollout")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")
flags.DEFINE_bool("dry_run", False, "Build env/agent/replay setup and exit.")

flags.DEFINE_string("reward_classifier_ckpt_path", None, "Path to reward classifier checkpoint.")
flags.DEFINE_integer("reward_classifier_ckpt_step", None, "Checkpoint step to restore; if None, use latest.")
flags.DEFINE_float("reward_classifier_threshold", 0.5, "Sigmoid threshold for success.")
flags.DEFINE_bool("terminate_on_classifier_success", True, "Terminate episode when classifier says success.")
flags.DEFINE_bool("zero_env_reward", False, "If true, ignore base environment reward and use only classifier reward.")
flags.DEFINE_bool("use_classifier_reward", False, "Whether to use classifier-based reward.")
flags.DEFINE_multi_string("classifier_image_keys", None, "Image keys used by classifier. If None, infer all non-state keys.")
flags.DEFINE_bool("classifier_use_proprio", False, "Whether classifier uses proprioceptive state.")

devices = None
num_devices = None
sharding = None


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def concat_batches(offline_batch, online_batch, axis=0):
    return jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=axis), offline_batch, online_batch
    )


##############################################################################


VISION_ENVS = {
    "PandaPickCubeVision-v0",
    "PandaReachCubeVision-v0",
    "PandaPickSparseCubeVision-v0",
    "PandaReachSparseCubeVision-v0",
}


def wrap_sim_env(env):
    if isinstance(env.observation_space, gym.spaces.Dict) and "images" in env.observation_space.spaces:
        env = SERLObsWrapper(env)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    elif FLAGS.env == "PandaReachSparseCube-v0":
        env = SERLObsWrapper(env)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    else:
        env = gym.wrappers.FlattenObservation(env)
    return env


def maybe_wrap_classifier_reward(env):
    if not FLAGS.use_classifier_reward:
        return env, None
    if not FLAGS.reward_classifier_ckpt_path:
        raise ValueError(
            "--reward_classifier_ckpt_path is required when --use_classifier_reward=True"
        )
    image_keys = FLAGS.classifier_image_keys or infer_image_keys(env.observation_space)
    validate_image_keys(env.observation_space, image_keys)
    classifier_fn = load_classifier_func(
        ckpt_path=FLAGS.reward_classifier_ckpt_path,
        sample_obs=env.observation_space.sample(),
        image_keys=image_keys,
        checkpoint_step=FLAGS.reward_classifier_ckpt_step,
        threshold=FLAGS.reward_classifier_threshold,
        use_proprio=FLAGS.classifier_use_proprio,
    )
    env = ClassifierRewardWrapper(
        env,
        classifier_fn=classifier_fn,
        threshold=FLAGS.reward_classifier_threshold,
        terminate_on_success=FLAGS.terminate_on_classifier_success,
        zero_env_reward=FLAGS.zero_env_reward,
    )
    print_green(
        "classifier reward enabled with "
        f"image_keys={list(image_keys)}, use_proprio={FLAGS.classifier_use_proprio}"
    )
    return env, classifier_fn


def actor(agent: Any, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    from agentlace.trainer import TrainerClient
    from serl_launcher.utils.launcher import make_trainer_config

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    eval_env = gym.make(FLAGS.env, disable_env_checker=True)
    eval_env = wrap_sim_env(eval_env)
    if FLAGS.use_classifier_reward:
        eval_env, _ = maybe_wrap_classifier_reward(eval_env)
    eval_env = RecordEpisodeStatistics(eval_env)

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done or truncated,
            )
            data_store.insert(transition)

            obs = next_obs
            if done or truncated:
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0:
            from serl_launcher.common.evaluation import evaluate

            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(
    rng,
    agent: Any,
    replay_buffer: MemoryEfficientReplayBufferDataStore,
    demo_buffer: Optional[MemoryEfficientReplayBufferDataStore] = None,
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    from serl_launcher.utils.launcher import make_trainer_config, make_wandb_logger

    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project=FLAGS.exp_name,
        name=FLAGS.run_name,
        description=FLAGS.exp_name or FLAGS.env,
        # wandb_output_dir=FLAGS.wandb_output_dir,
        debug=FLAGS.debug,
        # offline=FLAGS.wandb_offline,
    )

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    from agentlace.trainer import TrainerServer

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience if
    # demo_buffer is provided
    if demo_buffer is None:
        single_buffer_batch_size = FLAGS.batch_size
        demo_iterator = None
    else:
        if FLAGS.batch_size % 2 != 0:
            raise ValueError(
                "--batch_size must be even when demo data is provided so RLPD "
                "can sample 50/50 online and demonstration batches."
            )
        single_buffer_batch_size = FLAGS.batch_size // 2
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": single_buffer_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )

    # create replay buffer iterator
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

                # we will concatenate the demo data with the online data
                # if demo_buffer is provided
                if demo_iterator is not None:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(
                    batch,
                )

        with timer.context("train"):
            batch = next(replay_iterator)

            # we will concatenate the demo data with the online data
            # if demo_buffer is provided
            if demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        # publish the updated network
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoint_path = os.path.abspath(os.path.expanduser(FLAGS.checkpoint_path))
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoints.save_checkpoint(
                checkpoint_path, agent.state, step=update_steps, keep=20
            )

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


##############################################################################


def main(_):
    global devices, num_devices, sharding
    configure_jax_cpu_only()

    # create env and load dataset
    if FLAGS.render:
        env = gym.make(FLAGS.env, render_mode="human", disable_env_checker=True)
    else:
        env = gym.make(FLAGS.env, disable_env_checker=True)

    if FLAGS.env in {"PandaPickCube-v0", "PandaReachCube-v0", "PandaPickSparseCube-v0", "PandaReachSparseCube-v0", "PandaPickCubeVision-v0", "PandaReachCubeVision-v0", "PandaPickSparseCubeVision-v0", "PandaReachSparseCubeVision-v0"}:
        x_obs_idx=np.array([0,4])
        y_obs_idx=np.array([1,5])
    else:
        raise NotImplementedError(f"Unknown observation layout for {FLAGS.env}")
    
    env = wrap_sim_env(env)
    env, _ = maybe_wrap_classifier_reward(env)

    image_keys = infer_image_keys(env.observation_space)

    if FLAGS.dry_run:
        print_green("dry run complete")
        print(env.observation_space)
        print(env.action_space)
        print(f"image_keys={image_keys}")
        return

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % num_devices == 0

    from serl_launcher.utils.launcher import make_drq_agent

    agent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.learner:
        from serl_launcher.utils.launcher import make_replay_buffer

        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path,
            type=FLAGS.replay_buffer_type,
            branch_method=FLAGS.branch_method,
            split_method=FLAGS.split_method,
            branching_factor=FLAGS.branching_factor,
            starting_branch_count=FLAGS.starting_branch_count,
            workspace_width=FLAGS.workspace_width,
            max_traj_length=FLAGS.max_traj_length,
            x_obs_idx=x_obs_idx,
            y_obs_idx=y_obs_idx,
            preload_rlds_path=FLAGS.preload_rlds_path,
            max_depth=FLAGS.max_depth,
            alpha=FLAGS.alpha,
            disassociated_type=FLAGS.disassociated_type,
            min_branch_count=FLAGS.min_branch_count,
            max_branch_count=FLAGS.max_branch_count,
            image_keys=image_keys,
        )

        print_green("replay buffer created")
        print_green(f"replay_buffer size: {len(replay_buffer)}")

        # if demo data is provided, load it into the demo buffer
        # in the learner node, we support 2 ways to load demo data:
        # 1. load from pickle file; 2. load from tf rlds data
        if FLAGS.demo_path or FLAGS.preload_rlds_path:

            def preload_data_transform(data, metadata) -> Optional[Dict[str, Any]]:
                # NOTE: Create your own custom data transform function here if you
                # are loading this via with --preload_rlds_path with tf rlds data
                # This default does nothing
                # See:  https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_tutorial.ipynb#scrollTo=X1KXM8IGecRO
                #       https://www.tensorflow.org/guide/data 
                #       https://github.com/google-research/rlds/blob/main/docs/transformations.md
                #       Batch: rlds.transformations.batch (https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_tutorial.ipynb#scrollTo=TGT3YfzFOrBm)
                #       Reverb: rlds.transformations.pattern_map (https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_dataset_patterns.ipynb )
                #       Nested data set manipulation: rlds.transformations.episode_length/.sum_dataset/.final_step/.map_nested_steps
                #       Concatenation: rlds.transformations.concatenate / .concat_if_terminal (https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_examples.ipynb#scrollTo=pWNhxwJzOUJv)
                #       Stats: rlds.transformations.mean_and_std (https://colab.research.google.com/github/google-research/rlds/blob/main/rlds/examples/rlds_tutorial.ipynb#scrollTo=Z0TITfo_4oZr)
                #       Truncation: rlds.transformations.truncate_after_condition 
                #       Alignment: rlds.transformations.shift_keys
                #       Zero Init: rlds.transformations.zeros_from_spec
                return data

            demo_buffer = make_replay_buffer(
                env,
                capacity=FLAGS.replay_buffer_capacity,
                rlds_logger_path=FLAGS.log_rlds_path,
                type=FLAGS.replay_buffer_type,
                branch_method=FLAGS.branch_method,
                split_method=FLAGS.split_method,
                branching_factor=FLAGS.branching_factor,
                starting_branch_count=FLAGS.starting_branch_count,
                workspace_width=FLAGS.workspace_width,
                max_traj_length=FLAGS.max_traj_length,
                x_obs_idx=x_obs_idx,
                y_obs_idx=y_obs_idx,
                preload_rlds_path=FLAGS.preload_rlds_path,
                max_depth=FLAGS.max_depth,
                alpha=FLAGS.alpha,
                disassociated_type=FLAGS.disassociated_type,
                min_branch_count=FLAGS.min_branch_count,
                max_branch_count=FLAGS.max_branch_count,
                image_keys=image_keys,
                preload_data_transform=preload_data_transform,
            )

            if FLAGS.demo_path:
                # Check if the file exists
                if not os.path.exists(FLAGS.demo_path):
                    raise FileNotFoundError(f"File {FLAGS.demo_path} not found")

                with open(FLAGS.demo_path, "rb") as f:
                    trajs = pkl.load(f)
                    for traj in trajs:
                        demo_buffer.insert(traj)

            print(f"demo buffer size: {len(demo_buffer)}")
        else:
            demo_buffer = None

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,  # None if no demo data is provided
        )

    elif FLAGS.actor:
        from agentlace.data.data_store import QueuedDataStore

        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
