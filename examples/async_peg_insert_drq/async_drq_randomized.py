#!/usr/bin/env python3

import copy
import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer
)
# from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
)

import franka_env

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "FrankaEnv-Vision-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")

flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_float(
    "updates_per_min_period_sec",
    30.0,
    "Wall-clock period (seconds) to report learner updates/min.",
)

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

# replay buffer flags
flags.DEFINE_string("replay_buffer_type", "memory_efficient_replay_buffer", "Which replay buffer to use")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")
flags.DEFINE_integer("branching_factor", None, "Factor by which branch count is changed")
flags.DEFINE_integer("max_depth", None, "Maximum number of splits that may occur in one episode")
flags.DEFINE_string("branch_method", "constant", "Method for how many branches to generate")
flags.DEFINE_string("split_method", "never", "Method for when to change number of branches")
flags.DEFINE_float("alpha", 0.2, "Rate of change of max_traj_length")
flags.DEFINE_float("workspace_width", 0.3, "Workspace width in meters")
flags.DEFINE_integer("starting_branch_count", 27, "Initial number of branches")

flags.DEFINE_integer(
    "eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step"
)
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def flip_transition_horizontally(
    transition,
    image_keys,
    y_obs_idx,
    invert_state_indices=None,
    invert_action_indices=None,
):
    flipped_transition = copy.deepcopy(transition)
    
    # Flip images horizontally (assumes width is second-to-last dimension)
    for key in image_keys:
        if key in flipped_transition["observations"]:
            flipped_transition["observations"][key] = np.flip(
                flipped_transition["observations"][key], axis=-2
            )
        if key in flipped_transition["next_observations"]:
            flipped_transition["next_observations"][key] = np.flip(
                flipped_transition["next_observations"][key], axis=-2
            )
    
    # Invert specified state indices (negate values)
    # Use ellipsis to handle any leading dimensions (batch, time, etc.)
    if invert_state_indices is not None:
        flipped_transition["observations"]["state"][..., invert_state_indices] = -flipped_transition["observations"]["state"][..., invert_state_indices]
        flipped_transition["next_observations"]["state"][..., invert_state_indices] = -flipped_transition["next_observations"]["state"][..., invert_state_indices]
    
    # Also invert y-position (backward compatibility)
    flipped_transition["observations"]["state"][..., y_obs_idx] = -flipped_transition["observations"]["state"][..., y_obs_idx]
    flipped_transition["next_observations"]["state"][..., y_obs_idx] = -flipped_transition["next_observations"]["state"][..., y_obs_idx]

    if invert_action_indices is not None:
        flipped_transition["actions"][..., invert_action_indices] = -flipped_transition["actions"][..., invert_action_indices]
    
    return flipped_transition


##############################################################################


def actor(agent: DrQAgent, data_store, env, sampling_rng, image_keys, y_obs_idx):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)
        env.reset(joint_reset=True)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1} ({success_counter / (episode + 1):.2f})")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit

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

    obs, _ = env.reset(joint_reset=True)
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

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")

            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            # Insert original transition
            data_store.insert(transition)
            
            # Insert horizontally flipped transition for augmentation
            # Invert mirrored state components and mirrored action components.
            invert_indices = np.array([2, 7, 9, 10, 12, 14, 16, 18], dtype=np.int32)
            invert_action_indices = np.array([1, 3, 5], dtype=np.int32)
            flipped_transition = flip_transition_horizontally(
                transition,
                image_keys,
                y_obs_idx,
                invert_state_indices=invert_indices,
                invert_action_indices=invert_action_indices,
            )
            data_store.insert(flipped_transition)

            obs = next_obs
            if done or truncated:
                stats = {"train": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent: DrQAgent, replay_buffer, demo_buffer):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project=FLAGS.exp_name,
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0
    upm_start_time = time.time()
    upm_last_time = upm_start_time
    upm_last_step = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
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

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(
                    batch,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
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
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=100
            )

        update_steps += 1

        # Report rolling learner throughput in updates/min using wall-clock time.
        now = time.time()
        dt = now - upm_last_time
        if dt >= FLAGS.updates_per_min_period_sec:
            d_updates = update_steps - upm_last_step
            rolling_upm = (d_updates / dt) * 60.0 if dt > 0 else 0.0

            total_dt = now - upm_start_time
            lifetime_upm = (update_steps / total_dt) * 60.0 if total_dt > 0 else 0.0

            perf_stats = {
                "perf/updates_per_min": rolling_upm,
                "perf/updates_per_min_lifetime": lifetime_upm,
                "perf/learner_updates": update_steps,
                "perf/learner_elapsed_min": total_dt / 60.0,
            }
            if wandb_logger:
                wandb_logger.log(perf_stats, step=update_steps)

            print(
                f"learner update_steps={update_steps} "
                f"updates/min={rolling_upm:.2f} "
                f"lifetime_updates/min={lifetime_upm:.2f}"
            )

            upm_last_time = now
            upm_last_step = update_steps


##############################################################################


def main(_):
    assert FLAGS.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    env = gym.make(
        FLAGS.env,
        fake_env=FLAGS.learner,
        save_video=FLAGS.eval_checkpoint_step,
    )
    print(env.observation_space)
    env = GripperCloseEnv(env)
    if FLAGS.actor:
        env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)

    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    rng, sampling_rng = jax.random.split(rng)
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: DrQAgent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    ## Set indices to be transformed by fractal class for the serl_robot_infra/robot_env/envs/franka_env
    # Note that observation_space[state] willb e sorted and set as an ordered dict by SerlObservationWrapper
    # gripper_pose:0
    # tcp_force.x: 1
    # tcp_force.y: 2
    # tcp_force.z: 3
    # tcp_pose.x:  4 <-- rel_frame.x points to base.+y
    # tcp_pose.y:  5 <-- rel_frame.y points to base.+x
    # tcp_pose.z:  6 <-- rel_frame.z points to base.-z
    x_obs_idx = np.array([4])
    y_obs_idx = np.array([5])
    
    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            # rlds_logger_path=FLAGS.log_rlds_path,
            type=FLAGS.replay_buffer_type,
            branch_method=FLAGS.branch_method,
            branching_factor=FLAGS.branching_factor,
            max_depth=FLAGS.max_depth,
            max_traj_length=FLAGS.max_traj_length,
            split_method=FLAGS.split_method,
            alpha=FLAGS.alpha,
            starting_branch_count=FLAGS.starting_branch_count,
            workspace_width=FLAGS.workspace_width,
            x_obs_idx=x_obs_idx,
            y_obs_idx=y_obs_idx,
            # preload_rlds_path=FLAGS.preload_rlds_path,
            image_keys=image_keys,
        )
        demo_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            # rlds_logger_path=FLAGS.log_rlds_path,
            type=FLAGS.replay_buffer_type,
            branch_method=FLAGS.branch_method,
            branching_factor=FLAGS.branching_factor,
            max_depth=FLAGS.max_depth,
            max_traj_length=FLAGS.max_traj_length,
            split_method=FLAGS.split_method,
            alpha=FLAGS.alpha,
            starting_branch_count=FLAGS.starting_branch_count,
            workspace_width=FLAGS.workspace_width,
            x_obs_idx=x_obs_idx,
            y_obs_idx=y_obs_idx,
            # preload_rlds_path=FLAGS.preload_rlds_path,
            image_keys=image_keys,
        )
        print(demo_buffer._size)
        import pickle as pkl

        with open(FLAGS.demo_path, "rb") as f:
            trajs = pkl.load(f)
            for traj in trajs:
                demo_buffer.insert(traj)
        print(f"demo buffer size: {len(demo_buffer)}")

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng, image_keys, y_obs_idx)

    else:
        raise NotImplementedError("Must be either a learner or an actor")
    return

if __name__ == "__main__":
    app.run(main)
