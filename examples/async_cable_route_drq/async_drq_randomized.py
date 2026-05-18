#!/usr/bin/env python3

import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints

import pickle as pkl
import os
import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_replay_buffer,
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
    BinaryRewardClassifierWrapper,
)

import franka_env

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "FrankaCableRoute-Vision-v0", "Name of environment.")
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
flags.DEFINE_string(
    "reward_classifier_ckpt_path", None, "Path to reward classifier ckpt."
)

# replay buffer flags
flags.DEFINE_string("replay_buffer_type", "memory_efficient_replay_buffer", "Which replay buffer to use")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")
flags.DEFINE_integer("branching_factor", None, "Factor by which branch count is changed")
flags.DEFINE_integer("max_depth", None, "Maximum number of splits that may occur in one episode")
flags.DEFINE_string("branch_method", "constant", "Method for how many branches to generate")
flags.DEFINE_string("split_method", "never", "Method for when to change number of branches")
flags.DEFINE_float("alpha", 0.2, "Rate of change of max_traj_length")
flags.DEFINE_float("workspace_width", 0.5, "Workspace width in meters")
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


def actor(agent: DrQAgent, data_store, env, sampling_rng):
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
                # Use deterministic action selection for evaluation.
                # `argmax=True` tells the agent to pick the highest-value
                # / highest-probability action instead of sampling stochastically.
                # This produces consistent, reproducible evaluation metrics
                # (success rate and completion time) by removing exploration noise.
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
                    print(f"{success_counter}/{episode + 1}")

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

    # Learner pushes new network parameters; keep this actor's policy in sync.
    def update_params(params):
        """Replace only model parameters while preserving the rest of agent state."""
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    # registers update_params as the handler for incoming network parameter 
    # messages from the learner. When an update arrives, update_params(params) 
    # runs and swaps in new model weights
    client.recv_network_callback(update_params)

    # Initialize first rollout episode.
    obs, _ = env.reset(joint_reset=True)
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        # Start manual 'total' stopwatch
        timer.tick("total") 

        # Start automatic 'sample_actions' logging:
        with timer.context("sample_actions"):

            # Collect random steps to bootstrap the replay buffer. Initial policy poor.
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()

            # Sample actions from agent.
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                # Move actions from JAX device (GPU/TPU) to host NumPy for Gym env.step.
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            # Collect tuple info and packet in transition dict
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

            # Push transition to learner replay buffer
            data_store.insert(transition)

            obs = next_obs
            if done or truncated:
                if reward:
                    print("cable route success!")
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
        project="CableRoute-april_2026",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request.
        Will log what actor sends via client.request("send-stats", payload).
        i.e. from RecordEpisodeStatistics, usually info["episode"] = {"r": return, "l": length, "t": elapsed_time}
        i.e. wrapper flags like left/right from SpacemouseIntervention
        i.e. Timing stats inside timer: averages for keys like "sample_actions", "step_env", "total" from your Timer
        Training info and log_period steps are typically captured"""

        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create the Training server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)                          # Learner registers where incoming actor data should go:
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    # Do not start gradient updates until enough actor data has arrived.
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    
        # Advance only by newly added samples since last refresh.
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    
    # Final sync so bar reaches the latest replay size before closing.
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    # Initial send: broadcast learner's current params so actors start rollouts with synced weights.
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # Build two equal-size iterators: one for online actor data and one for demos.
    # Each learner step concatenates these batches for 50/50 mixed training.
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            # Half batch from online replay; the other half comes from demo_iterator.
            "batch_size": FLAGS.batch_size // 2,
            # Keep (obs, next_obs) together in one packed sample for agent updates.
            "pack_obs_and_next_obs": True,
        },
        # Pre-shard onto local devices to match replicated learner state.
        # Will prepare outputs as 'JAX device arrays' ahead of time vs NumPy. 
        # Results in less host-2-device overhead in hot training loop, fewer shape mismatches, more stable throughput.
        device=sharding.replicate(),
    )

    # Same for demo replay buffer.
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # Main learner loop
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):

        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.critic_actor_ratio - 1):

            with timer.context("sample_replay_buffer"):
                batch      = next(replay_iterator)
                demo_batch = next(demo_iterator)
                
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                # agent: updates the learner agent state (new critic params, targets, rng state, etc.).
                # critis_info: contains logging metrics from that critic update (e.g., critic losses/stats).
                agent, critics_info = agent.update_critics(
                    batch,
                )

        # Run utd "full update" steps (critic + actor + temperature). Actor weights not yet sent.       
        with timer.context("train"):
            batch      = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        # publish the updated network to the actor.
        if step > 0 and step % (FLAGS.steps_per_update) == 0:

            # Force synchronization by waiting for pending JAX computations that produce agent.
            # Key since JAX is asynchronous. Otherwise could publish params before latest updates.
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # Log {critic/actor/temperature/optimizer} and {timer info:sample_replay_buffer/train_critics/train} info
        if update_steps % FLAGS.log_period == 0 and wandb_logger:

            # Comes from agent.updated_high_utd
            wandb_logger.log(update_info, step=update_steps)

            # Timer class will return replay_buffer/critic avg wall-clock times for sections:
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        # Save checkpoint/model
        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=100
            )

        update_steps += 1


##############################################################################


def main(_):
    assert FLAGS.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # create env and load dataset
    env = gym.make(
        FLAGS.env,
        fake_env=FLAGS.learner,
        save_video=FLAGS.eval_checkpoint_step,
    )
    env = GripperCloseEnv(env)
    if FLAGS.actor:
        env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    if FLAGS.actor:
        # Actor uses a learned binary reward classifier to turn sparse visual success
        # detection into a runtime reward signal. We require a checkpoint path here
        # because actor/eval runs depend on classifier inference at every env step.

        if FLAGS.reward_classifier_ckpt_path is None:
            raise ValueError("reward_classifier_ckpt_path must be specified for actor")

        # Build classifier model definition, restore its parameters from checkpoint,
        # and return a jitted function: obs -> success logit.
        # - key: RNG for model initialization scaffolding before checkpoint restore.
        # - sample: example observation to initialize classifier parameter shapes.
        # - image_keys: camera streams consumed by the classifier encoder.
        # - checkpoint_path: directory containing saved classifier checkpoints.
        reward_func = load_classifier_func(
            key=sampling_rng,
            sample=env.observation_space.sample(),
            image_keys=image_keys,
            checkpoint_path=FLAGS.reward_classifier_ckpt_path,
        )
        
        # Inject classifier-based reward computation into env.step(...):
        # wrapper thresholds classifier logits into binary success and adds it to reward.
        env = BinaryRewardClassifierWrapper(env, reward_func)
    env = RecordEpisodeStatistics(env)

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
        
        x_obs_idx = np.array([4])
        y_obs_idx = np.array([5])

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

        if FLAGS.demo_path:
            # Check if the file exists
            if not os.path.exists(FLAGS.demo_path):
                raise FileNotFoundError(f"File {FLAGS.demo_path} not found")

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
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")
    return

if __name__ == "__main__":
    app.run(main)
