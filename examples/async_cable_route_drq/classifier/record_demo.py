import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os

import franka_env

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
)

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
import jax

if __name__ == "__main__":

    ## Initializes Enviroment 
    env = gym.make("FrankaCableRoute-Vision-v0", save_video=False)
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    image_keys = [k for k in env.observation_space.keys() if "state" not in k]

    ## Initialize RNG, POS/NEG Containers, and Counters 
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    obs, _ = env.reset()

    pos_transitions = [] 
    neg_transitions = []
    transition_batch = []

    pos_count = 0
    neg_count = 0
    pos_needed = 20   # Define a positive reward max
    neg_needed = 20

    neg_transition_count = 0
    pos_transition_count = 0


    ## Define Output file and safety checks
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pos_file_name = f"demos/positive_cable_route_{pos_needed}_demos_{uuid}.pkl"
    neg_file_name = f"demos/negative_cable_route_{neg_needed}_demos_{uuid}.pkl"

    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    pos_file_path = os.path.join(file_dir, pos_file_name)
    neg_file_path = os.path.join(file_dir, neg_file_name)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(pos_file_path):
        raise FileExistsError(f"{pos_file_name} already exists in {file_dir}")
    if os.path.exists(neg_file_path):
        raise FileExistsError(f"{neg_file_name} already exists in {file_dir}")
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    ## Record Negative demos
    print("Recording negative demos:\n")
    while neg_count < neg_needed:
        actions = np.zeros((6,))
        next_obs, rew, done, truncated, info = env.step(action=actions)
        rew = 0
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations = obs,
                actions = actions,
                next_observations = next_obs,
                rewards = rew,
                masks = 1.0 - done,
                dones = done,
            )
        )

        transition_batch.append(transition)
        neg_transition_count += 1
        print(f"neg transitions: {neg_transition_count} | demos completed: {neg_count}")
        obs = next_obs

        if done:
            neg_transitions += transition_batch
            neg_count += 1

            print(
                f"{neg_needed - neg_count} negative demos left."
            )
            obs, _ = env.reset(pos_reset=False)
            neg_transition_count = 0
            transition_batch.clear()

    ## Move to positive position | Asks to confirm every 50 steps
    userInput = "n"
    while (userInput != "y"):
        prep_transition_count = 0
        while prep_transition_count < 50:
            actions = np.zeros((6,))
            next_obs, rew, done, truncated, info = env.step(action=actions)
            if "intervene_action" in info:
                actions = info["intervene_action"]
            prep_transition_count += 1
            print(f"Transition count: {prep_transition_count} / 50")
        userInput = input("Is the robot in a successful pose? (y/n)")

    env.reset(pos_reset=False)
    ## Record Positive demos
    print("Recording positive demos:\n")
    print("Please put robot in successful pose and press Enter...")
    input()     # pause, wait for user
    while pos_count < pos_needed:
        actions = np.zeros((6,))
        next_obs, rew, done, truncated, info = env.step(action=actions)
        rew = 0
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        transition_batch.append(transition)
        
        obs = next_obs
        pos_transition_count += 1
        print(f"pos transitions: {pos_transition_count} | demos completed: {pos_count}")

        if done:
            pos_transitions += transition_batch
            pos_count += 1
            pos_transition_count = 0

            print(
                f"{pos_needed - pos_count} positive demos left."
            )
            obs, _ = env.reset(pos_reset=False)
            transition_batch.clear()

    with open(pos_file_path, "wb") as f:
        pkl.dump(pos_transitions, f)
        print(
            f"saved {pos_needed} demos and {len(pos_transitions)} transitions to {pos_file_path}"
        )

    with open(neg_file_path, "wb") as f:
        pkl.dump(neg_transitions, f)
        print(
            f"saved {neg_needed} demos and {len(neg_transitions)} transitions to {neg_file_path}"
        )

    env.close()
