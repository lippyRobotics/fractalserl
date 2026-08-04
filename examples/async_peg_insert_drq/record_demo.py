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


def flip_transition_horizontally(
    transition,
    image_keys,
    y_obs_idx,
    invert_state_indices=None,
    invert_action_indices=None,
):
    flipped_transition = copy.deepcopy(transition)

    # Check if both images are present
    assert "wrist_1" in image_keys and "wrist_2" in image_keys, (
        f"horizontal flip needs both wrist cameras, got image_keys={image_keys}"
    )

    # Flip images horizontally (assumes width is second-to-last dimension)
    for key in image_keys:
        if key in flipped_transition["observations"]:
            if key == "wrist_1":
                flipped_transition["observations"][key] = np.flip(
                    transition["observations"]["wrist_1"], axis=2
                ).copy()
                if key in flipped_transition["next_observations"]:
                    flipped_transition["next_observations"][key] = np.flip(
                        transition["next_observations"]["wrist_1"], axis=2
                    ).copy()
            if key == "wrist_2":
                flipped_transition["observations"][key] = np.flip(
                    transition["observations"]["wrist_2"], axis=2
                ).copy()
                if key in flipped_transition["next_observations"]:
                    flipped_transition["next_observations"][key] = np.flip(
                        transition["next_observations"]["wrist_2"], axis=2
                    ).copy()
    
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

if __name__ == "__main__":
    env = gym.make("FrankaPegInsert-Vision-v0")
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    # Variables for flipping transitions
    image_keys = [k for k in env.observation_space.keys() if k != "state"]
    invert_state_indices = np.array([2, 7, 9, 10, 12, 14, 16, 18], dtype=np.int32)
    invert_action_indices = np.array([1, 3, 5], dtype=np.int32)
    y_obs_idx = np.array([5], dtype=np.int32)

    batch = []
    transitions = []
    success_count = 0
    success_needed = 20
    total_count = 0
    pbar = tqdm(total=success_needed)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"peg_insert_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(file_path):
        raise FileExistsError(f"{file_name} already exists in {file_dir}")
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    while success_count < success_needed:
        actions = np.zeros((6,))
        next_obs, rew, done, truncated, info = env.step(action=actions)
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
        batch.append(transition)
        obs = next_obs

        if done:
            if rew:
                transitions += batch

                # Perform horizontal flip augmentation on the batch of transitions
                for t in batch:
                    transitions.append(
                        flip_transition_horizontally(
                            t,
                            image_keys,
                            y_obs_idx,
                            invert_state_indices=invert_state_indices,
                            invert_action_indices=invert_action_indices,
                        )
                    )

                success_count += 1
            total_count += 1

            batch.clear()
            print(
                f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
            )
            pbar.update(rew)
            obs, _ = env.reset()


    with open(file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_path}")

    env.close()
    pbar.close()
