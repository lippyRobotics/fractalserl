import numpy as np
import copy

# This file contains all required functions for performing reflections on transitions.
# It works by calling reflect_transitions() and passing in EITHER a list of
# transitions OR a single transition, image_keys, along with the required robot
# state/action indices for reflections about the desired axis.

# Main function call
# Handles either single or list transitions
def reflect_transitions(
    transitions,
    image_keys,
    invert_state_indices=None,
    invert_action_indices=None,
):
    flipped_transitions = []
    if isinstance(transitions, list):
        for t in transitions:
            flipped_transitions.append(
                reflect_single_transition(
                    t,
                    image_keys,
                    invert_state_indices,
                    invert_action_indices,
                )
            )
    else:
        flipped_transitions = reflect_single_transition(
            transitions,
            image_keys,
            invert_state_indices,
            invert_action_indices,
        )
        
    return flipped_transitions



def reflect_single_transition(
    transition,
    image_keys,
    invert_state_indices=None,
    invert_action_indices=None,
):
    flipped_transition = copy.deepcopy(transition)
    image_keys_copy = copy.deepcopy(image_keys)

    # Check if both images are present
    assert "wrist_1" in image_keys and "wrist_2" in image_keys, (
        f"horizontal flip needs both wrist cameras, got image_keys={image_keys}"
    )
    # Flip images horizontally (assumes width is second-to-last dimension)
    for key in image_keys_copy:
        if key in flipped_transition["observations"]:
            # Swap camera 1 with camera 2; Also swap the pixels of each camera
            if key == "wrist_1":
                flipped_transition["observations"][key] = np.flip(
                    transition["observations"]["wrist_2"], axis=2
                ).copy()
                if key in flipped_transition["next_observations"]:
                    flipped_transition["next_observations"][key] = np.flip(
                        transition["next_observations"]["wrist_2"], axis=2
                    ).copy()
            # Swap camera 2 with camera 1; Also swap the pixels of each camera
            if key == "wrist_2":
                flipped_transition["observations"][key] = np.flip(
                    transition["observations"]["wrist_1"], axis=2
                ).copy()
                if key in flipped_transition["next_observations"]:
                    flipped_transition["next_observations"][key] = np.flip(
                        transition["next_observations"]["wrist_1"], axis=2
                    ).copy()
    
    # Invert specified state indices (negate values)
    # Use ellipsis to handle any leading dimensions (batch, time, etc.)
    if invert_state_indices is not None:
        flipped_transition["observations"]["state"][..., invert_state_indices] = -flipped_transition["observations"]["state"][..., invert_state_indices]
        flipped_transition["next_observations"]["state"][..., invert_state_indices] = -flipped_transition["next_observations"]["state"][..., invert_state_indices]
    
    # Also invert y-position (backward compatibility)
    # flipped_transition["observations"]["state"][..., y_obs_idx] = -flipped_transition["observations"]["state"][..., y_obs_idx]
    # flipped_transition["next_observations"]["state"][..., y_obs_idx] = -flipped_transition["next_observations"]["state"][..., y_obs_idx]

    if invert_action_indices is not None:
        flipped_transition["actions"][..., invert_action_indices] = -flipped_transition["actions"][..., invert_action_indices]
    
    return flipped_transition


# Will deprecate this soon (after testing)
def insert_reflected_transitions(pendingBuffer, data_store):
    for flipped_transition in pendingBuffer:
        data_store.insert(flipped_transition)
    pendingBuffer.clear()