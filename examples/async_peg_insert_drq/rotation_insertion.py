import numpy as np
import copy
from scipy.spatial.transform import Rotation

# def rotate_transitions():

def rotate_transitions(
    transitions,
    radians,
):
    rotated_transitions = []
    if isinstance(transitions, list):
        for t in transitions:
            rotated_transitions.append(
                rotate_single_transition(
                    t,
                    radians,
                )
            )
    else:
        rotated_transitions = rotate_single_transition(
            transitions,
            radians,
        )
        
    return rotated_transitions

def rotate_single_transition(
    transition,
    radians,
):
    rotated_transition = copy.deepcopy(transition)
    
    #==========================================
    # Gather state/action indices to rotate
    #==========================================
    # States
    state_item_count = 6 # Unique items in robot states (pos, vel, torque, etc.)
    state_item_count_array = np.arange(state_item_count)
    dimensions = 3  # Includes (x,y,z) for each item
    state_indices = dimensions*state_item_count_array[:, None] + np.array([1, 2, 3])

    # Actions
    # print("=======================================")
    # print(rotated_transition["actions"])
    action_item_count = 2 # Unique items in robot actions (pos_delta & rot_delta)
    action_item_count_array = np.arange(action_item_count)
    action_indices = dimensions*action_item_count_array[:, None] + np.array([0, 1, 2])
    rotation_matrix = Rotation.from_euler('z', radians).as_matrix()
    print(action_indices)

    #==========================================
    # Rotate states/actions
    #==========================================
    # States
    # print("Original State:")
    # print(rotated_transition["observations"]["state"][..., state_indices])
    state_items = rotated_transition["observations"]["state"][..., state_indices]
    state_items = state_items.transpose().reshape(dimensions,state_item_count)
    new_state_items = rotation_matrix @ state_items
    new_state_items = new_state_items.transpose().reshape(1,state_item_count,dimensions)
    rotated_transition["observations"]["state"][..., state_indices] = new_state_items
    # print("Rotated State:")
    # print(rotated_transition["observations"]["state"][..., state_indices])

    # Actions
    # print("Original Action:")
    # print(rotated_transition["actions"][..., action_indices])
    action_items = rotated_transition["actions"][..., action_indices]
    action_items = action_items.transpose().reshape(dimensions,action_item_count)
    new_action_items = rotation_matrix @ action_items
    new_action_items = new_action_items.transpose().reshape(1,action_item_count,dimensions)
    rotated_transition["actions"][..., action_indices] = new_action_items
    # print("Rotated Action:")
    # print(rotated_transition["actions"][..., action_indices])

    #==========================================
    # Re-place states/actions
    #==========================================
    # States
    rotated_transition["observations"]["state"][..., state_indices] = new_state_items
    # print(rotated_transition["observations"]["state"][..., state_indices])
    # print("=====================================")
    # Actions
    rotated_transition["actions"][..., action_indices] = new_action_items
    return rotated_transition