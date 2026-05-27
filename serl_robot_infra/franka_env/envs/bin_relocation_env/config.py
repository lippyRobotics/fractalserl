import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class BinEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""
    WAIT_FOR_GRIPPER_SETTLED: bool = True
    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1":  "218622274083",
        "front":    "218622276001",
    }
    # TARGET_POSE is the center of the two trays relative to the base of the robot
    # For Lipscomb setup, these numbers make sense
    TARGET_POSE = np.array(
        [
            0.5725, #0.485,
            -0.025,
            0.047555915476419935,
            3.1331234,
            0.0182487,
            1.5824805,
        ]
    )
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.zeros(6)
    APPLY_GRIPPER_PENALTY = True
    ACTION_SCALE = np.array([0.1, 0.2, 1])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.1
    RANDOM_RZ_RANGE = np.pi / 6
    # All the upper and lower adjustments happen in franka_bin_relocation.py:FrankBinRelocation:30 
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - 0.13,   # -x axis
            TARGET_POSE[1] - 0.24,   # -y axis
            TARGET_POSE[2] - 0.03,   # -z axis
            TARGET_POSE[3] - 0.01,
            TARGET_POSE[4] - 0.01,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + 0.15,   # +x axis
            TARGET_POSE[1] + 0.25,   # +y axis
            TARGET_POSE[2] + 0.1,     # +z axis
            TARGET_POSE[3] + 0.01,
            TARGET_POSE[4] + 0.01,
            TARGET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.006,
        "translational_clip_y": 0.006,
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.006,
        "translational_clip_neg_y": 0.006,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }