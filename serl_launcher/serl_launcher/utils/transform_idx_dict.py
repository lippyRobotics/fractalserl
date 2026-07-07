import numpy as np

observation_idx = {
    "real": {
        "FV": {
            "x_obs_idx": np.array([]),
            "y_obs_idx": np.array([]),
        },
        "FH": {
            "x_obs_idx": np.array([]),
            "y_obs_idx": np.array([]),
        },
        "FR": {
            "x_obs_idx": np.array([]),
            "y_obs_idx": np.array([]),
        },
    },
    "sim": {
        "proprio": {
            "FV": {
                "x_obs_idx": np.array([1, 4]),
                "y_obs_idx": np.array([2, 5]),
                "y_state_reflect_idx": np.array([2, 5]),
                "y_action_reflect_idx": np.array([1]),
            },
            "FH": {
                "x_obs_idx": np.array([]),
                "y_obs_idx": np.array([]),
            },
            "NF": { # No Reflection
                "x_obs_idx": np.array([]),
                "y_obs_idx": np.array([]),
            }  # 
        },
        "vision": {
            "FV": {
                "x_obs_idx": np.array([1]),
                "y_obs_idx": np.array([2]),
                "y_state_reflect_idx": np.array([2]),
                "y_action_reflect_idx": np.array([1]),
            },
            "FH": {
                "x_obs_idx": np.array([]),
                "y_obs_idx": np.array([]),
            },
            "NF": {
                "x_obs_idx": np.array([1]),
                "y_obs_idx": np.array([2]),
            }  # Note: Used "None" as a string key here per your structure
        },
    }
}