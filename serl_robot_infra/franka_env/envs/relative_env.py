from scipy.spatial.transform import Rotation as R
import gym
import numpy as np
from gym import Env
from franka_env.utils.transformations import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)


class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame at reset.
    All measurements are thus relative to the starting reset frame.

    Consider the following frames and nomenclatures:
    o: base robot frame
    r: the frozen tcp frame that happens only at reset time.
    b: the current end-effector frame

    Notation: b_T_a would represent a transformation from b to a.
    T: robot_base_frame -> reset_tcp_frame:         o_T_r
    T: end_effector_frame -> robot_base_frame:      b_T_o
    Transformation: reset_tcp_frame -> end_effector_frame:
        r_T_b = inv(o_T_r) * o_T_b
        r_T_b = r_T_o * b_T_o

    This wrapper is expected to be used on top of the Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...).
    By convention, the 7th dimension of the action space is used for the gripper.

    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)

        # Adjoint matrix used to convert tcp_vel or actions from base frame to end-effector frame via inv(Adj(T))*tcp_vel
        self.adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # o: base frame
            # r: the frozen tcp frame that happens only at reset time.
            # b: end-effector frame
            # Transformation from base to tcp: o_T_r (or inv(T_r_o below)
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from base frame to end-effector frame
        transformed_action = self.transform_action(action)

        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        """
        Reset the wrapped environment and choose the new relative coordinate frame.

        ``RelativeFrame`` does not directly read ``TARGET_POSE``, ``RESET_POSE``,
        or the base environment's persistent reset-pose attribute. In this repo
        that attribute is named ``self.resetpos``; if a future environment uses
        a name like ``self._reset_pose``, it would play the same indirect role.
        Instead, ``RelativeFrame`` first calls ``self.env.reset(**kwargs)`` and
        then uses the returned
        ``obs["state"]["tcp_pose"]`` as the origin and orientation of the
        relative frame for this episode.

        In the Franka environments, those config values still affect this
        wrapper indirectly:

        * ``FrankaEnv`` converts ``config.RESET_POSE`` into ``self.resetpos``.
        * ``FrankaEnv.reset()`` moves the robot to ``self.resetpos`` when
          ``RANDOM_RESET`` is disabled.
        * When ``RANDOM_RESET`` is enabled, ``FrankaEnv.reset()`` starts from
          ``self.resetpos`` but randomizes xy position and yaw using
          ``TARGET_POSE`` orientation values.
        * Some task envs can modify ``self.resetpos`` before the parent reset.
          For example, bin relocation shifts ``self.resetpos[:2]`` from
          ``TARGET_POSE`` differently for the forward and backward task ids.

        Therefore, the relative frame is anchored to the actual post-reset TCP
        pose. For homography code, this distinction matters: calibration points
        from ``/getpos`` are in the robot base/world frame, while observations
        after this wrapper are expressed relative to the post-reset TCP frame.
        """
        obs, info = self.env.reset(**kwargs)

        # Update adjoint matrix wrt to new reset position
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Convert the environment's observation into the frames expected by the policy.

        * Linear/angular velocities are provided by the wrapped env in the spatial (base) frame; we left-multiply them by ``inv(Adj(T))`` so they are expressed in the end-effector frame.
        * When ``include_relative_pose`` is enabled, tcp poses are re-expressed
          relative to the pose at reset.

          That is, we compute T_r^b = T_r^o @ T_o^b and return
          the position/quaternion extracted from T_r^b.

        * Image observations pass through untouched.
        """
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action
