import numpy as np
import time
import requests
import copy
import cv2
import queue
import gym

from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.bin_relocation_env.config import BinEnvConfig


class FrankaBinRelocation(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=BinEnvConfig)
        self.observation_space["images"] = gym.spaces.Dict(
            {
                "wrist_1": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                "front": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
            }
        )
        self.task_id = 0  # 0 for forward task, 1 for backward task
        """
        the inner safety box is used to prevent the gripper from hitting the two walls of the bins in the center.
        it is particularly useful when there is things you want to avoid running into within the bounding box.
        it uses the intersect_line_bbox function to detect whether the gripper is going to hit the wall
        and clips actions that will lead to collision.
        """
        self.inner_safety_box = gym.spaces.Box(
            self._TARGET_POSE[:3] - np.array([0.5, 0.05, 0.05]),
            self._TARGET_POSE[:3] + np.array([0.5, 0.05, 0.0275]),
            dtype=np.float64,
        )

    def intersect_line_bbox(self, p1, p2, bbox_min, bbox_max):
        # Define the parameterized line segment
        # P(t) = p1 + t(p2 - p1)
        tmin = 0
        tmax = 1

        for i in range(3):
            if p1[i] < bbox_min[i] and p2[i] < bbox_min[i]:
                return None
            if p1[i] > bbox_max[i] and p2[i] > bbox_max[i]:
                return None

            # For each axis (x, y, z), compute t values at the intersection points
            if abs(p2[i] - p1[i]) > 1e-10:  # To prevent division by zero
                t1 = (bbox_min[i] - p1[i]) / (p2[i] - p1[i])
                t2 = (bbox_max[i] - p1[i]) / (p2[i] - p1[i])

                # Ensure t1 is smaller than t2
                if t1 > t2:
                    t1, t2 = t2, t1

                tmin = max(tmin, t1)
                tmax = min(tmax, t2)

                if tmin > tmax:
                    return None

        # Compute the intersection point using the t value
        intersection = p1 + tmin * (p2 - p1)

        return intersection

    def clip_safety_box(self, pose):
        pose = super().clip_safety_box(pose)
        # Clip xyz to inner box
        if self.inner_safety_box.contains(pose[:3]):
            # print(f'Command: {pose[:3]}')
            pose[:3] = self.intersect_line_bbox(
                self.currpos[:3],
                pose[:3],
                self.inner_safety_box.low,
                self.inner_safety_box.high,
            )
            # print(f'Clipped: {pose[:3]}')
        return pose

    def crop_image(self, name, image):
        """Crop realsense images to be a square."""
        if name == "wrist_1":
            return image[:, 80:560, :]
        elif name == "front":
            # return image[:, 80:560, :]
            return image
        else:
            return ValueError(f"Camera {name} not recognized in cropping")

    def get_im(self):
        images = {}
        display_images = {}
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.crop_image(key, rgb)
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                if key == "front":
                    display_images[key + "_full"] = cv2.resize(cropped_rgb, (480, 480))
                else:
                    display_images[key + "_full"] = cropped_rgb
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        self.recording_frames.append(
            np.concatenate([display_images[f"{k}_full"] for k in self.cap], axis=0)
        )  # only record wrist images since front image is not used for training
        self.img_queue.put(display_images)
        return images

    def task_graph(self, obs=None):
        if obs is None:
            return (self.task_id + 1) % 2

    def set_task_id(self, task_id):
        self.task_id = task_id

    def reset(self, joint_reset=False, **kwargs):
        '''
        Set resest position for end-effector based on TARGET_POSE.
        Select values for forward and backward policy that maximize
        your viewing range from global camera.

        This is experiment-setup specific and will depend on where 
        your global camera is positioned and the size of your trays.
        '''
        # Forward policy offset
        if self.task_id == 0:
            X_OFFSET_FW = -0.025
            Y_OFFSET_FW = 0.025
            self.resetpos[0] = self._TARGET_POSE[0] + X_OFFSET_FW
            self.resetpos[1] = self._TARGET_POSE[1] + Y_OFFSET_FW
        # Backward policy offset
        elif self.task_id == 1:
            X_OFFSET_BW = 0.2
            Y_OFFSET_BW = 0.05
            self.resetpos[0] = self._TARGET_POSE[0] - Y_OFFSET_BW
            self.resetpos[1] = self._TARGET_POSE[1] - X_OFFSET_BW
        else:
            raise ValueError(f"Task id {self.task_id} should be 0 or 1")

        return super().reset(joint_reset, **kwargs)

    def go_to_rest(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
       # Open gripper
        self._send_gripper_command(1)
        
        # Get current position
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.5)

        # Move up 0.05m in Z-axis to clear any objects in the slot
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.05
        self.interpolate_move(reset_pose, timeout=1)

        # Call parent to finish reset
        super().go_to_rest(joint_reset)