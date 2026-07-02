import copy
from typing import Iterable, Optional

import cv2
import gym
import numpy as np
from serl_launcher.data.dataset import DatasetDict, _sample
from serl_launcher.data.replay_buffer import ReplayBuffer
from flax.core import frozen_dict


class IndexedSymmetryReplayBuffer(ReplayBuffer):
    """Fractal-symmetry replay buffer that applies the grid transform at sample
    time, keyed by a per-sample transformation_index, instead of tiling every
    transition into one copy per branch at insert time.

    capacity means number of real (source) transitions (unlike
    FractalSymmetryReplayBuffer, where capacity is multiplied by the branch
    count internally).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        workspace_width: int,
        x_obs_idx: np.ndarray,
        y_obs_idx: np.ndarray,
        branch_method: str,
        split_method: str,
        img_keys: list,
        kwargs: dict,
        front_M: Optional[np.ndarray] = None,
        world_fixed_img_keys: Iterable[str] = (),
    ):
        assert branch_method == "constant", (
            "\033[31mERROR: \033[0mIndexedSymmetryReplayBuffer supports only "
            "branch_method='constant' / split_method='never'; use "
            "FractalSymmetryReplayBuffer for variable branching"
        )
        assert split_method in (None, "", "never"), (
            "\033[31mERROR: \033[0mIndexedSymmetryReplayBuffer supports only "
            "branch_method='constant' / split_method='never'; use "
            "FractalSymmetryReplayBuffer for variable branching"
        )

        self.workspace_width = workspace_width
        self.img_keys = img_keys
        self._img_insert_index_ = 0
        self.timestep = 0

        self.front_M = None if front_M is None else np.asarray(front_M, np.float64)
        self.world_fixed_img_keys = tuple(world_fixed_img_keys)
        self.front_map = None

        self.x_obs_idx = x_obs_idx
        self.y_obs_idx = y_obs_idx

        self.branch_method = branch_method
        self.split_method = split_method

        assert "starting_branch_count" in kwargs, (
            "\033[31mERROR: \033[0mstarting_branch_count must be defined for "
            "branch_method \"constant\""
        )
        self.starting_branch_count = kwargs.pop("starting_branch_count")
        self.total_branches = self.starting_branch_count ** 2

        # Warn about unused kwargs
        for k in kwargs.keys():
            print(f"\033[33mWARNING \033[0m argument \"{k}\" not used")

        # Account for images
        self._num_stack = None
        observation_space = copy.deepcopy(observation_space)
        next_observation_space = None
        if self.img_keys:
            self.img_buffer = {}
            next_observation_space_dict = copy.deepcopy(observation_space.spaces)
            for k in img_keys:
                img_obs_space = observation_space.spaces[k]
                if self._num_stack is None:
                    self._num_stack = img_obs_space.shape[0]
                    self.img_buffer_size = capacity * (self._num_stack + 1)

                buffer_shape = list(img_obs_space.shape[1:])
                buffer_shape.insert(0, self.img_buffer_size)
                self.img_buffer[k] = np.empty(buffer_shape, img_obs_space.dtype)

                observation_space.spaces[k] = gym.spaces.Box(
                    low=float("-inf"), high=float("inf"), shape=(), dtype=np.int32
                )
                next_observation_space_dict.pop(k)
            next_observation_space = gym.spaces.Dict(next_observation_space_dict)

        # Init replay buffer class - capacity is source transitions, not tiled
        super().__init__(
            observation_space=observation_space,
            next_observation_space=next_observation_space,
            action_space=action_space,
            capacity=capacity,
        )

        self.generate_transform_deltas()

    def generate_transform_deltas(self):
        obs_state = self.dataset_dict["observations"]
        if self.img_keys:
            obs_state = self.dataset_dict["observations"]["state"]

        obs_size = obs_state.shape[-1]

        self.transform_deltas = np.zeros(
            shape=(self.total_branches, obs_size), dtype=np.float32
        )

        idx = np.arange(self.total_branches)
        x_deltas, y_deltas = np.divmod(idx, self.starting_branch_count)

        base_diff = -self.workspace_width / 2.0

        x_deltas = (2 * x_deltas + 1) * self.workspace_width / (
            2 * self.starting_branch_count
        ) + base_diff
        y_deltas = (2 * y_deltas + 1) * self.workspace_width / (
            2 * self.starting_branch_count
        ) + base_diff
        x_deltas = np.repeat(x_deltas, self.x_obs_idx.size)
        y_deltas = np.repeat(y_deltas, self.y_obs_idx.size)
        x_deltas = np.reshape(x_deltas, (self.total_branches, self.x_obs_idx.size))
        y_deltas = np.reshape(y_deltas, (self.total_branches, self.y_obs_idx.size))

        self.transform_deltas[..., self.x_obs_idx] = x_deltas
        self.transform_deltas[..., self.y_obs_idx] = y_deltas

        if self._num_stack:
            # singleton stack dim; broadcasting handles the rest (avoids the
            # np.repeat-across-stack pattern used by FractalSymmetryReplayBuffer)
            self.transform_deltas = np.expand_dims(self.transform_deltas, axis=1)

        self.generate_front_maps()

    def generate_front_maps(self):
        if self.front_M is None or not self.world_fixed_img_keys or not self.img_keys:
            return

        deltas = self.transform_deltas
        if self._num_stack:
            deltas = deltas[:, 0, :]  # (n, obs_size)
        n = deltas.shape[0]

        dx = deltas[:, self.x_obs_idx[0]]  # (n,) - already centered
        dy = deltas[:, self.y_obs_idx[0]]  # (n,)

        sample_img = next(iter(self.img_buffer.values()))
        H, W = sample_img.shape[1:3]

        Minv = np.linalg.inv(self.front_M)

        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        dst = np.stack([uu.ravel(), vv.ravel(), np.ones(H * W)])

        self.front_map = np.empty((n, 2, H, W), np.float32)
        for b in range(n):
            T = np.array(
                [
                    [1.0, 0.0, dx[b]],
                    [0.0, 1.0, dy[b]],
                    [0.0, 0.0, 1.0],
                ]
            )
            Hb = self.front_M @ T @ Minv
            src = np.linalg.inv(Hb) @ dst  # (3, H*W)
            src = src[:2] / src[2]
            self.front_map[b, 0] = src[0].reshape(H, W)  # map_x
            self.front_map[b, 1] = src[1].reshape(H, W)  # map_y

    def insert_images(self, observation: dict, frame_idx: int = -1):
        for k in self.img_keys:
            if self._num_stack:
                self.img_buffer[k][self._img_insert_index_] = observation[k][frame_idx, ...]
            else:
                self.img_buffer[k][self._img_insert_index_] = observation[k]
        self._img_insert_index_ = (self._img_insert_index_ + 1) % self.img_buffer_size

    def insert(self, data: DatasetDict):
        data_dict = data.copy()
        data_dict["observations"] = data_dict["observations"].copy()
        data_dict["next_observations"] = data_dict["next_observations"].copy()

        if self.img_keys:
            if self.timestep == 0:
                for i in range(self._num_stack):
                    self.insert_images(data_dict["observations"], frame_idx=i)
            self.insert_images(data_dict["next_observations"], frame_idx=-1)

        for k in self.img_keys:
            data_dict["observations"][k] = np.int32(
                (self._img_insert_index_ - 1) % self.img_buffer_size
            )
            data_dict["next_observations"].pop(k)

        super().insert(data_dict)

        self.timestep += 1
        if data_dict["dones"]:
            self.timestep = 0

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        pack_obs_and_next_obs: bool = False,
        transformation_index: Optional[np.ndarray] = None,
    ) -> frozen_dict.FrozenDict:
        """Samples from the replay buffer, applying the fractal-symmetry
        transform at sample time via transformation_index.

        Args:
            batch_size: Minibatch size.
            keys: Keys to sample.
            indx: Take indices instead of sampling.
            pack_obs_and_next_obs: whether to pack img and next_img into one
                image stack. Useful when they have overlapping frames.
            transformation_index: explicit per-sample transform ids (for
                tests/debugging); drawn uniformly at random if None.

        Returns:
            A frozen dictionary, including a "transformation_index" key.
        """
        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)
        else:
            indx = np.asarray(indx)
            assert np.all(indx < self._size)

        if transformation_index is None:
            if hasattr(self.np_random, "integers"):
                transformation_index = self.np_random.integers(
                    self.total_branches, size=batch_size
                ).astype(np.int32)
            else:
                transformation_index = self.np_random.randint(
                    self.total_branches, size=batch_size
                ).astype(np.int32)
        else:
            transformation_index = np.asarray(transformation_index, dtype=np.int32)
            assert transformation_index.shape == (batch_size,)
            assert np.all(transformation_index >= 0) and np.all(
                transformation_index < self.total_branches
            )

        if not self.img_keys:
            if keys is None:
                sample_keys = None
            else:
                sample_keys = [k for k in keys if k != "transformation_index"]
            batch = super().sample(batch_size, sample_keys, indx)
            batch = batch.unfreeze()

            deltas = self.transform_deltas[transformation_index]
            if "observations" in batch:
                obs = np.array(batch["observations"])
                obs = obs.copy()
                obs[..., self.x_obs_idx] += deltas[..., self.x_obs_idx]
                obs[..., self.y_obs_idx] += deltas[..., self.y_obs_idx]
                batch["observations"] = obs
            if "next_observations" in batch:
                n_obs = np.array(batch["next_observations"])
                n_obs = n_obs.copy()
                n_obs[..., self.x_obs_idx] += deltas[..., self.x_obs_idx]
                n_obs[..., self.y_obs_idx] += deltas[..., self.y_obs_idx]
                batch["next_observations"] = n_obs

            batch["transformation_index"] = transformation_index
            return frozen_dict.freeze(batch)

        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            assert "observations" in keys

        keys = list(keys)
        keys.remove("observations")
        if "transformation_index" in keys:
            keys.remove("transformation_index")

        batch = super().sample(batch_size, keys, indx)
        batch = batch.unfreeze()

        obs_keys = self.dataset_dict["observations"].keys()
        obs_keys = list(obs_keys)
        for k in self.img_keys:
            obs_keys.remove(k)

        batch["observations"] = {}
        for k in obs_keys:
            batch["observations"][k] = _sample(self.dataset_dict["observations"][k], indx)

        # State transform - obs and next_obs of one sample share one transform
        deltas = self.transform_deltas[transformation_index]
        if "state" in obs_keys:
            state = np.array(batch["observations"]["state"]).copy()
            state[..., self.x_obs_idx] += deltas[..., self.x_obs_idx]
            state[..., self.y_obs_idx] += deltas[..., self.y_obs_idx]
            batch["observations"]["state"] = state

            if "next_observations" in batch and "state" in batch["next_observations"]:
                n_state = np.array(batch["next_observations"]["state"]).copy()
                n_state[..., self.x_obs_idx] += deltas[..., self.x_obs_idx]
                n_state[..., self.y_obs_idx] += deltas[..., self.y_obs_idx]
                batch["next_observations"]["state"] = n_state

        # Image reconstruction via explicit circular gather (avoids the
        # non-circular sliding_window_view-at-negative-index bug near ring wraps)
        for k in self.img_keys:
            ptrs = self.dataset_dict["observations"][k][indx]  # (B,)
            window = (
                ptrs[:, None] - self._num_stack + np.arange(self._num_stack + 1)
            ) % self.img_buffer_size
            obs_imgs = self.img_buffer[k][window]  # (B, T+1, H, W, C)

            if (
                k in self.world_fixed_img_keys
                and self.front_map is not None
            ):
                B, Tp1 = obs_imgs.shape[:2]
                out = np.empty_like(obs_imgs)
                for i in range(B):
                    mx = self.front_map[transformation_index[i], 0]
                    my = self.front_map[transformation_index[i], 1]
                    for t in range(Tp1):
                        out[i, t] = cv2.remap(
                            obs_imgs[i, t],
                            mx,
                            my,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE,
                        )
                obs_imgs = out

            if pack_obs_and_next_obs:
                batch["observations"][k] = obs_imgs
            else:
                batch["observations"][k] = obs_imgs[:, :-1, ...]
                if "next_observations" in keys:
                    batch["next_observations"][k] = obs_imgs[:, 1:, ...]

        batch["transformation_index"] = transformation_index

        return frozen_dict.freeze(batch)
