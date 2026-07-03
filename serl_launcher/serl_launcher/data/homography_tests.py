"""Standalone tests for homography-aware fractal replay-buffer sampling.

Run from the repository root with:

    PYTHONPATH=serl_launcher \
        /home/bison/miniconda3/envs/serl/bin/python \
        serl_launcher/serl_launcher/data/homography_tests.py

The tests intentionally use the typical 0.3-meter workspace width. They cover
both K=9 and K=27 grid divisions per axis, corresponding to 81 and 729 spatial
transformations respectively. Image sizes are kept small so the K=27 dense-map
test remains inexpensive.
"""

import unittest

import gym
import numpy as np

from serl_launcher.data.fractal_symmetry_replay_buffer import (
    FractalSymmetryReplayBuffer,
)


WORKSPACE_WIDTH = 0.3
BRANCH_COUNTS = (9, 27)
DEFAULT_FRONT_M = object()


def pixel_scaled_homography(branch_count: int) -> np.ndarray:
    """Return a calibration for which adjacent grid cells differ by one pixel."""
    pixels_per_meter = branch_count / WORKSPACE_WIDTH
    return np.array(
        [
            [pixels_per_meter, 0.0, 0.0],
            [0.0, pixels_per_meter, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_observation_space(
    stack_size: int = 1,
    image_height: int = 8,
    image_width: int = 8,
    image_keys: tuple[str, ...] = ("front",),
) -> gym.spaces.Dict:
    """Construct the stacked state/image space expected by the replay buffer."""
    spaces = {
        "state": gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(stack_size, 2),
            dtype=np.float32,
        )
    }
    for key in image_keys:
        spaces[key] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(stack_size, image_height, image_width, 3),
            dtype=np.uint8,
        )
    return gym.spaces.Dict(spaces)


def make_buffer(
    branch_count: int = 9,
    capacity: int = 2,
    stack_size: int = 1,
    image_height: int = 8,
    image_width: int = 8,
    image_keys: tuple[str, ...] = ("front",),
    front_M=DEFAULT_FRONT_M,
    world_fixed_img_keys: tuple[str, ...] = ("front",),
    branch_method: str = "constant",
) -> FractalSymmetryReplayBuffer:
    """Build a small replay buffer with production-compatible observation shapes."""
    if front_M is DEFAULT_FRONT_M:
        front_M = (
            pixel_scaled_homography(branch_count)
            if world_fixed_img_keys
            else None
        )

    if branch_method == "constant":
        method_kwargs = {"starting_branch_count": branch_count}
    elif branch_method == "fractal":
        method_kwargs = {"max_depth": 1, "branching_factor": branch_count}
    else:
        raise ValueError(f"Unsupported test branch method: {branch_method}")

    return FractalSymmetryReplayBuffer(
        observation_space=make_observation_space(
            stack_size=stack_size,
            image_height=image_height,
            image_width=image_width,
            image_keys=image_keys,
        ),
        action_space=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        ),
        capacity=capacity,
        workspace_width=WORKSPACE_WIDTH,
        x_obs_idx=np.array([0]),
        y_obs_idx=np.array([1]),
        branch_method=branch_method,
        split_method="never",
        img_keys=list(image_keys),
        kwargs=method_kwargs,
        front_M=front_M,
        world_fixed_img_keys=world_fixed_img_keys,
    )


def make_transition(
    stack_size: int,
    image_height: int,
    image_width: int,
    image_keys: tuple[str, ...] = ("front",),
    state_value: float = 0.0,
    obs_images: dict[str, np.ndarray] | None = None,
    next_images: dict[str, np.ndarray] | None = None,
) -> dict:
    """Construct one transition with explicit stacked observations."""
    if obs_images is None:
        obs_images = {
            key: np.zeros(
                (stack_size, image_height, image_width, 3),
                dtype=np.uint8,
            )
            for key in image_keys
        }
    if next_images is None:
        next_images = {
            key: np.ones(
                (stack_size, image_height, image_width, 3),
                dtype=np.uint8,
            )
            for key in image_keys
        }

    observations = {
        "state": np.full(
            (stack_size, 2),
            state_value,
            dtype=np.float32,
        ),
        **obs_images,
    }
    next_observations = {
        "state": np.full(
            (stack_size, 2),
            state_value + 1.0,
            dtype=np.float32,
        ),
        **next_images,
    }
    return {
        "observations": observations,
        "next_observations": next_observations,
        "actions": np.array([state_value], dtype=np.float32),
        "rewards": np.float32(state_value),
        "masks": np.float32(1.0),
        "dones": np.bool_(False),
    }


class FixedIndexRng:
    """Minimal RNG interface that makes sample() select deterministic indices."""

    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)

    def integers(self, high: int, size: int | None = None):
        del high
        if size is None:
            return int(self.indices[0])
        if size != len(self.indices):
            raise ValueError(
                f"Requested {size} indices, configured {len(self.indices)}."
            )
        return self.indices.copy()


def sample_at(
    buffer: FractalSymmetryReplayBuffer,
    indices: np.ndarray,
):
    """Sample specified replay entries without changing production sample()."""
    previous_rng = buffer._np_random
    buffer._np_random = FixedIndexRng(indices)
    try:
        return buffer.sample(batch_size=len(indices))
    finally:
        buffer._np_random = previous_rng


class HomographyConfigurationTests(unittest.TestCase):
    def test_requires_matrix_and_keys_together(self):
        with self.assertRaisesRegex(ValueError, "both be configured"):
            make_buffer(
                branch_count=9,
                front_M=np.eye(3),
                world_fixed_img_keys=(),
            )

        with self.assertRaisesRegex(ValueError, "both be configured"):
            make_buffer(
                branch_count=9,
                front_M=None,
                world_fixed_img_keys=("front",),
            )

    def test_rejects_invalid_matrices(self):
        invalid_matrices = (
            np.eye(2),
            np.array(
                [[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]]
            ),
            np.zeros((3, 3)),
        )
        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                make_buffer(branch_count=9, front_M=matrix)

    def test_rejects_variable_branching_and_unknown_image_keys(self):
        with self.assertRaisesRegex(ValueError, "branch_method='constant'"):
            make_buffer(branch_count=9, branch_method="fractal")

        with self.assertRaisesRegex(ValueError, "missing keys"):
            make_buffer(
                branch_count=9,
                world_fixed_img_keys=("missing_camera",),
            )


class HomographyGeometryTests(unittest.TestCase):
    def test_translation_order_and_inverse_homographies_for_k9_and_k27(self):
        for branch_count in BRANCH_COUNTS:
            with self.subTest(branch_count=branch_count):
                buffer = make_buffer(
                    branch_count=branch_count,
                    image_height=4,
                    image_width=4,
                )

                axis_deltas = (
                    (np.arange(branch_count) + 0.5)
                    * WORKSPACE_WIDTH
                    / branch_count
                    - WORKSPACE_WIDTH / 2.0
                )
                transform_indices = np.arange(branch_count**2)
                x_indices, y_indices = np.divmod(
                    transform_indices,
                    branch_count,
                )
                expected_deltas = np.stack(
                    (
                        axis_deltas[x_indices],
                        axis_deltas[y_indices],
                    ),
                    axis=1,
                )
                np.testing.assert_allclose(
                    buffer.translation_deltas_xy,
                    expected_deltas,
                    atol=1e-12,
                )

                expected_inverse_translations = np.broadcast_to(
                    np.eye(3),
                    (branch_count**2, 3, 3),
                ).copy()
                expected_inverse_translations[:, 0, 2] = (
                    -expected_deltas[:, 0]
                )
                expected_inverse_translations[:, 1, 2] = (
                    -expected_deltas[:, 1]
                )
                M = buffer.front_M
                expected_inverse_homographies = (
                    M[None]
                    @ expected_inverse_translations
                    @ np.linalg.inv(M)[None]
                )
                np.testing.assert_allclose(
                    buffer.front_inverse_homographies,
                    expected_inverse_homographies,
                    atol=1e-12,
                )

    def test_dense_maps_for_k9_and_k27(self):
        image_height, image_width = 6, 7
        destination_x, destination_y = np.meshgrid(
            np.arange(image_width),
            np.arange(image_height),
        )

        for branch_count in BRANCH_COUNTS:
            with self.subTest(branch_count=branch_count):
                buffer = make_buffer(
                    branch_count=branch_count,
                    image_height=image_height,
                    image_width=image_width,
                )
                maps = buffer.front_maps["front"]
                self.assertEqual(
                    maps.shape,
                    (
                        branch_count**2,
                        image_height,
                        image_width,
                        2,
                    ),
                )
                self.assertEqual(maps.dtype, np.float32)
                self.assertTrue(np.all(np.isfinite(maps)))

                center = branch_count // 2
                center_transform = center * branch_count + center
                np.testing.assert_allclose(
                    maps[center_transform, ..., 0],
                    destination_x,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    maps[center_transform, ..., 1],
                    destination_y,
                    atol=1e-6,
                )

                # Moving one cell in +x is exactly one pixel under the test
                # calibration, so backward sampling reads destination_x - 1.
                right_transform = (center + 1) * branch_count + center
                np.testing.assert_allclose(
                    maps[right_transform, ..., 0],
                    destination_x - 1,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    maps[right_transform, ..., 1],
                    destination_y,
                    atol=1e-6,
                )


class HomographyReplayIntegrationTests(unittest.TestCase):
    def test_disabled_homography_preserves_base_sampling_behavior(self):
        height, width = 3, 4
        buffer = make_buffer(
            branch_count=9,
            capacity=1,
            image_height=height,
            image_width=width,
            front_M=None,
            world_fixed_img_keys=(),
        )
        obs_image = np.full((1, height, width, 3), 10, dtype=np.uint8)
        next_image = np.full((1, height, width, 3), 20, dtype=np.uint8)
        buffer.insert(
            make_transition(
                stack_size=1,
                image_height=height,
                image_width=width,
                obs_images={"front": obs_image},
                next_images={"front": next_image},
            )
        )

        batch = sample_at(buffer, np.array([0]))
        self.assertNotIn("transformation_index", buffer.dataset_dict)
        np.testing.assert_array_equal(
            batch["observations"]["front"][0],
            obs_image,
        )
        np.testing.assert_array_equal(
            batch["next_observations"]["front"][0],
            next_image,
        )

    def test_transformation_indices_and_state_alignment_for_k27(self):
        branch_count = 27
        buffer = make_buffer(
            branch_count=branch_count,
            capacity=1,
            image_height=2,
            image_width=2,
        )
        buffer.insert(
            make_transition(
                stack_size=1,
                image_height=2,
                image_width=2,
                state_value=5.0,
            )
        )

        num_transforms = branch_count**2
        self.assertEqual(len(buffer), num_transforms)
        np.testing.assert_array_equal(
            buffer.dataset_dict["transformation_index"][:num_transforms],
            np.arange(num_transforms, dtype=np.int32),
        )
        np.testing.assert_allclose(
            buffer.dataset_dict["observations"]["state"][
                :num_transforms, 0, 0
            ],
            5.0 + buffer.translation_deltas_xy[:, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            buffer.dataset_dict["observations"]["state"][
                :num_transforms, 0, 1
            ],
            5.0 + buffer.translation_deltas_xy[:, 1],
            atol=1e-6,
        )

    def test_circular_rollover_keeps_indices_and_states_aligned(self):
        branch_count = 9
        buffer = make_buffer(
            branch_count=branch_count,
            capacity=2,
            image_height=2,
            image_width=2,
        )
        for state_value in (10.0, 20.0, 30.0):
            buffer.insert(
                make_transition(
                    stack_size=1,
                    image_height=2,
                    image_width=2,
                    state_value=state_value,
                )
            )

        num_transforms = branch_count**2
        expected_indices = np.arange(num_transforms, dtype=np.int32)
        np.testing.assert_array_equal(
            buffer.dataset_dict["transformation_index"][:num_transforms],
            expected_indices,
        )
        np.testing.assert_array_equal(
            buffer.dataset_dict["transformation_index"][
                num_transforms : 2 * num_transforms
            ],
            expected_indices,
        )

        # The third insertion overwrites the first half of the ring; the second
        # insertion remains in the second half.
        for start, state_value in (
            (0, 30.0),
            (num_transforms, 20.0),
        ):
            stored_x = buffer.dataset_dict["observations"]["state"][
                start : start + num_transforms, 0, 0
            ]
            np.testing.assert_allclose(
                stored_x,
                state_value + buffer.translation_deltas_xy[:, 0],
                atol=1e-6,
            )

    def test_identity_sampling_and_public_batch_contract(self):
        height, width = 3, 4
        buffer = make_buffer(
            branch_count=1,
            capacity=2,
            image_height=height,
            image_width=width,
            front_M=np.eye(3),
        )
        obs_image = np.arange(
            height * width * 3,
            dtype=np.uint8,
        ).reshape(1, height, width, 3)
        next_image = (obs_image + 40).astype(np.uint8)
        buffer.insert(
            make_transition(
                stack_size=1,
                image_height=height,
                image_width=width,
                obs_images={"front": obs_image},
                next_images={"front": next_image},
            )
        )

        batch = sample_at(buffer, np.array([0]))
        np.testing.assert_array_equal(
            batch["observations"]["front"][0],
            obs_image,
        )
        np.testing.assert_array_equal(
            batch["next_observations"]["front"][0],
            next_image,
        )
        self.assertIn("transformation_index", buffer.dataset_dict)
        self.assertNotIn("transformation_index", batch)
        self.assertEqual(
            set(batch.keys()),
            {
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "masks",
                "dones",
            },
        )

    def test_known_front_translation_and_wrist_bypass(self):
        branch_count = 9
        height, width = 6, 8
        image_keys = ("front", "wrist")
        buffer = make_buffer(
            branch_count=branch_count,
            capacity=1,
            image_height=height,
            image_width=width,
            image_keys=image_keys,
            world_fixed_img_keys=("front",),
        )

        column_values = np.arange(width, dtype=np.uint8)[None, :, None]
        base_image = np.broadcast_to(
            column_values,
            (height, width, 3),
        ).copy()
        obs_images = {
            "front": base_image[None],
            "wrist": (base_image + 50)[None],
        }
        next_images = {
            "front": (base_image + 20)[None],
            "wrist": (base_image + 70)[None],
        }
        buffer.insert(
            make_transition(
                stack_size=1,
                image_height=height,
                image_width=width,
                image_keys=image_keys,
                obs_images=obs_images,
                next_images=next_images,
            )
        )

        center = branch_count // 2
        right_transform = (center + 1) * branch_count + center
        batch = sample_at(buffer, np.array([right_transform]))

        expected_front = np.concatenate(
            (base_image[:, :1], base_image[:, :-1]),
            axis=1,
        )
        np.testing.assert_array_equal(
            batch["observations"]["front"][0, 0],
            expected_front,
        )
        np.testing.assert_array_equal(
            batch["observations"]["wrist"][0],
            obs_images["wrist"],
        )

    def test_three_frame_temporal_stack_reconstruction(self):
        """A realistic T=3 transition should preserve its overlapping frames."""
        stack_size = 3
        height, width = 3, 4
        buffer = make_buffer(
            branch_count=1,
            capacity=2,
            stack_size=stack_size,
            image_height=height,
            image_width=width,
            front_M=np.eye(3),
        )

        def frame(value: int) -> np.ndarray:
            return np.full((height, width, 3), value, dtype=np.uint8)

        # This is the sequence produced by ChunkingWrapper after reset and one
        # environment step: [f0, f0, f0] -> [f0, f0, f1].
        obs_stack = np.stack((frame(10), frame(10), frame(10)))
        next_stack = np.stack((frame(10), frame(10), frame(20)))
        buffer.insert(
            make_transition(
                stack_size=stack_size,
                image_height=height,
                image_width=width,
                obs_images={"front": obs_stack},
                next_images={"front": next_stack},
            )
        )

        batch = sample_at(buffer, np.array([0]))
        np.testing.assert_array_equal(
            batch["observations"]["front"][0],
            obs_stack,
        )
        np.testing.assert_array_equal(
            batch["next_observations"]["front"][0],
            next_stack,
        )

        # Advance one more environment step:
        # [f0, f0, f1] -> [f0, f1, f2].
        second_obs_stack = next_stack
        second_next_stack = np.stack((frame(10), frame(20), frame(30)))
        buffer.insert(
            make_transition(
                stack_size=stack_size,
                image_height=height,
                image_width=width,
                obs_images={"front": second_obs_stack},
                next_images={"front": second_next_stack},
            )
        )

        second_batch = sample_at(buffer, np.array([1]))
        np.testing.assert_array_equal(
            second_batch["observations"]["front"][0],
            second_obs_stack,
        )
        np.testing.assert_array_equal(
            second_batch["next_observations"]["front"][0],
            second_next_stack,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
