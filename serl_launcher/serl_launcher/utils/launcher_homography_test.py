"""Tests for explicit homography configuration through replay launchers."""

import unittest
from unittest import mock

import numpy as np

from serl_launcher.data.data_store import (
    FractalSymmetryReplayBufferDataStore,
)
from serl_launcher.utils import launcher


class FakeEnvironment:
    """Minimal environment interface required by ``make_replay_buffer``."""

    observation_space = "observation-space"
    action_space = "action-space"


class ReplayBufferLauncherHomographyTest(unittest.TestCase):
    """Verify launcher defaults, validation, and datastore forwarding."""

    @mock.patch.object(launcher, "FractalSymmetryReplayBufferDataStore")
    def test_enabled_configuration_is_forwarded_explicitly(self, datastore):
        """Matrix and front key should reach the fractal datastore unchanged."""

        matrix = np.array(
            [
                [100.0, 0.0, 64.0],
                [0.0, 100.0, 64.0],
                [0.0, 0.0, 1.0],
            ]
        )
        expected_buffer = object()
        datastore.return_value = expected_buffer

        result = launcher.make_replay_buffer(
            FakeEnvironment(),
            type="fractal_symmetry_replay_buffer",
            capacity=10,
            image_keys=["front"],
            branch_method="constant",
            split_method="never",
            workspace_width=0.3,
            x_obs_idx=np.array([0]),
            y_obs_idx=np.array([1]),
            front_M=matrix,
            world_fixed_img_keys=("front",),
            starting_branch_count=1,
        )

        self.assertIs(result, expected_buffer)
        call_kwargs = datastore.call_args.kwargs
        self.assertIs(call_kwargs["front_M"], matrix)
        self.assertEqual(call_kwargs["world_fixed_img_keys"], ("front",))
        self.assertEqual(
            call_kwargs["kwargs"],
            {"starting_branch_count": 1},
        )

    @mock.patch.object(launcher, "FractalSymmetryReplayBufferDataStore")
    def test_disabled_defaults_are_forwarded(self, datastore):
        """Omitted homography arguments should preserve disabled behavior."""

        launcher.make_replay_buffer(
            FakeEnvironment(),
            type="fractal_symmetry_replay_buffer",
            branch_method="constant",
            split_method="never",
            workspace_width=0.3,
            x_obs_idx=np.array([0]),
            y_obs_idx=np.array([1]),
            starting_branch_count=1,
        )

        call_kwargs = datastore.call_args.kwargs
        self.assertIsNone(call_kwargs["front_M"])
        self.assertEqual(call_kwargs["world_fixed_img_keys"], ())

    def test_matrix_and_keys_are_required_together(self):
        """Partial homography configuration should fail before allocation."""

        with self.assertRaisesRegex(ValueError, "configured together"):
            launcher.make_replay_buffer(
                FakeEnvironment(),
                type="fractal_symmetry_replay_buffer",
                front_M=np.eye(3),
            )
        with self.assertRaisesRegex(ValueError, "configured together"):
            launcher.make_replay_buffer(
                FakeEnvironment(),
                type="fractal_symmetry_replay_buffer",
                world_fixed_img_keys=("front",),
            )

    def test_nonfractal_buffer_rejects_homography(self):
        """A calibrated matrix must not be silently ignored by other buffers."""

        with self.assertRaisesRegex(ValueError, "supported only"):
            launcher.make_replay_buffer(
                FakeEnvironment(),
                type="memory_efficient_replay_buffer",
                front_M=np.eye(3),
                world_fixed_img_keys=("front",),
            )


class FractalDatastoreHomographyTest(unittest.TestCase):
    """Verify the datastore forwards explicit fields to its replay buffer."""

    @mock.patch(
        "serl_launcher.data.data_store.DataStoreBase.__init__",
        return_value=None,
    )
    @mock.patch(
        "serl_launcher.data.data_store.FractalSymmetryReplayBuffer.__init__",
        return_value=None,
    )
    def test_datastore_forwards_matrix_and_keys(
        self,
        fractal_init,
        datastore_base_init,
    ):
        """Datastore construction should not bury calibration in ``kwargs``."""

        matrix = np.eye(3)
        FractalSymmetryReplayBufferDataStore(
            observation_space="observation-space",
            action_space="action-space",
            capacity=10,
            workspace_width=0.3,
            x_obs_idx=np.array([0]),
            y_obs_idx=np.array([1]),
            branch_method="constant",
            split_method="never",
            image_keys=("front",),
            front_M=matrix,
            world_fixed_img_keys=("front",),
            kwargs={"starting_branch_count": 1},
        )

        fractal_kwargs = fractal_init.call_args.kwargs
        self.assertIs(fractal_kwargs["front_M"], matrix)
        self.assertEqual(
            fractal_kwargs["world_fixed_img_keys"],
            ("front",),
        )
        self.assertEqual(
            fractal_kwargs["kwargs"],
            {"starting_branch_count": 1},
        )
        datastore_base_init.assert_called_once_with(
            mock.ANY,
            10,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
