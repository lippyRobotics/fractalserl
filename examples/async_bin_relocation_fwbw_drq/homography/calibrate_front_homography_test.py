"""Tests for front-camera point collection that do not require hardware."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import calibrate_front_homography as calibration


TEST_POSE = np.array([0.5, -0.1, 0.025, 0.0, 0.0, 0.0, 1.0])
KNOWN_HOMOGRAPHY = np.array(
    [
        [420.0, 35.0, 300.0],
        [-25.0, 310.0, 240.0],
        [0.4, -0.2, 1.0],
    ],
    dtype=np.float64,
)
WORLD_XY = np.array(
    [
        [-0.20, -0.15],
        [-0.20, 0.15],
        [0.00, -0.15],
        [0.00, 0.00],
        [0.00, 0.15],
        [0.20, -0.15],
        [0.20, 0.00],
        [0.20, 0.15],
    ],
    dtype=np.float64,
)


class CalibrationSessionTest(unittest.TestCase):
    """Verify resumable point collection and physical-plane checks."""

    def setUp(self):
        """Create a fresh small session for each test."""

        self.session = calibration.CalibrationSession.create(
            camera_serial="test-camera",
            raw_size_hw=(480, 640),
            target_count=4,
            plane_tolerance_m=0.002,
            nominal_z_m=None,
        )

    def test_first_point_selects_plane_and_undo_resets_it(self):
        """The first pose should select Z until its point is removed."""

        point = self.session.add_point(
            (100, 200),
            TEST_POSE,
            captured_at="2026-07-06T12:00:00+00:00",
        )
        self.assertEqual(self.session.nominal_z_m, TEST_POSE[2])
        np.testing.assert_allclose(point["world_xyz_m"], TEST_POSE[:3])

        removed = self.session.undo_last()
        self.assertEqual(removed, point)
        self.assertIsNone(self.session.nominal_z_m)
        self.assertIsNone(self.session.nominal_z_source)

    def test_undo_preserves_configured_plane(self):
        """Removing all points must retain a plane supplied by the operator."""

        session = calibration.CalibrationSession.create(
            camera_serial="test-camera",
            raw_size_hw=(480, 640),
            target_count=4,
            plane_tolerance_m=0.002,
            nominal_z_m=TEST_POSE[2],
        )
        session.add_point((100, 200), TEST_POSE)
        session.undo_last()
        self.assertEqual(session.nominal_z_m, TEST_POSE[2])
        self.assertEqual(session.nominal_z_source, "configured")

    def test_plane_and_pixel_bounds_are_enforced(self):
        """Out-of-plane robot poses and off-image clicks should not be saved."""

        self.session.add_point((100, 200), TEST_POSE)
        out_of_plane = TEST_POSE.copy()
        out_of_plane[2] += 0.003
        with self.assertRaisesRegex(ValueError, "outside the calibration plane"):
            self.session.add_point((200, 200), out_of_plane)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.session.add_point((640, 200), TEST_POSE)
        self.assertEqual(len(self.session.points), 1)

    def test_session_round_trip_and_resume_compatibility(self):
        """A saved session should reload only with matching hardware settings."""

        self.session.add_point((100, 200), TEST_POSE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.json"
            self.session.save(path)
            loaded = calibration.CalibrationSession.load(path)
            self.assertEqual(loaded.to_dict(), self.session.to_dict())
            loaded.assert_compatible(
                camera_serial="test-camera",
                raw_size_hw=(480, 640),
                target_count=4,
                plane_tolerance_m=0.002,
                nominal_z_m=None,
            )
            with self.assertRaisesRegex(ValueError, "camera serial"):
                loaded.assert_compatible(
                    camera_serial="other-camera",
                    raw_size_hw=(480, 640),
                    target_count=4,
                    plane_tolerance_m=0.002,
                    nominal_z_m=None,
                )

    def test_corrupt_world_position_is_rejected(self):
        """Saved XYZ must remain identical to the full pose position."""

        self.session.add_point((100, 200), TEST_POSE)
        data = copy.deepcopy(self.session.to_dict())
        data["points"][0]["world_xyz_m"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "must equal"):
            calibration.CalibrationSession.from_dict(data)


class PoseRequestTest(unittest.TestCase):
    """Verify the Franka HTTP contract without contacting a real server."""

    @mock.patch.object(calibration.requests, "post")
    def test_fetch_robot_pose_uses_post_getpos(self, post):
        """The pose helper should POST to /getpos and validate seven values."""

        response = post.return_value
        response.json.return_value = {"pose": TEST_POSE.tolist()}
        pose = calibration.fetch_robot_pose(
            "http://127.0.0.1:5000/",
            timeout_s=1.5,
        )
        post.assert_called_once_with(
            "http://127.0.0.1:5000/getpos",
            timeout=1.5,
        )
        response.raise_for_status.assert_called_once_with()
        np.testing.assert_allclose(pose, TEST_POSE)

    @mock.patch.object(calibration.requests, "post")
    def test_invalid_server_pose_is_rejected(self, post):
        """Malformed /getpos data should fail before entering the session."""

        post.return_value.json.return_value = {"pose": [1.0, 2.0]}
        with self.assertRaisesRegex(ValueError, "seven"):
            calibration.fetch_robot_pose(
                "http://127.0.0.1:5000",
                timeout_s=1.0,
            )


class SessionPreparationTest(unittest.TestCase):
    """Verify explicit overwrite and resume behavior around point files."""

    def test_existing_file_requires_explicit_mode(self):
        """Collection should never replace an existing session implicitly."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.json"
            session = calibration.prepare_session(
                path=path,
                resume=False,
                overwrite=False,
                camera_serial="test-camera",
                raw_size_hw=(480, 640),
                target_count=4,
                plane_tolerance_m=0.002,
                nominal_z_m=None,
            )
            self.assertTrue(path.exists())
            self.assertEqual(len(session.points), 0)

            with self.assertRaises(FileExistsError):
                calibration.prepare_session(
                    path=path,
                    resume=False,
                    overwrite=False,
                    camera_serial="test-camera",
                    raw_size_hw=(480, 640),
                    target_count=4,
                    plane_tolerance_m=0.002,
                    nominal_z_m=None,
                )

            resumed = calibration.prepare_session(
                path=path,
                resume=True,
                overwrite=False,
                camera_serial="test-camera",
                raw_size_hw=(480, 640),
                target_count=4,
                plane_tolerance_m=0.002,
                nominal_z_m=None,
            )
            self.assertEqual(resumed.to_dict(), session.to_dict())


class CalibrationFinalizationTest(unittest.TestCase):
    """Verify completed-session fitting and diagnostic output without a GUI."""

    def setUp(self):
        """Create a completed session generated by one known homography."""

        self.session = calibration.CalibrationSession.create(
            camera_serial="test-camera",
            raw_size_hw=(480, 640),
            target_count=len(WORLD_XY),
            plane_tolerance_m=0.002,
            nominal_z_m=0.025,
        )
        pixels = calibration.project_points(KNOWN_HOMOGRAPHY, WORLD_XY)
        for world_xy, pixel_uv in zip(WORLD_XY, pixels):
            pose = np.array(
                [
                    world_xy[0],
                    world_xy[1],
                    0.025,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            )
            self.session.add_point(pixel_uv, pose)

    def test_completed_session_builds_raw_and_target_matrices(self):
        """Fitting should recover M_raw and derive a valid 128x128 M_target."""

        artifact = calibration.build_calibration_artifact(
            self.session,
            target_size_hw=(128, 128),
            ransac_threshold_px=0.5,
        )
        np.testing.assert_allclose(
            artifact["homographies"]["M_raw"],
            KNOWN_HOMOGRAPHY,
            atol=2e-4,
        )
        self.assertEqual(
            artifact["image_geometry"]["target_size_hw"],
            [128, 128],
        )
        self.assertEqual(
            np.count_nonzero(artifact["fit"]["inlier_mask"]),
            len(WORLD_XY),
        )

    def test_incomplete_session_cannot_be_finalized(self):
        """A partial checkpoint should remain a session rather than an artifact."""

        incomplete = calibration.CalibrationSession.create(
            camera_serial="test-camera",
            raw_size_hw=(480, 640),
            target_count=4,
            plane_tolerance_m=0.002,
            nominal_z_m=0.025,
        )
        incomplete.add_point((100, 100), TEST_POSE)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            calibration.build_calibration_artifact(
                incomplete,
                target_size_hw=(128, 128),
                ransac_threshold_px=3.0,
            )

    def test_diagnostic_draws_raw_resolution_residuals(self):
        """Diagnostic rendering should preserve size and add visible overlays."""

        artifact = calibration.build_calibration_artifact(
            self.session,
            target_size_hw=(128, 128),
            ransac_threshold_px=0.5,
        )
        background = np.zeros((480, 640, 3), dtype=np.uint8)
        diagnostic = calibration.draw_calibration_diagnostic(
            artifact,
            background=background,
        )
        self.assertEqual(diagnostic.shape, background.shape)
        self.assertEqual(diagnostic.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(diagnostic), 0)
        self.assertIn("M_raw", calibration.format_calibration_report(artifact))

    def test_outputs_require_explicit_overwrite(self):
        """Existing final outputs should not be replaced without permission."""

        artifact = calibration.build_calibration_artifact(
            self.session,
            target_size_hw=(128, 128),
            ransac_threshold_px=0.5,
        )
        diagnostic = calibration.draw_calibration_diagnostic(artifact)
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "front_homography.json"
            diagnostic_path = Path(directory) / "diagnostic.png"
            calibration.save_calibration_outputs(
                artifact=artifact,
                diagnostic=diagnostic,
                artifact_path=artifact_path,
                diagnostic_path=diagnostic_path,
                overwrite=False,
            )
            self.assertTrue(artifact_path.exists())
            self.assertTrue(diagnostic_path.exists())

            with self.assertRaises(FileExistsError):
                calibration.save_calibration_outputs(
                    artifact=artifact,
                    diagnostic=diagnostic,
                    artifact_path=artifact_path,
                    diagnostic_path=diagnostic_path,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
