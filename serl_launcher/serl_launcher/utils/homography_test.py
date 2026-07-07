"""Unit tests for hardware-independent homography utilities."""

import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np

from serl_launcher.utils.homography import (
    convert_world_homography_to_relative,
    create_calibration_artifact,
    fit_planar_homography,
    load_calibration_artifact,
    opencv_resize_transform,
    project_points,
    relative_xy_to_world_xy_from_reset_transform,
    resize_homography,
    save_calibration_artifact,
    validate_calibration_artifact,
    validate_homography,
)


KNOWN_HOMOGRAPHY = np.array(
    [
        [420.0, 35.0, 80.0],
        [-25.0, 310.0, 55.0],
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


class HomographyGeometryTest(unittest.TestCase):
    """Verify fitting and projection behavior without robot hardware."""

    def test_projection_and_exact_fit(self):
        """An exact synthetic mapping should be recovered within roundoff."""

        pixels = project_points(KNOWN_HOMOGRAPHY, WORLD_XY)
        fit = fit_planar_homography(
            WORLD_XY,
            pixels,
            ransac_threshold_px=0.1,
        )

        np.testing.assert_allclose(fit.matrix, KNOWN_HOMOGRAPHY, atol=1e-7)
        np.testing.assert_array_equal(
            fit.inlier_mask,
            np.ones(WORLD_XY.shape[0], dtype=bool),
        )
        np.testing.assert_allclose(fit.reprojection_errors_px, 0.0, atol=1e-5)
        self.assertLess(fit.inlier_rmse_px, 1e-5)
        self.assertLess(fit.all_points_max_error_px, 1e-5)

    def test_ransac_rejects_obvious_outlier(self):
        """RANSAC should mark one deliberately incorrect click as an outlier."""

        pixels = project_points(KNOWN_HOMOGRAPHY, WORLD_XY)
        pixels[3] += np.array([90.0, -75.0])

        fit = fit_planar_homography(
            WORLD_XY,
            pixels,
            ransac_threshold_px=1.0,
        )

        self.assertFalse(fit.inlier_mask[3])
        self.assertGreater(fit.reprojection_errors_px[3], 50.0)
        self.assertLess(fit.inlier_rmse_px, 1e-5)
        self.assertGreater(
            fit.all_points_rmse_px,
            fit.inlier_rmse_px,
        )

    def test_degenerate_correspondences_are_rejected(self):
        """Collinear points should fail because they cannot constrain a plane."""

        collinear = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        )
        with self.assertRaisesRegex(ValueError, "collinear"):
            fit_planar_homography(collinear, collinear)

    def test_invalid_homographies_are_rejected(self):
        """Wrong-shaped, non-finite, and singular matrices should fail early."""

        with self.assertRaisesRegex(ValueError, "shape"):
            validate_homography(np.eye(4))
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid = np.eye(3)
            invalid[0, 0] = np.nan
            validate_homography(invalid)
        with self.assertRaisesRegex(ValueError, "invertible"):
            validate_homography(np.zeros((3, 3)))

    def test_opencv_pixel_center_resize_conversion(self):
        """Raw projections should follow OpenCV pixel centers after resizing."""

        source_size = (480, 640)
        target_size = (128, 128)
        resize_transform = opencv_resize_transform(
            source_size,
            target_size,
        )
        expected = np.array(
            [
                [0.2, 0.0, -0.4],
                [0.0, 128.0 / 480.0, (128.0 / 480.0 - 1.0) / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(resize_transform, expected)

        target_homography = resize_homography(
            KNOWN_HOMOGRAPHY,
            source_size,
            target_size,
        )
        raw_pixels = project_points(KNOWN_HOMOGRAPHY, WORLD_XY)
        expected_target_pixels = (
            raw_pixels + np.array([0.5, 0.5])
        ) * np.array([0.2, 128.0 / 480.0]) - np.array([0.5, 0.5])
        np.testing.assert_allclose(
            project_points(target_homography, WORLD_XY),
            expected_target_pixels,
            atol=1e-10,
        )


class RelativeFrameHomographyTest(unittest.TestCase):
    """Verify conversion between world-frame and RelativeFrame homographies."""

    def test_rx_pi_reset_basis_flips_relative_y(self):
        """A reset x-rotation by pi maps relative +y to world -y."""

        reset_transform = np.array(
            [
                [1.0, 0.0, 0.0, 0.55],
                [0.0, -1.0, 0.0, 0.025],
                [0.0, 0.0, -1.0, 0.10],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        relativeXY_to_worldXY = relative_xy_to_world_xy_from_reset_transform(
            reset_transform
        )

        np.testing.assert_allclose(
            relativeXY_to_worldXY,
            np.array([[1.0, 0.0], [0.0, -1.0]]),
        )

        relative_points = np.array(
            [
                [0.00, 0.00],
                [0.02, 0.00],
                [0.00, 0.03],
                [-0.04, -0.02],
            ],
            dtype=np.float64,
        )
        origin_world_xy = reset_transform[:2, 3]
        world_points = relative_points @ relativeXY_to_worldXY.T + origin_world_xy

        relative_matrix = convert_world_homography_to_relative(
            KNOWN_HOMOGRAPHY,
            relativeXY_to_worldXY,
            relative_origin_world_xy=origin_world_xy,
        )

        np.testing.assert_allclose(
            project_points(relative_matrix, relative_points),
            project_points(KNOWN_HOMOGRAPHY, world_points),
            atol=1e-10,
        )

    def test_identity_basis_keeps_world_homography(self):
        """With identical coordinate bases, the converted matrix is unchanged."""

        relative_matrix = convert_world_homography_to_relative(
            KNOWN_HOMOGRAPHY,
            np.eye(2),
        )

        np.testing.assert_allclose(relative_matrix, KNOWN_HOMOGRAPHY)

    def test_general_planar_rotation_basis_projects_same_pixels(self):
        """A nontrivial planar basis should still produce equivalent pixels."""

        theta = np.deg2rad(30.0)
        relativeXY_to_worldXY = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float64,
        )
        origin_world_xy = np.array([0.40, -0.12], dtype=np.float64)
        relative_points = np.array(
            [
                [0.00, 0.00],
                [0.03, 0.01],
                [-0.02, 0.04],
                [0.05, -0.03],
            ],
            dtype=np.float64,
        )
        world_points = relative_points @ relativeXY_to_worldXY.T + origin_world_xy

        relative_matrix = convert_world_homography_to_relative(
            KNOWN_HOMOGRAPHY,
            relativeXY_to_worldXY,
            relative_origin_world_xy=origin_world_xy,
        )

        np.testing.assert_allclose(
            project_points(relative_matrix, relative_points),
            project_points(KNOWN_HOMOGRAPHY, world_points),
            atol=1e-10,
        )

    def test_relative_origin_is_optional_for_translation_conjugation(self):
        """Translation image homographies do not depend on relative origin."""

        relativeXY_to_worldXY = np.array([[1.0, 0.0], [0.0, -1.0]])
        delta_relative = np.array([0.02, -0.03], dtype=np.float64)

        relative_matrix_without_origin = convert_world_homography_to_relative(
            KNOWN_HOMOGRAPHY,
            relativeXY_to_worldXY,
        )
        relative_matrix_with_origin = convert_world_homography_to_relative(
            KNOWN_HOMOGRAPHY,
            relativeXY_to_worldXY,
            relative_origin_world_xy=np.array([0.55, 0.025]),
        )

        translation = np.eye(3, dtype=np.float64)
        translation[:2, 2] = delta_relative
        image_homography_without_origin = (
            relative_matrix_without_origin
            @ translation
            @ np.linalg.inv(relative_matrix_without_origin)
        )
        image_homography_with_origin = (
            relative_matrix_with_origin
            @ translation
            @ np.linalg.inv(relative_matrix_with_origin)
        )

        np.testing.assert_allclose(
            image_homography_without_origin,
            image_homography_with_origin,
            atol=1e-10,
        )

    def test_tilted_reset_transform_is_rejected(self):
        """Relative xy axes with world-z leakage cannot use one xy homography."""

        reset_transform = np.eye(4, dtype=np.float64)
        reset_transform[2, 0] = 0.01

        with self.assertRaisesRegex(ValueError, "not planar enough"):
            relative_xy_to_world_xy_from_reset_transform(
                reset_transform,
                planar_tolerance=1e-4,
            )

    def test_invalid_relative_basis_is_rejected(self):
        """The basis must be finite, 2x2, and invertible."""

        with self.assertRaisesRegex(ValueError, "shape"):
            convert_world_homography_to_relative(KNOWN_HOMOGRAPHY, np.eye(3))
        with self.assertRaisesRegex(ValueError, "finite"):
            convert_world_homography_to_relative(
                KNOWN_HOMOGRAPHY,
                np.array([[1.0, np.nan], [0.0, 1.0]]),
            )
        with self.assertRaisesRegex(ValueError, "invertible"):
            convert_world_homography_to_relative(
                KNOWN_HOMOGRAPHY,
                np.array([[1.0, 0.0], [2.0, 0.0]]),
            )


class CalibrationArtifactTest(unittest.TestCase):
    """Verify creation, persistence, and validation of calibration JSON."""

    def setUp(self):
        """Create one valid synthetic artifact for each validation test."""

        pixels = project_points(KNOWN_HOMOGRAPHY, WORLD_XY)
        world_xyz = np.concatenate(
            (
                WORLD_XY,
                np.full((WORLD_XY.shape[0], 1), 0.025),
            ),
            axis=1,
        )
        self.artifact = create_calibration_artifact(
            camera_serial="front-camera-test",
            world_xyz_m=world_xyz,
            pixels_uv_raw=pixels,
            raw_size_hw=(480, 640),
            target_size_hw=(128, 128),
            nominal_z_m=0.025,
            plane_tolerance_m=0.001,
            ransac_threshold_px=0.5,
            created_at="2026-07-06T12:00:00+00:00",
        )

    def test_artifact_creation_and_validation(self):
        """Creation should preserve the known matrix and aligned point data."""

        validate_calibration_artifact(self.artifact)
        np.testing.assert_allclose(
            self.artifact["homographies"]["M_raw"],
            KNOWN_HOMOGRAPHY,
            atol=1e-7,
        )
        self.assertEqual(
            len(self.artifact["fit"]["inlier_mask"]),
            len(self.artifact["points"]),
        )

    def test_json_round_trip_and_compatibility_checks(self):
        """Saved JSON should reload and reject unexpected camera hardware."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "front_homography.json"
            save_calibration_artifact(path, self.artifact)
            loaded = load_calibration_artifact(
                path,
                expected_camera_serial="front-camera-test",
                expected_raw_size_hw=(480, 640),
                expected_target_size_hw=(128, 128),
            )
            self.assertEqual(loaded, self.artifact)

            with self.assertRaisesRegex(ValueError, "camera serial"):
                load_calibration_artifact(
                    path,
                    expected_camera_serial="different-camera",
                )
            loaded_override = load_calibration_artifact(
                path,
                expected_camera_serial="different-camera",
                allow_incompatible=True,
            )
            self.assertEqual(loaded_override, self.artifact)

    def test_inconsistent_errors_are_rejected(self):
        """A stored error that disagrees with the matrix should be rejected."""

        invalid = copy.deepcopy(self.artifact)
        invalid["fit"]["reprojection_errors_px"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "reprojection_errors_px"):
            validate_calibration_artifact(invalid)

    def test_inconsistent_target_matrix_is_rejected(self):
        """M_target must equal the resized form of the stored raw matrix."""

        invalid = copy.deepcopy(self.artifact)
        invalid["homographies"]["M_target"][0][2] += 1.0
        with self.assertRaisesRegex(ValueError, "M_target resize conversion"):
            validate_calibration_artifact(invalid)

    def test_boolean_schema_version_is_rejected(self):
        """Boolean true must not be mistaken for integer schema version one."""

        invalid = copy.deepcopy(self.artifact)
        invalid["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "schema_version"):
            validate_calibration_artifact(invalid)

    def test_plane_tolerance_is_enforced(self):
        """Points too far from the selected physical plane should be rejected."""

        pixels = project_points(KNOWN_HOMOGRAPHY, WORLD_XY)
        world_xyz = np.concatenate(
            (WORLD_XY, np.full((WORLD_XY.shape[0], 1), 0.025)),
            axis=1,
        )
        world_xyz[-1, 2] = 0.030
        with self.assertRaisesRegex(ValueError, "plane tolerance"):
            create_calibration_artifact(
                camera_serial="front-camera-test",
                world_xyz_m=world_xyz,
                pixels_uv_raw=pixels,
                raw_size_hw=(480, 640),
                target_size_hw=(128, 128),
                nominal_z_m=0.025,
                plane_tolerance_m=0.001,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
