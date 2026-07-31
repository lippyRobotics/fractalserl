"""Utilities for fitting and storing planar camera homographies.

This module is intentionally independent of robot and camera hardware. World
points use robot-base ``(X, Y)`` coordinates in meters, while image points use
OpenCV ``(u, v)`` pixel coordinates.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


SCHEMA_VERSION = 1
COORDINATE_FRAME = "robot_base_xy"
PIXEL_CONVENTION = "upper_left_pixel_center_uv"
RESIZE_METHOD = "opencv_pixel_center"
FIT_METHOD = "opencv_find_homography_ransac"

PathLike = Union[str, os.PathLike]
ImageSize = Tuple[int, int]


@dataclass(frozen=True)
class HomographyFit:
    """Result of robustly fitting workspace-plane points to image pixels."""

    matrix: np.ndarray
    inlier_mask: np.ndarray
    reprojection_errors_px: np.ndarray
    inlier_rmse_px: float
    inlier_max_error_px: float
    all_points_rmse_px: float
    all_points_max_error_px: float


def _as_points(
    points: Any,
    *,
    name: str,
    dimensions: int,
    minimum_count: int = 1,
) -> np.ndarray:
    """Convert a collection of points to a checked floating-point array.

    A point collection is represented as one row per point and one column per
    coordinate. For example, five image points have shape ``(5, 2)`` because
    each row contains ``(u, v)``.

    Args:
        points: Array-like point data supplied by the caller.
        name: Human-readable name included in validation errors.
        dimensions: Required number of coordinates in each point.
        minimum_count: Smallest permitted number of points.

    Returns:
        A finite ``float64`` NumPy array with shape ``(N, dimensions)``.

    Raises:
        ValueError: If the shape, count, or numeric values are invalid.
    """

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != dimensions:
        raise ValueError(
            f"{name} must have shape (N, {dimensions}), got {array.shape}."
        )
    if array.shape[0] < minimum_count:
        raise ValueError(
            f"{name} must contain at least {minimum_count} points, "
            f"got {array.shape[0]}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_correspondences(
    world_xy: Any,
    pixels_uv: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Check that world points and clicked pixels can define a homography.

    Each world point must have exactly one corresponding image point. Four
    unique, non-collinear pairs are the mathematical minimum for fitting a
    planar homography.

    Args:
        world_xy: Robot-base plane coordinates with shape ``(N, 2)``.
        pixels_uv: Raw image coordinates with shape ``(N, 2)``.

    Returns:
        The validated ``(world_xy, pixels_uv)`` arrays as ``float64``.

    Raises:
        ValueError: If counts differ or either point layout is degenerate.
    """

    world = _as_points(
        world_xy,
        name="world_xy",
        dimensions=2,
        minimum_count=4,
    )
    pixels = _as_points(
        pixels_uv,
        name="pixels_uv",
        dimensions=2,
        minimum_count=4,
    )
    if world.shape[0] != pixels.shape[0]:
        raise ValueError(
            "world_xy and pixels_uv must contain the same number of points, "
            f"got {world.shape[0]} and {pixels.shape[0]}."
        )

    for name, points in (("world_xy", world), ("pixels_uv", pixels)):
        if np.unique(points, axis=0).shape[0] < 4:
            raise ValueError(f"{name} must contain at least four unique points.")
        centered = points - np.mean(points, axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered) < 2:
            raise ValueError(f"{name} points must not be collinear.")

    return world, pixels


def validate_homography(matrix: Any, *, normalize: bool = True) -> np.ndarray:
    """Validate an invertible 3x3 homography and optionally normalize it.

    Homographies are defined only up to a nonzero scale: ``M`` and ``2 * M``
    describe the same projection. Normalization removes this ambiguity,
    usually by making the bottom-right entry equal to one.

    Args:
        matrix: Array-like candidate homography.
        normalize: Whether to return a consistently scaled matrix.

    Returns:
        A validated ``float64`` matrix with shape ``(3, 3)``.

    Raises:
        ValueError: If the matrix has the wrong shape, contains invalid
            numbers, is singular, or cannot be normalized safely.
    """

    homography = np.asarray(matrix, dtype=np.float64)
    if homography.shape != (3, 3):
        raise ValueError(
            f"Homography must have shape (3, 3), got {homography.shape}."
        )
    if not np.all(np.isfinite(homography)):
        raise ValueError("Homography must contain only finite values.")
    if np.linalg.matrix_rank(homography) < 3:
        raise ValueError("Homography must be invertible.")

    if not normalize:
        return homography.copy()

    scale = homography[2, 2]
    scale_tolerance = (
        np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(homography))))
    )
    if abs(scale) <= scale_tolerance:
        scale = float(np.linalg.norm(homography))
    if not np.isfinite(scale) or abs(scale) <= scale_tolerance:
        raise ValueError("Homography cannot be normalized safely.")
    return homography / scale


def project_points(matrix: Any, world_xy: Any) -> np.ndarray:
    """Project planar world points through a homography into image pixels.

    The function appends the homogeneous coordinate ``1`` to every ``(X, Y)``
    point, multiplies by the matrix, and divides by the resulting third
    component. That final division converts homogeneous coordinates back to
    ordinary ``(u, v)`` pixel coordinates.

    Args:
        matrix: Homography mapping robot-base plane coordinates to pixels.
        world_xy: Plane coordinates with shape ``(N, 2)``.

    Returns:
        Projected pixel coordinates with shape ``(N, 2)``.

    Raises:
        ValueError: If the inputs are invalid or a point projects to infinity.
    """

    homography = validate_homography(matrix)
    world = _as_points(
        world_xy,
        name="world_xy",
        dimensions=2,
    )
    homogeneous_world = np.concatenate(
        (world, np.ones((world.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    homogeneous_pixels = (homography @ homogeneous_world.T).T
    denominators = homogeneous_pixels[:, 2]
    tolerance = np.finfo(np.float64).eps * np.maximum(
        1.0,
        np.max(np.abs(homogeneous_pixels), axis=1),
    )
    if np.any(np.abs(denominators) <= tolerance):
        invalid = np.flatnonzero(np.abs(denominators) <= tolerance)
        raise ValueError(
            "Homography projects points to infinity at indices "
            f"{invalid.tolist()}."
        )
    return homogeneous_pixels[:, :2] / denominators[:, None]


def reprojection_errors(
    matrix: Any,
    world_xy: Any,
    pixels_uv: Any,
) -> np.ndarray:
    """Measure how far each projected point is from its recorded click.

    Reprojection error is ordinary two-dimensional Euclidean distance in
    pixels. An error of ``3`` means the prediction is three pixels away from
    the manually selected image location.

    Args:
        matrix: World-plane-to-image homography to evaluate.
        world_xy: Robot-base plane coordinates with shape ``(N, 2)``.
        pixels_uv: Recorded image coordinates with shape ``(N, 2)``.

    Returns:
        A one-dimensional array containing one pixel error per point.

    Raises:
        ValueError: If the point arrays have invalid or unequal sizes.
    """

    world = _as_points(world_xy, name="world_xy", dimensions=2)
    pixels = _as_points(pixels_uv, name="pixels_uv", dimensions=2)
    if world.shape[0] != pixels.shape[0]:
        raise ValueError(
            "world_xy and pixels_uv must contain the same number of points."
        )
    projected = project_points(matrix, world)
    return np.linalg.norm(projected - pixels, axis=1)


def _error_summary(errors: np.ndarray) -> Tuple[float, float]:
    """Calculate root-mean-square and maximum error for one point group.

    Args:
        errors: One-dimensional collection of nonempty reprojection errors.

    Returns:
        A pair ``(rmse, maximum)`` expressed in pixels.

    Raises:
        ValueError: If no errors are provided.
    """

    if errors.size == 0:
        raise ValueError("At least one reprojection error is required.")
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    maximum = float(np.max(errors))
    return rmse, maximum


def fit_planar_homography(
    world_xy: Any,
    pixels_uv: Any,
    *,
    ransac_threshold_px: float = 3.0,
    max_iterations: int = 2000,
    confidence: float = 0.995,
) -> HomographyFit:
    """Fit a robust world-plane-to-image homography with OpenCV RANSAC.

    RANSAC repeatedly fits candidate matrices from small point subsets. A
    recorded point is marked as an inlier when its reprojection error is close
    enough to the candidate matrix. This makes the result less sensitive to an
    occasional incorrect click or mismatched robot pose.

    Args:
        world_xy: Robot-base ``(X, Y)`` coordinates with shape ``(N, 2)``.
        pixels_uv: Matching raw image ``(u, v)`` coordinates.
        ransac_threshold_px: Maximum pixel error used by RANSAC when deciding
            whether a point supports a candidate matrix.
        max_iterations: Maximum number of candidate-fitting attempts.
        confidence: Desired probability that RANSAC finds a valid model.

    Returns:
        A :class:`HomographyFit` containing the normalized matrix, Boolean
        inlier mask, individual errors, and summary statistics.

    Raises:
        ValueError: If parameters or correspondences are invalid, or OpenCV
            cannot find a matrix supported by at least four inliers.
    """

    world, pixels = _validate_correspondences(world_xy, pixels_uv)
    if (
        isinstance(ransac_threshold_px, bool)
        or not np.isfinite(ransac_threshold_px)
        or ransac_threshold_px <= 0
    ):
        raise ValueError("ransac_threshold_px must be a positive finite number.")
    if isinstance(max_iterations, bool) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise ValueError("max_iterations must be a positive integer.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    if (
        isinstance(confidence, bool)
        or not np.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be a finite number between 0 and 1.")

    matrix, mask = cv2.findHomography(
        world,
        pixels,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_px),
        maxIters=int(max_iterations),
        confidence=float(confidence),
    )
    if matrix is None or mask is None:
        raise ValueError("OpenCV could not fit a homography to the points.")

    matrix = validate_homography(matrix)
    inlier_mask = np.asarray(mask, dtype=bool).reshape(-1)
    if inlier_mask.shape != (world.shape[0],):
        raise ValueError(
            "OpenCV returned an inlier mask with an unexpected shape "
            f"{np.asarray(mask).shape}."
        )
    if np.count_nonzero(inlier_mask) < 4:
        raise ValueError("RANSAC found fewer than four inlier correspondences.")

    errors = reprojection_errors(matrix, world, pixels)
    inlier_rmse, inlier_max = _error_summary(errors[inlier_mask])
    all_rmse, all_max = _error_summary(errors)
    return HomographyFit(
        matrix=matrix,
        inlier_mask=inlier_mask,
        reprojection_errors_px=errors,
        inlier_rmse_px=inlier_rmse,
        inlier_max_error_px=inlier_max,
        all_points_rmse_px=all_rmse,
        all_points_max_error_px=all_max,
    )


def _validate_image_size(size_hw: Any, *, name: str) -> ImageSize:
    """Validate an image size written in ``(height, width)`` order.

    Args:
        size_hw: Two positive integer dimensions.
        name: Human-readable name included in validation errors.

    Returns:
        The dimensions as a standard Python ``(height, width)`` tuple.

    Raises:
        ValueError: If either dimension is missing, non-integer, or nonpositive.
    """

    if (
        not isinstance(size_hw, Sequence)
        or isinstance(size_hw, (str, bytes))
        or len(size_hw) != 2
    ):
        raise ValueError(f"{name} must be a two-element (height, width) size.")
    height, width = size_hw
    for dimension_name, value in (("height", height), ("width", width)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} {dimension_name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} {dimension_name} must be positive.")
    return int(height), int(width)


def opencv_resize_transform(
    source_size_hw: Sequence[int],
    target_size_hw: Sequence[int],
) -> np.ndarray:
    """Return the source-to-target pixel transform used by ``cv2.resize``.

    OpenCV aligns pixel centers according to
    ``target = scale * (source + 0.5) - 0.5``.

    Args:
        source_size_hw: Original ``(height, width)``.
        target_size_hw: Resized ``(height, width)``.

    Returns:
        A 3x3 affine matrix mapping source pixel centers to target pixel
        centers.

    Raises:
        ValueError: If either image size is invalid.
    """

    source_height, source_width = _validate_image_size(
        source_size_hw,
        name="source_size_hw",
    )
    target_height, target_width = _validate_image_size(
        target_size_hw,
        name="target_size_hw",
    )
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return np.array(
        [
            [scale_x, 0.0, (scale_x - 1.0) / 2.0],
            [0.0, scale_y, (scale_y - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def resize_homography(
    matrix: Any,
    source_size_hw: Sequence[int],
    target_size_hw: Sequence[int],
) -> np.ndarray:
    """Convert a homography from source pixels to resized target pixels.

    This left-multiplies the calibration matrix by the OpenCV resize transform.
    For this project, it converts ``M_raw`` for the 640x480 camera frame into
    ``M_target`` for the 128x128 replay-buffer frame.

    Args:
        matrix: Homography whose output uses the source image coordinates.
        source_size_hw: Original image size in ``(height, width)`` order.
        target_size_hw: Desired image size in ``(height, width)`` order.

    Returns:
        A normalized homography whose output uses target image coordinates.

    Raises:
        ValueError: If the matrix or either image size is invalid.
    """

    resize_transform = opencv_resize_transform(
        source_size_hw,
        target_size_hw,
    )
    return validate_homography(resize_transform @ validate_homography(matrix))


def _validate_relative_xy_to_world_xy(relativeXY_to_worldXY: Any) -> np.ndarray:
    """Validate the 2D basis that maps relative-frame xy into world-frame xy.

    Args:
        relativeXY_to_worldXY: Candidate ``2x2`` matrix. It is interpreted as
            ``world_xy = relativeXY_to_worldXY @ relative_xy`` for column
            vectors.

    Returns:
        The validated matrix as ``float64``.

    Raises:
        ValueError: If the matrix is the wrong shape, non-finite, or singular.
    """

    basis = np.asarray(relativeXY_to_worldXY, dtype=np.float64)
    if basis.shape != (2, 2):
        raise ValueError(
            "relativeXY_to_worldXY must have shape (2, 2), "
            f"got {basis.shape}."
        )
    if not np.all(np.isfinite(basis)):
        raise ValueError("relativeXY_to_worldXY must contain only finite values.")
    if np.linalg.matrix_rank(basis) < 2:
        raise ValueError("relativeXY_to_worldXY must be invertible.")
    return basis


def _validate_optional_xy(value: Optional[Any], *, name: str) -> np.ndarray:
    """Validate an optional two-element xy vector.

    Args:
        value: Optional candidate xy vector. ``None`` means ``[0, 0]``.
        name: Human-readable field name for validation errors.

    Returns:
        A finite ``float64`` vector with shape ``(2,)``.

    Raises:
        ValueError: If the supplied value is not a finite two-vector.
    """

    if value is None:
        return np.zeros(2, dtype=np.float64)
    xy = np.asarray(value, dtype=np.float64)
    if xy.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {xy.shape}.")
    if not np.all(np.isfinite(xy)):
        raise ValueError(f"{name} must contain only finite values.")
    return xy


def relative_xy_to_world_xy_from_reset_transform(
    reset_transform: Any,
    *,
    planar_tolerance: float = 1e-6,
) -> np.ndarray:
    """Extract the relative-frame xy basis used by ``RelativeFrame``.

    ``RelativeFrame`` is important because it changes the coordinate system
    seen by the policy and replay buffer. Calibration points collected from
    Franka server ``/getpos`` are in the robot base/world frame, but
    ``RelativeFrame`` rewrites ``obs["state"]["tcp_pose"]`` so positions are
    expressed relative to the TCP pose observed immediately after reset.

    The wrapper does not directly read ``TARGET_POSE``, ``RESET_POSE``, or a
    reset-pose attribute. Those values affect it indirectly: the task env uses
    them to move the robot during ``env.reset()``, then ``RelativeFrame`` uses
    the actual post-reset TCP pose as the relative frame. In this repo the
    persistent reset-pose attribute is named ``self.resetpos``; if a future env
    uses a name like ``self._reset_pose``, it plays the same indirect role.

    ``reset_transform`` should be the post-reset TCP pose written as a
    homogeneous matrix, i.e. ``T_world_reset_tcp``. It represents the reset TCP
    frame's origin and axes expressed in the robot base/world frame. When used
    for coordinate multiplication, it maps coordinates from the reset TCP frame
    into base/world coordinates:

    ``p_world = T_world_reset_tcp @ p_reset_tcp``.

    Its inverse performs the opposite reference-frame change:

    ``p_reset_tcp = inv(T_world_reset_tcp) @ p_world``.

    This function needs ``T_world_reset_tcp`` rather than its inverse because it
    extracts ``relativeXY_to_worldXY``. This is the matrix returned by
    ``construct_homogeneous_matrix(obs["state"]["tcp_pose"])`` before
    ``RelativeFrame`` rewrites the observation.

    For the cleaned bin-relocation reset orientation ``[pi, 0, 0]``, the
    current Franka pose convention yields an ``Rx(pi)`` basis:

    ``relative +x -> world +x``
    ``relative +y -> world -y``

    so the extracted planar matrix is approximately ``[[1, 0], [0, -1]]``.
    If random yaw is enabled, this matrix becomes a true 2D rotation/reflection
    instead of only a y sign flip.

    Args:
        reset_transform: Homogeneous ``4x4`` post-reset TCP pose matrix
            ``T_world_reset_tcp``.
        planar_tolerance: Maximum allowed magnitude of the world-z component
            in the relative x/y axes. Nonzero z leakage means planar relative
            xy shifts are not exactly representable by a homography calibrated
            on one constant-z world plane.

    Returns:
        ``relativeXY_to_worldXY``, a ``2x2`` matrix satisfying
        ``world_xy_delta = relativeXY_to_worldXY @ relative_xy_delta`` for
        column vectors.

    Raises:
        ValueError: If the transform is invalid, singular in the xy plane, or
            tilted out of the calibrated plane beyond ``planar_tolerance``.
    """

    transform = np.asarray(reset_transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(
            "reset_transform must have shape (4, 4), "
            f"got {transform.shape}."
        )
    if not np.all(np.isfinite(transform)):
        raise ValueError("reset_transform must contain only finite values.")
    if (
        isinstance(planar_tolerance, bool)
        or not np.isfinite(planar_tolerance)
        or planar_tolerance < 0
    ):
        raise ValueError("planar_tolerance must be a nonnegative finite number.")

    vertical_leak = np.abs(transform[2, :2])
    if np.any(vertical_leak > planar_tolerance):
        raise ValueError(
            "reset_transform relative x/y axes are not planar enough for the "
            "calibrated world-xy homography; world-z leakage is "
            f"{vertical_leak.tolist()}, tolerance is {planar_tolerance}."
        )

    return _validate_relative_xy_to_world_xy(transform[:2, :2])


def convert_world_homography_to_relative(
    matrix: Any,
    relativeXY_to_worldXY: Any,
    *,
    relative_origin_world_xy: Optional[Any] = None,
) -> np.ndarray:
    """Convert a world/base-frame homography into the relative-frame basis.

    The calibration script fits ``matrix`` from robot base/world plane
    coordinates to front-camera pixels:

    ``pixel_uv ~= matrix @ [world_x, world_y, 1]``.

    The fractal replay buffer, however, builds its translation matrices from
    ``transform_delta`` values attached to observations. If the environment is
    wrapped by ``RelativeFrame``, those deltas are relative-frame deltas rather
    than world/base deltas. The homography must therefore use the same
    coordinate basis as those deltas.

    This function performs the basis change:

    ``world_xy = relativeXY_to_worldXY @ relative_xy + relative_origin_world_xy``

    and returns:

    ``M_relative = M_world @ relative_to_world_affine``.

    For the common bin-relocation reset orientation ``Rx(pi)``, this means:

    ``dx_world = dx_relative``
    ``dy_world = -dy_relative``

    so the y sign is corrected before the replay buffer constructs image
    homographies. If reset yaw is randomized, ``relativeXY_to_worldXY`` should
    be recomputed for the actual reset pose instead of hardcoded.

    ``relative_origin_world_xy`` is optional because pure translation
    homographies of the form ``M @ T @ inv(M)`` cancel the origin term. Passing
    the true reset origin is still useful when you want ``M_relative`` itself
    to project absolute relative-frame points into pixels.

    Args:
        matrix: Homography mapping robot base/world xy points to pixels.
        relativeXY_to_worldXY: ``2x2`` matrix mapping relative xy deltas into
            world/base xy deltas.
        relative_origin_world_xy: Optional world/base xy location of the
            relative-frame origin.

    Returns:
        Normalized homography mapping relative-frame xy points to pixels.

    Raises:
        ValueError: If any matrix, basis, or origin value is invalid.
    """

    world_matrix = validate_homography(matrix)
    basis = _validate_relative_xy_to_world_xy(relativeXY_to_worldXY)
    origin = _validate_optional_xy(
        relative_origin_world_xy,
        name="relative_origin_world_xy",
    )

    relative_to_world = np.eye(3, dtype=np.float64)
    relative_to_world[:2, :2] = basis
    relative_to_world[:2, 2] = origin
    return validate_homography(world_matrix @ relative_to_world)


def _finite_number(value: Any, *, name: str, minimum: Optional[float] = None) -> float:
    """Convert one scalar to ``float`` after checking its numeric range.

    Args:
        value: Candidate scalar value.
        name: Human-readable name included in validation errors.
        minimum: Optional inclusive lower bound.

    Returns:
        The validated value as a Python ``float``.

    Raises:
        ValueError: If the value is Boolean, nonnumeric, infinite, NaN, or
            below the requested minimum.
    """

    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return number


def create_calibration_artifact(
    *,
    camera_serial: str,
    world_xyz_m: Any,
    pixels_uv_raw: Any,
    raw_size_hw: Sequence[int],
    target_size_hw: Sequence[int],
    nominal_z_m: float,
    plane_tolerance_m: float,
    ransac_threshold_px: float = 3.0,
    created_at: Optional[str] = None,
) -> dict:
    """Fit correspondences and build a complete calibration artifact.

    This is the main hardware-independent entry point used after point
    collection. It verifies that all robot points belong to one physical
    plane, fits ``M_raw``, derives ``M_target``, and records the measurements
    and quality statistics needed to audit the result. Unlike the collector's
    resumable point session, the returned artifact is a completed calibration
    suitable for validation and later training-time loading.

    Args:
        camera_serial: Serial number of the calibrated front camera.
        world_xyz_m: Base-frame robot points in meters, shaped ``(N, 3)``.
        pixels_uv_raw: Matching clicks in the raw image, shaped ``(N, 2)``.
        raw_size_hw: Raw camera size in ``(height, width)`` order.
        target_size_hw: Replay-buffer image size in ``(height, width)`` order.
        nominal_z_m: Intended physical height of the calibration plane.
        plane_tolerance_m: Maximum permitted deviation from ``nominal_z_m``.
        ransac_threshold_px: RANSAC inlier threshold in raw-image pixels.
        created_at: Optional timezone-aware ISO-8601 timestamp. The current UTC
            time is used when this argument is omitted.

    Returns:
        A JSON-compatible dictionary following schema version 1.

    Raises:
        ValueError: If metadata, points, plane consistency, or fitting fails.
    """

    if not isinstance(camera_serial, str) or not camera_serial:
        raise ValueError("camera_serial must be a non-empty string.")
    world_xyz = _as_points(
        world_xyz_m,
        name="world_xyz_m",
        dimensions=3,
        minimum_count=4,
    )
    pixels = _as_points(
        pixels_uv_raw,
        name="pixels_uv_raw",
        dimensions=2,
        minimum_count=4,
    )
    if world_xyz.shape[0] != pixels.shape[0]:
        raise ValueError(
            "world_xyz_m and pixels_uv_raw must contain the same number of points."
        )
    raw_size = _validate_image_size(raw_size_hw, name="raw_size_hw")
    target_size = _validate_image_size(target_size_hw, name="target_size_hw")
    nominal_z = _finite_number(nominal_z_m, name="nominal_z_m")
    tolerance = _finite_number(
        plane_tolerance_m,
        name="plane_tolerance_m",
        minimum=0.0,
    )
    if tolerance <= 0:
        raise ValueError("plane_tolerance_m must be positive.")

    z_values = world_xyz[:, 2]
    max_z_deviation = float(np.max(np.abs(z_values - nominal_z)))
    if max_z_deviation > tolerance:
        raise ValueError(
            "Recorded Z coordinates exceed the plane tolerance: "
            f"{max_z_deviation:.6g} m > {tolerance:.6g} m."
        )

    fit = fit_planar_homography(
        world_xyz[:, :2],
        pixels,
        ransac_threshold_px=ransac_threshold_px,
    )
    target_matrix = resize_homography(
        fit.matrix,
        raw_size,
        target_size,
    )
    timestamp = created_at or datetime.now(timezone.utc).isoformat()

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "created_at": timestamp,
        "camera": {
            "name": "front",
            "serial": camera_serial,
        },
        "coordinate_frame": COORDINATE_FRAME,
        "pixel_convention": PIXEL_CONVENTION,
        "image_geometry": {
            "raw_size_hw": list(raw_size),
            "target_size_hw": list(target_size),
            "resize_method": RESIZE_METHOD,
        },
        "plane": {
            "nominal_z_m": nominal_z,
            "mean_z_m": float(np.mean(z_values)),
            "std_z_m": float(np.std(z_values)),
            "max_deviation_m": max_z_deviation,
            "tolerance_m": tolerance,
        },
        "points": [
            {
                "world_xyz_m": world_point.tolist(),
                "pixel_uv_raw": pixel.tolist(),
            }
            for world_point, pixel in zip(world_xyz, pixels)
        ],
        "homographies": {
            "M_raw": fit.matrix.tolist(),
            "M_target": target_matrix.tolist(),
        },
        "fit": {
            "method": FIT_METHOD,
            "ransac_threshold_px": float(ransac_threshold_px),
            "inlier_mask": fit.inlier_mask.tolist(),
            "reprojection_errors_px": fit.reprojection_errors_px.tolist(),
            "inlier_rmse_px": fit.inlier_rmse_px,
            "inlier_max_error_px": fit.inlier_max_error_px,
            "all_points_rmse_px": fit.all_points_rmse_px,
            "all_points_max_error_px": fit.all_points_max_error_px,
        },
    }
    validate_calibration_artifact(artifact)
    return artifact


_TOP_LEVEL_KEYS = {
    "schema_version",
    "created_at",
    "camera",
    "coordinate_frame",
    "pixel_convention",
    "image_geometry",
    "plane",
    "points",
    "homographies",
    "fit",
}


def _require_mapping(
    value: Any,
    *,
    name: str,
    keys: set,
) -> Mapping[str, Any]:
    """Require a dictionary-like object to contain exactly the expected fields.

    Strict field checking catches spelling mistakes and prevents silently
    accepting calibration data written for another schema version.

    Args:
        value: Candidate dictionary-like object.
        name: Object name included in validation errors.
        keys: Exact set of required field names.

    Returns:
        The validated mapping.

    Raises:
        ValueError: If the value is not a mapping or its fields differ.
    """

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    
    actual_keys = set(value)
    if actual_keys != keys:
        missing = sorted(keys - actual_keys)
        extra = sorted(actual_keys - keys)
        raise ValueError(
            f"{name} has invalid fields; missing={missing}, extra={extra}."
        )
    return value


def _validate_timestamp(value: Any) -> None:
    """Require a timezone-aware ISO-8601 calibration timestamp.

    Args:
        value: Candidate timestamp, such as
            ``"2026-07-06T12:00:00+00:00"``.

    Raises:
        ValueError: If the value is not parseable or omits its timezone.
    """

    if not isinstance(value, str):
        raise ValueError("created_at must be an ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at must be a valid ISO-8601 timestamp.") from error
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone.")


def _assert_close(name: str, actual: Any, expected: Any) -> None:
    """Raise an informative error when stored numeric data is inconsistent.

    Args:
        name: Name of the field being checked.
        actual: Value stored in the artifact.
        expected: Value recomputed from the underlying calibration points.

    Raises:
        ValueError: If the values differ beyond floating-point roundoff.
    """

    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-9):
        raise ValueError(f"{name} is inconsistent with the calibration data.")


def validate_calibration_artifact(artifact: Any) -> None:
    """Perform strict structural and numerical validation of an artifact.

    Validation checks more than JSON structure. It recomputes plane statistics,
    projections, error metrics, and ``M_target`` so that internally
    inconsistent or manually corrupted calibration files fail before training.

    Args:
        artifact: Parsed calibration dictionary to validate.

    Raises:
        ValueError: If any required field, convention, measurement, matrix, or
            derived statistic violates the version-1 contract.
    """

    root = _require_mapping(
        artifact,
        name="artifact",
        keys=_TOP_LEVEL_KEYS,
    )
    if (
        isinstance(root["schema_version"], bool)
        or root["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported schema_version {root['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}."
        )
    _validate_timestamp(root["created_at"])

    camera = _require_mapping(
        root["camera"],
        name="camera",
        keys={"name", "serial"},
    )
    if camera["name"] != "front":
        raise ValueError("camera.name must be 'front'.")
    if not isinstance(camera["serial"], str) or not camera["serial"]:
        raise ValueError("camera.serial must be a non-empty string.")
    if root["coordinate_frame"] != COORDINATE_FRAME:
        raise ValueError(
            f"coordinate_frame must be {COORDINATE_FRAME!r}."
        )
    if root["pixel_convention"] != PIXEL_CONVENTION:
        raise ValueError(
            f"pixel_convention must be {PIXEL_CONVENTION!r}."
        )

    geometry = _require_mapping(
        root["image_geometry"],
        name="image_geometry",
        keys={"raw_size_hw", "target_size_hw", "resize_method"},
    )
    raw_size = _validate_image_size(
        geometry["raw_size_hw"],
        name="image_geometry.raw_size_hw",
    )
    target_size = _validate_image_size(
        geometry["target_size_hw"],
        name="image_geometry.target_size_hw",
    )
    if geometry["resize_method"] != RESIZE_METHOD:
        raise ValueError(
            f"image_geometry.resize_method must be {RESIZE_METHOD!r}."
        )

    plane = _require_mapping(
        root["plane"],
        name="plane",
        keys={
            "nominal_z_m",
            "mean_z_m",
            "std_z_m",
            "max_deviation_m",
            "tolerance_m",
        },
    )
    nominal_z = _finite_number(plane["nominal_z_m"], name="plane.nominal_z_m")
    mean_z = _finite_number(plane["mean_z_m"], name="plane.mean_z_m")
    std_z = _finite_number(
        plane["std_z_m"],
        name="plane.std_z_m",
        minimum=0.0,
    )
    max_z_deviation = _finite_number(
        plane["max_deviation_m"],
        name="plane.max_deviation_m",
        minimum=0.0,
    )
    tolerance = _finite_number(
        plane["tolerance_m"],
        name="plane.tolerance_m",
        minimum=0.0,
    )
    if tolerance <= 0:
        raise ValueError("plane.tolerance_m must be positive.")

    if not isinstance(root["points"], list) or len(root["points"]) < 4:
        raise ValueError("points must be a list containing at least four points.")
    world_points = []
    pixel_points = []
    for index, point_value in enumerate(root["points"]):
        point = _require_mapping(
            point_value,
            name=f"points[{index}]",
            keys={"world_xyz_m", "pixel_uv_raw"},
        )
        world = np.asarray(point["world_xyz_m"], dtype=np.float64)
        pixel = np.asarray(point["pixel_uv_raw"], dtype=np.float64)
        if world.shape != (3,) or not np.all(np.isfinite(world)):
            raise ValueError(
                f"points[{index}].world_xyz_m must contain three finite values."
            )
        if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
            raise ValueError(
                f"points[{index}].pixel_uv_raw must contain two finite values."
            )
        world_points.append(world)
        pixel_points.append(pixel)
    world_xyz = np.stack(world_points)
    pixels_uv = np.stack(pixel_points)
    _validate_correspondences(world_xyz[:, :2], pixels_uv)

    z_values = world_xyz[:, 2]
    computed_max_z_deviation = float(np.max(np.abs(z_values - nominal_z)))
    _assert_close("plane.mean_z_m", mean_z, np.mean(z_values))
    _assert_close("plane.std_z_m", std_z, np.std(z_values))
    _assert_close(
        "plane.max_deviation_m",
        max_z_deviation,
        computed_max_z_deviation,
    )
    if max_z_deviation > tolerance:
        raise ValueError(
            "Recorded Z coordinates exceed plane.tolerance_m."
        )

    homographies = _require_mapping(
        root["homographies"],
        name="homographies",
        keys={"M_raw", "M_target"},
    )
    raw_matrix = validate_homography(homographies["M_raw"])
    target_matrix = validate_homography(homographies["M_target"])
    _assert_close(
        "homographies.M_raw normalization",
        np.asarray(homographies["M_raw"], dtype=np.float64),
        raw_matrix,
    )
    _assert_close(
        "homographies.M_target normalization",
        np.asarray(homographies["M_target"], dtype=np.float64),
        target_matrix,
    )
    _assert_close(
        "homographies.M_target resize conversion",
        target_matrix,
        resize_homography(raw_matrix, raw_size, target_size),
    )

    fit = _require_mapping(
        root["fit"],
        name="fit",
        keys={
            "method",
            "ransac_threshold_px",
            "inlier_mask",
            "reprojection_errors_px",
            "inlier_rmse_px",
            "inlier_max_error_px",
            "all_points_rmse_px",
            "all_points_max_error_px",
        },
    )
    if fit["method"] != FIT_METHOD:
        raise ValueError(f"fit.method must be {FIT_METHOD!r}.")
    threshold = _finite_number(
        fit["ransac_threshold_px"],
        name="fit.ransac_threshold_px",
        minimum=0.0,
    )
    if threshold <= 0:
        raise ValueError("fit.ransac_threshold_px must be positive.")

    point_count = world_xyz.shape[0]
    mask_value = fit["inlier_mask"]
    if (
        not isinstance(mask_value, list)
        or len(mask_value) != point_count
        or any(not isinstance(value, bool) for value in mask_value)
    ):
        raise ValueError(
            "fit.inlier_mask must contain one Boolean per calibration point."
        )
    inlier_mask = np.asarray(mask_value, dtype=bool)
    if np.count_nonzero(inlier_mask) < 4:
        raise ValueError("fit.inlier_mask must contain at least four inliers.")

    stored_errors = np.asarray(
        fit["reprojection_errors_px"],
        dtype=np.float64,
    )
    if (
        stored_errors.shape != (point_count,)
        or not np.all(np.isfinite(stored_errors))
        or np.any(stored_errors < 0)
    ):
        raise ValueError(
            "fit.reprojection_errors_px must contain one nonnegative finite "
            "number per calibration point."
        )
    computed_errors = reprojection_errors(
        raw_matrix,
        world_xyz[:, :2],
        pixels_uv,
    )
    _assert_close(
        "fit.reprojection_errors_px",
        stored_errors,
        computed_errors,
    )

    expected_inlier_rmse, expected_inlier_max = _error_summary(
        computed_errors[inlier_mask]
    )
    expected_all_rmse, expected_all_max = _error_summary(computed_errors)
    metrics = (
        ("inlier_rmse_px", expected_inlier_rmse),
        ("inlier_max_error_px", expected_inlier_max),
        ("all_points_rmse_px", expected_all_rmse),
        ("all_points_max_error_px", expected_all_max),
    )
    for field, expected in metrics:
        actual = _finite_number(
            fit[field],
            name=f"fit.{field}",
            minimum=0.0,
        )
        _assert_close(f"fit.{field}", actual, expected)


def save_calibration_artifact(path: PathLike, artifact: Mapping[str, Any]) -> None:
    """Validate and atomically save a calibration artifact as JSON.

    Atomic saving first writes a temporary file beside the destination and
    then replaces the destination in one filesystem operation. This avoids
    leaving a partly written calibration if the process stops unexpectedly.

    Args:
        path: Destination JSON path.
        artifact: Complete artifact created by
            :func:`create_calibration_artifact`.

    Raises:
        ValueError: If the artifact is invalid.
        OSError: If the destination cannot be written or replaced.
    """

    validate_calibration_artifact(artifact)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                artifact,
                temporary_file,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            temporary_file.write("\n")
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_calibration_artifact(
    path: PathLike,
    *,
    expected_camera_serial: Optional[str] = None,
    expected_raw_size_hw: Optional[Sequence[int]] = None,
    expected_target_size_hw: Optional[Sequence[int]] = None,
    allow_incompatible: bool = False,
) -> dict:
    """Load and validate a calibration before it is used for training.

    Expected camera and image values let the caller detect a valid calibration
    that belongs to a different physical setup. ``allow_incompatible`` bypasses
    only those caller-supplied compatibility checks; it never bypasses schema
    or numerical validation.

    Args:
        path: Source JSON path.
        expected_camera_serial: Optional serial required by the caller.
        expected_raw_size_hw: Optional required raw ``(height, width)``.
        expected_target_size_hw: Optional required target ``(height, width)``.
        allow_incompatible: Whether to permit mismatches against the three
            optional expectations.

    Returns:
        A validated deep copy of the parsed calibration dictionary.

    Raises:
        ValueError: If JSON, artifact data, or compatibility checks fail.
        OSError: If the source file cannot be opened.
    """

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as artifact_file:
            artifact = json.load(artifact_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid calibration JSON in {source}.") from error
    validate_calibration_artifact(artifact)

    mismatches = []
    if (
        expected_camera_serial is not None
        and artifact["camera"]["serial"] != expected_camera_serial
    ):
        mismatches.append(
            "camera serial "
            f"{artifact['camera']['serial']!r} != {expected_camera_serial!r}"
        )
    for field, expected in (
        ("raw_size_hw", expected_raw_size_hw),
        ("target_size_hw", expected_target_size_hw),
    ):
        if expected is None:
            continue
        expected_size = _validate_image_size(expected, name=f"expected_{field}")
        actual_size = tuple(artifact["image_geometry"][field])
        if actual_size != expected_size:
            mismatches.append(
                f"{field} {actual_size!r} != {expected_size!r}"
            )
    if mismatches and not allow_incompatible:
        raise ValueError(
            "Calibration artifact is incompatible: " + "; ".join(mismatches)
        )
    return copy.deepcopy(artifact)


__all__ = [
    "COORDINATE_FRAME",
    "FIT_METHOD",
    "HomographyFit",
    "PIXEL_CONVENTION",
    "RESIZE_METHOD",
    "SCHEMA_VERSION",
    "convert_world_homography_to_relative",
    "create_calibration_artifact",
    "fit_planar_homography",
    "load_calibration_artifact",
    "opencv_resize_transform",
    "project_points",
    "relative_xy_to_world_xy_from_reset_transform",
    "reprojection_errors",
    "resize_homography",
    "save_calibration_artifact",
    "validate_calibration_artifact",
    "validate_homography",
]
