#!/usr/bin/env python3
"""Collect robot-base and raw-image points to later create a front-camera  
calibration matrix for a homography. See more at: https://bit.ly/4p6zIfP

The script reads the RealSense camera directly, so neither the environment's
128x128 resize nor ``SERLObsWrapper`` changes the image geometry. It saves a
resumable point-collection session, robustly fits the completed session, and
shows a diagnostic view before the operator explicitly saves the final
calibration artifact.

A point session is not mathematically required to compute a homography. It is
an operational checkpoint for a slow manual hardware procedure. Saving after
each accepted click prevents a camera, server, or process failure from losing
the preceding measurements. Session metadata also prevents points collected
with different cameras, image sizes, or physical planes from being mixed
accidentally. Once fitting succeeds, the final calibration artifact contains
the matrices and quality metrics; the session remains useful as the raw,
auditable input from which that artifact can be regenerated.

Interactive controls:

* ``Space`` or ``f`` freezes the current raw frame.
* Left-clicking a frozen frame records the pixel and calls ``POST /getpos``.
* ``c`` cancels a frozen frame and returns to the live view.
* ``u`` removes the most recently saved point.
* ``q`` quits while keeping the resumable session.

After fitting:

* ``s`` saves the reviewed calibration artifact and diagnostic image.
* ``q`` exits without saving a final calibration.

Keep the robot stationary between freezing the frame and clicking the visible
TCP reference or pointer tip.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests

from serl_launcher.utils.homography import (
    create_calibration_artifact,
    project_points,
    save_calibration_artifact,
    validate_calibration_artifact,
)


SESSION_SCHEMA_VERSION = 1
COORDINATE_FRAME = "robot_base_xy"
PIXEL_CONVENTION = "upper_left_pixel_center_uv"
DEFAULT_SESSION_PATH = (
    Path(__file__).resolve().parent
    / "calibrations"
    / "front_homography_points.json"
)
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "calibrations"
    / "front_homography.json"
)
DEFAULT_DIAGNOSTIC_PATH = (
    Path(__file__).resolve().parent
    / "calibrations"
    / "front_homography_diagnostic.png"
)


def utc_timestamp() -> str:
    """Return a timezone-aware timestamp suitable for saved JSON metadata."""

    return datetime.now(timezone.utc).isoformat()


def _validate_timestamp(value: Any, *, name: str) -> str:
    """Validate one timezone-aware ISO-8601 timestamp.

    Args:
        value: Candidate timestamp string.
        name: Field name included in a validation error.

    Returns:
        The original validated timestamp string.

    Raises:
        ValueError: If the timestamp is malformed or omits a timezone.
    """

    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be a valid ISO-8601 timestamp.") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone.")
    return value


def _finite_float(
    value: Any,
    *,
    name: str,
    positive: bool = False,
) -> float:
    """Convert a numeric scalar to a finite float.

    Args:
        value: Candidate numeric value.
        name: Field name included in a validation error.
        positive: Whether the number must be strictly greater than zero.

    Returns:
        The validated Python float.

    Raises:
        ValueError: If the value is Boolean, nonnumeric, non-finite, or fails
            the positivity requirement.
    """

    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive.")
    return number


def _validate_size_hw(value: Any, *, name: str) -> Tuple[int, int]:
    """Validate an image size expressed in ``(height, width)`` order.

    Args:
        value: Two positive integer dimensions.
        name: Field name included in a validation error.

    Returns:
        A standard Python ``(height, width)`` tuple.

    Raises:
        ValueError: If the value does not contain two positive integers.
    """

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must contain [height, width].")
    dimensions = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(
            dimension, (int, np.integer)
        ):
            raise ValueError(f"{name} dimensions must be integers.")
        if dimension <= 0:
            raise ValueError(f"{name} dimensions must be positive.")
        dimensions.append(int(dimension))
    return dimensions[0], dimensions[1]


def _validate_pose(value: Any, *, name: str = "robot_pose_xyz_quat") -> np.ndarray:
    """Validate the seven-value pose returned by Franka ``/getpos``.

    Args:
        value: Pose ordered as ``x, y, z, qx, qy, qz, qw``.
        name: Field name included in a validation error.

    Returns:
        A finite ``float64`` NumPy array with shape ``(7,)``.

    Raises:
        ValueError: If the pose has the wrong shape or invalid numbers.
    """

    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"{name} must contain seven values, got {pose.shape}.")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must contain only finite values.")
    return pose


def _atomic_save_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON atomically so interruption cannot leave a partial file.

    Args:
        path: Destination file.
        data: JSON-compatible mapping to serialize.

    Raises:
        OSError: If the directory, temporary file, or final replacement fails.
        ValueError: If ``data`` contains non-JSON values such as NaN.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                data,
                temporary_file,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


@dataclass
class CalibrationSession:
    """Resumable raw correspondences collected before homography fitting.

    A session is an intermediate collection checkpoint, not the final
    calibration. Keeping it allows a 20-point manual procedure to resume after
    interruption, supports immediate undo/checkpoint operations, and retains
    the original poses and clicks for auditing or refitting. The final
    calibration JSON cannot be written during the first few clicks because its
    homographies, RANSAC inlier mask, and reprojection metrics do not exist
    until enough correspondences have been collected and fitted.

    Attributes:
        camera_serial: Serial number of the front RealSense camera.
        raw_size_hw: Unmodified camera frame size in ``(height, width)`` order.
        target_count: Number of correspondences the operator plans to collect.
        plane_tolerance_m: Permitted physical Z deviation in meters.
        nominal_z_m: Selected plane height, or ``None`` before the first point.
        nominal_z_source: Whether Z was configured or inferred from a point.
        points: Recorded robot poses and raw pixel clicks.
        created_at: Timestamp at which collection began.
        updated_at: Timestamp of the most recent change.
    """

    camera_serial: str
    raw_size_hw: Tuple[int, int]
    target_count: int
    plane_tolerance_m: float
    nominal_z_m: Optional[float]
    nominal_z_source: Optional[str]
    points: list
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        """Validate all session fields after construction or JSON loading."""

        if not isinstance(self.camera_serial, str) or not self.camera_serial:
            raise ValueError("camera_serial must be a non-empty string.")
        self.raw_size_hw = _validate_size_hw(
            self.raw_size_hw,
            name="raw_size_hw",
        )
        if isinstance(self.target_count, bool) or not isinstance(
            self.target_count, (int, np.integer)
        ):
            raise ValueError("target_count must be a positive integer.")
        self.target_count = int(self.target_count)
        if self.target_count < 4:
            raise ValueError("target_count must be at least four.")
        self.plane_tolerance_m = _finite_float(
            self.plane_tolerance_m,
            name="plane_tolerance_m",
            positive=True,
        )
        if self.nominal_z_m is not None:
            self.nominal_z_m = _finite_float(
                self.nominal_z_m,
                name="nominal_z_m",
            )
        if self.nominal_z_source not in (None, "configured", "first_point"):
            raise ValueError(
                "nominal_z_source must be null, 'configured', or 'first_point'."
            )
        if (self.nominal_z_m is None) != (self.nominal_z_source is None):
            raise ValueError(
                "nominal_z_m and nominal_z_source must either both be set "
                "or both be null."
            )
        self.created_at = _validate_timestamp(
            self.created_at,
            name="created_at",
        )
        self.updated_at = _validate_timestamp(
            self.updated_at,
            name="updated_at",
        )
        if not isinstance(self.points, list):
            raise ValueError("points must be a list.")
        validated_points = [
            self._validate_point(point, index)
            for index, point in enumerate(self.points)
        ]
        if len(validated_points) > self.target_count:
            raise ValueError("points cannot exceed target_count.")
        self.points = validated_points

        height, width = self.raw_size_hw
        for index, point in enumerate(self.points):
            u_value, v_value = point["pixel_uv_raw"]
            if not (0 <= u_value < width and 0 <= v_value < height):
                raise ValueError(
                    f"points[{index}].pixel_uv_raw lies outside the raw frame."
                )

        if self.points and self.nominal_z_m is None:
            raise ValueError("nominal_z_m is required once points exist.")
        if self.nominal_z_m is not None:
            for index, point in enumerate(self.points):
                z_value = point["world_xyz_m"][2]
                if abs(z_value - self.nominal_z_m) > self.plane_tolerance_m:
                    raise ValueError(
                        f"points[{index}] exceeds the plane tolerance."
                    )

    @classmethod
    def create(
        cls,
        *,
        camera_serial: str,
        raw_size_hw: Sequence[int],
        target_count: int,
        plane_tolerance_m: float,
        nominal_z_m: Optional[float],
    ) -> "CalibrationSession":
        """Create a new empty point-collection session.

        If ``nominal_z_m`` is omitted, the first accepted robot pose selects
        the plane height. Later points must remain within
        ``plane_tolerance_m`` of that height.

        Args:
            camera_serial: Serial number of the front camera.
            raw_size_hw: Raw ``(height, width)`` frame size.
            target_count: Planned number of correspondences.
            plane_tolerance_m: Allowed Z deviation in meters.
            nominal_z_m: Optional preselected plane height.

        Returns:
            An empty validated :class:`CalibrationSession`.
        """

        timestamp = utc_timestamp()
        return cls(
            camera_serial=camera_serial,
            raw_size_hw=tuple(raw_size_hw),
            target_count=target_count,
            plane_tolerance_m=plane_tolerance_m,
            nominal_z_m=nominal_z_m,
            nominal_z_source=(
                "configured" if nominal_z_m is not None else None
            ),
            points=[],
            created_at=timestamp,
            updated_at=timestamp,
        )

    @staticmethod
    def _validate_point(point: Any, index: int) -> dict:
        """Validate one saved correspondence from the partial session.

        Args:
            point: Candidate point dictionary.
            index: Position used to make errors easy to locate.

        Returns:
            A normalized JSON-compatible point dictionary.

        Raises:
            ValueError: If fields, pose, pixel, or timestamp are invalid.
        """

        required_keys = {
            "world_xyz_m",
            "robot_pose_xyz_quat",
            "pixel_uv_raw",
            "captured_at",
        }
        if not isinstance(point, Mapping) or set(point) != required_keys:
            raise ValueError(
                f"points[{index}] must contain exactly "
                f"{sorted(required_keys)}."
            )
        pose = _validate_pose(
            point["robot_pose_xyz_quat"],
            name=f"points[{index}].robot_pose_xyz_quat",
        )
        world = np.asarray(point["world_xyz_m"], dtype=np.float64)
        if (
            world.shape != (3,)
            or not np.all(np.isfinite(world))
            or not np.allclose(world, pose[:3], rtol=0.0, atol=1e-12)
        ):
            raise ValueError(
                f"points[{index}].world_xyz_m must equal the pose position."
            )
        pixel = np.asarray(point["pixel_uv_raw"], dtype=np.float64)
        if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
            raise ValueError(
                f"points[{index}].pixel_uv_raw must contain two finite values."
            )
        timestamp = _validate_timestamp(
            point["captured_at"],
            name=f"points[{index}].captured_at",
        )
        return {
            "world_xyz_m": world.tolist(),
            "robot_pose_xyz_quat": pose.tolist(),
            "pixel_uv_raw": pixel.tolist(),
            "captured_at": timestamp,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CalibrationSession":
        """Reconstruct and validate a session loaded from JSON.

        Args:
            data: Parsed session dictionary.

        Returns:
            A validated :class:`CalibrationSession`.

        Raises:
            ValueError: If the schema version, conventions, or data are invalid.
        """

        required_keys = {
            "schema_version",
            "camera",
            "coordinate_frame",
            "pixel_convention",
            "raw_size_hw",
            "target_count",
            "plane_tolerance_m",
            "nominal_z_m",
            "nominal_z_source",
            "points",
            "created_at",
            "updated_at",
        }
        if not isinstance(data, Mapping) or set(data) != required_keys:
            raise ValueError(
                "Point session fields do not match schema version 1."
            )
        if (
            isinstance(data["schema_version"], bool)
            or data["schema_version"] != SESSION_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Unsupported point-session schema {data['schema_version']!r}."
            )
        camera = data["camera"]
        if (
            not isinstance(camera, Mapping)
            or set(camera) != {"name", "serial"}
            or camera["name"] != "front"
        ):
            raise ValueError("camera must identify the front camera.")
        if data["coordinate_frame"] != COORDINATE_FRAME:
            raise ValueError(
                f"coordinate_frame must be {COORDINATE_FRAME!r}."
            )
        if data["pixel_convention"] != PIXEL_CONVENTION:
            raise ValueError(
                f"pixel_convention must be {PIXEL_CONVENTION!r}."
            )
        return cls(
            camera_serial=camera["serial"],
            raw_size_hw=tuple(data["raw_size_hw"]),
            target_count=data["target_count"],
            plane_tolerance_m=data["plane_tolerance_m"],
            nominal_z_m=data["nominal_z_m"],
            nominal_z_source=data["nominal_z_source"],
            points=list(data["points"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    @classmethod
    def load(cls, path: Path) -> "CalibrationSession":
        """Load a resumable point session from disk.

        Loading a session lets collection continue without repeating valid
        manual measurements from an earlier run. Compatibility is checked
        separately against the newly opened camera and command-line settings
        before any additional point is accepted.

        Args:
            path: Existing point-session JSON file.

        Returns:
            A validated :class:`CalibrationSession`.

        Raises:
            OSError: If the file cannot be opened.
            ValueError: If JSON or session contents are invalid.
        """

        try:
            with path.open("r", encoding="utf-8") as session_file:
                data = json.load(session_file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid point-session JSON in {path}.") from error
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert this session into the documented JSON representation."""

        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "camera": {
                "name": "front",
                "serial": self.camera_serial,
            },
            "coordinate_frame": COORDINATE_FRAME,
            "pixel_convention": PIXEL_CONVENTION,
            "raw_size_hw": list(self.raw_size_hw),
            "target_count": self.target_count,
            "plane_tolerance_m": self.plane_tolerance_m,
            "nominal_z_m": self.nominal_z_m,
            "nominal_z_source": self.nominal_z_source,
            "points": self.points,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def save(self, path: Path) -> None:
        """Validate and atomically checkpoint this session.

        The checkpoint is updated after every accepted click and undo. This
        limits data loss to the operation currently in progress if hardware or
        the process fails.

        Args:
            path: Destination JSON file.
        """

        self.__post_init__()
        _atomic_save_json(path, self.to_dict())

    def assert_compatible(
        self,
        *,
        camera_serial: str,
        raw_size_hw: Sequence[int],
        target_count: int,
        plane_tolerance_m: float,
        nominal_z_m: Optional[float],
    ) -> None:
        """Ensure resume arguments describe the same calibration session.

        Args:
            camera_serial: Serial number requested for this run.
            raw_size_hw: Size of the newly opened raw camera frame.
            target_count: Requested number of points.
            plane_tolerance_m: Requested Z tolerance.
            nominal_z_m: Optional plane height supplied on the command line.

        Raises:
            ValueError: If continuing would mix incompatible collection data.
        """

        mismatches = []
        if self.camera_serial != camera_serial:
            mismatches.append(
                f"camera serial {self.camera_serial!r} != {camera_serial!r}"
            )
        actual_size = _validate_size_hw(raw_size_hw, name="raw_size_hw")
        if self.raw_size_hw != actual_size:
            mismatches.append(
                f"raw size {self.raw_size_hw!r} != {actual_size!r}"
            )
        if self.target_count != target_count:
            mismatches.append(
                f"target count {self.target_count} != {target_count}"
            )
        requested_tolerance = _finite_float(
            plane_tolerance_m,
            name="plane_tolerance_m",
            positive=True,
        )
        if not math.isclose(
            self.plane_tolerance_m,
            requested_tolerance,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            mismatches.append(
                "plane tolerance "
                f"{self.plane_tolerance_m} != {requested_tolerance}"
            )
        if nominal_z_m is not None:
            requested_z = _finite_float(nominal_z_m, name="nominal_z_m")
            if self.nominal_z_m is None or not math.isclose(
                self.nominal_z_m,
                requested_z,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                mismatches.append(
                    f"nominal Z {self.nominal_z_m!r} != {requested_z!r}"
                )
        if mismatches:
            raise ValueError(
                "Cannot resume incompatible point session: "
                + "; ".join(mismatches)
            )

    @property
    def complete(self) -> bool:
        """Return whether the requested number of points has been collected."""

        return len(self.points) >= self.target_count

    def add_point(
        self,
        pixel_uv_raw: Sequence[float],
        robot_pose_xyz_quat: Sequence[float],
        *,
        captured_at: Optional[str] = None,
    ) -> dict:
        """Add one clicked pixel and its matching robot-base pose.

        Args:
            pixel_uv_raw: Clicked ``(u, v)`` in the unmodified camera frame.
            robot_pose_xyz_quat: Seven-value pose returned by ``/getpos``.
            captured_at: Optional timezone-aware timestamp.

        Returns:
            The JSON-compatible point that was appended.

        Raises:
            ValueError: If collection is complete, the click is outside the
                image, or the pose violates the selected physical plane.
        """

        if self.complete:
            raise ValueError("The requested point count is already complete.")
        pixel = np.asarray(pixel_uv_raw, dtype=np.float64)
        if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
            raise ValueError("pixel_uv_raw must contain two finite values.")
        height, width = self.raw_size_hw
        u_value, v_value = pixel
        if not (0 <= u_value < width and 0 <= v_value < height):
            raise ValueError(
                f"Pixel {pixel.tolist()} lies outside the {width}x{height} frame."
            )

        pose = _validate_pose(robot_pose_xyz_quat)
        if self.nominal_z_m is None:
            self.nominal_z_m = float(pose[2])
            self.nominal_z_source = "first_point"
        z_deviation = abs(float(pose[2]) - self.nominal_z_m)
        if z_deviation > self.plane_tolerance_m:
            raise ValueError(
                "Robot Z is outside the calibration plane: "
                f"deviation {z_deviation:.6g} m exceeds "
                f"{self.plane_tolerance_m:.6g} m."
            )

        point = {
            "world_xyz_m": pose[:3].tolist(),
            "robot_pose_xyz_quat": pose.tolist(),
            "pixel_uv_raw": pixel.tolist(),
            "captured_at": captured_at or utc_timestamp(),
        }
        point = self._validate_point(point, len(self.points))
        self.points.append(point)
        self.updated_at = utc_timestamp()
        return point

    def undo_last(self) -> Optional[dict]:
        """Remove and return the newest point, or ``None`` if no points exist."""

        if not self.points:
            return None
        point = self.points.pop()
        if not self.points and self.nominal_z_source == "first_point":
            self.nominal_z_m = None
            self.nominal_z_source = None
        self.updated_at = utc_timestamp()
        return point


def fetch_robot_pose(server_url: str, timeout_s: float) -> np.ndarray:
    """Fetch the current global TCP pose from Franka ``POST /getpos``.

    Args:
        server_url: Franka server base URL, with or without a trailing slash.
        timeout_s: HTTP request timeout in seconds.

    Returns:
        Pose ordered as ``x, y, z, qx, qy, qz, qw`` in the robot base frame.

    Raises:
        requests.RequestException: If communication or HTTP status fails.
        ValueError: If the server response does not contain a valid pose.
    """

    timeout = _finite_float(timeout_s, name="timeout_s", positive=True)
    endpoint = f"{server_url.rstrip('/')}/getpos"
    response = requests.post(endpoint, timeout=timeout)
    response.raise_for_status()
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as error:
        raise ValueError("/getpos returned invalid JSON.") from error
    if not isinstance(payload, Mapping) or set(payload) != {"pose"}:
        raise ValueError("/getpos must return exactly {'pose': [...]}.")
    return _validate_pose(payload["pose"], name="/getpos pose")


def read_raw_frame(capture: Any) -> np.ndarray:
    """Read and validate one unmodified BGR frame from ``RSCapture``.

    Args:
        capture: Object whose ``read`` method returns ``(success, image)``.

    Returns:
        A three-channel NumPy image without cropping or resizing.

    Raises:
        RuntimeError: If capture fails or returns an invalid image.
    """

    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError("Front camera did not return a frame.")
    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        shape = getattr(frame, "shape", None)
        raise RuntimeError(
            f"Expected a three-channel raw camera frame, got {shape}."
        )
    return frame


class CalibrationCollector:
    """OpenCV user interface that pairs frozen-frame clicks with robot poses."""

    def __init__(
        self,
        *,
        capture: Any,
        session: CalibrationSession,
        session_path: Path,
        server_url: str,
        request_timeout_s: float,
        initial_frame: np.ndarray,
        window_name: str = "Front homography calibration",
    ) -> None:
        """Configure the collection interface without starting its event loop.

        Args:
            capture: Open raw-camera capture object.
            session: New or resumed point session.
            session_path: JSON checkpoint destination.
            server_url: Franka server base URL.
            request_timeout_s: Timeout for each ``/getpos`` request.
            initial_frame: First validated raw camera frame.
            window_name: Title for the OpenCV display window.
        """

        self.capture = capture
        self.session = session
        self.session_path = session_path
        self.server_url = server_url
        self.request_timeout_s = request_timeout_s
        self.latest_frame = initial_frame
        self.frozen_frame = None
        self.pending_click = None
        self.window_name = window_name
        self.status = "Live view: move robot, then press Space to freeze."

    def _mouse_callback(
        self,
        event: int,
        x_position: int,
        y_position: int,
        flags: int,
        parameter: Any,
    ) -> None:
        """Queue a left-click only while a raw frame is frozen.

        OpenCV invokes this callback from its GUI event loop. The callback only
        stores the click; the slower HTTP request is handled afterward by the
        main loop so errors can be reported cleanly.
        """

        del flags, parameter
        if event == cv2.EVENT_LBUTTONDOWN and self.frozen_frame is not None:
            self.pending_click = (x_position, y_position)

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw saved points, progress, controls, and status on a frame copy."""

        display = frame.copy()
        for index, point in enumerate(self.session.points, start=1):
            u_value, v_value = np.rint(point["pixel_uv_raw"]).astype(int)
            cv2.circle(display, (u_value, v_value), 5, (0, 255, 0), 2)
            cv2.putText(
                display,
                str(index),
                (u_value + 7, v_value - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        mode = "FROZEN" if self.frozen_frame is not None else "LIVE"
        lines = [
            (
                f"{mode} | points {len(self.session.points)}/"
                f"{self.session.target_count}"
            ),
            "Space/f: freeze | click: record | c: cancel | u: undo | q: quit",
            self.status,
        ]
        for line_index, line in enumerate(lines):
            y_position = 24 + line_index * 22
            cv2.putText(
                display,
                line,
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                line,
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return display

    def _record_pending_click(self) -> None:
        """Call ``/getpos``, append the queued click, and save immediately."""

        click = self.pending_click
        self.pending_click = None
        if click is None:
            return
        try:
            pose = fetch_robot_pose(
                self.server_url,
                self.request_timeout_s,
            )
            self.session.add_point(click, pose)
            self.session.save(self.session_path)
        except (ValueError, OSError, requests.RequestException) as error:
            self.status = (
                f"Point not saved: {error}. Press c and recapture if needed."
            )
            print(self.status)
            return

        point_number = len(self.session.points)
        self.status = (
            f"Saved point {point_number}: pixel={click}, "
            f"xyz={pose[:3].round(6).tolist()}"
        )
        print(self.status)
        self.frozen_frame = None

    def run(self) -> None:
        """Run collection until the target count is reached or the user quits.

        Raises:
            RuntimeError: If the raw camera fails during live collection.
        """

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        try:
            while not self.session.complete:
                if self.frozen_frame is None:
                    self.latest_frame = read_raw_frame(self.capture)
                    frame = self.latest_frame
                else:
                    frame = self.frozen_frame

                if self.pending_click is not None:
                    self._record_pending_click()
                    frame = (
                        self.latest_frame
                        if self.frozen_frame is None
                        else self.frozen_frame
                    )

                cv2.imshow(self.window_name, self._draw_overlay(frame))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self.status = "Collection stopped; session remains resumable."
                    print(self.status)
                    break
                if key in (ord(" "), ord("f")) and self.frozen_frame is None:
                    self.frozen_frame = self.latest_frame.copy()
                    self.status = (
                        "Frame frozen: click the TCP reference or press c."
                    )
                elif key == ord("c") and self.frozen_frame is not None:
                    self.frozen_frame = None
                    self.pending_click = None
                    self.status = "Frozen frame cancelled; returned to live view."
                elif key == ord("u"):
                    removed = self.session.undo_last()
                    if removed is None:
                        self.status = "No saved point is available to undo."
                    else:
                        self.session.save(self.session_path)
                        self.status = (
                            f"Removed point {len(self.session.points) + 1}."
                        )
                    print(self.status)
        finally:
            cv2.destroyWindow(self.window_name)

        if self.session.complete:
            print(
                f"Collected all {self.session.target_count} points in "
                f"{self.session_path}."
            )


def build_calibration_artifact(
    session: CalibrationSession,
    *,
    target_size_hw: Sequence[int],
    ransac_threshold_px: float,
) -> dict:
    """Fit a completed point session and construct the final artifact.

    The session contains raw measurements only. This function extracts their
    base-frame XYZ positions and raw-image UV clicks, then delegates fitting,
    resize conversion, quality statistics, and artifact validation to the
    hardware-independent homography utility.

    Args:
        session: Completed and validated point-collection session.
        target_size_hw: Replay-buffer image size in ``(height, width)`` order.
        ransac_threshold_px: RANSAC inlier threshold in raw-image pixels.

    Returns:
        A validated artifact containing ``M_raw``, ``M_target``, points,
        inliers, and reprojection metrics.

    Raises:
        ValueError: If collection is incomplete or fitting/validation fails.
    """

    if not session.complete:
        raise ValueError(
            "Point session is incomplete: "
            f"{len(session.points)}/{session.target_count} points."
        )
    if session.nominal_z_m is None:
        raise ValueError("Completed session has no nominal calibration plane.")

    world_xyz_m = np.asarray(
        [point["world_xyz_m"] for point in session.points],
        dtype=np.float64,
    )
    pixels_uv_raw = np.asarray(
        [point["pixel_uv_raw"] for point in session.points],
        dtype=np.float64,
    )
    artifact = create_calibration_artifact(
        camera_serial=session.camera_serial,
        world_xyz_m=world_xyz_m,
        pixels_uv_raw=pixels_uv_raw,
        raw_size_hw=session.raw_size_hw,
        target_size_hw=target_size_hw,
        nominal_z_m=session.nominal_z_m,
        plane_tolerance_m=session.plane_tolerance_m,
        ransac_threshold_px=ransac_threshold_px,
    )
    validate_calibration_artifact(artifact)
    return artifact


def format_calibration_report(artifact: Mapping[str, Any]) -> str:
    """Format fitted matrices and quality statistics for terminal review.

    Args:
        artifact: Valid completed calibration artifact.

    Returns:
        A multiline report showing point counts, pixel errors, plane
        statistics, ``M_raw``, and ``M_target``.

    Raises:
        ValueError: If the artifact is invalid.
    """

    validate_calibration_artifact(artifact)
    fit = artifact["fit"]
    inlier_count = int(np.count_nonzero(fit["inlier_mask"]))
    point_count = len(artifact["points"])
    raw_matrix = np.asarray(artifact["homographies"]["M_raw"])
    target_matrix = np.asarray(artifact["homographies"]["M_target"])
    matrix_options = {
        "precision": 8,
        "suppress_small": False,
    }
    return "\n".join(
        [
            "Front-camera homography fit",
            f"  inliers: {inlier_count}/{point_count}",
            (
                "  inlier error: "
                f"RMSE={fit['inlier_rmse_px']:.4f}px, "
                f"max={fit['inlier_max_error_px']:.4f}px"
            ),
            (
                "  all-point error: "
                f"RMSE={fit['all_points_rmse_px']:.4f}px, "
                f"max={fit['all_points_max_error_px']:.4f}px"
            ),
            (
                "  plane Z: "
                f"mean={artifact['plane']['mean_z_m']:.6f}m, "
                f"std={artifact['plane']['std_z_m']:.6f}m, "
                "max deviation="
                f"{artifact['plane']['max_deviation_m']:.6f}m"
            ),
            "  M_raw:",
            np.array2string(raw_matrix, **matrix_options),
            "  M_target:",
            np.array2string(target_matrix, **matrix_options),
        ]
    )


def _pixel_is_drawable(
    pixel_uv: np.ndarray,
    *,
    image_size_hw: Sequence[int],
) -> bool:
    """Return whether a projected pixel can safely become OpenCV integers.

    A badly fitted homography can project a point very far outside the image.
    Avoiding huge integer conversions keeps the diagnostic renderer safe while
    the numerical error is still reported beside the measured point.

    Args:
        pixel_uv: Candidate projected ``(u, v)``.
        image_size_hw: Diagnostic image size in ``(height, width)`` order.

    Returns:
        Whether the point lies within a bounded margin around the image.
    """

    image_height, image_width = _validate_size_hw(
        image_size_hw,
        name="image_size_hw",
    )
    margin = 4 * max(image_height, image_width)
    return bool(
        np.all(np.isfinite(pixel_uv))
        and -margin <= pixel_uv[0] <= image_width + margin
        and -margin <= pixel_uv[1] <= image_height + margin
    )


def draw_calibration_diagnostic(
    artifact: Mapping[str, Any],
    *,
    background: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render measured clicks, fitted projections, and residual errors.

    Measured clicks are circles: green for RANSAC inliers and red for outliers.
    Fitted projections are cyan crosses. A yellow line connects each measured
    click to its prediction, making both the magnitude and direction of error
    visible.

    Args:
        artifact: Valid completed calibration artifact.
        background: Optional raw BGR camera frame. When omitted, a dark canvas
            with the artifact's raw dimensions is used.

    Returns:
        A ``uint8`` BGR diagnostic image with raw camera dimensions.

    Raises:
        ValueError: If the artifact or supplied background is invalid.
    """

    validate_calibration_artifact(artifact)
    raw_height, raw_width = artifact["image_geometry"]["raw_size_hw"]
    if background is None:
        display = np.full(
            (raw_height, raw_width, 3),
            35,
            dtype=np.uint8,
        )
    else:
        if (
            not isinstance(background, np.ndarray)
            or background.shape != (raw_height, raw_width, 3)
        ):
            shape = getattr(background, "shape", None)
            raise ValueError(
                "Diagnostic background must have shape "
                f"{(raw_height, raw_width, 3)}, got {shape}."
            )
        if background.dtype == np.uint8:
            display = background.copy()
        else:
            if not np.all(np.isfinite(background)):
                raise ValueError(
                    "Diagnostic background must contain finite values."
                )
            display = np.clip(background, 0, 255).astype(np.uint8)

    world_xy = np.asarray(
        [point["world_xyz_m"][:2] for point in artifact["points"]],
        dtype=np.float64,
    )
    measured_pixels = np.asarray(
        [point["pixel_uv_raw"] for point in artifact["points"]],
        dtype=np.float64,
    )
    projected_pixels = project_points(
        artifact["homographies"]["M_raw"],
        world_xy,
    )
    inlier_mask = np.asarray(artifact["fit"]["inlier_mask"], dtype=bool)
    errors = np.asarray(
        artifact["fit"]["reprojection_errors_px"],
        dtype=np.float64,
    )

    for index, (measured, projected, is_inlier, error_px) in enumerate(
        zip(measured_pixels, projected_pixels, inlier_mask, errors),
        start=1,
    ):
        measured_point = tuple(np.rint(measured).astype(int))
        measured_color = (0, 210, 0) if is_inlier else (0, 0, 255)
        cv2.circle(display, measured_point, 6, measured_color, 2)

        if _pixel_is_drawable(
            projected,
            image_size_hw=(raw_height, raw_width),
        ):
            projected_point = tuple(np.rint(projected).astype(int))
            cv2.line(
                display,
                measured_point,
                projected_point,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.drawMarker(
                display,
                projected_point,
                (255, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        label_position = (
            max(0, min(measured_point[0] + 8, raw_width - 100)),
            max(measured_point[1] - 8, 82),
        )
        cv2.putText(
            display,
            f"{index}:{error_px:.1f}px",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            measured_color,
            1,
            cv2.LINE_AA,
        )

    fit = artifact["fit"]
    inlier_count = int(np.count_nonzero(inlier_mask))
    header_lines = [
        (
            f"inliers {inlier_count}/{len(inlier_mask)} | "
            f"inlier RMSE {fit['inlier_rmse_px']:.2f}px | "
            f"all RMSE {fit['all_points_rmse_px']:.2f}px"
        ),
        "circle=measured (green inlier/red outlier) | cyan cross=projected",
        "yellow line=reprojection residual",
    ]
    cv2.rectangle(display, (0, 0), (raw_width - 1, 68), (0, 0, 0), -1)
    for line_index, line in enumerate(header_lines):
        cv2.putText(
            display,
            line,
            (8, 19 + line_index * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return display


def load_diagnostic_background(
    path: Optional[Path],
    *,
    expected_size_hw: Sequence[int],
) -> Optional[np.ndarray]:
    """Load an optional raw BGR image for offline diagnostic rendering.

    Args:
        path: Image path, or ``None`` to request a blank diagnostic canvas.
        expected_size_hw: Required raw ``(height, width)``.

    Returns:
        Loaded BGR image, or ``None``.

    Raises:
        ValueError: If OpenCV cannot read the image or dimensions differ.
    """

    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read diagnostic background {path}.")
    expected_height, expected_width = _validate_size_hw(
        expected_size_hw,
        name="expected_size_hw",
    )
    if image.shape != (expected_height, expected_width, 3):
        raise ValueError(
            f"Diagnostic background has shape {image.shape}, expected "
            f"{(expected_height, expected_width, 3)}."
        )
    return image


def save_calibration_outputs(
    *,
    artifact: Mapping[str, Any],
    diagnostic: np.ndarray,
    artifact_path: Path,
    diagnostic_path: Path,
    overwrite: bool,
) -> None:
    """Save the reviewed JSON artifact and diagnostic PNG.

    Existing outputs are protected unless the operator supplied
    ``--overwrite-artifact``. The JSON writer performs full numerical
    validation and uses an atomic replacement.

    Args:
        artifact: Valid completed calibration artifact.
        diagnostic: Rendered raw-resolution BGR diagnostic.
        artifact_path: Destination for the final calibration JSON.
        diagnostic_path: Destination for the diagnostic PNG.
        overwrite: Whether existing output files may be replaced.

    Raises:
        FileExistsError: If an output exists and overwrite is disabled.
        ValueError: If the artifact or diagnostic is invalid.
        OSError: If either output cannot be written.
    """

    validate_calibration_artifact(artifact)
    raw_height, raw_width = artifact["image_geometry"]["raw_size_hw"]
    if (
        not isinstance(diagnostic, np.ndarray)
        or diagnostic.shape != (raw_height, raw_width, 3)
        or diagnostic.dtype != np.uint8
    ):
        raise ValueError(
            "Diagnostic must be a uint8 BGR image with raw camera dimensions."
        )
    existing = [
        path for path in (artifact_path, diagnostic_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Calibration output already exists: "
            + ", ".join(str(path) for path in existing)
            + ". Use --overwrite-artifact to replace it."
        )

    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(diagnostic_path), diagnostic):
        raise OSError(f"Could not write diagnostic image {diagnostic_path}.")
    save_calibration_artifact(artifact_path, artifact)


def review_and_save_calibration(
    *,
    artifact: Mapping[str, Any],
    diagnostic: np.ndarray,
    artifact_path: Path,
    diagnostic_path: Path,
    overwrite: bool,
    window_name: str = "Review front homography",
) -> bool:
    """Show a diagnostic until the operator saves or rejects the fit.

    Args:
        artifact: Fitted calibration artifact.
        diagnostic: Diagnostic image created from the same artifact.
        artifact_path: Destination JSON path.
        diagnostic_path: Destination PNG path.
        overwrite: Whether existing outputs may be replaced.
        window_name: OpenCV review-window title.

    Returns:
        ``True`` when outputs were saved, or ``False`` when review was closed
        without saving.
    """

    print(format_calibration_report(artifact))
    print("Review controls: s=save calibration, q/Esc=leave unsaved")
    status = (
        f"s: save {artifact_path.name} | q: leave final calibration unsaved"
    )
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frame = diagnostic.copy()
            cv2.rectangle(
                frame,
                (0, frame.shape[0] - 28),
                (frame.shape[1] - 1, frame.shape[0] - 1),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                status,
                (8, frame.shape[0] - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Review window closed; final calibration was not saved.")
                return False
            if key in (ord("q"), 27):
                print("Final calibration was not saved; point session remains.")
                return False
            if key == ord("s"):
                try:
                    save_calibration_outputs(
                        artifact=artifact,
                        diagnostic=diagnostic,
                        artifact_path=artifact_path,
                        diagnostic_path=diagnostic_path,
                        overwrite=overwrite,
                    )
                except (ValueError, OSError) as error:
                    status = f"Not saved: {error}"
                    print(status)
                    continue
                print(f"Saved calibration artifact: {artifact_path}")
                print(f"Saved diagnostic image: {diagnostic_path}")
                return True
    finally:
        cv2.destroyWindow(window_name)


def fit_review_and_save(
    *,
    session: CalibrationSession,
    target_size_hw: Sequence[int],
    ransac_threshold_px: float,
    background: Optional[np.ndarray],
    artifact_path: Path,
    diagnostic_path: Path,
    overwrite: bool,
) -> bool:
    """Fit a completed session, render it, and request operator approval.

    Args:
        session: Completed raw point session.
        target_size_hw: Replay-buffer ``(height, width)``.
        ransac_threshold_px: Raw-pixel RANSAC threshold.
        background: Optional raw BGR frame for the diagnostic.
        artifact_path: Final JSON destination.
        diagnostic_path: Diagnostic PNG destination.
        overwrite: Whether existing outputs may be replaced.

    Returns:
        Whether the operator saved the final calibration.
    """

    artifact = build_calibration_artifact(
        session,
        target_size_hw=target_size_hw,
        ransac_threshold_px=ransac_threshold_px,
    )
    diagnostic = draw_calibration_diagnostic(
        artifact,
        background=background,
    )
    return review_and_save_calibration(
        artifact=artifact,
        diagnostic=diagnostic,
        artifact_path=artifact_path,
        diagnostic_path=diagnostic_path,
        overwrite=overwrite,
    )


def configured_front_camera_serial() -> str:
    """Read the bin-relocation front-camera serial from its environment config.

    The import is delayed until runtime so unit tests can exercise collection
    logic on machines without RealSense or robot packages.

    Returns:
        Configured front-camera serial number.

    Raises:
        RuntimeError: If the task configuration has no usable front camera.
    """

    try:
        from franka_env.envs.bin_relocation_env.config import BinEnvConfig

        serial = BinEnvConfig.REALSENSE_CAMERAS["front"]
    except (ImportError, KeyError, TypeError) as error:
        raise RuntimeError(
            "Could not read the front-camera serial from BinEnvConfig."
        ) from error
    if not isinstance(serial, str) or not serial:
        raise RuntimeError("Configured front-camera serial is invalid.")
    return serial


def open_front_camera(
    *,
    serial: str,
    width: int,
    height: int,
    fps: int,
) -> Any:
    """Open the task's RealSense color stream without environment wrappers.

    Args:
        serial: RealSense device serial number.
        width: Requested raw frame width.
        height: Requested raw frame height.
        fps: Requested capture rate.

    Returns:
        An open ``RSCapture`` instance.

    Raises:
        ImportError: If RealSense support is unavailable.
        RuntimeError: If the device cannot be found or opened.
    """

    from franka_env.camera.rs_capture import RSCapture

    return RSCapture(
        name="front",
        serial_number=serial,
        dim=(width, height),
        fps=fps,
        depth=False,
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for collection and final fitting."""

    parser = argparse.ArgumentParser(
        description=(
            "Collect raw front-camera pixels and matching Franka base-frame "
            "poses for planar homography calibration."
        )
    )
    parser.add_argument(
        "--camera-serial",
        default=None,
        help="Front RealSense serial; defaults to BinEnvConfig.",
    )
    parser.add_argument("--raw-width", type=int, default=640)
    parser.add_argument("--raw-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:5000",
        help="Franka server base URL.",
    )
    parser.add_argument("--request-timeout-s", type=float, default=2.0)
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument(
        "--plane-tolerance-m",
        type=float,
        default=0.005,
        help="Maximum TCP Z deviation from the selected plane.",
    )
    parser.add_argument(
        "--nominal-z-m",
        type=float,
        default=None,
        help="Optional plane height; the first point selects it when omitted.",
    )
    parser.add_argument(
        "--session-path",
        type=Path,
        default=DEFAULT_SESSION_PATH,
        help="Resumable point-session JSON path.",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help=(
            "Fit and review an already completed session without opening "
            "the camera or Franka server."
        ),
    )
    parser.add_argument("--target-width", type=int, default=128)
    parser.add_argument("--target-height", type=int, default=128)
    parser.add_argument(
        "--ransac-threshold-px",
        type=float,
        default=3.0,
        help="RANSAC inlier threshold measured in raw-image pixels.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Reviewed final calibration JSON destination.",
    )
    parser.add_argument(
        "--diagnostic-path",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_PATH,
        help="Measured-versus-projected diagnostic PNG destination.",
    )
    parser.add_argument(
        "--background-image",
        type=Path,
        default=None,
        help=(
            "Optional raw-size image used behind --fit-only diagnostics; "
            "otherwise a dark canvas is used."
        ),
    )
    parser.add_argument(
        "--overwrite-artifact",
        action="store_true",
        help="Allow reviewed final JSON/PNG outputs to be replaced.",
    )
    session_mode = parser.add_mutually_exclusive_group()
    session_mode.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing compatible session.",
    )
    session_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing point session.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line values before opening hardware.

    Args:
        args: Parsed command-line namespace.

    Raises:
        ValueError: If dimensions, counts, rate, timeout, or tolerance fail.
    """

    _validate_size_hw(
        (args.raw_height, args.raw_width),
        name="requested raw size",
    )
    _validate_size_hw(
        (args.target_height, args.target_width),
        name="target image size",
    )
    if isinstance(args.fps, bool) or args.fps <= 0:
        raise ValueError("fps must be a positive integer.")
    if isinstance(args.target_count, bool) or args.target_count < 4:
        raise ValueError("target-count must be at least four.")
    _finite_float(
        args.request_timeout_s,
        name="request-timeout-s",
        positive=True,
    )
    _finite_float(
        args.plane_tolerance_m,
        name="plane-tolerance-m",
        positive=True,
    )
    _finite_float(
        args.ransac_threshold_px,
        name="ransac-threshold-px",
        positive=True,
    )
    if args.nominal_z_m is not None:
        _finite_float(args.nominal_z_m, name="nominal-z-m")
    if not isinstance(args.server_url, str) or not args.server_url.strip():
        raise ValueError("server-url must be non-empty.")
    if args.fit_only and (args.resume or args.overwrite):
        raise ValueError(
            "--fit-only loads the completed session directly; do not combine "
            "it with --resume or --overwrite."
        )


def prepare_session(
    *,
    path: Path,
    resume: bool,
    overwrite: bool,
    camera_serial: str,
    raw_size_hw: Sequence[int],
    target_count: int,
    plane_tolerance_m: float,
    nominal_z_m: Optional[float],
) -> CalibrationSession:
    """Create, replace, or resume a collection session safely.

    Existing data is never overwritten implicitly. Requiring an explicit
    resume or overwrite choice prevents a new invocation from silently
    destroying an unfinished manual calibration.

    Args:
        path: Point-session JSON path.
        resume: Whether an existing file must be resumed.
        overwrite: Whether an existing file may be replaced.
        camera_serial: Serial of the camera opened for this run.
        raw_size_hw: Actual first-frame size.
        target_count: Requested point count.
        plane_tolerance_m: Requested Z tolerance.
        nominal_z_m: Optional requested plane height.

    Returns:
        A compatible new or resumed session.

    Raises:
        FileExistsError: If a file exists without ``resume`` or ``overwrite``.
        FileNotFoundError: If ``resume`` is requested but no file exists.
        ValueError: If an existing session is incompatible.
    """

    if path.exists() and resume:
        session = CalibrationSession.load(path)
        session.assert_compatible(
            camera_serial=camera_serial,
            raw_size_hw=raw_size_hw,
            target_count=target_count,
            plane_tolerance_m=plane_tolerance_m,
            nominal_z_m=nominal_z_m,
        )
        return session
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; use --resume or --overwrite."
        )
    if resume:
        raise FileNotFoundError(f"Cannot resume missing session {path}.")

    session = CalibrationSession.create(
        camera_serial=camera_serial,
        raw_size_hw=raw_size_hw,
        target_count=target_count,
        plane_tolerance_m=plane_tolerance_m,
        nominal_z_m=nominal_z_m,
    )
    session.save(path)
    return session


def main() -> None:
    """Run point collection or offline fitting, then review the calibration.

    The calls flow in this order:

    1. :func:`create_argument_parser` defines and parses the command-line
       settings.
    2. :func:`validate_arguments` rejects invalid dimensions, point counts,
       timeouts, and plane settings before hardware is opened.
    3. With ``--fit-only``, :meth:`CalibrationSession.load` reads an already
       completed checkpoint, :func:`load_diagnostic_background` optionally
       reads a raw image, and execution skips directly to fitting and review.
    4. Otherwise, :func:`configured_front_camera_serial` supplies the serial from
       ``BinEnvConfig`` when ``--camera-serial`` was not provided.
    5. :func:`open_front_camera` creates a direct ``RSCapture`` at the requested
       raw resolution, without constructing a Gym environment or wrapper.
    6. :func:`read_raw_frame` retrieves the first frame. Its actual dimensions
       are compared with the requested dimensions to prevent calibrating the
       wrong image geometry.
    7. :func:`prepare_session` either creates an empty checkpoint, explicitly
       replaces one, or validates and resumes an existing one.
    8. :class:`CalibrationCollector` displays live frames. Its ``run`` method
       freezes frames, receives mouse clicks, calls :func:`fetch_robot_pose`,
       validates each point through :meth:`CalibrationSession.add_point`, and
       checkpoints changes through :meth:`CalibrationSession.save`.
    9. The ``finally`` block closes the RealSense capture even if collection
       finishes, the operator quits, or an exception occurs.
    10. For a completed session, :func:`fit_review_and_save` fits ``M_raw``,
        derives ``M_target``, renders measured-versus-projected errors, and
        saves JSON/PNG outputs only after the operator presses ``s``.
    """

    args = create_argument_parser().parse_args()
    validate_arguments(args)
    target_size_hw = (args.target_height, args.target_width)

    if args.fit_only:
        session = CalibrationSession.load(args.session_path)
        if (
            args.camera_serial is not None
            and session.camera_serial != args.camera_serial
        ):
            raise ValueError(
                "Completed session camera serial "
                f"{session.camera_serial!r} does not match "
                f"--camera-serial {args.camera_serial!r}."
            )
        background = load_diagnostic_background(
            args.background_image,
            expected_size_hw=session.raw_size_hw,
        )
        fit_review_and_save(
            session=session,
            target_size_hw=target_size_hw,
            ransac_threshold_px=args.ransac_threshold_px,
            background=background,
            artifact_path=args.artifact_path,
            diagnostic_path=args.diagnostic_path,
            overwrite=args.overwrite_artifact,
        )
        return

    camera_serial = args.camera_serial or configured_front_camera_serial()

    capture = open_front_camera(
        serial=camera_serial,
        width=args.raw_width,
        height=args.raw_height,
        fps=args.fps,
    )
    try:
        initial_frame = read_raw_frame(capture)
        actual_size_hw = tuple(initial_frame.shape[:2])
        requested_size_hw = (args.raw_height, args.raw_width)
        if actual_size_hw != requested_size_hw:
            raise RuntimeError(
                f"Camera returned {actual_size_hw}, requested "
                f"{requested_size_hw}; calibration stopped."
            )

        session = prepare_session(
            path=args.session_path,
            resume=args.resume,
            overwrite=args.overwrite,
            camera_serial=camera_serial,
            raw_size_hw=actual_size_hw,
            target_count=args.target_count,
            plane_tolerance_m=args.plane_tolerance_m,
            nominal_z_m=args.nominal_z_m,
        )
        print(
            f"Using raw {args.raw_width}x{args.raw_height} front camera "
            f"{camera_serial}; session={args.session_path}"
        )
        collector = CalibrationCollector(
            capture=capture,
            session=session,
            session_path=args.session_path,
            server_url=args.server_url,
            request_timeout_s=args.request_timeout_s,
            initial_frame=initial_frame,
        )
        collector.run()
        diagnostic_background = collector.latest_frame.copy()
    finally:
        capture.close()

    if session.complete:
        fit_review_and_save(
            session=session,
            target_size_hw=target_size_hw,
            ransac_threshold_px=args.ransac_threshold_px,
            background=diagnostic_background,
            artifact_path=args.artifact_path,
            diagnostic_path=args.diagnostic_path,
            overwrite=args.overwrite_artifact,
        )
    else:
        print(
            "Point session is incomplete; no final homography was fitted. "
            "Run again with --resume."
        )


if __name__ == "__main__":
    main()
