from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class EuclideanAffine2D:
    """A 2D Euclidean transform in image coordinates."""

    rotation_rad: float = 0.0
    translation_px: Tuple[float, float] = (0.0, 0.0)
    center_px: Optional[Tuple[float, float]] = None


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def build_affine_matrix(shape_hw: Tuple[int, int], transform: EuclideanAffine2D) -> np.ndarray:
    """Builds a 2x3 forward affine matrix [[A|t]] in x-y convention."""
    h, w = shape_hw
    center = (
        np.asarray(transform.center_px, dtype=np.float64)
        if transform.center_px is not None
        else np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=np.float64)
    )
    a = _rotation_matrix(transform.rotation_rad)
    t = np.asarray(transform.translation_px, dtype=np.float64)
    effective_t = center + t - (a @ center)
    return np.concatenate([a, effective_t[:, None]], axis=1)


def preprocess_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Denoises a binary mask with close+open morphology."""
    binary = (mask > 0)
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    clean = ndimage.binary_closing(binary, structure=structure)
    clean = ndimage.binary_opening(clean, structure=structure)
    return (clean.astype(np.uint8) * 255)


def _warp_by_inverse_mapping(image: np.ndarray, transform: EuclideanAffine2D, order: int) -> np.ndarray:
    h, w = image.shape[:2]
    center = (
        np.asarray(transform.center_px, dtype=np.float64)
        if transform.center_px is not None
        else np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=np.float64)
    )
    r = _rotation_matrix(transform.rotation_rad)
    t = np.asarray(transform.translation_px, dtype=np.float64)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    output_xy = np.stack([xx, yy], axis=0).reshape(2, -1)
    input_xy = (r.T @ (output_xy - t[:, None] - center[:, None])) + center[:, None]

    coords = np.stack([input_xy[1], input_xy[0]], axis=0)
    if image.ndim == 2:
        warped = ndimage.map_coordinates(image, coords, order=order, mode="constant", cval=0.0)
        return warped.reshape(h, w)

    channels = []
    for c in range(image.shape[2]):
        ch = ndimage.map_coordinates(image[..., c], coords, order=order, mode="constant", cval=0.0)
        channels.append(ch.reshape(h, w))
    return np.stack(channels, axis=-1)


def _fill_removed_region(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Simple background fill using nearest valid pixel."""
    valid = mask == 0
    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    filled = image.copy()
    if image.ndim == 3:
        filled[~valid] = image[indices[0][~valid], indices[1][~valid]]
    else:
        filled[~valid] = image[indices[0][~valid], indices[1][~valid]]
    return filled


def transform_segmented_object(
    image: np.ndarray,
    mask: np.ndarray,
    transform: EuclideanAffine2D,
    *,
    preprocess: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim != 3:
        raise ValueError("Expected image with shape [H, W, C].")
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError("Mask and image spatial dimensions must match.")

    clean_mask = preprocess_mask(mask) if preprocess else (mask > 0).astype(np.uint8) * 255
    object_rgb = image * (clean_mask[..., None] > 0)
    background = _fill_removed_region(image, clean_mask > 0)

    moved_object = _warp_by_inverse_mapping(object_rgb.astype(np.float32), transform, order=1)
    moved_mask = _warp_by_inverse_mapping(clean_mask.astype(np.float32), transform, order=0)
    moved_mask = (moved_mask > 0.5).astype(np.uint8) * 255

    composite = background.copy()
    moved_mask_bool = moved_mask > 0
    composite[moved_mask_bool] = moved_object[moved_mask_bool].astype(image.dtype)
    return composite, moved_mask, build_affine_matrix(image.shape[:2], transform)


def update_planar_robot_state(
    robot_state_xytheta: np.ndarray,
    transform: EuclideanAffine2D,
    meters_per_pixel: float,
    *,
    camera_yaw_in_robot_rad: float = 0.0,
) -> np.ndarray:
    robot_state_xytheta = np.asarray(robot_state_xytheta, dtype=np.float64)
    if robot_state_xytheta.shape != (3,):
        raise ValueError("robot_state_xytheta must have shape (3,).")

    tx_px, ty_px = transform.translation_px
    delta_cam = np.array([tx_px, ty_px], dtype=np.float64) * float(meters_per_pixel)

    c, s = np.cos(camera_yaw_in_robot_rad), np.sin(camera_yaw_in_robot_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    delta_robot = rot @ delta_cam

    updated = robot_state_xytheta.copy()
    updated[:2] += delta_robot
    updated[2] += float(transform.rotation_rad)
    return updated


def benchmark_affine_strategies(
    image: np.ndarray,
    mask: np.ndarray,
    transform: EuclideanAffine2D,
    *,
    num_runs: int = 100,
) -> Dict[str, float]:
    clean_mask = preprocess_mask(mask)

    def full_frame():
        _warp_by_inverse_mapping(image.astype(np.float32), transform, order=1)
        _warp_by_inverse_mapping(clean_mask.astype(np.float32), transform, order=0)

    ys, xs = np.where(clean_mask > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty. Cannot benchmark ROI strategy.")
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    roi = image[y0:y1, x0:x1]

    local_transform = EuclideanAffine2D(
        rotation_rad=transform.rotation_rad,
        translation_px=transform.translation_px,
        center_px=((roi.shape[1] - 1) / 2.0, (roi.shape[0] - 1) / 2.0),
    )

    def roi_only():
        _warp_by_inverse_mapping(roi.astype(np.float32), local_transform, order=1)

    start = perf_counter()
    for _ in range(num_runs):
        full_frame()
    full_ms = (perf_counter() - start) * 1e3 / num_runs

    start = perf_counter()
    for _ in range(num_runs):
        roi_only()
    roi_ms = (perf_counter() - start) * 1e3 / num_runs

    return {"full_frame_ms": full_ms, "roi_only_ms": roi_ms, "speedup_x": full_ms / max(roi_ms, 1e-9)}
