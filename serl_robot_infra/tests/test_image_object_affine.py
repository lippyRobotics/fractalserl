import numpy as np

from franka_env.utils.image_object_affine import (
    EuclideanAffine2D,
    benchmark_affine_strategies,
    transform_segmented_object,
    update_planar_robot_state,
)


def test_transform_segmented_object_moves_mask():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[20:30, 20:30] = (255, 0, 0)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:30, 20:30] = 255

    transform = EuclideanAffine2D(rotation_rad=0.0, translation_px=(10, 5), center_px=(24.5, 24.5))
    output, moved_mask, _ = transform_segmented_object(image, mask, transform)

    assert output.shape == image.shape
    assert moved_mask.sum() > 0
    ys, xs = np.where(moved_mask > 0)
    assert xs.min() >= 29
    assert ys.min() >= 24


def test_update_planar_robot_state_applies_translation_and_rotation():
    state = np.array([0.0, 0.0, 0.0])
    transform = EuclideanAffine2D(rotation_rad=np.pi / 4.0, translation_px=(20.0, 0.0))

    updated = update_planar_robot_state(state, transform, meters_per_pixel=0.001)

    np.testing.assert_allclose(updated[:2], np.array([0.02, 0.0]), atol=1e-6)
    np.testing.assert_allclose(updated[2], np.pi / 4.0, atol=1e-6)


def test_benchmark_affine_strategies_returns_metrics():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[40:80, 40:80] = 255
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:80, 40:80] = 255
    transform = EuclideanAffine2D(rotation_rad=0.3, translation_px=(5.0, -2.0))

    metrics = benchmark_affine_strategies(image, mask, transform, num_runs=5)

    assert set(metrics.keys()) == {"full_frame_ms", "roi_only_ms", "speedup_x"}
    assert metrics["full_frame_ms"] > 0
    assert metrics["roi_only_ms"] > 0
