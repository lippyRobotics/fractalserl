#!/usr/bin/env python3
"""Visualize every homography transform for one front-camera image.

This is a standalone diagnostic for the homography path used by
``FractalSymmetryReplayBuffer``. The learner debug dump samples random replay
entries, so it is useful for checking whether sampled batches look sane, but it
does not show the full branch grid for one original transition. This script
does exactly that: it takes one image, builds the same branch translations used
by the fractal replay buffer, applies the corresponding image homographies, and
writes a single grid image.

System will resize any incoming image to an 128x128 image as this is what is currently 
used in our system using cv2.resize(). It is best if the image is natively this dimension 
to avoid distortions.

    python examples/async_bin_relocation_fwbw_drq/homography/homography_transformed_image_stand_alone_test.py \
        --image_path /path/to/front.png \
        --branch_count 3 \
        --workspace_width 0.5

The calibration artifact stores a world/base-frame homography. Bin relocation
training uses ``RelativeFrame``, so this script converts the homography into
the relative-frame xy basis by default with:

    relativeXY_to_worldXY = [[1, 0], [0, -1]]

That default matches the cleaned bin-relocation reset orientation ``Rx(pi)``:
relative x maps to world x, while relative y maps to negative world y. To view
world-frame shifts instead, pass:

    --relative_xy_to_world_xy 1,0,0,1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from serl_launcher.utils.homography import (
    convert_world_homography_to_relative,
    validate_calibration_artifact,
    validate_homography,
)


DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] # Extract the absolute path of the immediate parent
    / "calibrations"
    / "front_homography.json"
)
INTERPOLATION_BY_NAME = {
    "min_distort": cv2.INTER_AREA,
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}


def parse_relative_xy_to_world_xy(value: str) -> np.ndarray:
    """Parse a command-line 2x2 basis matrix.

    The expected format is four comma-separated numbers in row-major order:
    ``a,b,c,d`` becomes ``[[a, b], [c, d]]``.

    Args:
        value: Text supplied to ``--relative_xy_to_world_xy``.

    Returns:
        A finite ``float64`` array with shape ``(2, 2)``.

    Raises:
        argparse.ArgumentTypeError: If the text cannot define a valid matrix.
    """

    try:
        numbers = [float(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--relative_xy_to_world_xy must contain four comma-separated numbers."
        ) from error
    if len(numbers) != 4:
        raise argparse.ArgumentTypeError(
            "--relative_xy_to_world_xy must contain exactly four numbers."
        )
    matrix = np.asarray(numbers, dtype=np.float64).reshape(2, 2)
    if not np.all(np.isfinite(matrix)):
        raise argparse.ArgumentTypeError(
            "--relative_xy_to_world_xy must contain only finite values."
        )
    if np.linalg.matrix_rank(matrix) < 2:
        raise argparse.ArgumentTypeError(
            "--relative_xy_to_world_xy must be invertible."
        )
    return matrix


def read_image(path: Path) -> np.ndarray:
    """Read one image from disk without changing its channels.

    Args:
        path: Image file readable by OpenCV.

    Returns:
        The loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If OpenCV cannot decode the image.
    """

    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) # load the image as is, including alpha channel if present

    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    
    if image.ndim == 2:
        image = image[..., None]

    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):

        raise ValueError(
            "Expected a grayscale, BGR, or BGRA image; "
            f"got shape {image.shape}."
        )
    
    # Resize the image
    image = cv2.resize(image,(128,128), cv2.INTER_AREA)
    return image


def load_front_homography(
    path: Path,
    *,
    homography_key: str,
    relativeXY_to_worldXY: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    """Load and optionally basis-convert the calibration matrix.

    Args:
        path: Calibration artifact JSON.
        homography_key: Either ``"M_target"`` for replay-sized images or
            ``"M_raw"`` for raw camera images.
        relativeXY_to_worldXY: Matrix mapping relative-frame xy deltas into
            world/base-frame xy deltas.

    Returns:
        ``(front_M, expected_size_hw, artifact)`` where ``front_M`` maps the
        coordinate basis used by replay-buffer transform deltas into pixels.

    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If the artifact is invalid.
    """

    if not path.exists():
        raise FileNotFoundError(f"Calibration artifact does not exist: {path}")
    
    artifact = json.loads(path.read_text(encoding="utf-8"))
    validate_calibration_artifact(artifact)

    world_M = validate_homography(artifact["homographies"][homography_key])
    front_M = convert_world_homography_to_relative(
        world_M,
        relativeXY_to_worldXY,
    )

    geometry_key = "target_size_hw" if homography_key == "M_target" else "raw_size_hw"
    expected_size_hw = tuple(int(v) for v in artifact["image_geometry"][geometry_key])
    return front_M, expected_size_hw, artifact


def generate_translation_deltas_xy(
    *,
    branch_count: int,
    workspace_width: float,
) -> np.ndarray:
    """Generate the same branch translations as the fractal replay buffer.

    Args:
        branch_count: Number of bins along each planar axis. The total number
            of image transforms is ``branch_count ** 2``.
        workspace_width: Width of the square workspace in meters.

    Returns:
        Array with shape ``(branch_count ** 2, 2)``. Each row is
        ``[delta_x, delta_y]`` in the replay-buffer transform basis.

    Raises:
        ValueError: If either setting is not positive.
    """

    if branch_count <= 0:
        raise ValueError("--branch_count must be positive.")
    
    if workspace_width <= 0:
        raise ValueError("--workspace_width must be positive.")

    total_branches = branch_count**2
    indices = np.arange(total_branches)
    x_indices, y_indices = np.divmod(indices, branch_count)

    x_deltas = (2 * x_indices + 1) * workspace_width / (2 * branch_count)
    y_deltas = (2 * y_indices + 1) * workspace_width / (2 * branch_count)
    base_diff = -workspace_width / 2.0

    return np.stack(
        (base_diff + x_deltas, base_diff + y_deltas),
        axis=1,
    ).astype(np.float64)


def build_inverse_image_homographies(
    front_M: np.ndarray,
    translation_deltas_xy: np.ndarray,
) -> np.ndarray:
    """Build destination-to-source image homographies for ``cv2.remap``.

    Args:
        front_M: Calibration matrix mapping workspace-plane coordinates to
            image pixels in the same basis as ``translation_deltas_xy``.
        translation_deltas_xy: Branch translations from
            :func:`generate_translation_deltas_xy`.

    Returns:
        Array with shape ``(N, 3, 3)``. Each matrix maps destination pixels
        backward to source pixels for one branch transform.
    """

    front_M = validate_homography(front_M)
    translation_deltas_xy = np.asarray(translation_deltas_xy, dtype=np.float64)
    if translation_deltas_xy.ndim != 2 or translation_deltas_xy.shape[1] != 2:
        raise ValueError(
            "translation_deltas_xy must have shape (N, 2), "
            f"got {translation_deltas_xy.shape}."
        )

    inverse_translations = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (translation_deltas_xy.shape[0], 3, 3),
    ).copy()
    inverse_translations[:, 0, 2] = -translation_deltas_xy[:, 0]
    inverse_translations[:, 1, 2] = -translation_deltas_xy[:, 1]

    front_M_inverse = np.linalg.inv(front_M)
    return front_M[None, :, :] @ inverse_translations @ front_M_inverse[None, :, :]


def build_dense_remap(
    inverse_homographies: np.ndarray,
    *,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Convert inverse homographies into dense OpenCV remap coordinates.

    Args:
        inverse_homographies: Array with shape ``(N, 3, 3)``.
        image_height: Output image height in pixels.
        image_width: Output image width in pixels.

    Returns:
        Float32 array with shape ``(N, image_height, image_width, 2)`` where
        the last dimension stores source ``(x, y)`` pixel coordinates.
    """

    inverse_homographies = np.asarray(inverse_homographies, dtype=np.float64)
    if inverse_homographies.ndim != 3 or inverse_homographies.shape[1:] != (3, 3):
        raise ValueError(
            "inverse_homographies must have shape (N, 3, 3), "
            f"got {inverse_homographies.shape}."
        )

    pixel_x, pixel_y = np.meshgrid(
        np.arange(image_width, dtype=np.float64),
        np.arange(image_height, dtype=np.float64),
    )
    destination_pixels = np.stack(
        (
            pixel_x.ravel(),
            pixel_y.ravel(),
            np.ones(image_height * image_width, dtype=np.float64),
        ),
        axis=0,
    )
    source_pixels = inverse_homographies @ destination_pixels
    source_pixels[:, :2, :] /= source_pixels[:, 2:3, :]
    return (
        source_pixels[:, :2, :]
        .transpose(0, 2, 1)
        .reshape(inverse_homographies.shape[0], image_height, image_width, 2)
        .astype(np.float32)
    )


def apply_remaps(
    image: np.ndarray,
    maps: np.ndarray,
    *,
    interpolation: int,
) -> list[np.ndarray]:
    """Apply every dense remap to one image.

    Args:
        image: Source image.
        maps: Dense source-coordinate maps from :func:`build_dense_remap`.
        interpolation: OpenCV interpolation constant.

    Returns:
        One transformed image per map. Border pixels use ``BORDER_REPLICATE``,
        matching the replay-buffer homography path.
    """

    transformed = []
    for map_xy in maps:
        transformed.append(
            cv2.remap(
                image,
                map_xy,
                None,
                interpolation=interpolation,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )
    return transformed


def draw_label(image: np.ndarray, label: str) -> np.ndarray:
    """Draw a small branch label on a copy of an image."""

    output = image.copy()
    if output.shape[2] == 1:
        text_color = 255
        background_color = 0
    else:
        text_color = (255, 255, 255, 255)[: output.shape[2]]
        background_color = (0, 0, 0, 255)[: output.shape[2]]
    cv2.rectangle(output, (0, 0), (min(output.shape[1] - 1, 88), 17), background_color, -1)
    cv2.putText(
        output,
        label,
        (3, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        text_color,
        1,
        cv2.LINE_AA,
    )
    return output


def make_image_grid(
    images: Sequence[np.ndarray],
    *,
    columns: int,
    gutter: int = 2,
) -> np.ndarray:
    """Arrange transformed images into one grid image.

    Args:
        images: Non-empty list of same-shaped images.
        columns: Number of images per row.
        gutter: Pixel spacing between cells.

    Returns:
        A new image containing the grid.
    """

    if not images:
        raise ValueError("images must not be empty.")
    columns = max(1, min(columns, len(images)))
    rows = int(np.ceil(len(images) / columns))
    height, width = images[0].shape[:2]
    channels = images[0].shape[2]
    fill_value = 255 if images[0].dtype == np.uint8 else 1.0
    grid = np.full(
        (
            rows * height + (rows - 1) * gutter,
            columns * width + (columns - 1) * gutter,
            channels,
        ),
        fill_value,
        dtype=images[0].dtype,
    )

    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        y0 = row * (height + gutter)
        x0 = column * (width + gutter)
        grid[y0 : y0 + height, x0 : x0 + width] = image
    return grid


def default_output_path(image_path: Path, branch_count: int) -> Path:
    """Return a readable default output path next to the input image."""

    return image_path.with_name(
        f"{image_path.stem}_homography_branches{branch_count}_grid.png"
    )


def save_metadata(
    output_path: Path,
    *,
    args: argparse.Namespace,
    artifact: dict[str, Any],
    front_M: np.ndarray,
    translation_deltas_xy: np.ndarray,
) -> None:
    """Write a small JSON sidecar describing how the grid was generated."""

    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata = {
        "image_path": str(args.image_path),
        "calibration_path": str(args.calibration_path),
        "output_path": str(output_path),
        "homography_key": args.homography_key,
        "branch_count": args.branch_count,
        "workspace_width": args.workspace_width,
        "relativeXY_to_worldXY": args.relative_xy_to_world_xy.tolist(),
        "camera": artifact["camera"],
        "front_M_used": front_M.tolist(),
        "translation_deltas_xy": translation_deltas_xy.tolist(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the standalone diagnostic."""

    parser = argparse.ArgumentParser(
        description=(
            "Apply every fractal-branch homography transform to one image and "
            "save the transformed images as a grid."
        )
    )
    parser.add_argument(
        "--image_path",
        type=Path,
        required=True,
        help="Input front-camera image to transform.",
    )
    parser.add_argument(
        "--calibration_path",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help=f"Calibration artifact JSON. Default: {DEFAULT_CALIBRATION_PATH}",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output grid image path. Defaults next to the input image.",
    )
    parser.add_argument(
        "--branch_count",
        type=int,
        default=3,
        help="Number of branch cells per axis. Total images = branch_count^2.",
    )
    parser.add_argument(
        "--workspace_width",
        type=float,
        default=0.5,
        help="Workspace width in meters, matching the learner flag.",
    )
    parser.add_argument(
        "--homography_key",
        choices=("M_target", "M_raw"),
        default="M_target",
        help="Use M_target for replay-sized images or M_raw for raw images.",
    )
    parser.add_argument(
        "--relative_xy_to_world_xy",
        type=parse_relative_xy_to_world_xy,
        default=parse_relative_xy_to_world_xy("1,0,0,-1"),
        help=(
            "2x2 basis as 'a,b,c,d'. Default '1,0,0,-1' matches "
            "bin-relocation RelativeFrame after Rx(pi). Use '1,0,0,1' for "
            "world-frame shifts."
        ),
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=None,
        help="Grid columns. Defaults to branch_count.",
    )
    parser.add_argument(
        "--interpolation",
        choices=tuple(INTERPOLATION_BY_NAME.keys()),
        default="linear",
        help="OpenCV interpolation mode.",
    )
    parser.add_argument(
        "--allow_size_mismatch",
        action="store_true",
        help="Do not fail if image size differs from the selected artifact size.",
    )
    parser.add_argument(
        "--draw_labels",
        action="store_true",
        help="Overlay index and delta values on each transformed image.",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="Do not write the JSON metadata sidecar.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the standalone transform-grid diagnostic."""

    args = parse_args()
    image = read_image(args.image_path)
    image_height, image_width = image.shape[:2]

    front_M, expected_size_hw, artifact = load_front_homography(
        args.calibration_path,
        homography_key=args.homography_key,
        relativeXY_to_worldXY=args.relative_xy_to_world_xy,
    )
    if (image_height, image_width) != expected_size_hw and not args.allow_size_mismatch:
        raise ValueError(
            f"Input image size {(image_height, image_width)} does not match "
            f"{args.homography_key} artifact size {expected_size_hw}. Use "
            "--allow_size_mismatch only if this is intentional."
        )

    translation_deltas_xy = generate_translation_deltas_xy(
        branch_count=args.branch_count,
        workspace_width=args.workspace_width,
    )
    inverse_homographies = build_inverse_image_homographies(
        front_M,
        translation_deltas_xy,
    )
    remap_coordinates = build_dense_remap(
        inverse_homographies,
        image_height=image_height,
        image_width=image_width,
    )
    transformed_images = apply_remaps(
        image,
        remap_coordinates,
        interpolation=INTERPOLATION_BY_NAME[args.interpolation],
    )

    if args.draw_labels:
        transformed_images = [
            draw_label(
                transformed_image,
                f"{index}: {delta[0]:+.2f},{delta[1]:+.2f}",
            )
            for index, (transformed_image, delta) in enumerate(
                zip(transformed_images, translation_deltas_xy)
            )
        ]

    output_path = args.output_path or default_output_path(
        args.image_path,
        args.branch_count,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_image_grid(
        transformed_images,
        columns=args.columns or args.branch_count,
    )
    if not cv2.imwrite(str(output_path), grid):
        raise OSError(f"OpenCV failed to write output image: {output_path}")
    print(f"wrote grid: {output_path}")

    if not args.no_metadata:
        save_metadata(
            output_path,
            args=args,
            artifact=artifact,
            front_M=front_M,
            translation_deltas_xy=translation_deltas_xy,
        )


if __name__ == "__main__":
    main()
