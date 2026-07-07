# Front-camera homography calibration

This directory contains the data contract for the bin-relocation front-camera
calibration. Generated calibration artifacts are machine-specific and are
ignored by Git.

## Coordinate convention

The calibration uses robot-base coordinates in meters. A workspace point is
represented by the planar homogeneous vector `[X, Y, 1]`. The final component is a homogeneous coordinate, not the robot's physical `Z` coordinate.

All sampled points must lie on one physical plane at approximately
`plane.nominal_z_m`. The recorded physical `Z` values are retained so the
calibration tool can detect violations of this assumption. This is achieved
by placing a pin at the center/mid-point of the robot's parallel jaw grippers along with a fixed orientaiton (See FR3's programming mode).

Raw image points use `[u, v]` coordinates:

- the origin is the upper-left pixel center;
- `u` increases to the right;
- `v` increases downward;
- image sizes are stored as `[height, width]`.

`homographies.M_raw` satisfies

```text
scale * [u_raw, v_raw, 1]^T = M_raw * [X, Y, 1]^T
```

`homographies.M_target` represents the same mapping after the configured
OpenCV resize from the raw camera size to the replay-buffer image size. The
resize conversion must account for OpenCV's pixel-center convention. Both
matrices are normalized to make their bottom-right entry equal to `1` when
that normalization is numerically valid.

The saved calibration remains in the robot-base frame. Conversion to any
reset-relative augmentation frame is a separate runtime operation and must
not be encoded implicitly in this artifact.

## Artifact

The default generated artifact is `front_homography.json`. Its required
structure is defined by `front_homography.schema.json`.

The artifact records:

- schema version and creation time;
- camera name and serial number;
- coordinate and pixel conventions;
- raw and target image geometry;
- physical plane-height statistics;
- every robot/image correspondence;
- the raw and target homographies;
- RANSAC inliers and reprojection-error statistics for both the inlier set and
  the complete collected point set.

`fit.inlier_rmse_px` and `fit.inlier_max_error_px` describe only the points
accepted by RANSAC. `fit.all_points_rmse_px` and
`fit.all_points_max_error_px` also include rejected points, making bad clicks
or mismatched robot poses visible in the saved results. Individual errors are
stored in point order in `fit.reprojection_errors_px`; `fit.inlier_mask` uses
the same order and identifies which errors contribute to the inlier metrics.

A loader must reject incompatible schema versions, coordinate frames, camera
serials, or image dimensions unless the caller explicitly requests an
override.

## Collecting correspondences

Run `calibrate_front_homography.py` when the Franka server is available and no
other process is consuming the front RealSense camera. The script opens the
raw 640x480 color stream directly; it does not construct the Gym environment
or use an observation wrapper.

The point-collection workflow is:

1. Move the robot until the TCP reference or attached pointer is at the desired
   location on the calibration plane.
2. Keep the robot stationary and press `Space` or `f` to freeze the raw frame.
3. Left-click the visible TCP reference. The script calls `POST /getpos` and
   pairs its global robot pose with that raw pixel.
4. Repeat until the requested point count is reached.

Press `c` to discard a frozen frame, `u` to remove the latest point, or `q` to
stop safely. Every accepted point and undo is saved atomically to
`front_homography_points.json`. Use `--resume` to continue that file or
`--overwrite` to explicitly replace it. Generated point sessions are ignored
by Git because they are tied to one physical setup.

The first accepted pose selects the nominal TCP Z height unless
`--nominal-z-m` is supplied. A new point is rejected when its Z deviation
exceeds `--plane-tolerance-m`. The partial session records whether its nominal
height came from the command line or the first point, so undoing all points
does not accidentally discard an explicitly configured plane.

## Fitting and reviewing

Once the requested point count is reached, the script:

1. fits `M_raw` with OpenCV RANSAC;
2. derives `M_target` for the configured target image size;
3. prints both matrices and the inlier/all-point error statistics;
4. displays a raw-resolution diagnostic image.

In the diagnostic, green circles are measured inliers, red circles are
measured outliers, cyan crosses are homography projections, and yellow lines
show reprojection residuals. Press `s` to save the reviewed
`front_homography.json` artifact and diagnostic PNG. Press `q` or `Esc` to
leave the final calibration unsaved while retaining the point session.

An existing completed session can be fitted again without opening the camera
or contacting the Franka server:

```bash
python examples/async_bin_relocation_fwbw_drq/calibrate_front_homography.py \
    --fit-only
```

Use `--background-image` to place the offline diagnostic over a raw 640x480
image; otherwise it uses a dark canvas. Existing final outputs are protected
unless `--overwrite-artifact` is provided. The RANSAC threshold defaults to
three raw-image pixels and can be changed with `--ransac-threshold-px`.

## Calibration quality

Use at least four non-collinear points. Approximately 20 points distributed
across the complete usable workspace are recommended. Calibration should be
rejected when the point layout is degenerate, matrix values are non-finite,
the matrix is singular, or recorded `Z` variation exceeds the configured
tolerance.
