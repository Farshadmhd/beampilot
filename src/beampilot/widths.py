"""ISO 11146 second-moment beam widths from camera frames.

Background handling follows ISO/TR 11146-3: subtract a mean dark frame, then
a fine offset estimated from the image corners, and keep negative pixel values.
Image coordinates: x along columns, y along rows, in metres relative to the
image centre (the same convention as :mod:`beampilot.simulator`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

MIN_DARK_FRAMES = 10


@dataclass(frozen=True)
class BackgroundInfo:
    """Residual background found by the fine offset correction.

    Parameters
    ----------
    offset_counts : float
        Mean of the four corner patches after dark subtraction, in counts.
        This value was subtracted from the whole image.
    std_counts : float
        Standard deviation of the corner pixels after dark subtraction, in
        counts (pixel noise of the background).
    corner_means_counts : tuple of float
        Mean of each corner patch in counts, ordered top-left, top-right,
        bottom-left, bottom-right. A spread between them indicates a
        background gradient.
    """

    offset_counts: float
    std_counts: float
    corner_means_counts: tuple[float, float, float, float]


def mean_dark(dark_frames: Sequence[ArrayLike] | NDArray) -> NDArray[np.float64]:
    """Average dark frames pixel by pixel.

    Parameters
    ----------
    dark_frames : sequence of 2D arrays, or 3D array of shape (n, ny, nx)
        Dark frames in counts, all with the same shape.

    Returns
    -------
    ndarray of float64, shape (ny, nx)
        Mean dark frame in counts.

    Raises
    ------
    ValueError
        If fewer than ``MIN_DARK_FRAMES`` (10) frames are given, or the
        frames are not 2D.
    """
    stack = np.asarray(dark_frames, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"expected a stack of 2D frames, got shape {stack.shape}")
    if stack.shape[0] < MIN_DARK_FRAMES:
        raise ValueError(
            f"need at least {MIN_DARK_FRAMES} dark frames, got {stack.shape[0]}"
        )
    return stack.mean(axis=0)


def subtract_background(
    frame: ArrayLike,
    dark_mean: ArrayLike,
    corner_fraction: float = 0.035,
) -> tuple[NDArray[np.float64], BackgroundInfo]:
    """Subtract the mean dark frame and the residual corner offset.

    Each corner patch is ``corner_fraction`` of the image width by
    ``corner_fraction`` of the image height (at least 1 pixel). The corners
    must be free of beam light. Negative pixel values are kept (ISO/TR 11146-3);
    clipping them at zero would bias the second moments.

    Parameters
    ----------
    frame : 2D array
        Raw frame in counts.
    dark_mean : 2D array
        Mean dark frame in counts, same shape as `frame` (see :func:`mean_dark`).
    corner_fraction : float, optional
        Corner patch size as a fraction of the image size, in (0, 0.5).
        Default 0.035 (the design allows 2-5 %).

    Returns
    -------
    image : ndarray of float64
        Background-corrected image in counts; may contain negative values.
    info : BackgroundInfo
        Residual offset that was removed and its pixel noise.

    Raises
    ------
    ValueError
        If the shapes differ or `corner_fraction` is outside (0, 0.5).
    """
    frame = np.asarray(frame, dtype=np.float64)
    dark_mean = np.asarray(dark_mean, dtype=np.float64)
    if frame.shape != dark_mean.shape or frame.ndim != 2:
        raise ValueError(
            f"frame {frame.shape} and dark_mean {dark_mean.shape} must be equal 2D"
        )
    if not 0 < corner_fraction < 0.5:
        raise ValueError(f"corner_fraction must be in (0, 0.5), got {corner_fraction}")

    image = frame - dark_mean
    ny, nx = image.shape
    cy = max(1, round(corner_fraction * ny))
    cx = max(1, round(corner_fraction * nx))
    corners = (
        image[:cy, :cx],
        image[:cy, -cx:],
        image[-cy:, :cx],
        image[-cy:, -cx:],
    )
    corner_pixels = np.concatenate([c.ravel() for c in corners])
    offset = float(corner_pixels.mean())
    info = BackgroundInfo(
        offset_counts=offset,
        std_counts=float(corner_pixels.std()),
        corner_means_counts=tuple(float(c.mean()) for c in corners),
    )
    return image - offset, info


@dataclass(frozen=True)
class Roi:
    """Rectangular region of interest in pixel indices (half-open, like slices).

    Parameters
    ----------
    x_start, x_stop : int
        Column range ``[x_start, x_stop)``.
    y_start, y_stop : int
        Row range ``[y_start, y_stop)``.
    """

    x_start: int
    x_stop: int
    y_start: int
    y_stop: int


@dataclass(frozen=True)
class Moments:
    """First and second moments of an intensity distribution.

    Parameters
    ----------
    x_c_m, y_c_m : float
        Centroid in m, relative to the image centre.
    var_x_m2, var_y_m2 : float
        Second moments sigma_x² and sigma_y² about the centroid, in m².
    cov_xy_m2 : float
        Mixed second moment sigma_xy about the centroid, in m².
    """

    x_c_m: float
    y_c_m: float
    var_x_m2: float
    var_y_m2: float
    cov_xy_m2: float


@dataclass(frozen=True)
class WidthResult:
    """ISO 11146 beam widths of one frame.

    Parameters
    ----------
    d_x_m, d_y_m : float
        Second-moment (4 sigma) beam widths along the lab x and y axes, in m.
        NaN if the measurement failed.
    x_c_m, y_c_m : float
        Centroid in m, relative to the image centre.
    azimuth_rad : float
        Angle between the x axis and the major principal axis, in rad, in
        (-pi/2, pi/2]. Meaningless when the profile is nearly circular.
    ellipticity : float
        Ratio of the minor to the major principal-axis width (0 to 1).
    iterations : int
        Number of moment calculations used.
    converged : bool
        True if d_x and d_y changed by less than `rtol` between the last two
        integration areas.
    """

    d_x_m: float
    d_y_m: float
    x_c_m: float
    y_c_m: float
    azimuth_rad: float
    ellipticity: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class WidthStatistics:
    """Mean and spread of beam widths over repeated frames at one position.

    Parameters
    ----------
    d_x_mean_m, d_y_mean_m : float
        Mean of d_x and d_y in m.
    d_x_std_m, d_y_std_m : float
        Sample standard deviation (ddof=1) of d_x and d_y in m; NaN for a
        single frame.
    n : int
        Number of frames.
    """

    d_x_mean_m: float
    d_x_std_m: float
    d_y_mean_m: float
    d_y_std_m: float
    n: int


def second_moments(image: ArrayLike, pixel_size_m: float, roi: Roi) -> Moments:
    """Centroid and second moments of the image inside a rectangular ROI.

    Pixels are treated as point samples at their centres. Negative pixel
    values are used as they are.

    Parameters
    ----------
    image : 2D array
        Background-corrected image in counts.
    pixel_size_m : float
        Pixel pitch in m.
    roi : Roi
        Integration area in pixel indices.

    Returns
    -------
    Moments
        Centroid in m and second moments in m². All NaN if the summed
        intensity inside the ROI is not positive.
    """
    image = np.asarray(image, dtype=np.float64)
    ny, nx = image.shape
    sub = image[roi.y_start : roi.y_stop, roi.x_start : roi.x_stop]
    x_m = (np.arange(roi.x_start, roi.x_stop) - (nx - 1) / 2.0) * pixel_size_m
    y_m = (np.arange(roi.y_start, roi.y_stop) - (ny - 1) / 2.0) * pixel_size_m

    total = sub.sum()
    if not total > 0:
        return Moments(np.nan, np.nan, np.nan, np.nan, np.nan)
    profile_x = sub.sum(axis=0)
    profile_y = sub.sum(axis=1)
    x_c = float(profile_x @ x_m / total)
    y_c = float(profile_y @ y_m / total)
    dx = x_m - x_c
    dy = y_m - y_c
    return Moments(
        x_c_m=x_c,
        y_c_m=y_c,
        var_x_m2=float(profile_x @ dx**2 / total),
        var_y_m2=float(profile_y @ dy**2 / total),
        cov_xy_m2=float(dy @ sub @ dx / total),
    )


def measure_widths(
    image: ArrayLike,
    pixel_size_m: float,
    max_iter: int = 25,
    rtol: float = 1e-4,
) -> WidthResult:
    """ISO 11146-1 second-moment beam widths with an iterated integration area.

    The first pass uses the whole image. Each further pass uses a rectangle
    centred on the last centroid, 3 * d_x wide and 3 * d_y high (clipped to
    the image), until d_x and d_y both change by less than `rtol`.

    Widths are measured along the lab x and y axes, not along the principal
    axes. The design assumes beams aligned to x/y; for a rotated beam use
    `azimuth_rad` and `ellipticity` to detect it.

    Parameters
    ----------
    image : 2D array
        Background-corrected image in counts (negative values allowed).
    pixel_size_m : float
        Pixel pitch in m.
    max_iter : int, optional
        Maximum number of moment calculations. Default 25.
    rtol : float, optional
        Relative change of d_x and d_y below which the iteration stops.
        Default 1e-4.

    Returns
    -------
    WidthResult
        Widths in m, centroid in m, azimuth in rad. If a second moment comes
        out <= 0 (e.g. no beam, or a badly corrected background), widths are
        NaN and ``converged`` is False.

    Raises
    ------
    ValueError
        If `max_iter` is below 1.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    image = np.asarray(image, dtype=np.float64)
    ny, nx = image.shape
    roi = Roi(0, nx, 0, ny)
    previous: tuple[float, float] | None = None
    converged = False

    for iteration in range(1, max_iter + 1):
        moments = second_moments(image, pixel_size_m, roi)
        # Written as "not x > 0" so that NaN moments also fail here.
        if not (moments.var_x_m2 > 0 and moments.var_y_m2 > 0):
            return _failed_result(moments, iteration)
        d_x = 4.0 * np.sqrt(moments.var_x_m2)
        d_y = 4.0 * np.sqrt(moments.var_y_m2)
        if previous is not None and (
            abs(d_x - previous[0]) <= rtol * previous[0]
            and abs(d_y - previous[1]) <= rtol * previous[1]
        ):
            converged = True
            break
        previous = (d_x, d_y)
        roi = _integration_area(moments, d_x, d_y, pixel_size_m, nx, ny)

    azimuth, ellipticity = _azimuth_and_ellipticity(moments)
    return WidthResult(
        d_x_m=float(d_x),
        d_y_m=float(d_y),
        x_c_m=moments.x_c_m,
        y_c_m=moments.y_c_m,
        azimuth_rad=azimuth,
        ellipticity=ellipticity,
        iterations=iteration,
        converged=converged,
    )


def width_statistics(results: Sequence[WidthResult]) -> WidthStatistics:
    """Mean and sample standard deviation of d_x and d_y over repeats.

    A failed frame (NaN width) makes the corresponding mean and std NaN, so a
    bad repeat cannot go unnoticed.

    Parameters
    ----------
    results : sequence of WidthResult
        Width results of repeated frames at one stage position.

    Returns
    -------
    WidthStatistics
        Means and standard deviations in m.

    Raises
    ------
    ValueError
        If `results` is empty.
    """
    if len(results) == 0:
        raise ValueError("need at least one width result")
    d_x = np.array([r.d_x_m for r in results])
    d_y = np.array([r.d_y_m for r in results])
    n = len(results)
    return WidthStatistics(
        d_x_mean_m=float(d_x.mean()),
        d_x_std_m=float(d_x.std(ddof=1)) if n > 1 else np.nan,
        d_y_mean_m=float(d_y.mean()),
        d_y_std_m=float(d_y.std(ddof=1)) if n > 1 else np.nan,
        n=n,
    )


def _integration_area(
    moments: Moments, d_x: float, d_y: float, pixel_size_m: float, nx: int, ny: int
) -> Roi:
    """Rectangle of 3 * d_x by 3 * d_y centred on the centroid, clipped."""
    x_c_px = moments.x_c_m / pixel_size_m + (nx - 1) / 2.0
    y_c_px = moments.y_c_m / pixel_size_m + (ny - 1) / 2.0
    half_x_px = 1.5 * d_x / pixel_size_m
    half_y_px = 1.5 * d_y / pixel_size_m
    # Include every pixel whose centre lies inside the rectangle.
    return Roi(
        x_start=max(0, int(np.ceil(x_c_px - half_x_px))),
        x_stop=min(nx, int(np.floor(x_c_px + half_x_px)) + 1),
        y_start=max(0, int(np.ceil(y_c_px - half_y_px))),
        y_stop=min(ny, int(np.floor(y_c_px + half_y_px)) + 1),
    )


def _azimuth_and_ellipticity(moments: Moments) -> tuple[float, float]:
    """Major-axis azimuth in rad and minor/major principal width ratio."""
    var_x, var_y, cov = moments.var_x_m2, moments.var_y_m2, moments.cov_xy_m2
    azimuth = 0.5 * np.arctan2(2.0 * cov, var_x - var_y)
    mean = 0.5 * (var_x + var_y)
    half_diff = np.hypot(0.5 * (var_x - var_y), cov)
    var_major, var_minor = mean + half_diff, mean - half_diff
    ellipticity = np.sqrt(var_minor / var_major) if var_minor > 0 else np.nan
    return float(azimuth), float(ellipticity)


def _failed_result(moments: Moments, iterations: int) -> WidthResult:
    """Result for a frame whose second moments are not positive."""
    return WidthResult(
        d_x_m=np.nan,
        d_y_m=np.nan,
        x_c_m=moments.x_c_m,
        y_c_m=moments.y_c_m,
        azimuth_rad=np.nan,
        ellipticity=np.nan,
        iterations=iterations,
        converged=False,
    )
