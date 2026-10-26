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
