"""Simulated laser beam and camera.

:class:`BeamParams` is the hidden ground truth: the agent never sees it, only
the camera frames rendered from it. All lengths are in metres (SI). Beams with
M² > 1 are modelled as an "embedded Gaussian", i.e. a Gaussian profile whose
width follows the M² caustic. Frames have shape ``(ny, nx)``: rows are y,
columns are x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BeamParams:
    """Ground-truth parameters of a beam with principal axes along x and y.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in m.
    d0x_m, d0y_m : float
        Second-moment (4 sigma) waist diameters along x and y, in m.
    z0x_m, z0y_m : float
        Waist positions along the propagation axis for x and y, in m.
    m2x, m2y : float
        Beam propagation ratios M² along x and y (dimensionless, >= 1).

    Raises
    ------
    ValueError
        If the wavelength or a waist diameter is not positive, or M² < 1.
    """

    wavelength_m: float
    d0x_m: float
    d0y_m: float
    z0x_m: float
    z0y_m: float
    m2x: float
    m2y: float

    def __post_init__(self) -> None:
        # Written as "not x > limit" so that NaN is rejected too.
        for name in ("wavelength_m", "d0x_m", "d0y_m"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("m2x", "m2y"):
            if not getattr(self, name) >= 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")


@dataclass(frozen=True)
class CameraConfig:
    """Simulated camera sensor.

    Parameters
    ----------
    pixel_size_m : float
        Pixel pitch in m (square pixels).
    nx, ny : int
        Number of pixels along x (columns) and y (rows).
    bit_depth : int
        ADC bit depth; full scale is ``2**bit_depth - 1`` counts.
    offset_counts : float
        Constant dark offset in counts. Keep ``offset_counts >= 4 *
        read_noise_counts``: pixel values are clipped at 0, so a smaller offset
        biases the background upwards.
    read_noise_counts : float
        Standard deviation of the Gaussian read noise in counts.
    signal_counts_per_ms : float
        Brightness: total beam signal summed over an unbounded sensor, in
        counts per ms of exposure (before clipping). A beam larger than the
        sensor is truncated, not rescaled: signal outside the sensor is lost.
    shot_noise : bool, optional
        If True (default), apply Poisson shot noise to the beam signal. Assumes
        a gain of one count per photo-electron.

    Raises
    ------
    ValueError
        If the pixel size is not positive or the bit depth is below 1.
    """

    pixel_size_m: float
    nx: int
    ny: int
    bit_depth: int
    offset_counts: float
    read_noise_counts: float
    signal_counts_per_ms: float
    shot_noise: bool = True

    def __post_init__(self) -> None:
        if not self.pixel_size_m > 0:
            raise ValueError(f"pixel_size_m must be > 0, got {self.pixel_size_m}")
        if not self.bit_depth >= 1:
            raise ValueError(f"bit_depth must be >= 1, got {self.bit_depth}")

    @property
    def full_scale_counts(self) -> int:
        """Largest pixel value in counts, ``2**bit_depth - 1``."""
        return 2**self.bit_depth - 1


@dataclass(frozen=True)
class Ghost:
    """A second, weaker Gaussian spot on the sensor, e.g. a reflection.

    The ghost follows the main beam's caustic, scaled by `size_factor`, and
    sits at an offset from the main beam's centre. The offset can change
    linearly with z, and the ghost can be limited to a z window.

    Parameters
    ----------
    relative_power : float
        Ghost power relative to the main beam (> 0), e.g. 0.1.
    offset_x_m, offset_y_m : float
        Ghost centre relative to the main beam centre at z = 0, in m.
    offset_x_per_m, offset_y_per_m : float, optional
        Change of the offset per m of z (dimensionless). Default 0.
    size_factor : float, optional
        Ghost width relative to the main beam width at the same z (> 0).
        Default 1.
    z_min_m, z_max_m : float, optional
        The ghost is present only for z_min <= z <= z_max, in m. Default:
        everywhere.

    Raises
    ------
    ValueError
        If the power or size factor is not positive or the z window is empty.
    """

    relative_power: float
    offset_x_m: float
    offset_y_m: float
    offset_x_per_m: float = 0.0
    offset_y_per_m: float = 0.0
    size_factor: float = 1.0
    z_min_m: float = -np.inf
    z_max_m: float = np.inf

    def __post_init__(self) -> None:
        if not self.relative_power > 0:
            raise ValueError(f"relative_power must be > 0, got {self.relative_power}")
        if not self.size_factor > 0:
            raise ValueError(f"size_factor must be > 0, got {self.size_factor}")
        if not self.z_max_m >= self.z_min_m:
            raise ValueError("z_max_m must be >= z_min_m")

    def present_at(self, z_m: float) -> bool:
        """True if the ghost is on the sensor at stage position `z_m` (m)."""
        return bool(self.z_min_m <= z_m <= self.z_max_m)

    def offset_at(self, z_m: float) -> tuple[float, float]:
        """Ghost offset from the main beam centre at `z_m`, in m."""
        return (
            self.offset_x_m + self.offset_x_per_m * z_m,
            self.offset_y_m + self.offset_y_per_m * z_m,
        )


def rayleigh_length(d0_m: float, m2: float, wavelength_m: float) -> float:
    """Rayleigh length zR = pi * d0**2 / (4 * M² * wavelength).

    Parameters
    ----------
    d0_m : float
        Second-moment (4 sigma) waist diameter in m.
    m2 : float
        Beam propagation ratio M² (dimensionless).
    wavelength_m : float
        Wavelength in m.

    Returns
    -------
    float
        Rayleigh length in m.
    """
    return np.pi * d0_m**2 / (4.0 * m2 * wavelength_m)


def beam_diameter(
    z_m: float | NDArray[np.floating],
    d0_m: float,
    z0_m: float,
    m2: float,
    wavelength_m: float,
) -> float | NDArray[np.floating]:
    """Second-moment (4 sigma) beam diameter d(z) = d0 * sqrt(1 + ((z - z0)/zR)**2).

    Parameters
    ----------
    z_m : float or ndarray
        Position(s) along the propagation axis in m.
    d0_m : float
        Second-moment waist diameter in m.
    z0_m : float
        Waist position in m.
    m2 : float
        Beam propagation ratio M² (dimensionless).
    wavelength_m : float
        Wavelength in m.

    Returns
    -------
    float or ndarray
        Beam diameter in m, same shape as `z_m`.
    """
    z_r = rayleigh_length(d0_m, m2, wavelength_m)
    return d0_m * np.sqrt(1.0 + ((z_m - z0_m) / z_r) ** 2)


def render_frame(
    beam: BeamParams,
    camera: CameraConfig,
    z_m: float,
    exposure_ms: float,
    rng: np.random.Generator,
    x0_m: float = 0.0,
    y0_m: float = 0.0,
    ghost: Ghost | None = None,
) -> NDArray[np.int32]:
    """Render one camera frame of the beam.

    The total beam signal is proportional to the exposure, so the peak pixel
    value is highest near the waist and saturation happens there first. Parts
    of the beam outside the sensor are truncated, not rescaled, so the frame
    sum drops when the beam outgrows the sensor (ISO clipping loss). An
    optional ghost adds a second, weaker spot (see :class:`Ghost`).

    Parameters
    ----------
    beam : BeamParams
        Ground-truth beam.
    camera : CameraConfig
        Camera sensor.
    z_m : float
        Stage position along the propagation axis in m.
    exposure_ms : float
        Exposure time in ms.
    rng : numpy.random.Generator
        Random generator for the shot and read noise.
    x0_m, y0_m : float, optional
        Beam centre relative to the sensor centre in m. Positive x0 moves the
        beam towards higher column indices, positive y0 towards higher row
        indices. Default 0.0 (centred).
    ghost : Ghost, optional
        Second, weaker spot (e.g. a reflection). Default None.

    Returns
    -------
    ndarray of int32, shape (ny, nx)
        Pixel values in counts, clipped to ``[0, full_scale_counts]``.

    Raises
    ------
    ValueError
        If `exposure_ms` is not positive.
    """
    if not exposure_ms > 0:
        raise ValueError(f"exposure_ms must be > 0, got {exposure_ms}")

    d_x = beam_diameter(z_m, beam.d0x_m, beam.z0x_m, beam.m2x, beam.wavelength_m)
    d_y = beam_diameter(z_m, beam.d0y_m, beam.z0y_m, beam.m2y, beam.wavelength_m)
    total_counts = camera.signal_counts_per_ms * exposure_ms
    signal_counts = _gaussian_spot(camera, total_counts, d_x, d_y, x0_m, y0_m)
    if ghost is not None and ghost.present_at(z_m):
        offset_x, offset_y = ghost.offset_at(z_m)
        signal_counts += _gaussian_spot(
            camera,
            total_counts * ghost.relative_power,
            d_x * ghost.size_factor,
            d_y * ghost.size_factor,
            x0_m + offset_x,
            y0_m + offset_y,
        )
    if camera.shot_noise:
        signal_counts = rng.poisson(signal_counts).astype(np.float64)
    return _digitise(signal_counts, camera, rng)


def _gaussian_spot(
    camera: CameraConfig,
    total_counts: float,
    d_x_m: float,
    d_y_m: float,
    x0_m: float,
    y0_m: float,
) -> NDArray[np.float64]:
    """Gaussian spot with 4-sigma widths d_x, d_y and `total_counts` in total.

    The peak is set so that the profile integrates to `total_counts` over an
    unbounded sensor; parts outside the sensor are lost.
    """
    sigma_x, sigma_y = d_x_m / 4.0, d_y_m / 4.0
    # Pixel-centre coordinates relative to the spot centre.
    x_m = (np.arange(camera.nx) - (camera.nx - 1) / 2.0) * camera.pixel_size_m - x0_m
    y_m = (np.arange(camera.ny) - (camera.ny - 1) / 2.0) * camera.pixel_size_m - y0_m
    peak_counts = (
        total_counts * camera.pixel_size_m**2 / (2.0 * np.pi * sigma_x * sigma_y)
    )
    profile = np.exp(
        -0.5 * (x_m[np.newaxis, :] / sigma_x) ** 2
        - 0.5 * (y_m[:, np.newaxis] / sigma_y) ** 2
    )
    return peak_counts * profile


def dark_frame(camera: CameraConfig, rng: np.random.Generator) -> NDArray[np.int32]:
    """Render a frame with no beam: offset plus read noise.

    Parameters
    ----------
    camera : CameraConfig
        Camera sensor.
    rng : numpy.random.Generator
        Random generator for the read noise.

    Returns
    -------
    ndarray of int32, shape (ny, nx)
        Pixel values in counts, clipped to ``[0, full_scale_counts]``.
    """
    return _digitise(np.zeros((camera.ny, camera.nx)), camera, rng)


def _digitise(
    signal_counts: NDArray[np.floating],
    camera: CameraConfig,
    rng: np.random.Generator,
) -> NDArray[np.int32]:
    """Add offset and read noise, round to integers, clip to the ADC range.

    Returns signed integers so that later dark-frame subtraction cannot wrap.
    """
    noise = rng.normal(0.0, camera.read_noise_counts, size=signal_counts.shape)
    counts = np.rint(signal_counts + camera.offset_counts + noise)
    return np.clip(counts, 0, camera.full_scale_counts).astype(np.int32)
