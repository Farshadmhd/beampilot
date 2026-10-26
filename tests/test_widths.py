"""Known-answer tests for ISO 11146 beam widths, using the simulator."""

from dataclasses import replace

import numpy as np
import pytest

from beampilot.simulator import (
    BeamParams,
    CameraConfig,
    beam_diameter,
    dark_frame,
    render_frame,
)
from beampilot.widths import (
    mean_dark,
    measure_widths,
    subtract_background,
    width_statistics,
)

# Astigmatic test beam: different waist sizes, positions and M² per axis.
BEAM = BeamParams(
    wavelength_m=1064e-9,
    d0x_m=0.5e-3,
    d0y_m=0.4e-3,
    z0x_m=0.10,
    z0y_m=0.12,
    m2x=1.2,
    m2y=1.5,
)
CAMERA = CameraConfig(
    pixel_size_m=5.5e-6,
    nx=512,
    ny=512,
    bit_depth=12,
    offset_counts=100.0,
    read_noise_counts=5.0,
    signal_counts_per_ms=1e6,
)


NOISELESS = replace(CAMERA, offset_counts=0.0, read_noise_counts=0.0, shot_noise=False)
PIXEL_M = CAMERA.pixel_size_m
Z_M = 0.25  # away from both waists: d_x ~ 698 um, d_y ~ 772 um
EXPOSURE_MS = 20.0  # peak ~ 2900 of 4095 counts at Z_M: unsaturated

# 4 sigma of the corner-patch mean: 4 patches of 18 x 18 px, ~5.2 counts noise each.
CORNER_MEAN_TOL = 4 * 5.2 / np.sqrt(4 * 18 * 18)


def _darks(camera: CameraConfig, n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [dark_frame(camera, rng) for _ in range(n)]


def _true_diameters(beam: BeamParams, z_m: float) -> tuple[float, float]:
    d_x = beam_diameter(z_m, beam.d0x_m, beam.z0x_m, beam.m2x, beam.wavelength_m)
    d_y = beam_diameter(z_m, beam.d0y_m, beam.z0y_m, beam.m2y, beam.wavelength_m)
    return float(d_x), float(d_y)


def _rotated_ellipse(
    sigma_major_px: float, sigma_minor_px: float, angle_rad: float, n: int = 512
) -> np.ndarray:
    """Noiseless elliptical Gaussian, major axis at `angle_rad` from x."""
    y, x = np.indices((n, n)) - (n - 1) / 2.0
    u = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    v = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return 1000.0 * np.exp(
        -0.5 * (u / sigma_major_px) ** 2 - 0.5 * (v / sigma_minor_px) ** 2
    )


def test_noiseless_centred_beam() -> None:
    frame = render_frame(BEAM, NOISELESS, Z_M, EXPOSURE_MS, np.random.default_rng(0))
    result = measure_widths(frame, PIXEL_M)
    true_x, true_y = _true_diameters(BEAM, Z_M)
    assert result.converged
    assert result.d_x_m == pytest.approx(true_x, rel=0.005)
    assert result.d_y_m == pytest.approx(true_y, rel=0.005)


def test_off_centre_beam_gives_same_width_and_correct_centroid() -> None:
    x0_m, y0_m = 300e-6, -200e-6
    rng = np.random.default_rng(0)
    frame = render_frame(BEAM, NOISELESS, Z_M, EXPOSURE_MS, rng, x0_m=x0_m, y0_m=y0_m)
    result = measure_widths(frame, PIXEL_M)
    true_x, true_y = _true_diameters(BEAM, Z_M)
    assert result.d_x_m == pytest.approx(true_x, rel=0.005)
    assert result.d_y_m == pytest.approx(true_y, rel=0.005)
    assert result.x_c_m == pytest.approx(x0_m, abs=0.05 * PIXEL_M)
    assert result.y_c_m == pytest.approx(y0_m, abs=0.05 * PIXEL_M)


def test_astigmatic_beam_recovers_both_widths() -> None:
    beam = replace(
        BEAM, d0x_m=0.3e-3, d0y_m=0.6e-3, z0x_m=0.0, z0y_m=0.0, m2x=1.0, m2y=1.0
    )
    frame = render_frame(beam, NOISELESS, 0.0, 4.0, np.random.default_rng(0))
    result = measure_widths(frame, PIXEL_M)
    assert result.d_x_m == pytest.approx(0.3e-3, rel=0.005)
    assert result.d_y_m == pytest.approx(0.6e-3, rel=0.005)
    assert result.ellipticity == pytest.approx(0.5, rel=0.01)
    assert abs(result.azimuth_rad) == pytest.approx(np.pi / 2, abs=np.radians(0.5))


@pytest.mark.parametrize("angle_deg", [25.0, -60.0])
def test_rotated_ellipse_azimuth_recovered(angle_deg: float) -> None:
    angle = np.radians(angle_deg)
    sigma_major, sigma_minor = 40.0, 20.0
    image = _rotated_ellipse(sigma_major, sigma_minor, angle)
    result = measure_widths(image, PIXEL_M)

    assert result.azimuth_rad == pytest.approx(angle, abs=np.radians(0.5))
    assert result.ellipticity == pytest.approx(sigma_minor / sigma_major, rel=0.01)
    # Lab-axis width of a rotated ellipse: sigma_x² = a² cos² + b² sin².
    var_x_px = (sigma_major * np.cos(angle)) ** 2 + (sigma_minor * np.sin(angle)) ** 2
    assert result.d_x_m == pytest.approx(4 * np.sqrt(var_x_px) * PIXEL_M, rel=0.005)


def test_iteration_converges_before_max_iter() -> None:
    dark = mean_dark(_darks(CAMERA, 10, seed=0))
    frame = render_frame(BEAM, CAMERA, Z_M, EXPOSURE_MS, np.random.default_rng(1))
    image, _ = subtract_background(frame, dark)
    result = measure_widths(image, PIXEL_M, max_iter=25)
    assert result.converged
    assert result.iterations < 25


def test_realistic_frames_mean_of_five_repeats() -> None:
    dark = mean_dark(_darks(CAMERA, 10, seed=0))
    rng = np.random.default_rng(1)
    results = []
    for _ in range(5):
        frame = render_frame(BEAM, CAMERA, Z_M, EXPOSURE_MS, rng)
        image, _ = subtract_background(frame, dark)
        results.append(measure_widths(image, PIXEL_M))

    stats = width_statistics(results)
    true_x, true_y = _true_diameters(BEAM, Z_M)
    assert stats.n == 5
    assert stats.d_x_mean_m == pytest.approx(true_x, rel=0.02)
    assert stats.d_y_mean_m == pytest.approx(true_y, rel=0.02)
    assert 0 < stats.d_x_std_m < 0.02 * true_x
    assert 0 < stats.d_y_std_m < 0.02 * true_y


def test_skipping_fine_offset_makes_beam_look_larger() -> None:
    """Negative control: why the corner offset correction matters."""
    dark = mean_dark(_darks(CAMERA, 10, seed=0))
    # Background 3 counts higher than in the dark frames (0.1 % of the peak).
    brighter = replace(CAMERA, offset_counts=CAMERA.offset_counts + 3.0)
    frame = render_frame(BEAM, brighter, Z_M, EXPOSURE_MS, np.random.default_rng(1))
    true_x, true_y = _true_diameters(BEAM, Z_M)

    dark_only = measure_widths(frame - dark, PIXEL_M)
    corrected_image, _ = subtract_background(frame, dark)
    corrected = measure_widths(corrected_image, PIXEL_M)

    assert dark_only.d_x_m > 1.15 * true_x
    assert dark_only.d_y_m > 1.15 * true_y
    assert corrected.d_x_m == pytest.approx(true_x, rel=0.02)
    assert corrected.d_y_m == pytest.approx(true_y, rel=0.02)


def test_width_statistics_single_and_empty() -> None:
    image = _rotated_ellipse(40.0, 20.0, 0.0)
    single = width_statistics([measure_widths(image, PIXEL_M)])
    assert single.n == 1
    assert np.isnan(single.d_x_std_m)
    with pytest.raises(ValueError):
        width_statistics([])


@pytest.mark.parametrize("level", [0.0, -1.0])
def test_no_beam_gives_nan_not_crash(level: float) -> None:
    image = np.full((64, 64), level)
    result = measure_widths(image, PIXEL_M)
    assert np.isnan(result.d_x_m) and np.isnan(result.d_y_m)
    assert not result.converged


def test_fewer_than_ten_dark_frames_raises() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        mean_dark(_darks(CAMERA, 9, seed=0))


def test_mean_dark_averages_frames() -> None:
    dark = mean_dark(_darks(CAMERA, 10, seed=0))
    assert dark.shape == (CAMERA.ny, CAMERA.nx)
    assert dark.mean() == pytest.approx(CAMERA.offset_counts, abs=0.1)
    # Averaging 10 frames reduces the pixel noise by sqrt(10).
    assert dark.std() == pytest.approx(CAMERA.read_noise_counts / np.sqrt(10), rel=0.05)


def test_negative_values_survive_background_subtraction() -> None:
    dark = mean_dark(_darks(CAMERA, 10, seed=0))
    frame = dark_frame(CAMERA, np.random.default_rng(1))  # no beam: pure noise
    image, _ = subtract_background(frame, dark)
    assert image.dtype == np.float64
    assert (image < 0).sum() > 0.4 * image.size  # about half the pixels
    assert image.mean() == pytest.approx(0.0, abs=CORNER_MEAN_TOL)


def test_fine_offset_removes_residual_background() -> None:
    dark = mean_dark(_darks(CAMERA, 10, seed=0))
    extra_counts = 3.0  # e.g. room light switched on after the dark frames
    frame = dark_frame(CAMERA, np.random.default_rng(1)) + extra_counts
    image, info = subtract_background(frame, dark)
    assert info.offset_counts == pytest.approx(extra_counts, abs=CORNER_MEAN_TOL)
    assert info.std_counts == pytest.approx(
        CAMERA.read_noise_counts * np.sqrt(1 + 1 / 10), rel=0.1
    )
    assert len(info.corner_means_counts) == 4
    assert image.mean() == pytest.approx(0.0, abs=CORNER_MEAN_TOL)
