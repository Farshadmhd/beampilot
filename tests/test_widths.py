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
    Roi,
    mean_dark,
    measure_widths,
    second_moments,
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


@pytest.mark.parametrize(
    "roi", [Roi(0, 512, 0, 512), Roi(18, 453, 89, 408)], ids=["full", "8sigma"]
)
def test_second_moments_of_analytic_gaussian(roi: Roi) -> None:
    """Rotated, off-centre Gaussian sampled at pixel centres.

    Expected: centroid (x0, y0); sigma_x² = a² cos² + b² sin²,
    sigma_y² = a² sin² + b² cos², sigma_xy = (a² - b²) sin cos; sum = 2 pi a b A.
    Projected sigmas are 27.0 px (x) and 19.8 px (y); both ROIs reach at least
    8 sigma from the centroid, and sampling at >= 15 px sigma is exact for a
    Gaussian, so both effects stay far below 1e-9 relative.
    """
    n, peak = 512, 1000.0
    a_px, b_px, angle = 30.0, 15.0, np.radians(30.0)
    x0_px, y0_px = -20.3, -7.6  # relative to the image centre
    y, x = np.indices((n, n)) - (n - 1) / 2.0
    u = (x - x0_px) * np.cos(angle) + (y - y0_px) * np.sin(angle)
    v = -(x - x0_px) * np.sin(angle) + (y - y0_px) * np.cos(angle)
    image = peak * np.exp(-0.5 * (u / a_px) ** 2 - 0.5 * (v / b_px) ** 2)

    moments = second_moments(image, PIXEL_M, roi)

    c, s = np.cos(angle), np.sin(angle)
    p2 = PIXEL_M**2
    assert moments.x_c_m == pytest.approx(x0_px * PIXEL_M, rel=1e-9)
    assert moments.y_c_m == pytest.approx(y0_px * PIXEL_M, rel=1e-9)
    assert moments.var_x_m2 == pytest.approx((a_px**2 * c**2 + b_px**2 * s**2) * p2)
    assert moments.var_y_m2 == pytest.approx((a_px**2 * s**2 + b_px**2 * c**2) * p2)
    assert moments.cov_xy_m2 == pytest.approx((a_px**2 - b_px**2) * s * c * p2)
    assert moments.total_counts == pytest.approx(2 * np.pi * a_px * b_px * peak)


def test_noiseless_centred_beam() -> None:
    frame = render_frame(BEAM, NOISELESS, Z_M, EXPOSURE_MS, np.random.default_rng(0))
    result = measure_widths(frame, PIXEL_M)
    true_x, true_y = _true_diameters(BEAM, Z_M)
    assert result.converged
    assert result.d_x_m == pytest.approx(true_x, rel=0.005)
    assert result.d_y_m == pytest.approx(true_y, rel=0.005)


def test_centred_beam_roi_not_clipped_and_holds_the_signal() -> None:
    frame = render_frame(BEAM, NOISELESS, Z_M, EXPOSURE_MS, np.random.default_rng(0))
    result = measure_widths(frame, PIXEL_M)
    assert not result.roi_clipped
    roi = result.roi
    assert 0 < roi.x_start < roi.x_stop < NOISELESS.nx
    assert 0 < roi.y_start < roi.y_stop < NOISELESS.ny
    # The 3 x d rectangle spans +-6 sigma, so it holds practically all signal.
    total_counts = NOISELESS.signal_counts_per_ms * EXPOSURE_MS
    assert result.roi_sum_counts == pytest.approx(total_counts, rel=0.005)
    assert result.roi_sum_counts == pytest.approx(
        frame[roi.y_start : roi.y_stop, roi.x_start : roi.x_stop].sum()
    )


def test_beam_cut_by_sensor_edge_flags_roi_clipped() -> None:
    """A beam 1.35 mm off-centre on a 2.8 mm sensor converges ~34 % too small.

    Nothing else in the result reveals this, so roi_clipped must.
    """
    rng = np.random.default_rng(0)
    frame = render_frame(BEAM, NOISELESS, Z_M, EXPOSURE_MS, rng, x0_m=1.35e-3)
    result = measure_widths(frame, PIXEL_M)
    true_x, _ = _true_diameters(BEAM, Z_M)
    assert result.converged
    assert result.d_x_m < 0.8 * true_x  # the bias the flag warns about
    assert result.roi_clipped
    assert result.roi.x_stop == NOISELESS.nx


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


def test_dim_noisy_frames_all_converge() -> None:
    """At peak ~430 counts the 3 x d rectangle can flip by one pixel each pass.

    Each frame gets its own dark set, as in a real run. Before the
    repeated-rectangle stopping rule, 20 of the first 100 seeds hit max_iter.
    """
    dim_exposure_ms = 3.0  # peak ~ 430 counts at Z_M
    # Only convergence is checked: at this SNR single-frame widths scatter by
    # ~5.5 % (1 sigma, up to ~20 %), which a quality flag has to catch.
    for seed in range(50):
        dark = mean_dark(_darks(CAMERA, 10, seed=seed + 10_000))
        rng = np.random.default_rng(seed)
        frame = render_frame(BEAM, CAMERA, Z_M, dim_exposure_ms, rng)
        image, _ = subtract_background(frame, dark)
        result = measure_widths(image, PIXEL_M, max_iter=25)
        assert result.converged, f"seed {seed}"
        assert result.iterations < 25, f"seed {seed}"
        assert np.isfinite(result.d_x_m) and np.isfinite(result.d_y_m)


def test_small_beam_with_residual_offset_is_measured() -> None:
    """A 200 um beam holds only ~3e6 counts at 70 % of full scale.

    A residual offset of ~0.2 counts (the corner estimate's standard error is
    ~0.15) summed with r² weights over the full sensor outweighs the beam's
    sigma², so a full-image first pass gave negative second moments on
    several of these frames. The 1/e² start area avoids that.
    """
    small = replace(BEAM, d0x_m=0.2e-3, d0y_m=0.2e-3, z0x_m=0.0, z0y_m=0.0)
    small = replace(small, m2x=1.0, m2y=1.0)
    exposure_ms = 1.5  # peak ~ 2900 counts
    dark = mean_dark(_darks(CAMERA, 10, seed=0))
    rng = np.random.default_rng(7)
    results = []
    for _ in range(20):
        frame = render_frame(small, CAMERA, 0.0, exposure_ms, rng)
        image, _ = subtract_background(frame, dark)
        result = measure_widths(image, PIXEL_M)
        assert result.converged and not result.roi_clipped
        results.append(result)
    stats = width_statistics(results)
    assert stats.d_x_mean_m == pytest.approx(0.2e-3, rel=0.01)
    assert stats.d_y_mean_m == pytest.approx(0.2e-3, rel=0.01)


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


@pytest.mark.parametrize(
    "roi",
    [
        Roi(-1, 10, 0, 10),  # negative start
        Roi(0, 65, 0, 10),  # past the right edge
        Roi(0, 10, 0, 65),  # past the bottom edge
        Roi(10, 10, 0, 10),  # empty
        Roi(20, 10, 0, 10),  # reversed
    ],
)
def test_second_moments_rejects_bad_roi(roi: Roi) -> None:
    with pytest.raises(ValueError, match="roi"):
        second_moments(np.ones((64, 64)), PIXEL_M, roi)


@pytest.mark.parametrize("pixel_size_m", [0.0, -5.5e-6, np.nan])
def test_non_positive_pixel_size_raises(pixel_size_m: float) -> None:
    image = _rotated_ellipse(10.0, 10.0, 0.0, n=64)
    with pytest.raises(ValueError, match="pixel_size_m"):
        second_moments(image, pixel_size_m, Roi(0, 64, 0, 64))
    with pytest.raises(ValueError, match="pixel_size_m"):
        measure_widths(image, pixel_size_m)


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
    # 4 corner patches of 18 x 18 px on a 512 x 512 image.
    n_corner = 4 * 18 * 18
    assert info.offset_std_err_counts == pytest.approx(
        info.std_counts / np.sqrt(n_corner)
    )
    assert len(info.corner_means_counts) == 4
    assert image.mean() == pytest.approx(0.0, abs=CORNER_MEAN_TOL)
