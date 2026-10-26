"""Known-answer tests for the beam simulator."""

from dataclasses import replace

import numpy as np
import pytest

from beampilot.simulator import (
    BeamParams,
    CameraConfig,
    beam_diameter,
    dark_frame,
    rayleigh_length,
    render_frame,
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
NOISELESS = replace(CAMERA, offset_counts=0.0, read_noise_counts=0.0)


def _second_moment_diameters(
    frame: np.ndarray, pixel_size_m: float
) -> tuple[float, float]:
    """Return 4-sigma diameters (d_x, d_y) in m of an offset-free frame."""
    image = frame.astype(float)
    y, x = np.indices(image.shape)
    total = image.sum()
    x_c = (image * x).sum() / total
    y_c = (image * y).sum() / total
    var_x = (image * (x - x_c) ** 2).sum() / total
    var_y = (image * (y - y_c) ** 2).sum() / total
    return 4.0 * np.sqrt(var_x) * pixel_size_m, 4.0 * np.sqrt(var_y) * pixel_size_m


def test_diameter_at_waist_equals_d0() -> None:
    d = beam_diameter(BEAM.z0x_m, BEAM.d0x_m, BEAM.z0x_m, BEAM.m2x, BEAM.wavelength_m)
    assert d == pytest.approx(BEAM.d0x_m, rel=1e-12)


def test_diameter_one_rayleigh_length_from_waist() -> None:
    z_r = rayleigh_length(BEAM.d0x_m, BEAM.m2x, BEAM.wavelength_m)
    d = beam_diameter(
        BEAM.z0x_m + z_r, BEAM.d0x_m, BEAM.z0x_m, BEAM.m2x, BEAM.wavelength_m
    )
    assert d == pytest.approx(BEAM.d0x_m * np.sqrt(2.0), rel=1e-12)


def test_noiseless_frame_matches_diameters() -> None:
    z_m = 0.25  # away from both waists, so d_x and d_y differ clearly
    frame = render_frame(BEAM, NOISELESS, z_m, 20.0, np.random.default_rng(0))
    assert frame.max() < NOISELESS.full_scale_counts  # unsaturated

    d_x, d_y = _second_moment_diameters(frame, NOISELESS.pixel_size_m)
    expected_x = beam_diameter(z_m, BEAM.d0x_m, BEAM.z0x_m, BEAM.m2x, BEAM.wavelength_m)
    expected_y = beam_diameter(z_m, BEAM.d0y_m, BEAM.z0y_m, BEAM.m2y, BEAM.wavelength_m)
    assert d_x == pytest.approx(expected_x, rel=0.01)
    assert d_y == pytest.approx(expected_y, rel=0.01)


def test_frame_sum_is_proportional_to_exposure() -> None:
    z_m = 0.25  # beam edge is > 7 sigma from the sensor edge: nothing truncated
    single = render_frame(BEAM, NOISELESS, z_m, 5.0, np.random.default_rng(0))
    double = render_frame(BEAM, NOISELESS, z_m, 10.0, np.random.default_rng(0))
    assert double.max() < NOISELESS.full_scale_counts  # both unsaturated

    expected = NOISELESS.signal_counts_per_ms * 5.0
    assert single.sum() == pytest.approx(expected, rel=0.005)
    assert double.sum() == pytest.approx(2.0 * single.sum(), rel=0.005)


def test_saturation_depends_on_exposure_and_position() -> None:
    full_scale = CAMERA.full_scale_counts
    near_waist = BEAM.z0x_m
    far = 0.60

    high = render_frame(BEAM, CAMERA, near_waist, 100.0, np.random.default_rng(1))
    low = render_frame(BEAM, CAMERA, near_waist, 1.0, np.random.default_rng(1))
    high_far = render_frame(BEAM, CAMERA, far, 100.0, np.random.default_rng(1))

    assert (high == full_scale).any()
    assert not (low == full_scale).any()
    assert not (high_far == full_scale).any()


def test_same_seed_gives_identical_frame() -> None:
    a = render_frame(BEAM, CAMERA, 0.2, 5.0, np.random.default_rng(42))
    b = render_frame(BEAM, CAMERA, 0.2, 5.0, np.random.default_rng(42))
    c = render_frame(BEAM, CAMERA, 0.2, 5.0, np.random.default_rng(43))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wavelength_m", 0.0),
        ("d0x_m", 0.0),
        ("d0y_m", -1e-3),
        ("m2x", 0.99),
        ("m2y", float("nan")),
    ],
)
def test_invalid_beam_params_raise(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        replace(BEAM, **{field: value})


def test_m2_of_one_is_valid() -> None:
    beam = replace(BEAM, m2x=1.0, m2y=1.0)
    assert beam.m2x == beam.m2y == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("pixel_size_m", 0.0), ("pixel_size_m", -5e-6), ("bit_depth", 0)],
)
def test_invalid_camera_config_raises(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        replace(CAMERA, **{field: value})


@pytest.mark.parametrize("exposure_ms", [0.0, -1.0])
def test_non_positive_exposure_raises(exposure_ms: float) -> None:
    with pytest.raises(ValueError, match="exposure_ms"):
        render_frame(BEAM, CAMERA, 0.2, exposure_ms, np.random.default_rng(0))


def test_dark_frame_has_offset_and_read_noise() -> None:
    frame = dark_frame(CAMERA, np.random.default_rng(3))
    assert frame.shape == (CAMERA.ny, CAMERA.nx)
    assert frame.mean() == pytest.approx(CAMERA.offset_counts, abs=0.1)
    # Rounding adds a variance of 1/12 count², negligible against 5 counts.
    assert frame.std() == pytest.approx(CAMERA.read_noise_counts, rel=0.02)
