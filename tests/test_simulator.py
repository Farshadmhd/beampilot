"""Known-answer tests for the beam simulator."""

from dataclasses import replace

import numpy as np
import pytest

from beampilot.simulator import (
    BeamParams,
    CameraConfig,
    Ghost,
    beam_diameter,
    dark_frame,
    rayleigh_length,
    render_frame,
)
from beampilot.widths import Roi, measure_widths, second_moments

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


def _centroid(frame: np.ndarray) -> tuple[float, float]:
    """Return the centroid (x_c, y_c) in pixels of an offset-free frame."""
    image = frame.astype(float)
    y, x = np.indices(image.shape)
    total = image.sum()
    return (image * x).sum() / total, (image * y).sum() / total


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
    assert d_x == pytest.approx(expected_x, rel=0.002)
    assert d_y == pytest.approx(expected_y, rel=0.002)


def test_frame_sum_is_proportional_to_exposure() -> None:
    z_m = 0.25  # beam edge is > 7 sigma from the sensor edge: nothing truncated
    single = render_frame(BEAM, NOISELESS, z_m, 5.0, np.random.default_rng(0))
    double = render_frame(BEAM, NOISELESS, z_m, 10.0, np.random.default_rng(0))
    assert double.max() < NOISELESS.full_scale_counts  # both unsaturated

    expected = NOISELESS.signal_counts_per_ms * 5.0
    assert single.sum() == pytest.approx(expected, rel=0.005)
    assert double.sum() == pytest.approx(2.0 * single.sum(), rel=0.005)


def test_beam_position_moves_centroid() -> None:
    x0_m, y0_m = 200e-6, -100e-6
    rng = np.random.default_rng(0)
    centred = render_frame(BEAM, NOISELESS, 0.25, 20.0, rng)
    shifted = render_frame(BEAM, NOISELESS, 0.25, 20.0, rng, x0_m=x0_m, y0_m=y0_m)

    (xc0, yc0), (xc1, yc1) = _centroid(centred), _centroid(shifted)
    assert xc0 == pytest.approx((NOISELESS.nx - 1) / 2.0, abs=0.01)
    assert yc0 == pytest.approx((NOISELESS.ny - 1) / 2.0, abs=0.01)
    assert xc1 - xc0 == pytest.approx(x0_m / NOISELESS.pixel_size_m, abs=0.01)
    assert yc1 - yc0 == pytest.approx(y0_m / NOISELESS.pixel_size_m, abs=0.01)


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


def test_shot_noise_variance_matches_signal() -> None:
    # A 50 mm beam is flat to < 0.01 % over the central 64 x 64 pixels.
    flat_beam = replace(
        BEAM, d0x_m=50e-3, d0y_m=50e-3, z0x_m=0.0, z0y_m=0.0, m2x=1.0, m2y=1.0
    )
    bright = replace(CAMERA, signal_counts_per_ms=6.5e10)  # ~2000 counts per pixel
    patch = np.s_[224:288, 224:288]
    rng = np.random.default_rng(5)

    with_shot = render_frame(flat_beam, bright, 0.0, 1.0, rng)[patch]
    no_shot_camera = replace(bright, shot_noise=False)
    without_shot = render_frame(flat_beam, no_shot_camera, 0.0, 1.0, rng)[patch]

    signal = with_shot.mean() - bright.offset_counts
    read_variance = bright.read_noise_counts**2
    assert with_shot.var() == pytest.approx(signal + read_variance, rel=0.1)
    assert without_shot.var() == pytest.approx(read_variance, rel=0.1)


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


# Ghost reflection


GHOST_Z_M = 0.25  # d_x ~ 698 um, d_y ~ 772 um


def _frame(ghost=None, z_m=GHOST_Z_M) -> np.ndarray:
    return render_frame(
        BEAM, NOISELESS, z_m, 10.0, np.random.default_rng(0), ghost=ghost
    ).astype(np.float64)


def test_ghost_second_moments_match_the_two_spot_mixture() -> None:
    """Two equal-width spots, power ratio r, offset dx: the mixture has
    sigma_x² = sigma² + w1 w2 dx² (w1 = 1/(1+r), w2 = r/(1+r)), its centroid
    moves by w2 dx, and the y moment is unchanged."""
    r, dx = 0.1, 0.3e-3
    full = Roi(0, NOISELESS.nx, 0, NOISELESS.ny)
    clean = second_moments(_frame(), NOISELESS.pixel_size_m, full)
    ghosted = second_moments(
        _frame(Ghost(relative_power=r, offset_x_m=dx, offset_y_m=0.0)),
        NOISELESS.pixel_size_m,
        full,
    )
    w1, w2 = 1 / (1 + r), r / (1 + r)
    # Integer rounding of the faint tails costs ~0.1 % of the shift.
    assert ghosted.x_c_m - clean.x_c_m == pytest.approx(w2 * dx, rel=2e-3)
    assert ghosted.var_x_m2 == pytest.approx(clean.var_x_m2 + w1 * w2 * dx**2, rel=1e-3)
    assert ghosted.var_y_m2 == pytest.approx(clean.var_y_m2, rel=1e-3)
    assert ghosted.total_counts == pytest.approx(clean.total_counts * (1 + r), rel=1e-3)


@pytest.mark.parametrize(
    ("offset_x_m", "offset_y_m"), [(0.3e-3, 0.0), (0.0, -0.5e-3), (0.4e-3, 0.4e-3)]
)
def test_ghost_makes_iso_widths_larger(offset_x_m: float, offset_y_m: float) -> None:
    pixel = NOISELESS.pixel_size_m
    clean = measure_widths(_frame(), pixel)
    ghosted = measure_widths(
        _frame(Ghost(relative_power=0.1, offset_x_m=offset_x_m, offset_y_m=offset_y_m)),
        pixel,
    )
    if offset_x_m:
        assert ghosted.d_x_m > 1.01 * clean.d_x_m
    if offset_y_m:
        assert ghosted.d_y_m > 1.01 * clean.d_y_m
    if not offset_x_m:
        assert ghosted.d_x_m == pytest.approx(clean.d_x_m, rel=2e-3)


def test_ghost_only_inside_its_z_window_and_offset_follows_z() -> None:
    ghost = Ghost(
        relative_power=0.2,
        offset_x_m=0.1e-3,
        offset_y_m=0.0,
        offset_x_per_m=2e-3,  # +2 um per mm of z
        z_min_m=0.20,
        z_max_m=0.30,
    )
    assert np.array_equal(_frame(ghost, z_m=0.15), _frame(None, z_m=0.15))
    assert not np.array_equal(_frame(ghost, z_m=0.25), _frame(None, z_m=0.25))
    assert ghost.offset_at(0.25) == pytest.approx((0.1e-3 + 0.5e-3, 0.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relative_power": 0.0},
        {"relative_power": 0.1, "size_factor": 0.0},
        {"relative_power": 0.1, "z_min_m": 0.2, "z_max_m": 0.1},
        {"relative_power": 0.1, "offset_x_m": float("nan")},
        {"relative_power": 0.1, "offset_y_per_m": float("inf")},
    ],
)
def test_ghost_rejects_bad_parameters(kwargs) -> None:
    kwargs = {"offset_x_m": 0.0, "offset_y_m": 0.0, **kwargs}
    with pytest.raises(ValueError):
        Ghost(**kwargs)
