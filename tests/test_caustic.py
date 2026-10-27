"""Known-answer tests for the caustic fit, ISO coverage and astigmatism check."""

import numpy as np
import pytest

from beampilot.caustic import coverage, fit_caustic, general_astigmatism
from beampilot.simulator import (
    BeamParams,
    CameraConfig,
    beam_diameter,
    dark_frame,
    rayleigh_length,
    render_frame,
)
from beampilot.widths import (
    mean_dark,
    measure_widths,
    subtract_background,
    width_statistics,
)

WAVELENGTH_M = 1064e-9
D0_M = 0.5e-3
Z0_M = 0.10

# Offsets from the waist in units of zR: 5 within +-zR, 5 beyond 2 zR.
ISO_OFFSETS = np.array([-3.0, -2.2, -0.8, -0.4, 0.0, 0.4, 0.8, 2.2, 3.0, 3.8])
# Near-waist-only sampling: the beam stays within 12 % of d0. The penalty
# shrinks as the span grows: +-0.7 zR gives ~3x, +-1 zR only ~1.8x.
NEAR_ONLY_OFFSETS = np.linspace(-0.5, 0.5, 10)


def _diameters(offsets: np.ndarray, m2: float) -> tuple[np.ndarray, np.ndarray]:
    """Stage positions and true diameters in m for offsets in units of zR."""
    z_r = rayleigh_length(D0_M, m2, WAVELENGTH_M)
    z = Z0_M + offsets * z_r
    return z, beam_diameter(z, D0_M, Z0_M, m2, WAVELENGTH_M)


@pytest.mark.parametrize("m2", [1.0, 1.5])
def test_noiseless_caustic_recovers_beam(m2: float) -> None:
    z, d = _diameters(ISO_OFFSETS, m2)
    fit = fit_caustic(z, d, wavelength_m=WAVELENGTH_M)

    assert fit.success
    assert fit.n_points == 10
    assert fit.d0_m == pytest.approx(D0_M, rel=1e-6)
    assert fit.z0_m == pytest.approx(Z0_M, rel=1e-6)
    assert fit.m2 == pytest.approx(m2, rel=1e-6)
    z_r = rayleigh_length(D0_M, m2, WAVELENGTH_M)
    assert fit.z_r_m == pytest.approx(z_r, rel=1e-6)
    # Full angle: theta = d0 / zR = 4 M² lambda / (pi d0).
    theta = 4 * m2 * WAVELENGTH_M / (np.pi * D0_M)
    assert fit.theta_rad == pytest.approx(theta, rel=1e-6)
    assert fit.residual_rms < 1e-9


def test_noiseless_caustic_with_repeat_std() -> None:
    m2 = 1.5
    z, d = _diameters(ISO_OFFSETS, m2)
    fit = fit_caustic(z, d, 0.01 * d, wavelength_m=WAVELENGTH_M)
    assert fit.success
    assert fit.m2 == pytest.approx(m2, rel=1e-6)


@pytest.mark.parametrize("m2", [1.0, 1.5])
def test_two_percent_noise_mean_m2_unbiased(m2: float) -> None:
    z, d_true = _diameters(ISO_OFFSETS, m2)
    m2_fits, m2_stds = [], []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        d = d_true * (1.0 + 0.02 * rng.standard_normal(d_true.size))
        fit = fit_caustic(z, d, wavelength_m=WAVELENGTH_M)
        assert fit.success, f"seed {seed}"
        m2_fits.append(fit.m2)
        m2_stds.append(fit.m2_std)
    m2_fits = np.array(m2_fits)
    print(
        f"\nM² = {m2}, 2 % noise, 100 seeds: mean {m2_fits.mean():.4f} "
        f"(error {m2_fits.mean() / m2 - 1:+.2%}), "
        f"scatter std {m2_fits.std(ddof=1):.4f}, "
        f"mean fit uncertainty {np.mean(m2_stds):.4f}"
    )
    assert m2_fits.mean() == pytest.approx(m2, rel=0.01)
    # The Jacobian uncertainty should describe the actual scatter.
    assert np.mean(m2_stds) == pytest.approx(m2_fits.std(ddof=1), rel=0.3)


def test_near_waist_only_sampling_inflates_m2_uncertainty() -> None:
    """Without far-field points the divergence, and so M², is badly constrained."""
    m2 = 1.2
    z_iso, d_iso = _diameters(ISO_OFFSETS, m2)
    z_near, d_near = _diameters(NEAR_ONLY_OFFSETS, m2)
    iso_stds, near_stds = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        noise = 1.0 + 0.02 * rng.standard_normal(10)
        iso = fit_caustic(z_iso, d_iso * noise, wavelength_m=WAVELENGTH_M)
        near = fit_caustic(z_near, d_near * noise, wavelength_m=WAVELENGTH_M)
        iso_stds.append(iso.m2_std)
        near_stds.append(near.m2_std)
    ratio = np.median(near_stds) / np.median(iso_stds)
    print(f"\nnear-waist / ISO median M² uncertainty: {ratio:.1f}x")
    assert ratio >= 3.0


def test_fit_needs_four_points() -> None:
    z, d = _diameters(ISO_OFFSETS[:3], 1.0)
    with pytest.raises(ValueError, match="at least 4"):
        fit_caustic(z, d, wavelength_m=WAVELENGTH_M)


def test_fit_rejects_bad_input() -> None:
    z, d = _diameters(ISO_OFFSETS, 1.0)
    with pytest.raises(ValueError, match="wavelength"):
        fit_caustic(z, d, wavelength_m=0.0)
    with pytest.raises(ValueError, match="> 0"):
        fit_caustic(z, -d, wavelength_m=WAVELENGTH_M)
    with pytest.raises(ValueError, match="d_std_m"):
        fit_caustic(z, d, d[:5], wavelength_m=WAVELENGTH_M)
    with pytest.raises(ValueError, match="distinct"):
        fit_caustic([0.1, 0.1, 0.2, 0.2], d[:4], wavelength_m=WAVELENGTH_M)


def test_tiny_repeat_std_is_floored() -> None:
    """A zero std must not make one point dominate (floor at 1e-3 * d)."""
    m2 = 1.0
    z, d_true = _diameters(ISO_OFFSETS, m2)
    d = d_true * (1.0 + 0.02 * np.random.default_rng(3).standard_normal(10))
    d_std = 0.02 * d
    d_std[4] = 0.0
    fit = fit_caustic(z, d, d_std, wavelength_m=WAVELENGTH_M)
    assert fit.success
    assert np.isfinite(fit.m2_std)


def test_coverage_counts_distinct_positions() -> None:
    # z0 = 0.5 m, zR = 0.125 m; all values exact in binary floating point.
    z0, z_r = 0.5, 0.125
    z = [
        0.5,
        0.5,  # repeat: counted once
        0.375,  # exactly zR: near
        0.625,  # exactly zR: near
        0.4375,  # near
        0.6875,  # 1.5 zR: between
        0.25,  # exactly 2 zR: far
        0.75,  # exactly 2 zR: far
        0.0,  # 4 zR: far
        1.0,  # 4 zR: far
        1.0,  # repeat
    ]
    cov = coverage(z, z0, z_r)
    assert (cov.n_near, cov.n_far, cov.n_between) == (4, 4, 1)
    assert not cov.compliant
    assert coverage(z, z0, z_r, min_near=4, min_far=4).compliant
    assert coverage([*z, 0.5625, 1.125], z0, z_r).compliant  # 5 near, 5 far


def test_coverage_rejects_non_positive_rayleigh_length() -> None:
    with pytest.raises(ValueError, match="z_r_m"):
        coverage([0.0, 0.1], 0.05, 0.0)


@pytest.mark.parametrize(
    ("azimuths_deg", "ellipticities", "expected"),
    [
        ([5.0, 20.0], [0.5, 0.5], True),  # 15 deg apart
        ([5.0, 12.0], [0.5, 0.5], False),  # 7 deg apart
        ([0.0, 45.0], [0.5, 0.95], False),  # second is circular: ignored
        ([0.0, 45.0, 70.0], [0.95, 0.9, 0.5], False),  # only one non-circular
        ([89.0, -89.0], [0.5, 0.5], False),  # 2 deg apart
        ([80.0, -80.0], [0.5, 0.5], True),  # 20 deg apart
        ([0.0, 90.0], [0.5, 0.5], False),  # major/minor swap: same axis pair
        ([0.0, 45.0], [0.5, 0.5], True),  # axes rotated by 45 deg
        ([0.0, 30.0], [0.5, np.nan], False),  # failed profile: ignored
        ([], [], False),
    ],
)
def test_general_astigmatism(
    azimuths_deg: list[float], ellipticities: list[float], expected: bool
) -> None:
    assert general_astigmatism(np.radians(azimuths_deg), ellipticities) is expected


def test_simple_astigmatic_beam_is_not_general_astigmatism() -> None:
    """x and y waists 2 zR apart: the major axis swaps from y to x along z."""
    z_r = rayleigh_length(0.2e-3, 1.0, WAVELENGTH_M)
    beam = BeamParams(
        wavelength_m=WAVELENGTH_M,
        d0x_m=0.2e-3,
        d0y_m=0.2e-3,
        z0x_m=0.0,
        z0y_m=2 * z_r,
        m2x=1.0,
        m2y=1.0,
    )
    camera = CameraConfig(
        pixel_size_m=5.5e-6,
        nx=512,
        ny=512,
        bit_depth=12,
        offset_counts=0.0,
        read_noise_counts=0.0,
        signal_counts_per_ms=1e6,
        shot_noise=False,
    )
    rng = np.random.default_rng(0)
    results = [
        measure_widths(render_frame(beam, camera, z, 1.0, rng), camera.pixel_size_m)
        for z in np.array([-1.0, -0.5, 0.0, 1.0, 2.0, 2.5, 3.0]) * z_r
    ]
    azimuths = np.array([r.azimuth_rad for r in results])
    ellipticities = np.array([r.ellipticity for r in results])
    non_circular = ellipticities <= 0.87
    # Both orientations occur among the non-circular profiles.
    assert np.any(np.abs(azimuths[non_circular]) < 0.01)
    assert np.any(np.abs(azimuths[non_circular]) > np.radians(89.0))
    assert not general_astigmatism(azimuths, ellipticities)


def test_general_astigmatism_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        general_astigmatism([0.0, 0.1], [0.5])


def test_end_to_end_simulated_frames_give_m2_per_axis() -> None:
    """Simulator -> dark subtraction -> widths -> repeat stats -> fit, per axis.

    The waists are sized so that both axes have the same zR (~24.6 mm), so one
    set of 10 positions satisfies the ISO point rule for x and y, and the
    3 x d rectangle still fits on the 512 px sensor at the farthest position.
    """
    m2x, m2y = 1.2, 1.5
    d0x = 0.2e-3
    d0y = d0x * np.sqrt(m2y / m2x)  # equal Rayleigh lengths
    beam = BeamParams(
        wavelength_m=WAVELENGTH_M,
        d0x_m=d0x,
        d0y_m=d0y,
        z0x_m=0.100,
        z0y_m=0.105,
        m2x=m2x,
        m2y=m2y,
    )
    camera = CameraConfig(
        pixel_size_m=5.5e-6,
        nx=512,
        ny=512,
        bit_depth=12,
        offset_counts=100.0,
        read_noise_counts=5.0,
        signal_counts_per_ms=1e6,
    )
    z_r = rayleigh_length(d0x, m2x, WAVELENGTH_M)
    z_mid = 0.5 * (beam.z0x_m + beam.z0y_m)
    z_positions = (
        z_mid + np.array([-3.0, -2.3, -0.8, -0.4, 0, 0.4, 0.8, 2.3, 3.0, 3.6]) * z_r
    )

    rng = np.random.default_rng(2026)
    dark = mean_dark([dark_frame(camera, rng) for _ in range(10)])
    d_x, d_x_std, d_y, d_y_std = [], [], [], []
    for z in z_positions:
        # The operator sets the exposure for a peak of ~75 % of full scale.
        dx = beam_diameter(z, d0x, beam.z0x_m, m2x, WAVELENGTH_M)
        dy = beam_diameter(z, d0y, beam.z0y_m, m2y, WAVELENGTH_M)
        peak_per_ms = (
            camera.signal_counts_per_ms
            * camera.pixel_size_m**2
            / (2 * np.pi * (dx / 4) * (dy / 4))
        )
        exposure_ms = 0.75 * camera.full_scale_counts / peak_per_ms
        results = []
        for _ in range(5):
            frame = render_frame(beam, camera, z, exposure_ms, rng)
            image, _ = subtract_background(frame, dark)
            result = measure_widths(image, camera.pixel_size_m)
            assert result.converged and not result.roi_clipped
            results.append(result)
        stats = width_statistics(results)
        d_x.append(stats.d_x_mean_m)
        d_x_std.append(stats.d_x_std_m)
        d_y.append(stats.d_y_mean_m)
        d_y_std.append(stats.d_y_std_m)

    fit_x = fit_caustic(z_positions, d_x, d_x_std, wavelength_m=WAVELENGTH_M)
    fit_y = fit_caustic(z_positions, d_y, d_y_std, wavelength_m=WAVELENGTH_M)
    print(
        f"\nend-to-end M²x {fit_x.m2:.4f} +- {fit_x.m2_std:.4f} "
        f"(error {fit_x.m2 / m2x - 1:+.2%}), "
        f"M²y {fit_y.m2:.4f} +- {fit_y.m2_std:.4f} (error {fit_y.m2 / m2y - 1:+.2%})"
    )
    print(
        f"d0x error {fit_x.d0_m / d0x - 1:+.2%}, "
        f"d0y error {fit_y.d0_m / d0y - 1:+.2%}, "
        f"z0x error {(fit_x.z0_m - beam.z0x_m) * 1e3:+.3f} mm, "
        f"z0y error {(fit_y.z0_m - beam.z0y_m) * 1e3:+.3f} mm"
    )
    assert fit_x.success and fit_y.success
    assert fit_x.m2 == pytest.approx(m2x, rel=0.03)
    assert fit_y.m2 == pytest.approx(m2y, rel=0.03)
    # The chosen positions meet the ISO point rule on both axes.
    assert coverage(z_positions, fit_x.z0_m, fit_x.z_r_m).compliant
    assert coverage(z_positions, fit_y.z0_m, fit_y.z_r_m).compliant
