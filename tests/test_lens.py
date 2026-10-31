"""Known-answer tests for the thin-lens optics (design section 7, items 1-5)."""

import math

import numpy as np
import pytest

from beampilot.caustic import fit_caustic
from beampilot.lens import (
    Waist,
    abc_from_waist,
    back_through_thin_lens,
    focal_sensitivity,
    invert_abcd,
    propagate,
    self_focus,
    thin_lens,
    through_thin_lens,
    waist_from_abc,
)
from beampilot.simulator import BeamParams, CameraConfig, render_frame
from beampilot.widths import measure_widths

LAMBDA_M = 1064e-9


def _gaussian(d0_m: float, z0_m: float, m2: float = 1.0) -> Waist:
    z_r = math.pi * d0_m**2 / (4.0 * m2 * LAMBDA_M)
    return Waist(d0_m, z0_m, z_r)


# 1. Textbook focus (Self 1983), both signs of f, including s = f.


@pytest.mark.parametrize("s_over_f", [0.3, 1.0, 1.5, 3.0, 10.0])
@pytest.mark.parametrize("f_m", [0.04, 0.3, -0.2])
def test_ideal_gaussian_matches_self_1983(s_over_f: float, f_m: float) -> None:
    beam = _gaussian(200e-6, 0.0)
    s = abs(f_m) * s_over_f
    lens_z = s  # the waist sits s before the lens
    out = through_thin_lens(beam, f_m, lens_z)
    s_out, m = self_focus(s, f_m, beam.z_r_m)
    assert out.z0_m - lens_z == pytest.approx(s_out, rel=1e-9, abs=1e-15)
    assert out.d0_m == pytest.approx(m * beam.d0_m, rel=1e-9)
    assert out.z_r_m == pytest.approx(m**2 * beam.z_r_m, rel=1e-9)


def test_waist_at_the_focal_plane_images_to_the_back_focal_plane() -> None:
    beam = _gaussian(300e-6, -0.1)
    f = 0.15
    out = through_thin_lens(beam, f, -0.1 + f)  # s = f
    assert out.z0_m - (-0.1 + f) == pytest.approx(f, rel=1e-12)


# 2. M² is unchanged.


@pytest.mark.parametrize("m2", [1.0, 1.4, 1.9])
def test_beam_quality_product_is_unchanged(m2: float) -> None:
    beam = _gaussian(250e-6, -0.06, m2)
    out = through_thin_lens(beam, 0.04, -0.02)
    m2_out = math.pi * out.d0_m**2 / (4.0 * LAMBDA_M * out.z_r_m)
    assert m2_out == pytest.approx(m2, rel=1e-12)


def test_simulated_caustic_after_the_lens_keeps_m2() -> None:
    """Noiseless frames of the transformed beam, ISO widths, caustic fit."""
    m2 = 1.3
    original = _gaussian(200e-6, -0.06, m2)
    after = through_thin_lens(original, 0.04, -0.02)  # waist moves into the stage
    assert 0.0 < after.z0_m < 0.25
    beam = BeamParams(LAMBDA_M, after.d0_m, after.d0_m, after.z0_m, after.z0_m, m2, m2)
    camera = CameraConfig(5.5e-6, 512, 512, 12, 0.0, 0.0, 1e6, shot_noise=False)
    # Keep the 3 d integration area on the 2.8 mm sensor (d <= ~0.9 mm) and the
    # peak near 3000 counts, as an operator would by scaling the exposure.
    z = np.linspace(0.0, 0.17, 12)
    widths = []
    for z_m in z:
        exposure_ms = 6.0 * (after.diameter_at(z_m) / after.d0_m) ** 2
        frame = render_frame(beam, camera, z_m, exposure_ms, np.random.default_rng(0))
        widths.append(measure_widths(frame.astype(np.float64), camera.pixel_size_m))
    fit = fit_caustic(z, [w.d_x_m for w in widths], wavelength_m=LAMBDA_M)
    assert fit.m2 == pytest.approx(m2, rel=1e-3)
    assert fit.z0_m == pytest.approx(after.z0_m, abs=0.2e-3)


# 3. Forward then inverse returns the original beam.


def test_forward_then_inverse_round_trip() -> None:
    rng = np.random.default_rng(42)
    for _ in range(300):
        beam = _gaussian(rng.uniform(100e-6, 400e-6), rng.uniform(-0.2, 0.5), 1.5)
        f = rng.choice([0.025, 0.04, 0.3, 0.75, -0.2])
        lens_z = rng.uniform(-0.3, -0.02)  # also upstream of the waist
        try:
            after = through_thin_lens(beam, f, lens_z)
        except ValueError:
            continue
        back = back_through_thin_lens(after, f, lens_z)
        assert back.d0_m == pytest.approx(beam.d0_m, rel=1e-9)
        assert back.z_r_m == pytest.approx(beam.z_r_m, rel=1e-9)
        assert abs(back.z0_m - beam.z0_m) <= 1e-9 * beam.z_r_m


def test_inverse_matrix_undoes_the_lens() -> None:
    matrix = thin_lens(0.04)
    assert invert_abcd(matrix) @ matrix == pytest.approx(np.eye(2))


# 4. A unit error (mm instead of m) is caught.


def test_focal_length_in_mm_instead_of_m_is_caught() -> None:
    beam = _gaussian(200e-6, -0.06)
    right = through_thin_lens(beam, 0.04, -0.02)
    wrong = through_thin_lens(beam, 40.0, -0.02)  # f in mm by mistake
    s_out, _ = self_focus(0.04, 0.04, beam.z_r_m)
    assert right.z0_m - (-0.02) == pytest.approx(s_out, rel=1e-9)
    assert abs(wrong.z0_m - right.z0_m) > 1e6 * 1e-9 * beam.z_r_m


# 5. Thin-lens limits.


@pytest.mark.parametrize("f_m", [0.04, 0.3, -0.2])
def test_lens_at_the_waist(f_m: float) -> None:
    beam = _gaussian(200e-6, -0.05)
    out = through_thin_lens(beam, f_m, beam.z0_m)
    expected = beam.z_r_m * f_m**2 / (f_m**2 + beam.z_r_m**2)
    assert out.z_r_m == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("f_m", [math.inf, -math.inf, 1e12])
def test_infinite_focal_length_leaves_the_beam_unchanged(f_m: float) -> None:
    beam = _gaussian(200e-6, -0.05)
    out = through_thin_lens(beam, f_m, -0.1)
    assert out.d0_m == pytest.approx(beam.d0_m, rel=1e-9)
    assert out.z0_m == pytest.approx(beam.z0_m, abs=1e-12)


def test_zero_or_nan_focal_length_is_rejected() -> None:
    for f in (0.0, math.nan):
        with pytest.raises(ValueError):
            thin_lens(f)


# Sensitivity, sigma propagation and the a, b, c form.


def test_focal_sensitivity_is_the_one_percent_difference() -> None:
    original = _gaussian(200e-6, -0.06, 1.2)
    after = through_thin_lens(original, 0.04, -0.02)
    dz0, dd0 = focal_sensitivity(after, 0.04, -0.02)
    shifted = back_through_thin_lens(after, 0.0404, -0.02)
    assert dz0 == pytest.approx(shifted.z0_m - original.z0_m, rel=1e-6)
    assert dd0 == pytest.approx(shifted.d0_m - original.d0_m, rel=1e-6)
    assert dz0 != 0.0 and dd0 != 0.0


def test_propagate_is_exact_for_a_linear_map() -> None:
    matrix = np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0]])
    cov = np.array([[1.0, 0.2, 0.0], [0.2, 2.0, -0.3], [0.0, -0.3, 0.5]])
    y, cov_y = propagate(lambda x: matrix @ x, [1.0, 2.0, 3.0], cov)
    assert y == pytest.approx(matrix @ [1.0, 2.0, 3.0])
    assert cov_y == pytest.approx(matrix @ cov @ matrix.T, rel=1e-6)


def test_abc_form_round_trip_and_invalid_curves() -> None:
    theta = 3e-3
    a, b, c = abc_from_waist(200e-6, -0.05, theta)
    waist = waist_from_abc(a, b, c)
    assert waist.d0_m == pytest.approx(200e-6, rel=1e-9)
    assert waist.z0_m == pytest.approx(-0.05, rel=1e-9)
    assert waist.z_r_m == pytest.approx(200e-6 / theta, rel=1e-9)
    assert waist_from_abc(a, b, 0.0) is None
    assert waist_from_abc(a, b, -c) is None
    assert waist_from_abc(b**2 / (4 * c) - 1e-12, b, c) is None
