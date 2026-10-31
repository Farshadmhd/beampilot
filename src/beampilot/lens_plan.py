"""Planning a lens from a caustic fit (change_lens; design sections 3 and 5).

Everything here works on the **fit** only, never on the hidden beam: lens
options, the prediction of the artificial waist and the back-calculated
original beam. All values are SI (m, rad); the tool layer converts.

Lens options score each kit x rail combination by the fraction of draws from
the fit covariance in which it is feasible. The draws are taken in the
coefficients of d(z)² = a + b z + c z² (the fit is parametrised by d0, z0 and
theta; a, b, c and their covariance follow by first-order propagation). A draw
with c <= 0 or 4ac - b² <= 0 has no real waist and counts as infeasible. The
random generator has a fixed seed, so the result is deterministic.

A far-field-only fit can leave the waist diameter undetermined (it collapses
towards d0 = 0 while z0 and theta stay well determined). For such an axis the
planning uses z0 and theta only: b and c are drawn from the linearised
covariance (theta = sqrt(c), z0 = -b / (2c)), M² is drawn uniformly in [1, 2]
and d0 = 4 lambda M² / (pi theta). The output is labelled so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from beampilot.caustic import CausticFit
from beampilot.lens import (
    Waist,
    abc_from_waist,
    back_through_thin_lens,
    focal_sensitivity,
    propagate,
    through_thin_lens,
)

DRAWS = 200
SEED = 20261031
MAX_OPTIONS = 5
ISO_POINTS = 5  # needed near (within zR) and far (beyond 2 zR), per axis
NEAR_WINDOW_ZR = 0.9  # usable near window, as in the evaluation oracle
FAR_WINDOW_ZR = 2.3  # far points comfortably beyond 2 zR
POINT_SPACING_ZR = 0.15  # smallest useful spacing of positions
# A fitted waist diameter counts as not determined if 4ac - b² <= 0, its
# relative std exceeds this, or the fitted M² is below MIN_FITTED_M2.
MAX_D0_REL_STD = 0.5
MIN_FITTED_M2 = 0.9
ASSUMED_M2_RANGE = (1.0, 2.0)  # for planning when d0 is not determined
NOMINAL_ASSUMED_M2 = 1.5
UNDETERMINED_LABEL = "waist diameter not determined; options assume M² between 1 and 2"


@dataclass(frozen=True)
class Geometry:
    """What limits an artificial waist (all in m).

    Parameters
    ----------
    stage_min_m, stage_max_m : float
        Stage travel.
    half_sensor_m : float
        Half the smaller sensor side: the largest far-field diameter.
    min_d0_m : float
        Smallest waist diameter (20 pixels across).
    max_beam_at_lens_m : float
        Largest beam diameter at the lens (no clipping).
    """

    stage_min_m: float
    stage_max_m: float
    half_sensor_m: float
    min_d0_m: float
    max_beam_at_lens_m: float


@dataclass(frozen=True)
class LensOption:
    """One kit x rail combination, scored from the fit (SI).

    Parameters
    ----------
    f_m, lens_z_m : float
        Focal length and lens position in m.
    fraction : float
        Share of the fit draws in which the option is feasible on both axes.
    margin : int
        For the nominal fit: the smallest spare number of near or far points
        over both axes (negative if the nominal fit is not feasible).
    predicted : dict of str to Waist
        Nominal predicted artificial waist per axis.
    """

    f_m: float
    lens_z_m: float
    fraction: float
    margin: int
    predicted: dict[str, Waist]


def fit_waist(fit: CausticFit) -> Waist:
    """The fitted waist of one axis (m), also when it is extrapolated."""
    return Waist(fit.d0_m, fit.z0_m, fit.z_r_m)


def waist_undetermined(fit: CausticFit) -> bool:
    """True if the fit does not determine the waist diameter (see module doc)."""
    a, b, c = abc_from_waist(fit.d0_m, fit.z0_m, fit.theta_rad)
    return bool(
        not 4.0 * a * c - b**2 > 0
        or not fit.d0_std_m <= MAX_D0_REL_STD * fit.d0_m
        or not fit.m2 >= MIN_FITTED_M2
    )


def _far_field_waist(theta, z0, m2, wavelength_m: float):
    """Waist from the divergence, waist position and an assumed M² (m, arrays ok)."""
    d0 = 4.0 * wavelength_m * m2 / (math.pi * theta)
    return d0, z0, d0 / theta


def planning_waist(fit: CausticFit, wavelength_m: float | None) -> Waist:
    """The waist used for planning: fitted, or from theta, z0 and M² = 1.5.

    Without a wavelength the fitted waist is used as it is.
    """
    if wavelength_m is None or not waist_undetermined(fit):
        return fit_waist(fit)
    return Waist(
        *_far_field_waist(fit.theta_rad, fit.z0_m, NOMINAL_ASSUMED_M2, wavelength_m)
    )


def _room(length: NDArray, z_r: NDArray) -> NDArray:
    """Positions that fit in a window: floor(L / (0.15 zR)) + 1, 0 if empty."""
    with np.errstate(invalid="ignore", divide="ignore"):
        count = np.floor(length / (POINT_SPACING_ZR * z_r) + 1e-9) + 1
    return np.where(length >= 0, count, 0)


def _points(geometry: Geometry, d0: NDArray, z0: NDArray, z_r: NDArray):
    """Room for near and far points per axis, for waists inside the stage."""
    lo, hi = geometry.stage_min_m, geometry.stage_max_m
    near = _room(
        np.minimum(hi, z0 + NEAR_WINDOW_ZR * z_r)
        - np.maximum(lo, z0 - NEAR_WINDOW_ZR * z_r),
        z_r,
    )
    with np.errstate(invalid="ignore"):
        reach = z_r * np.sqrt(np.maximum((geometry.half_sensor_m / d0) ** 2 - 1, 0))
    below = (z0 - FAR_WINDOW_ZR * z_r) - np.maximum(lo, z0 - reach)
    above = np.minimum(hi, z0 + reach) - (z0 + FAR_WINDOW_ZR * z_r)
    return near, _room(below, z_r) + _room(above, z_r)


def _through(d0, z0, z_r, f_m: float, lens_z_m: float):
    """Thin lens on arrays of waists: new d0, z0, zR and the diameter at the lens."""
    # Invalid draws are NaN; they stay NaN and fail every feasibility test.
    with np.errstate(invalid="ignore", divide="ignore"):
        q = (lens_z_m - z0) + 1j * z_r
        q_after = 1.0 / (1.0 / q - 1.0 / f_m)
        z_r_after = q_after.imag
        return (
            np.sqrt(d0**2 / z_r * z_r_after),
            lens_z_m - q_after.real,
            z_r_after,
            d0 * np.abs(q) / z_r,
        )


def _undo_lens(d0, z0, z_r, current_lens: tuple[float, float] | None):
    """Map waists measured after `current_lens` back to the original beam.

    A new lens replaces the current one, so planning starts from the original
    beam. Works on arrays; ``current_lens`` is (f, position) in m or None.
    """
    if current_lens is None:
        return d0, z0, z_r
    f_m, lens_z_m = current_lens
    with np.errstate(invalid="ignore", divide="ignore"):
        q_after = (lens_z_m - z0) + 1j * z_r
        q_before = 1.0 / (1.0 / q_after + 1.0 / f_m)
        z_r_before = q_before.imag
        return (
            np.sqrt(d0**2 / z_r * z_r_before),
            lens_z_m - q_before.real,
            z_r_before,
        )


def _original_waist(fit: CausticFit, current_lens, wavelength_m: float) -> Waist:
    waist = planning_waist(fit, wavelength_m)
    if current_lens is None:
        return waist
    return back_through_thin_lens(waist, *current_lens)


def _feasible(geometry: Geometry, waists, f_m: float, lens_z_m: float):
    """Feasible on this axis (bool array) and spare points (int array)."""
    d0, z0, z_r, valid = waists
    d0_new, z0_new, z_r_new, at_lens = _through(d0, z0, z_r, f_m, lens_z_m)
    near, far = _points(geometry, d0_new, z0_new, z_r_new)
    with np.errstate(invalid="ignore"):
        ok = (
            valid
            & (at_lens <= geometry.max_beam_at_lens_m)
            & (d0_new >= geometry.min_d0_m)
            & (z0_new >= geometry.stage_min_m)
            & (z0_new <= geometry.stage_max_m)
            & (near >= ISO_POINTS)
            & (far >= ISO_POINTS)
        )
    return ok, np.minimum(near, far) - ISO_POINTS


def fit_draws(
    fit: CausticFit,
    rng: np.random.Generator,
    n: int = DRAWS,
    wavelength_m: float | None = None,
):
    """Waists drawn from the fit covariance via a, b, c (arrays, m).

    If the waist diameter is not determined (and `wavelength_m` is given),
    only b and c are drawn, M² is drawn in [1, 2] and d0 follows from theta.

    Returns
    -------
    tuple of ndarray
        d0, z0, zR (NaN where invalid) and a validity mask; a draw with
        c <= 0 or 4ac - b² <= 0 has no real waist.
    """
    p = np.array([fit.d0_m, fit.z0_m, fit.theta_rad])
    mean, cov = propagate(lambda x: abc_from_waist(*x), p, fit.cov)
    if wavelength_m is not None and waist_undetermined(fit):
        b, c = rng.multivariate_normal(
            mean[1:], cov[1:, 1:], size=n, method="svd", check_valid="ignore"
        ).T
        m2 = rng.uniform(*ASSUMED_M2_RANGE, size=n)
        valid = c > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            theta = np.where(valid, np.sqrt(c), np.nan)
            z0 = np.where(valid, -b / (2.0 * c), np.nan)
            d0, z0, z_r = _far_field_waist(theta, z0, m2, wavelength_m)
        return d0, z0, z_r, valid
    a, b, c = rng.multivariate_normal(
        mean, cov, size=n, method="svd", check_valid="ignore"
    ).T
    disc = 4.0 * a * c - b**2
    valid = (c > 0) & (disc > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d0 = np.where(valid, np.sqrt(disc / (4.0 * c)), np.nan)
        z0 = np.where(valid, -b / (2.0 * c), np.nan)
        z_r = np.where(valid, d0 / np.sqrt(c), np.nan)
    return d0, z0, z_r, valid


def lens_options(
    fits: dict[str, CausticFit],
    focal_lengths_m: tuple[float, ...],
    rail_m: list[float],
    geometry: Geometry,
    current_lens: tuple[float, float] | None = None,
    wavelength_m: float | None = None,
) -> list[LensOption]:
    """Feasible lens options from the fits of both axes, best first.

    A new lens replaces the current one: if a lens is in the beam, the fits
    (of the beam after it) are first mapped back to the original beam.

    Sorted by the feasible fraction of the draws, then by the nominal margin
    (more spare points first), then by the larger |f| and the lens closer to
    the stage. At most ``MAX_OPTIONS``; options with fraction 0 are left out.

    Parameters
    ----------
    fits : dict
        ``{"x": CausticFit, "y": CausticFit}`` of the current segment.
    focal_lengths_m : tuple of float
        Lens kit in m.
    rail_m : list of float
        Allowed lens positions in m.
    geometry : Geometry
        Stage, sensor and lens limits.
    current_lens : tuple of float, optional
        (f, position) in m of the lens in the beam now, or None.
    wavelength_m : float, optional
        Wavelength in m; needed to plan an axis whose waist diameter is not
        determined (see :func:`waist_undetermined`).

    Returns
    -------
    list of LensOption
    """
    rng = np.random.default_rng(SEED)
    draws = {}
    for axis, fit in fits.items():
        d0, z0, z_r, valid = fit_draws(fit, rng, wavelength_m=wavelength_m)
        draws[axis] = (*_undo_lens(d0, z0, z_r, current_lens), valid)
    nominal = {}
    for axis, fit in fits.items():
        waist = _original_waist(fit, current_lens, wavelength_m)
        nominal[axis] = tuple(
            np.array([v]) for v in (waist.d0_m, waist.z0_m, waist.z_r_m)
        ) + (np.array([True]),)
    options = []
    for f in focal_lengths_m:
        for z_lens in rail_m:
            ok = np.ones(DRAWS, dtype=bool)
            for axis in fits:
                ok &= _feasible(geometry, draws[axis], f, z_lens)[0]
            fraction = float(ok.mean())
            if fraction == 0.0:
                continue
            margin = min(
                int(_feasible(geometry, nominal[axis], f, z_lens)[1][0])
                for axis in fits
            )
            predicted = {
                axis: through_thin_lens(
                    _original_waist(fit, current_lens, wavelength_m), f, z_lens
                )
                for axis, fit in fits.items()
            }
            options.append(LensOption(f, z_lens, fraction, margin, predicted))
    options.sort(key=lambda o: (-o.fraction, -o.margin, -abs(o.f_m), -o.lens_z_m))
    return options[:MAX_OPTIONS]


def _with_std(func, fit: CausticFit) -> tuple[NDArray, NDArray | None]:
    p = np.array([fit.d0_m, fit.z0_m, fit.theta_rad])
    if fit.cov is None:
        return np.asarray(func(p)), None
    values, cov = propagate(func, p, fit.cov)
    return values, np.sqrt(np.clip(np.diag(cov), 0.0, None))


def predict(
    fit: CausticFit,
    f_m: float,
    lens_z_m: float,
    current_lens: tuple[float, float] | None = None,
    wavelength_m: float | None = None,
):
    """Predicted artificial waist (d0', z0', zR' in m) and its std from the fit.

    With a lens in the beam (`current_lens`, (f, position) in m), the fit is
    first mapped back through it: the new lens replaces it. If the waist
    diameter is not determined (and `wavelength_m` is given), the waist comes
    from theta, z0 and M² = 1.5.
    """
    far_field = wavelength_m is not None and waist_undetermined(fit)

    def forward(p: NDArray) -> list[float]:
        if far_field:
            waist = Waist(
                *_far_field_waist(p[2], p[1], NOMINAL_ASSUMED_M2, wavelength_m)
            )
        else:
            waist = Waist(p[0], p[1], p[0] / p[2])
        if current_lens is not None:
            waist = back_through_thin_lens(waist, *current_lens)
        new = through_thin_lens(waist, f_m, lens_z_m)
        return [new.d0_m, new.z0_m, new.z_r_m]

    return _with_std(forward, fit)


def back_calculate(fit_after: CausticFit, f_m: float, lens_z_m: float):
    """Original waist from the fit after the lens, with std and f sensitivity.

    Returns
    -------
    tuple
        Values (d0, z0, zR in m), their std (or None), and the change of z0
        and of d0 in m for +1 % focal length.
    """

    def backward(p: NDArray) -> list[float]:
        old = back_through_thin_lens(Waist(p[0], p[1], p[0] / p[2]), f_m, lens_z_m)
        return [old.d0_m, old.z0_m, old.z_r_m]

    values, stds = _with_std(backward, fit_after)
    dz0, dd0 = focal_sensitivity(fit_waist(fit_after), f_m, lens_z_m)
    return values, stds, (dz0, dd0)
