"""Caustic fit, ISO point coverage and general-astigmatism check (ISO 11146-1).

All lengths are in m and angles in rad (SI). Diameters are second-moment
(4 sigma) diameters. Every function works on one axis; call it separately for
x and y, because a beam with simple astigmatism has a different waist and
Rayleigh length per axis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

MIN_FIT_POINTS = 4
# Repeat stds below this fraction of d are raised to it, so one lucky position
# with almost identical repeats cannot dominate the fit.
STD_FLOOR_REL = 1e-3


@dataclass(frozen=True)
class CausticFit:
    """Result of a caustic fit d(z)² = d0² + theta²(z - z0)² for one axis.

    Each ``*_std`` value is the standard uncertainty (1 sigma) from the fit
    Jacobian, scaled by the residual variance (see :func:`fit_caustic`).

    Parameters
    ----------
    d0_m, d0_std_m : float
        Waist diameter and its uncertainty, in m.
    z0_m, z0_std_m : float
        Waist position and its uncertainty, in m.
    theta_rad, theta_std_rad : float
        Full far-field divergence angle and its uncertainty, in rad.
    z_r_m, z_r_std_m : float
        Rayleigh length zR = d0 / theta and its uncertainty, in m.
    m2, m2_std : float
        Beam propagation ratio M² = pi d0 theta / (4 lambda) and its
        uncertainty (dimensionless).
    residual_rms : float
        RMS of the fit residuals (dimensionless): relative deviations
        (d_model - d) / d without `d_std_m`, deviations in units of the repeat
        std with it.
    n_points : int
        Number of (z, d) points used.
    success : bool
        True if the optimiser converged, d0 and theta are positive and all
        uncertainties are finite.
    cov : tuple of tuple of float, optional
        Covariance of (d0 in m, z0 in m, theta in rad), 3 x 3. Used to plan a
        lens and to propagate uncertainties through it.
    """

    d0_m: float
    d0_std_m: float
    z0_m: float
    z0_std_m: float
    theta_rad: float
    theta_std_rad: float
    z_r_m: float
    z_r_std_m: float
    m2: float
    m2_std: float
    residual_rms: float
    n_points: int
    success: bool
    cov: tuple[tuple[float, ...], ...] | None = None


@dataclass(frozen=True)
class Coverage:
    """ISO 11146-1 distribution of measurement positions for one axis.

    Parameters
    ----------
    n_near : int
        Distinct positions with |z - z0| <= zR.
    n_far : int
        Distinct positions with |z - z0| >= 2 zR.
    n_between : int
        Distinct positions with zR < |z - z0| < 2 zR.
    compliant : bool
        True if ``n_near >= min_near`` and ``n_far >= min_far``.
    """

    n_near: int
    n_far: int
    n_between: int
    compliant: bool


def fit_caustic(
    z_m: ArrayLike,
    d_m: ArrayLike,
    d_std_m: ArrayLike | None = None,
    *,
    wavelength_m: float,
) -> CausticFit:
    """Fit the hyperbolic caustic d(z)² = d0² + theta²(z - z0)² for one axis.

    The model is parametrised by d0, z0 and theta, so the fitted d(z)² is
    positive for every z. It is minimised with
    :func:`scipy.optimize.least_squares`, starting from a quadratic polyfit
    on d². Residuals are ``(d_model - d) / d_std`` when `d_std_m` is given
    (each std raised to at least ``1e-3 * d``), otherwise the relative
    deviations ``(d_model - d) / d`` that ISO 11146-1 recommends. Both are
    relative deviations, the first weighted by the repeat scatter.

    Uncertainties come from the covariance ``s² (J^T J)^-1`` at the solution,
    with ``s² = sum(residuals²) / (n - 3)``. The repeat stds therefore act as
    relative weights, as in :func:`scipy.optimize.curve_fit` with
    ``absolute_sigma=False``. zR and M² uncertainties include the
    d0-theta correlation.

    Parameters
    ----------
    z_m : 1D array
        Stage positions in m. Repeats at the same z are allowed.
    d_m : 1D array
        Beam diameters in m (e.g. the mean of the repeats at each position),
        same length as `z_m`.
    d_std_m : 1D array, optional
        Standard deviation of each diameter in m (e.g. the repeat std). If
        None, the fit uses relative residuals.
    wavelength_m : float
        Wavelength in m (keyword-only).

    Returns
    -------
    CausticFit
        Waist diameter, waist position, full divergence angle, Rayleigh
        length and M², with standard uncertainties.

    Raises
    ------
    ValueError
        If fewer than 4 points or fewer than 3 distinct z values are given,
        the arrays differ in length or contain non-finite values, a diameter
        is not positive, a std is negative, or the wavelength is not positive.
    """
    z = np.asarray(z_m, dtype=np.float64)
    d = np.asarray(d_m, dtype=np.float64)
    if z.ndim != 1 or z.shape != d.shape:
        raise ValueError(f"z_m {z.shape} and d_m {d.shape} must be equal 1D arrays")
    if z.size < MIN_FIT_POINTS:
        raise ValueError(f"need at least {MIN_FIT_POINTS} points, got {z.size}")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(d))):
        raise ValueError("z_m and d_m must be finite")
    if not np.all(d > 0):
        raise ValueError("all diameters must be > 0")
    if np.unique(z).size < 3:
        raise ValueError("need at least 3 distinct z positions")
    # Written as "not x > 0" so that NaN is rejected too.
    if not wavelength_m > 0:
        raise ValueError(f"wavelength_m must be > 0, got {wavelength_m}")
    if d_std_m is None:
        sigma = d
    else:
        d_std = np.asarray(d_std_m, dtype=np.float64)
        if d_std.shape != d.shape:
            raise ValueError(f"d_std_m {d_std.shape} must match d_m {d.shape}")
        if not (np.all(np.isfinite(d_std)) and np.all(d_std >= 0)):
            raise ValueError("d_std_m must be finite and >= 0")
        sigma = np.maximum(d_std, STD_FLOOR_REL * d)

    # Work in scaled units so that all three parameters are of order one:
    # u = (z - z_c) / z_s, delta = d / d_s, model delta² = a² + t²(u - u0)².
    z_c, z_s, d_s = float(z.mean()), float(z.std()), float(d.mean())
    u = (z - z_c) / z_s
    delta = d / d_s
    sigma_s = sigma / d_s

    residuals, jacobian = _scaled_model(u, delta, sigma_s)
    fit = least_squares(
        residuals,
        _start_values(u, delta),
        jac=jacobian,
        method="trf",
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    # The model depends on a² and t² only; report the positive branch and
    # take the Jacobian there so that the d0-theta covariance has the right sign.
    p = np.array([abs(fit.x[0]), fit.x[1], abs(fit.x[2])])
    r = residuals(p)
    dof = z.size - 3
    cov_scaled = _covariance(jacobian(p)) * float(r @ r) / dof

    # Back to SI: d0 = a d_s, z0 = z_c + u0 z_s, theta = t d_s / z_s.
    scale = np.array([d_s, z_s, d_s / z_s])
    d0, z0, theta = p * scale + np.array([0.0, z_c, 0.0])
    cov = cov_scaled * np.outer(scale, scale)

    z_r = d0 / theta
    m2 = np.pi * d0 * theta / (4.0 * wavelength_m)
    grad_z_r = np.array([1.0 / theta, 0.0, -d0 / theta**2])
    grad_m2 = np.pi / (4.0 * wavelength_m) * np.array([theta, 0.0, d0])
    stds = np.sqrt(np.diag(cov))
    z_r_std = np.sqrt(grad_z_r @ cov @ grad_z_r)
    m2_std = np.sqrt(grad_m2 @ cov @ grad_m2)

    success = bool(
        fit.success
        and d0 > 0
        and theta > 0
        and np.all(np.isfinite(stds))
        and np.isfinite(z_r_std)
        and np.isfinite(m2_std)
    )
    return CausticFit(
        d0_m=float(d0),
        d0_std_m=float(stds[0]),
        z0_m=float(z0),
        z0_std_m=float(stds[1]),
        theta_rad=float(theta),
        theta_std_rad=float(stds[2]),
        z_r_m=float(z_r),
        z_r_std_m=float(z_r_std),
        m2=float(m2),
        m2_std=float(m2_std),
        residual_rms=float(np.sqrt(np.mean(r**2))),
        n_points=int(z.size),
        success=success,
        cov=tuple(tuple(float(v) for v in row) for row in cov),
    )


def coverage(
    z_m: ArrayLike,
    z0_m: float,
    z_r_m: float,
    min_near: int = 5,
    min_far: int = 5,
) -> Coverage:
    """Count measurement positions near the waist and in the far field.

    ISO 11146-1 asks for at least 10 positions, about half within one Rayleigh
    length of the waist and about half beyond two Rayleigh lengths. We read
    "about half of 10" as at least 5 on each side (see docs/decisions.md).
    Positions are distinct z values: repeated frames at one z count once.
    Apply it per axis, with that axis's z0 and zR.

    Parameters
    ----------
    z_m : 1D array
        Stage positions in m; duplicates are counted once.
    z0_m : float
        Waist position of this axis in m.
    z_r_m : float
        Rayleigh length of this axis in m.
    min_near : int, optional
        Positions needed with |z - z0| <= zR. Default 5.
    min_far : int, optional
        Positions needed with |z - z0| >= 2 zR. Default 5.

    Returns
    -------
    Coverage
        Counts of near, far and in-between positions, and the compliance flag.

    Raises
    ------
    ValueError
        If `z_r_m` is not positive, or `z0_m` or any z is not finite.
    """
    z = np.unique(np.asarray(z_m, dtype=np.float64))
    if not z_r_m > 0 or not np.isfinite(z_r_m):
        raise ValueError(f"z_r_m must be finite and > 0, got {z_r_m}")
    if not (np.isfinite(z0_m) and np.all(np.isfinite(z))):
        raise ValueError("z_m and z0_m must be finite")
    distance = np.abs(z - z0_m)
    n_near = int(np.count_nonzero(distance <= z_r_m))
    n_far = int(np.count_nonzero(distance >= 2.0 * z_r_m))
    return Coverage(
        n_near=n_near,
        n_far=n_far,
        n_between=int(z.size) - n_near - n_far,
        compliant=n_near >= min_near and n_far >= min_far,
    )


def general_astigmatism(
    azimuths_rad: ArrayLike,
    ellipticities: ArrayLike,
    circular_threshold: float = 0.87,
    max_diff_deg: float = 10.0,
) -> bool:
    """Detect general astigmatism from the azimuths of elliptical profiles.

    If the principal axes of non-circular profiles rotate along z by more
    than `max_diff_deg`, the beam has general astigmatism and ISO 11146-2
    applies, which is out of scope for BeamPilot. Profiles with ellipticity
    above `circular_threshold` count as circular (their azimuth is noise) and
    are ignored, as are profiles with a non-finite azimuth or ellipticity.

    Azimuths are compared modulo 90 degrees, because what matters is the pair
    of principal axes, not which of them is the major one. The azimuth from
    :func:`beampilot.widths.measure_widths` is that of the major axis, and a
    beam with simple astigmatism swaps major and minor axis where d_x = d_y:
    its azimuth jumps from 90 to 0 degrees although the axis pair stays put.
    So 0 and 90 degrees count as the same axis pair, and 89 and -89 degrees
    differ by 2 degrees.

    Parameters
    ----------
    azimuths_rad : 1D array
        Azimuth of the major principal axis of each profile, in rad.
    ellipticities : 1D array
        Minor/major principal width ratio of each profile (0 to 1), same
        length as `azimuths_rad`.
    circular_threshold : float, optional
        Ellipticity above which a profile counts as circular. Default 0.87
        (unverified, from ISO 11146-1; check against the standard text).
    max_diff_deg : float, optional
        Largest allowed azimuth difference between two non-circular profiles,
        in degrees. Default 10.

    Returns
    -------
    bool
        True if any two non-circular profiles differ in azimuth by more than
        `max_diff_deg`.

    Raises
    ------
    ValueError
        If the two arrays differ in shape.
    """
    azimuths = np.asarray(azimuths_rad, dtype=np.float64)
    ellipticity = np.asarray(ellipticities, dtype=np.float64)
    if azimuths.shape != ellipticity.shape:
        raise ValueError(
            f"azimuths {azimuths.shape} and ellipticities {ellipticity.shape} differ"
        )
    usable = (
        np.isfinite(azimuths)
        & np.isfinite(ellipticity)
        & (ellipticity <= circular_threshold)
    )
    max_diff_rad = np.radians(max_diff_deg)
    quarter_turn = np.pi / 2.0
    for a, b in combinations(azimuths[usable], 2):
        diff = (a - b) % quarter_turn
        if min(diff, quarter_turn - diff) > max_diff_rad:
            return True
    return False


def _scaled_model(
    u: NDArray[np.float64], delta: NDArray[np.float64], sigma: NDArray[np.float64]
) -> tuple[
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
]:
    """Residual and Jacobian functions of delta(u) = sqrt(a² + t²(u - u0)²)."""

    def model(p: NDArray[np.float64]) -> NDArray[np.float64]:
        a, u0, t = p
        # A tiny floor keeps the Jacobian finite if a = 0 at a sample point.
        return np.maximum(np.sqrt(a**2 + t**2 * (u - u0) ** 2), np.finfo(float).tiny)

    def residuals(p: NDArray[np.float64]) -> NDArray[np.float64]:
        return (model(p) - delta) / sigma

    def jacobian(p: NDArray[np.float64]) -> NDArray[np.float64]:
        a, u0, t = p
        m = model(p)
        du = u - u0
        columns = (a / m, -(t**2) * du / m, t * du**2 / m)
        return np.column_stack(columns) / sigma[:, np.newaxis]

    return residuals, jacobian


def _start_values(u: NDArray[np.float64], delta: NDArray[np.float64]) -> NDArray:
    """Start (a, u0, t) from a quadratic polyfit of delta² over u."""
    c2, c1, c0 = np.polyfit(u, delta**2, 2)
    if c2 > 0:
        u0 = -c1 / (2.0 * c2)
        a2 = c0 - c1**2 / (4.0 * c2)
        a = np.sqrt(a2) if a2 > 0 else delta.min()
        return np.array([a, u0, np.sqrt(c2)])
    # No curvature: start at the smallest diameter with the mean slope.
    i_min = int(np.argmin(delta))
    slope = np.ptp(delta) / np.ptp(u)
    return np.array([delta[i_min], u[i_min], max(slope, 1e-3)])


def _covariance(jac: NDArray[np.float64]) -> NDArray[np.float64]:
    """(J^T J)^-1 via SVD; all NaN if J is rank deficient."""
    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    if s[-1] <= np.finfo(float).eps * max(jac.shape) * s[0]:
        return np.full((3, 3), np.nan)
    return (vt.T / s**2) @ vt
