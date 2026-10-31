"""Thin-lens optics for an artificial waist (docs/change-lens-design.md, section 1).

Each axis is treated as an embedded Gaussian with the complex beam parameter
q(z) = (z - z0) + i zR. Paraxial ABCD matrices transform q; for a real matrix
with determinant 1 the product d0² / zR = 4 M² lambda / pi is unchanged, so M²
is unchanged and the new waist diameter follows from the new zR alone. All
lengths are in m (SI); the tool layer converts at its boundary.

Simplifications: thin lens, paraxial optics, no aberrations, exact focal length
and position, no clipping at the lens aperture.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Waist:
    """Second-moment waist of one axis.

    Parameters
    ----------
    d0_m : float
        Waist diameter (4 sigma) in m.
    z0_m : float
        Waist position in m.
    z_r_m : float
        Rayleigh length in m.
    """

    d0_m: float
    z0_m: float
    z_r_m: float

    def q_at(self, z_m: float) -> complex:
        """Complex beam parameter q = (z - z0) + i zR at position `z_m` (m)."""
        return complex(z_m - self.z0_m, self.z_r_m)

    def diameter_at(self, z_m: float) -> float:
        """Second-moment diameter in m at position `z_m` (m)."""
        return self.d0_m * abs(self.q_at(z_m)) / self.z_r_m


def free_space(length_m: float) -> NDArray[np.float64]:
    """ABCD matrix of free space over `length_m` (m)."""
    return np.array([[1.0, length_m], [0.0, 1.0]])


def thin_lens(f_m: float) -> NDArray[np.float64]:
    """ABCD matrix of a thin lens of focal length `f_m` (m; > 0 converging).

    ``f_m = inf`` gives the identity (no lens).
    """
    if f_m == 0 or not math.isfinite(1.0 / f_m):
        raise ValueError(f"f_m must be non-zero, got {f_m}")
    return np.array([[1.0, 0.0], [-1.0 / f_m, 1.0]])


def apply_abcd(matrix: ArrayLike, q: complex) -> complex:
    """Transform q by an ABCD matrix: q' = (A q + B) / (C q + D)."""
    (a, b), (c, d) = np.asarray(matrix, dtype=np.float64)
    return (a * q + b) / (c * q + d)


def invert_abcd(matrix: ArrayLike) -> NDArray[np.float64]:
    """Inverse of an ABCD matrix with determinant 1: [[D, -B], [-C, A]]."""
    (a, b), (c, d) = np.asarray(matrix, dtype=np.float64)
    det = a * d - b * c
    if not math.isclose(det, 1.0, rel_tol=1e-9):
        raise ValueError(f"ABCD determinant must be 1, got {det}")
    return np.array([[d, -b], [-c, a]])


def _waist_from_q(q: complex, z_m: float, invariant_m: float) -> Waist:
    """Waist of the beam with parameter `q` at plane `z_m`.

    `invariant_m` is d0² / zR in m (unchanged by the lens).
    """
    z_r = q.imag
    if not z_r > 0:
        raise ValueError(f"transformed beam has no real waist (zR = {z_r})")
    return Waist(math.sqrt(invariant_m * z_r), z_m - q.real, z_r)


def through_thin_lens(waist: Waist, f_m: float, lens_z_m: float) -> Waist:
    """Waist of the beam after a thin lens at `lens_z_m` (forward).

    Parameters
    ----------
    waist : Waist
        Original waist (positions in the same z frame as the lens, m).
    f_m : float
        Focal length in m (> 0 converging, < 0 diverging).
    lens_z_m : float
        Lens position in m.

    Returns
    -------
    Waist
        The new (artificial) waist in m; M² is unchanged.
    """
    q_after = apply_abcd(thin_lens(f_m), waist.q_at(lens_z_m))
    return _waist_from_q(q_after, lens_z_m, waist.d0_m**2 / waist.z_r_m)


def back_through_thin_lens(after: Waist, f_m: float, lens_z_m: float) -> Waist:
    """Original waist from the waist measured after a thin lens (inverse).

    Uses the inverse ABCD matrix of the lens. Only this lens is undone: a
    second lens change replaces the first, it does not add to it.

    Parameters
    ----------
    after : Waist
        Waist measured after the lens, in m.
    f_m : float
        Focal length of the lens in m.
    lens_z_m : float
        Lens position in m.

    Returns
    -------
    Waist
        The original waist in m (calculated, not measured).
    """
    q_before = apply_abcd(invert_abcd(thin_lens(f_m)), after.q_at(lens_z_m))
    return _waist_from_q(q_before, lens_z_m, after.d0_m**2 / after.z_r_m)


def self_focus(s_m: float, f_m: float, z_r_m: float) -> tuple[float, float]:
    """Textbook thin-lens imaging of a Gaussian waist (Self 1983).

    Parameters
    ----------
    s_m : float
        Distance from the input waist to the lens in m (> 0: waist before it).
    f_m : float
        Focal length in m.
    z_r_m : float
        Input Rayleigh length in m.

    Returns
    -------
    tuple of float
        Distance s' from the lens to the output waist in m, and the waist
        magnification m (W0' = m W0, zR' = m² zR).
    """
    denominator = (s_m - f_m) ** 2 + z_r_m**2
    s_out = f_m + (s_m - f_m) * f_m**2 / denominator
    return s_out, abs(f_m) / math.sqrt(denominator)


def focal_sensitivity(
    after: Waist, f_m: float, lens_z_m: float, relative: float = 0.01
) -> tuple[float, float]:
    """Change of the back-calculated original waist for f -> f (1 + relative).

    Parameters
    ----------
    after : Waist
        Waist measured after the lens, in m.
    f_m : float
        Focal length in m.
    lens_z_m : float
        Lens position in m.
    relative : float, optional
        Relative change of f. Default 0.01 (+1 %).

    Returns
    -------
    tuple of float
        Change of the original z0 in m and of the original d0 in m.
    """
    nominal = back_through_thin_lens(after, f_m, lens_z_m)
    shifted = back_through_thin_lens(after, f_m * (1.0 + relative), lens_z_m)
    return shifted.z0_m - nominal.z0_m, shifted.d0_m - nominal.d0_m


def propagate(
    func: Callable[[NDArray[np.float64]], ArrayLike],
    x: ArrayLike,
    cov: ArrayLike,
    rel_step: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """First-order uncertainty propagation with a numerical Jacobian.

    Parameters
    ----------
    func : callable
        Maps a parameter vector to an output vector.
    x : 1D array
        Parameter values.
    cov : 2D array
        Covariance of `x`.
    rel_step : float, optional
        Central-difference step relative to each parameter (absolute for a
        zero parameter). Default 1e-6.

    Returns
    -------
    tuple of ndarray
        ``func(x)`` and its covariance ``J cov J^T``.
    """
    x = np.asarray(x, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    y = np.asarray(func(x), dtype=np.float64)
    jac = np.empty((y.size, x.size))
    for k in range(x.size):
        step = rel_step * (abs(x[k]) if x[k] != 0 else 1.0)
        up, down = x.copy(), x.copy()
        up[k] += step
        down[k] -= step
        jac[:, k] = (
            np.asarray(func(up), dtype=np.float64)
            - np.asarray(func(down), dtype=np.float64)
        ) / (2.0 * step)
    return y, jac @ cov @ jac.T


def abc_from_waist(d0_m: float, z0_m: float, theta_rad: float) -> NDArray[np.float64]:
    """Coefficients of d(z)² = a + b z + c z² for a waist and full divergence.

    a = d0² + theta² z0², b = -2 theta² z0, c = theta² (SI: m², m, 1).
    """
    c = theta_rad**2
    return np.array([d0_m**2 + c * z0_m**2, -2.0 * c * z0_m, c])


def waist_from_abc(a: float, b: float, c: float) -> Waist | None:
    """Waist from d(z)² = a + b z + c z², or None if it has no real waist.

    A curve with c <= 0 or 4ac - b² <= 0 is not a caustic (no real waist).
    """
    if not c > 0 or not 4.0 * a * c - b**2 > 0:
        return None
    z0 = -b / (2.0 * c)
    d0 = math.sqrt((4.0 * a * c - b**2) / (4.0 * c))
    return Waist(d0, z0, d0 / math.sqrt(c))
