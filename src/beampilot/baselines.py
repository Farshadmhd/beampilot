"""Two scripted baselines that use only the public agent tools.

Both strategies call exactly the tools the agent gets (``get_setup``,
``measure_at``, ``retake_darks``, ``set_status``, ``evaluate``, ``finish``)
and see only their JSON results, so the comparison with the agent is fair
(design section 8). Units follow the tool boundary: z in mm, diameters in um.

- :func:`naive_uniform`: an evenly spaced scan with one fixed exposure.
- :func:`scripted_iso`: a good-faith rule-based operator that follows
  ISO 11146-1 as closely as simple rules allow.
"""

from __future__ import annotations

from itertools import zip_longest
from typing import Any

import numpy as np

from beampilot.session import MeasurementSession
from beampilot.tools import (
    evaluate,
    finish,
    get_setup,
    measure_at,
    retake_darks,
    set_status,
)

NAIVE_POSITIONS = 10
NAIVE_EXPOSURE_MS = 1.0  # a mid-range exposure for this camera, not tuned

SCOUT_FRACTIONS = (0.125, 0.375, 0.625, 0.875)  # of the stage range
TARGET_PEAK_FRACTION = 0.6  # aim below full scale so one halving suffices
MAX_TRIES = 3  # exposure changes per position
NEAR_OFFSETS = (-0.8, -0.4, 0.0, 0.4, 0.8)  # in zR
FAR_OFFSETS = (2.1, -2.1, 2.4, -2.4, 2.7, -2.7, 3.0, -3.0, 3.5, -3.5, 4.0, -4.0)
MIN_SPACING_ZR = 0.15  # skip planned positions closer than this to others
# A planned position is skipped if the predicted d exceeds this fraction of the
# smaller sensor side (0.5: sensor edges beyond +-4 sigma, clipping < 0.02 %).
MAX_D_PER_SENSOR = 0.5
PLACEMENT_ROUNDS = 2
# Warning that makes the scripted operator reject a measurement anyway: beam
# light in the corner patches biases the background offset.
CAUTIOUS_REJECT_FLAGS = ("beam_in_corners",)


def suggest_verdict(evaluation: dict[str, Any]) -> str:
    """Verdict from an ``evaluate`` result by fixed rules.

    ``evaluate`` reports facts, not a verdict; both baselines use these rules:
    not enough data -> measurement_failed; waist outside the stage on an axis
    -> waist_not_accessible; general astigmatism -> out of scope; ISO coverage
    met on both axes -> iso_compliant; otherwise estimate_not_compliant.

    Parameters
    ----------
    evaluation : dict
        Result of :func:`beampilot.tools.evaluate`.

    Returns
    -------
    str
        One of the five ``finish`` verdicts.
    """
    if evaluation.get("status") != "ok":
        return "measurement_failed"
    axes = (evaluation["x"], evaluation["y"])
    if not all(axis["waist_in_stage_range"] for axis in axes):
        return "waist_not_accessible"
    if evaluation["general_astigmatism"]:
        return "general_astigmatism_out_of_scope"
    if evaluation["iso_coverage_met"]:
        return "iso_compliant"
    return "estimate_not_compliant"


def naive_uniform(
    session: MeasurementSession, exposure_ms: float = NAIVE_EXPOSURE_MS
) -> dict[str, Any]:
    """Naive scan: 10 evenly spaced positions, one fixed exposure.

    Accepts every measurement that is not saturated (and has valid widths),
    then evaluates and finishes with :func:`suggest_verdict`.

    Parameters
    ----------
    session : MeasurementSession
        The run.
    exposure_ms : float, optional
        Fixed exposure in ms for all positions. Default 1.0.

    Returns
    -------
    dict
        The ``finish`` result.
    """
    setup = get_setup(session, "Naive scan: read the stage range.")
    z_min, z_max = setup["stage_range_mm"]
    for z_mm in np.linspace(z_min, z_max, NAIVE_POSITIONS):
        result = measure_at(
            session, float(z_mm), exposure_ms, "Naive scan: evenly spaced position."
        )
        if "measurement_id" not in result:
            continue
        no_widths = result["d_x_um"]["mean"] is None or result["d_y_um"]["mean"] is None
        status = "reject" if result["flags"]["saturated"] or no_widths else "accept"
        set_status(
            session,
            result["measurement_id"],
            status,
            "Naive scan: accept everything that is not saturated.",
        )
    evaluation = evaluate(session, "Naive scan: fit all accepted positions.")
    verdict = suggest_verdict(evaluation)
    return _finish(session, verdict, "Naive scan: verdict from the evaluate rules.")


def scripted_iso(session: MeasurementSession) -> dict[str, Any]:
    """Good-faith rule-based ISO 11146-1 operator.

    1. Scout 4 positions spread over the stage. A scout rejected for a reason
       exposure cannot fix is replaced by a point halfway towards the
       smallest beam seen.
    2. At every position: halve the exposure on ``saturated``, double it on
       ``low_signal`` (at most 3 tries); retake darks on
       ``background_changed`` and measure again. The first exposure at a new
       position is scaled from a good earlier measurement and the predicted
       beam size.
    3. Evaluate, then per axis place up to 5 positions within +-zR and 5
       beyond 2 zR that are still missing, skipping positions off the stage
       or where the predicted d exceeds half the sensor. Repeat once if
       coverage is still short.
    4. Reject every measurement with a reject reason or ``beam_in_corners``;
       accept the rest. No exposure retries at a position where the beam is
       too large for the sensor.
    5. Evaluate again and finish with :func:`suggest_verdict`.

    Parameters
    ----------
    session : MeasurementSession
        The run.

    Returns
    -------
    dict
        The ``finish`` result.
    """
    operator = _ScriptedOperator(session)
    operator.scout()
    evaluation = evaluate(session, "Scripted: first fit from the scout positions.")
    for _ in range(PLACEMENT_ROUNDS):
        if evaluation.get("status") != "ok" or evaluation["iso_coverage_met"]:
            break
        planned = operator.plan_positions(evaluation)
        if not planned:
            break
        for z_mm in planned:
            operator.measure(z_mm, "Scripted: ISO position planned from the fit.")
        evaluation = evaluate(session, "Scripted: refit with the new positions.")
    verdict = suggest_verdict(evaluation)
    return _finish(session, verdict, f"Scripted: rules on evaluate gave {verdict}.")


class _ScriptedOperator:
    """State of the scripted baseline: setup and its good measurements."""

    def __init__(self, session: MeasurementSession) -> None:
        self.session = session
        self.setup = get_setup(session, "Scripted: read the setup.")
        self.full_scale = self.setup["full_scale_counts"]
        self.exposure_range = self.setup["exposure_range_ms"]
        self.sensor_mm = min(self.setup["sensor_size_mm"])
        # Budget as the tools report it (the session itself is only a handle).
        self.positions_left: int = self.setup["positions_left"]
        self.good: list[dict[str, Any]] = []  # accepted measure_at results
        self.measured_z: list[float] = []
        self.fit: dict[str, Any] | None = None

    def scout(self) -> None:
        """Measure 4 scout positions; refill towards the smallest beam.

        Once a scout is too large for the sensor, scouts further out in that
        direction are skipped (the beam only grows there). Missing scouts are
        refilled by bisecting between the smallest good beam and its nearest
        measured neighbours or the stage ends.
        """
        z_min, z_max = self.setup["stage_range_mm"]
        too_large: list[float] = []
        tried: list[float] = []  # refill candidates, each tried once
        for z_mm in [z_min + f * (z_max - z_min) for f in SCOUT_FRACTIONS]:
            if self._beyond_too_large(z_mm, too_large):
                continue
            result = self.measure(z_mm, "Scripted: scout position.")
            if result is not None and _too_large(result):
                too_large.append(z_mm)
        while len(self.good) < len(SCOUT_FRACTIONS) and self._has_budget(2):
            smallest = min(self.good, key=_beam_area, default=None)
            if smallest is None:
                break
            s = smallest["z_mm"]
            edges = sorted({*self.measured_z, z_min, z_max})
            lower = max((p for p in edges if p < s), default=None)
            upper = min((p for p in edges if p > s), default=None)
            candidates = [
                round(0.5 * (s + p), 1)
                for p in (lower, upper)
                if p is not None and abs(p - s) >= 2.0
            ]
            free = [
                z
                for z in candidates
                if all(abs(z - m) >= 1.0 for m in self.measured_z + tried)
                and not self._beyond_too_large(z, too_large)
            ]
            if not free:
                break
            tried.append(free[0])
            result = self.measure(
                free[0], "Scripted: refill a scout near the smallest beam."
            )
            if result is not None and _too_large(result):
                too_large.append(free[0])

    def _beyond_too_large(self, z_mm: float, too_large: list[float]) -> bool:
        """True if a too-large scout lies between the smallest beam and z."""
        smallest = min(self.good, key=_beam_area, default=None)
        if smallest is None:
            return False
        s = smallest["z_mm"]
        return any((s < t <= z_mm) or (z_mm <= t < s) for t in too_large)

    def plan_positions(self, evaluation: dict[str, Any]) -> list[float]:
        """Missing near and far positions per axis, from the current fit."""
        self.fit = evaluation
        z_min, z_max = self.setup["stage_range_mm"]
        taken = list(self._good_z())
        near: list[float] = []
        far: list[float] = []
        for axis in ("x", "y"):
            result = evaluation[axis]
            if not result["fit_success"] or not result["z_r_mm"]:
                continue
            z0, z_r = result["z0_mm"], result["z_r_mm"]
            for offsets, low, high, out in (
                (NEAR_OFFSETS, 0.0, 1.0, near),
                (FAR_OFFSETS, 2.0, np.inf, far),
            ):
                planned = taken + near + far
                have = sum(low <= abs(z - z0) / z_r <= high for z in planned)
                for offset in offsets:
                    if have >= 5:
                        break
                    z_mm = round(z0 + offset * z_r, 1)
                    if not z_min <= z_mm <= z_max:
                        continue
                    if not self._fits_sensor(z_mm, evaluation):
                        continue
                    spacing = MIN_SPACING_ZR * z_r
                    if any(abs(z_mm - z) < spacing for z in taken + near + far):
                        continue
                    out.append(z_mm)
                    have += 1
        # Alternate near and far so a short budget does not drop one kind.
        ordered = [z for pair in zip_longest(near, far) for z in pair if z is not None]
        return ordered[: max(self.positions_left, 0)]

    def measure(self, z_mm: float, reason: str) -> dict[str, Any] | None:
        """Measure with exposure correction; accept or reject. None on error."""
        exposure = self._first_exposure(z_mm)
        darks_retaken = False
        for _ in range(MAX_TRIES):
            if not self._has_budget(1):
                return None
            result = measure_at(self.session, z_mm, exposure, reason)
            if "measurement_id" not in result:
                return None
            self.positions_left = result["positions_left"]
            self.measured_z.append(z_mm)
            flags = result["flags"]
            if flags["background_changed"] and not darks_retaken:
                self._reject(result, "Background changed: retake darks first.")
                if not self._has_budget(2):
                    return result
                retake = retake_darks(self.session, "Scripted: background changed.")
                self.positions_left = retake.get("positions_left", 0)
                darks_retaken = True
                continue
            if _too_large(result):
                # No exposure fixes this position: do not spend retries on it.
                self._reject(result, "Beam too large for the sensor here.")
                return result
            if flags["saturated"]:
                self._reject(result, "Saturated: halve the exposure.")
                exposure = self._clamp(exposure / 2.0)
                continue
            if flags["low_signal"]:
                self._reject(result, "Low signal: double the exposure.")
                exposure = self._clamp(exposure * 2.0)
                continue
            cautious = [f for f in CAUTIOUS_REJECT_FLAGS if flags[f]]
            if result["suggested_verdict"] == "reject" or cautious:
                self._reject(result, f"Rejected: {result['reasons'] or cautious}.")
                return result
            set_status(
                self.session, result["measurement_id"], "accept", "No problem flags."
            )
            self.good.append(result)
            return result
        return None

    def _first_exposure(self, z_mm: float) -> float:
        """Exposure guess for a new position from a good earlier measurement."""
        if not self.good:
            return self._clamp(1.0)
        reference = min(self.good, key=lambda r: abs(r["z_mm"] - z_mm))
        scale = (
            TARGET_PEAK_FRACTION * self.full_scale / reference["details"]["peak_counts"]
        )
        predicted = self._predicted_area(z_mm)
        if predicted is not None:
            scale *= predicted / _beam_area(reference)
        else:
            # Without a fit the beam here may be much smaller and brighter
            # than at the reference: only ever shorten the exposure.
            scale = min(scale, 1.0)
        return self._clamp(reference["exposure_ms"] * scale)

    def _predicted_area(self, z_mm: float) -> float | None:
        if self.fit is None or self.fit.get("status") != "ok":
            return None
        d_x = _predicted_d_um(self.fit["x"], z_mm)
        d_y = _predicted_d_um(self.fit["y"], z_mm)
        if d_x is None or d_y is None:
            return None
        return d_x * d_y

    def _fits_sensor(self, z_mm: float, evaluation: dict[str, Any]) -> bool:
        """False if an axis with a usable fit predicts d above half the sensor."""
        for axis in ("x", "y"):
            d_um = _predicted_d_um(evaluation[axis], z_mm)
            if d_um is not None and d_um / 1e3 > MAX_D_PER_SENSOR * self.sensor_mm:
                return False
        return True

    def _good_z(self) -> list[float]:
        return [r["z_mm"] for r in self.good]

    def _has_budget(self, needed: int) -> bool:
        return self.positions_left >= needed

    def _reject(self, result: dict[str, Any], reason: str) -> None:
        set_status(self.session, result["measurement_id"], "reject", reason)

    def _clamp(self, exposure_ms: float) -> float:
        low, high = self.exposure_range
        return float(round(min(max(exposure_ms, low), high), 4))


def _predicted_d_um(axis: dict[str, Any], z_mm: float) -> float | None:
    """Predicted diameter in um from a fitted axis, or None without a fit."""
    # A collapsed fit (d0 or zR rounded to 0, e.g. far-field data only) predicts
    # nothing; treat it like no fit at all.
    if not axis["fit_success"] or not axis["d0_um"] or not axis["z_r_mm"]:
        return None
    if axis["z_r_mm"] <= 0:
        return None
    return axis["d0_um"] * float(np.hypot(1.0, (z_mm - axis["z0_mm"]) / axis["z_r_mm"]))


def _too_large(result: dict[str, Any]) -> bool:
    """True if the beam did not fit on the sensor at this position."""
    flags = result["flags"]
    return bool(flags["clipping_loss"] or flags["beam_in_corners"])


def _beam_area(result: dict[str, Any]) -> float:
    """Product of the mean widths in um² (proportional to the beam area)."""
    return result["d_x_um"]["mean"] * result["d_y_um"]["mean"]


def _finish(session: MeasurementSession, verdict: str, explanation: str) -> dict:
    result = finish(session, verdict, explanation)
    if "error" in result and verdict == "iso_compliant":
        # Should not happen (the rules use the same coverage), but never hang.
        result = finish(session, "estimate_not_compliant", explanation)
    return result
