"""Two scripted baselines that use only the public agent tools.

Both strategies call exactly the tools the agent gets (``get_setup``,
``measure_at``, ``retake_darks``, ``set_status``, ``evaluate``, ``finish``, and
``change_lens`` in setups with a lens kit) and see only their JSON results, so
the comparison with the agent is fair (design section 8). Units follow the
tool boundary: z in mm, diameters in um.

- :func:`naive_uniform`: an evenly spaced scan with one fixed exposure.
- :func:`scripted_iso`: a good-faith rule-based operator that follows
  ISO 11146-1 as closely as simple rules allow.
"""

from __future__ import annotations

from itertools import pairwise, zip_longest
from typing import Any

import numpy as np

from beampilot.session import MeasurementSession
from beampilot.tools import (
    STAGE_END_FRACTION,
    change_lens,
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
TARGET_PEAK_FRACTION = 0.7  # aimed peak, as a fraction of full scale
START_EXPOSURE_MS = 1.0  # first exposure before anything is known
MAX_EXPOSURE_STEP = 50.0  # largest exposure increase per retry
MAX_TRIES = 3  # measurements per position
# In zR. The first five are preferred; the rest are used only if some of those
# fall off the stage (waist close to a stage end).
NEAR_OFFSETS = (-0.8, -0.4, 0.0, 0.4, 0.8, -0.6, 0.6, -0.2, 0.2, -0.9, 0.9)
# Existing points count as near only within NEAR_COUNT_ZR when planning, so a
# refit (zR shifts by a few %) cannot push a counted point beyond 1 zR.
NEAR_COUNT_ZR = 0.9
# Far points start at 2.3 zR so a refit (zR shifts by a few %) keeps them
# beyond 2 zR; existing points count as far from FAR_COUNT_ZR when planning.
FAR_OFFSETS = (2.3, -2.3, 2.6, -2.6, 2.9, -2.9, 3.2, -3.2, 3.6, -3.6, 4.0, -4.0)
FAR_COUNT_ZR = 2.2
MIN_SPACING_ZR = 0.15  # skip planned positions closer than this to others
# A planned position is skipped if the predicted d exceeds this fraction of the
# smaller sensor side (0.5: sensor edges beyond +-4 sigma, clipping < 0.02 %).
MAX_D_PER_SENSOR = 0.5
PLACEMENT_ROUNDS = 3
EXTEND_ROUNDS = 3  # measurements towards a waist outside the measured span
# Warning that makes the scripted operator reject a measurement anyway: beam
# light in the corner patches biases the background offset.
CAUTIOUS_REJECT_FLAGS = ("beam_in_corners",)
# After a lens change: scout at the predicted waist and at these offsets in its
# predicted zR', clipped to the stage. Clipping can merge positions at a stage
# end, so missing ones are added by stepping into the stage until there are
# LENS_MIN_SCOUTS (the fit needs 4 positions).
LENS_SCOUT_ZR = (0.0, -1.0, 1.0, -2.5, 2.5)
LENS_MIN_SCOUTS = 4


def suggest_verdict(evaluation: dict[str, Any]) -> str:
    """Verdict from an ``evaluate`` result by fixed rules.

    ``evaluate`` reports facts, not a verdict; both baselines use these rules:
    not enough data or a failed fit on an axis -> measurement_failed; waist
    accessibility ``outside`` on an axis (z0 +- 2 sigma outside the stage;
    ``uncertain`` does not count) -> waist_not_accessible; general
    astigmatism -> out of scope; ISO coverage met on both axes and no M² < 1
    flag -> iso_compliant; otherwise estimate_not_compliant.

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
    if not all(axis["fit_success"] for axis in axes):
        return "measurement_failed"
    if any(axis["waist_accessibility"] == "outside" for axis in axes):
        return "waist_not_accessible"
    if evaluation["general_astigmatism"]:
        return "general_astigmatism_out_of_scope"
    if evaluation["iso_coverage_met"] and not evaluation.get("m2_below_1"):
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
    2. At every position: on ``saturated`` or ``low_signal`` set the next
       exposure from the measured peak (target 70 % of full scale; for a
       saturated frame the peak is estimated from the saturated area), at
       most 3 measurements per position; retake darks on
       ``background_changed`` and measure again. The first exposure at a new
       position is scaled from a good earlier measurement and the predicted
       beam size.
    3. Evaluate, then per axis place up to 5 positions within +-zR and 5
       beyond 2 zR (aimed at >= 2.3 zR, so a refit keeps them beyond 2 zR)
       that are still missing, skipping positions off the stage
       or where the predicted d exceeds half the sensor. Repeat once if
       coverage is still short.
    4. Reject every measurement with a reject reason or ``beam_in_corners``;
       accept the rest. No exposure retries at a position where the beam is
       too large for the sensor.
    5. Evaluate again and finish with :func:`suggest_verdict`.
    6. Lens rule (only in setups with a lens kit): if step 5 gives
       ``waist_not_accessible`` and ``evaluate`` lists lens options, insert
       the first one (``evaluate`` sorts them by a fixed rule: feasible
       fraction, then spare points, then the larger |f|, then the lens
       closer to the stage). On the new beam, scout at the predicted waist z0'
       and at +-1 and +-2.5 zR' (the prediction ``change_lens`` returns, the same the
       agent gets; clipped to the stage), with the first exposure scaled from
       the predicted waist, then repeat steps 3-5. At most one lens.

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
    evaluation = operator.run()
    verdict = suggest_verdict(evaluation)
    options = evaluation.get("lens_options") if evaluation.get("status") == "ok" else []
    if verdict == "waist_not_accessible" and options:
        option = options[0]
        result = change_lens(
            session,
            option["f_mm"],
            option["lens_z_mm"],
            "Scripted: insert the first lens option from evaluate (fixed rule).",
        )
        if "segment" in result:
            prediction = result["prediction"]
            hint = operator.exposure_for_waist(prediction)
            operator = _ScriptedOperator(session)
            if hint is not None:
                operator.exposure_hint = hint
            evaluation = operator.run(operator.scouts_from_prediction(prediction))
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
        self.exposure_hint = START_EXPOSURE_MS
        self.remeasured: set[float] = set()  # failed positions tried once more
        self.tried: set[tuple[float, float]] = set()  # (z_mm, exposure_ms) measured

    def run(self, scouts_mm: list[float] | None = None) -> dict[str, Any]:
        """Scout, extend towards the waist, place ISO positions; last evaluate.

        With `scouts_mm` (after a lens change) those positions replace the
        spread-out scout: they come from the predicted new waist.
        """
        session = self.session
        if scouts_mm is None:
            self.scout()
        else:
            for z_mm in scouts_mm:
                self.measure(z_mm, "Scripted: scout at the predicted new waist.")
        evaluation = evaluate(session, "Scripted: first fit from the scout positions.")
        # A waist inside the stage but outside the measured span is reported as
        # not determined: measure towards it until the waist lies inside the span.
        for _ in range(EXTEND_ROUNDS):
            if not self.extend_towards_waist(evaluation):
                break
            evaluation = evaluate(
                session, "Scripted: refit after measuring nearer the waist."
            )
        for _ in range(PLACEMENT_ROUNDS):
            if evaluation.get("status") != "ok" or evaluation["iso_coverage_met"]:
                break
            planned = self.plan_positions(evaluation)
            if not planned:
                break
            for z_mm in planned:
                self.measure(z_mm, "Scripted: ISO position planned from the fit.")
            evaluation = evaluate(session, "Scripted: refit with the new positions.")
        return evaluation

    def scouts_from_prediction(self, prediction: dict[str, Any]) -> list[float] | None:
        """Scout positions from the predicted new waist, the same numbers the agent
        gets: at z0', +-1 zR' and +-2.5 zR' (mean of both axes), clipped to the
        stage. Positions closer than the normal minimum spacing (0.15 zR', at
        least 1 mm) count once. If clipping leaves fewer than 4, positions are
        added stepping into the stage from the clipped end, every
        max(1 zR', minimum spacing), until there are 4.

        None (normal spread-out scout) if there is no prediction.
        """
        if prediction.get("basis") is None:
            return None
        z0 = 0.5 * (prediction["x"]["z0_mm"] + prediction["y"]["z0_mm"])
        z_r = 0.5 * (prediction["x"]["z_r_mm"] + prediction["y"]["z_r_mm"])
        z_min, z_max = self.setup["stage_range_mm"]
        spacing = max(MIN_SPACING_ZR * z_r, 1.0)
        scouts: list[float] = []
        clipped_ends: list[float] = []

        def add(z: float) -> None:
            if all(abs(z - s) >= spacing for s in scouts):
                scouts.append(z)

        for offset in LENS_SCOUT_ZR:
            wanted = z0 + offset * z_r
            z = round(float(np.clip(wanted, z_min, z_max)), 1)
            if z != round(wanted, 1):
                clipped_ends.append(z)
            add(z)
        step = max(z_r, spacing)
        for end in dict.fromkeys(clipped_ends):  # each clipped end once, in order
            direction = 1.0 if end <= z_min else -1.0
            z = end
            while len(scouts) < LENS_MIN_SCOUTS and z_min <= z <= z_max:
                z = round(z + direction * step, 1)
                if z_min <= z <= z_max:
                    add(z)
        return scouts

    def exposure_for_waist(self, prediction: dict[str, Any]) -> float | None:
        """Exposure in ms for the predicted new waist, from a good measurement.

        The peak scales with 1 / (d_x d_y) at fixed power, so the exposure that
        gave a good peak is scaled by the area ratio and to the target peak.
        """
        if not self.good or prediction.get("basis") is None:
            return None
        last = self.good[-1]
        area_now = last["d_x_um"]["mean"] * last["d_y_um"]["mean"]
        area_new = prediction["x"]["d0_um"] * prediction["y"]["d0_um"]
        peak = last["details"]["peak_counts"]
        if not (area_now > 0 and area_new > 0 and peak and peak > 0):
            return None
        exposure = (
            last["exposure_ms"]
            * area_new
            / area_now
            * TARGET_PEAK_FRACTION
            * self.full_scale
            / peak
        )
        return float(np.clip(exposure, *self.exposure_range))

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

    def extend_towards_waist(self, evaluation: dict[str, Any]) -> bool:
        """Measure towards a waist that is not determined or not surely inside.

        For ``waist_side`` below / above, or accessibility ``uncertain``,
        bisect between the outermost good position and that stage end; for
        ``within``, measure the middle of the gap. A waist reported
        ``outside`` is approached the same way, because a far-field-only fit
        can misplace it. Approaching stops once an accepted position lies
        within ``STAGE_END_FRACTION`` of that stage end (as ``finish`` checks
        for ``waist_not_accessible``).

        Returns True if a measurement was taken.
        """
        if evaluation.get("status") != "ok" or not self.good:
            return False
        if not self._has_budget(2):
            return False
        z_min, z_max = self.setup["stage_range_mm"]
        good_z = self._good_z()
        for axis in ("x", "y"):
            result = evaluation[axis]
            if not result["fit_success"]:
                continue
            side = result.get("waist_side")
            if side == "within" and result["waist_gap_mm"]:
                lower, upper = result["waist_gap_mm"]
                target = round(0.5 * (lower + upper), 1)
                if all(abs(target - z) >= 1.0 for z in self.measured_z):
                    self.measure(
                        target, "Scripted: the waist lies in a gap; measure there."
                    )
                    return True
                continue
            access = result["waist_accessibility"]
            if side is None and access == "uncertain":
                side = "below" if result["waist_near_stage_end"] == "start" else "above"
            if side is None and access == "outside" and result["z0_mm"] is not None:
                side = "below" if result["z0_mm"] < z_min else "above"
            if side not in ("below", "above"):
                continue
            end = z_min if side == "below" else z_max
            margin = STAGE_END_FRACTION * (z_max - z_min)
            if any(abs(end - z) <= margin for z in good_z):
                continue  # that stage end is reached (the margin finish checks)
            edge = min(good_z) if side == "below" else max(good_z)
            if self._measure_towards(edge, end, side):
                return True
        return False

    def _measure_towards(self, edge: float, end: float, side: str) -> bool:
        """Bisect from `edge` towards stage end `end`; True if measured.

        Skips positions that already have a good measurement; a position that
        only failed (e.g. a saturated first scout) may be measured once more,
        now with a known exposure.
        """
        good_z = self._good_z()
        candidates = []
        point = edge
        for _ in range(3):
            point = 0.5 * (point + end)
            candidates.append(round(point, 1))
        candidates.append(end)
        for target in candidates:
            if abs(target - edge) < 1.0 or any(abs(target - z) < 1.0 for z in good_z):
                continue
            failed_here = any(abs(target - z) < 1.0 for z in self.measured_z)
            if failed_here:
                if target in self.remeasured:
                    continue
                self.remeasured.add(target)
            self.measure(
                target, f"Scripted: the fit puts the waist {side} the measured span."
            )
            return True
        return False

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
                (NEAR_OFFSETS, 0.0, NEAR_COUNT_ZR, near),
                (FAR_OFFSETS, FAR_COUNT_ZR, np.inf, far),
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
                    # Spacing also against failed positions: no re-tries there.
                    spacing = MIN_SPACING_ZR * z_r
                    if any(
                        abs(z_mm - z) < spacing for z in self.measured_z + near + far
                    ):
                        continue
                    out.append(z_mm)
                    have += 1
            # The offset grid can miss free room in a one-sided near window
            # (waist close to a stage end): fill the largest gaps instead.
            planned = taken + near + far
            have = sum(abs(z - z0) / z_r <= NEAR_COUNT_ZR for z in planned)
            window = (
                max(z_min, z0 - NEAR_COUNT_ZR * z_r),
                min(z_max, z0 + NEAR_COUNT_ZR * z_r),
            )
            while have < 5:
                z_mm = self._largest_gap_midpoint(window, self.measured_z + near + far)
                if (
                    z_mm is None
                    or min(
                        (abs(z_mm - z) for z in self.measured_z + near + far),
                        default=np.inf,
                    )
                    < MIN_SPACING_ZR * z_r
                ):
                    break
                near.append(z_mm)
                have += 1
        # Alternate near and far so a short budget does not drop one kind.
        ordered = [z for pair in zip_longest(near, far) for z in pair if z is not None]
        return ordered[: max(self.positions_left, 0)]

    @staticmethod
    def _largest_gap_midpoint(
        window: tuple[float, float], points: list[float]
    ) -> float | None:
        """Middle of the largest free interval of `window` between `points`."""
        low, high = window
        if high <= low:
            return None
        edges = sorted({low, high, *(p for p in points if low < p < high)})
        width, start = max((b - a, a) for a, b in pairwise(edges))
        return round(start + 0.5 * width, 1)

    def measure(self, z_mm: float, reason: str) -> dict[str, Any] | None:
        """Measure with exposure correction; accept or reject. None on error."""
        exposure = self._first_exposure(z_mm)
        darks_retaken = False
        for attempt in range(MAX_TRIES):
            if not self._has_budget(1):
                return None
            # An identical repeat (same z, same exposure) cannot give anything new.
            if (z_mm, exposure) in self.tried:
                return None
            self.tried.add((z_mm, exposure))
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
            if flags["saturated"] or flags["low_signal"]:
                problem = "Saturated" if flags["saturated"] else "Low signal"
                new_exposure = self._exposure_after(result, exposure)
                self.exposure_hint = new_exposure
                if new_exposure == exposure:
                    # Clamped at the camera's exposure limit: a repeat would
                    # give the same result, so give the position up.
                    self._reject(result, f"{problem} at the exposure limit; giving up.")
                    return result
                if attempt == MAX_TRIES - 1:
                    self._reject(result, f"{problem}; no tries left here.")
                    return result
                exposure = new_exposure
                self._reject(result, f"{problem}: retry at {exposure} ms.")
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

    def _exposure_after(self, result: dict[str, Any], exposure: float) -> float:
        """Next exposure from the measured peak, aiming at 70 % of full scale.

        Peak counts scale with exposure. A saturated peak is clipped, so the
        true peak is estimated from the saturated area instead: for a
        Gaussian the pixels above full scale fill the ellipse where
        peak * exp(-r²/2σ²) > full scale, so
        peak / full scale = exp(N_sat p² / (2 pi σx σy)). Saturated frames
        have biased widths, so the step is at least a factor 2 down.
        """
        target = TARGET_PEAK_FRACTION * self.full_scale
        if result["flags"]["saturated"]:
            nx, ny = self.setup["sensor_px"]
            n_saturated = result["details"]["saturated_pixel_fraction"] * nx * ny
            pixel_um = self.setup["pixel_size_um"]
            d_x, d_y = result["d_x_um"]["mean"], result["d_y_um"]["mean"]
            if d_x and d_y:  # None if the widths could not be measured
                sigma_x, sigma_y = d_x / 4.0, d_y / 4.0
                exponent = n_saturated * pixel_um**2 / (2 * np.pi * sigma_x * sigma_y)
                peak = self.full_scale * float(np.exp(min(exponent, 50.0)))
            else:
                peak = 2.0 * self.full_scale
            factor = min(target / peak, 0.5)
        else:
            peak = result["details"]["peak_counts"] or 0.0
            # Noise alone gives a peak of a few tens of counts: cap the step.
            factor = target / peak if peak > 0 else MAX_EXPOSURE_STEP
            factor = min(factor, MAX_EXPOSURE_STEP)
        return self._clamp(exposure * factor)

    def _first_exposure(self, z_mm: float) -> float:
        """Exposure guess for a new position from a good earlier measurement."""
        if not self.good:
            # Nothing good yet: use the last suggested exposure, if any.
            return self._clamp(self.exposure_hint)
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
    if "error" in result:
        # The rules mirror finish's guards, so this should not happen; if it
        # does, say so instead of hiding it behind the original explanation.
        result = finish(
            session,
            "estimate_not_compliant",
            f"Scripted: finish refused {verdict} ({result['error']}).",
        )
    return result
