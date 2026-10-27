"""The six agent tools (design section 4), as plain functions on a session.

Every tool takes a required `reason` (``finish`` uses its `explanation`),
returns a JSON-serialisable dict, and never raises on bad input: it returns
``{"error": "..."}`` with a helpful message instead. Units at this boundary:
z in mm, diameters in um, wavelength in nm, exposure in ms. Inside the
session everything is SI. Every call, including failed ones, is appended to
``session.log`` with its arguments, reason and result.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

from beampilot.caustic import CausticFit, coverage, fit_caustic, general_astigmatism
from beampilot.session import N_DARK_FRAMES, Measurement, MeasurementSession
from beampilot.widths import measure_widths, subtract_background, width_statistics

# Quality-flag thresholds.
LOW_SIGNAL_COUNTS = 500.0  # background-subtracted peak in counts
MIN_PIXELS_ACROSS = 20  # ISO 11146-1: at least 20 px across the beam width
BACKGROUND_SIGMA_LIMIT = 4.0  # |residual offset| / std error (ISO/TR 11146-3)
ALIGNMENT_LIMIT_DEG = 10.0  # principal axes vs lab x/y
CIRCULAR_THRESHOLD = 0.87  # ellipticity above this: circular (unverified, ISO)
MIN_FIT_POSITIONS = 4

VERDICTS = (
    "iso_compliant",
    "estimate_not_compliant",
    "waist_not_accessible",
    "general_astigmatism_out_of_scope",
    "measurement_failed",
)
# Flags that make the suggested verdict "reject"; the others are warnings.
REJECT_FLAGS = {
    "saturated": "saturated: pixels at full scale, use a shorter exposure",
    "low_signal": f"low signal: peak below {LOW_SIGNAL_COUNTS:.0f} counts, "
    "use a longer exposure",
    "roi_clipped": "beam area cut by the sensor edge: widths biased low",
    "too_few_pixels": f"fewer than {MIN_PIXELS_ACROSS} pixels across the beam",
    "not_converged": "width iteration failed or did not converge",
}
WARNING_FLAGS = {
    "background_changed": "background differs from the dark frames "
    "(corrected by the corner offset, but check the setup)",
    "axes_not_aligned": "elliptical beam not aligned to x/y: lab-axis widths "
    "are not ISO widths",
}


class ToolInputError(ValueError):
    """Bad tool input; becomes the ``error`` field of the result."""


def _tool(reason_arg: str = "reason") -> Callable:
    """Wrap a tool: validate reason, catch bad input, log the call."""

    def decorate(func: Callable[..., dict[str, Any]]) -> Callable[..., dict]:
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(session: MeasurementSession, *args: Any, **kwargs: Any) -> dict:
            arguments: dict[str, Any] = {}
            try:
                try:
                    bound = signature.bind(session, *args, **kwargs)
                except TypeError as exc:
                    raise ToolInputError(
                        f"bad arguments for {func.__name__}: {exc}"
                    ) from None
                arguments = {k: v for k, v in bound.arguments.items() if k != "session"}
                reason = arguments.get(reason_arg)
                if not isinstance(reason, str) or not reason.strip():
                    raise ToolInputError(f"'{reason_arg}' must be a non-empty string")
                if session.finished:
                    raise ToolInputError("the run is finished; no further tool calls")
                result = func(session, *args, **kwargs)
            except ToolInputError as exc:
                result = {"error": str(exc)}
            session.log.append(
                {
                    "tool": func.__name__,
                    "arguments": arguments,
                    "reason": arguments.get(reason_arg),
                    "result": result,
                }
            )
            return result

        return wrapper

    return decorate


@_tool()
def get_setup(session: MeasurementSession, reason: str) -> dict[str, Any]:
    """Fixed setup data the agent needs to plan the run.

    Parameters
    ----------
    session : MeasurementSession
        The run.
    reason : str
        Why the agent calls this tool.

    Returns
    -------
    dict
        Wavelength in nm, stage range in mm, pixel size in um, sensor size in
        px and mm, full scale in counts, exposure range in ms, repeats per
        position, position budget, and the smallest beam width in um that
        still has 20 pixels across it.
    """
    scenario, camera = session.scenario, session.camera
    pixel_um = camera.pixel_size_m * 1e6
    return {
        "wavelength_nm": _num(session.wavelength_m * 1e9, 3),
        "stage_range_mm": [
            _num(scenario.stage_min_m * 1e3, 3),
            _num(scenario.stage_max_m * 1e3, 3),
        ],
        "pixel_size_um": _num(pixel_um, 3),
        "sensor_px": [camera.nx, camera.ny],
        "sensor_size_mm": [
            _num(camera.nx * pixel_um / 1e3, 3),
            _num(camera.ny * pixel_um / 1e3, 3),
        ],
        "full_scale_counts": camera.full_scale_counts,
        "exposure_range_ms": [scenario.min_exposure_ms, scenario.max_exposure_ms],
        "repeats_per_position": scenario.repeats,
        "max_positions": scenario.max_positions,
        "positions_left": session.positions_left,
        "min_beam_width_um": _num(MIN_PIXELS_ACROSS * pixel_um, 1),
    }


@_tool()
def measure_at(
    session: MeasurementSession, z_mm: float, exposure_ms: float, reason: str
) -> dict[str, Any]:
    """Move the stage, take the repeated frames and compute ISO widths.

    Each frame gets dark subtraction and the fine corner offset, then ISO
    second-moment widths. Quality flags (True means a problem):

    - ``saturated``: any raw pixel at full scale.
    - ``low_signal``: mean background-subtracted peak below 500 counts.
    - ``roi_clipped``: the 3 x d integration area hit the sensor edge.
    - ``too_few_pixels``: the smaller mean width spans fewer than 20 pixels.
    - ``not_converged``: a frame's width iteration failed or did not settle.
    - ``background_changed``: in some frame |residual offset| exceeds 4 times
      its standard error (ISO/TR 11146-3), i.e. the background is no longer
      the one in the dark frames.
    - ``axes_not_aligned``: for the non-circular frames (ellipticity <= 0.87),
      the median azimuth deviates from the x/y axes by more than 10 degrees
      (azimuth taken modulo 90 degrees).

    The first five make the suggested verdict ``"reject"``; the last two are
    warnings. The measurement is stored with status ``"pending"``.

    Parameters
    ----------
    session : MeasurementSession
        The run.
    z_mm : float
        Stage position in mm, inside the stage range.
    exposure_ms : float
        Exposure time in ms, inside the exposure range.
    reason : str
        Why the agent measures here.

    Returns
    -------
    dict
        ``measurement_id``, z in mm, exposure in ms, mean and std of d_x and
        d_y in um, ``status``, ``positions_left``, ``flags``,
        ``suggested_verdict`` with ``reasons`` and ``warnings``, and
        ``details`` (peak and offset in counts, centroid in um, ellipticity,
        azimuth in degrees, pixels across the beam).
    """
    scenario, camera = session.scenario, session.camera
    if session.positions_left <= 0:
        raise ToolInputError(
            f"position budget used up ({scenario.max_positions}); "
            "call evaluate and finish"
        )
    z_mm = _input_number(z_mm, "z_mm")
    exposure_ms = _input_number(exposure_ms, "exposure_ms")
    z_min_mm, z_max_mm = scenario.stage_min_m * 1e3, scenario.stage_max_m * 1e3
    if not z_min_mm <= z_mm <= z_max_mm:
        raise ToolInputError(
            f"z_mm = {z_mm} is outside the stage range [{z_min_mm:g}, {z_max_mm:g}] mm"
        )
    if not scenario.min_exposure_ms <= exposure_ms <= scenario.max_exposure_ms:
        raise ToolInputError(
            f"exposure_ms = {exposure_ms} is outside the range "
            f"[{scenario.min_exposure_ms:g}, {scenario.max_exposure_ms:g}] ms"
        )

    z_m = z_mm * 1e-3
    pixel_m = camera.pixel_size_m
    widths, peaks, offsets, offset_ratios, saturation = [], [], [], [], []
    for raw in session.take_frames(z_m, exposure_ms):
        saturation.append(float(np.mean(raw >= camera.full_scale_counts)))
        image, background = subtract_background(raw, session.dark_mean)
        widths.append(measure_widths(image, pixel_m))
        peaks.append(float(image.max()))
        offsets.append(background.offset_counts)
        offset_ratios.append(
            abs(background.offset_counts) / background.offset_std_err_counts
        )
    stats = width_statistics(widths)
    pixels_across = min(stats.d_x_mean_m, stats.d_y_mean_m) / pixel_m
    misalignment_deg = _median_misalignment_deg(widths)

    flags = {
        "saturated": max(saturation) > 0,
        "low_signal": float(np.mean(peaks)) < LOW_SIGNAL_COUNTS,
        "roi_clipped": any(w.roi_clipped for w in widths),
        "too_few_pixels": bool(pixels_across < MIN_PIXELS_ACROSS),
        "not_converged": not all(w.converged for w in widths),
        "background_changed": max(offset_ratios) > BACKGROUND_SIGMA_LIMIT,
        "axes_not_aligned": bool(misalignment_deg > ALIGNMENT_LIMIT_DEG),
    }
    flags = {name: bool(value) for name, value in flags.items()}  # no numpy bools
    reasons = [text for flag, text in REJECT_FLAGS.items() if flags[flag]]
    warnings = [text for flag, text in WARNING_FLAGS.items() if flags[flag]]
    details = {
        "peak_counts": _num(np.mean(peaks), 0),
        "saturated_pixel_fraction": _num(max(saturation), 6),
        "residual_offset_counts": _num(np.mean(offsets), 3),
        "max_offset_sigma": _num(max(offset_ratios), 1),
        "pixels_across": _num(pixels_across, 1),
        "centroid_um": [
            _num(np.mean([w.x_c_m for w in widths]) * 1e6, 1),
            _num(np.mean([w.y_c_m for w in widths]) * 1e6, 1),
        ],
        "ellipticity": _num(np.median([w.ellipticity for w in widths]), 3),
        "azimuth_deg": _num(np.degrees(np.median([w.azimuth_rad for w in widths])), 1),
        "misalignment_deg": _num(misalignment_deg, 1),
    }
    measurement = Measurement(
        measurement_id=session.next_measurement_id(),
        z_m=z_m,
        exposure_ms=exposure_ms,
        widths=widths,
        d_x_mean_m=stats.d_x_mean_m,
        d_x_std_m=stats.d_x_std_m,
        d_y_mean_m=stats.d_y_mean_m,
        d_y_std_m=stats.d_y_std_m,
        flags=flags,
        details=details,
        suggested_verdict="reject" if reasons else "accept",
    )
    session.add_measurement(measurement)
    return {
        "measurement_id": measurement.measurement_id,
        "z_mm": _num(z_mm, 3),
        "exposure_ms": _num(exposure_ms, 4),
        "d_x_um": {
            "mean": _num(stats.d_x_mean_m * 1e6, 1),
            "std": _num(stats.d_x_std_m * 1e6, 1),
        },
        "d_y_um": {
            "mean": _num(stats.d_y_mean_m * 1e6, 1),
            "std": _num(stats.d_y_std_m * 1e6, 1),
        },
        "status": measurement.status,
        "positions_left": session.positions_left,
        "flags": flags,
        "suggested_verdict": measurement.suggested_verdict,
        "reasons": reasons,
        "warnings": warnings,
        "details": details,
    }


@_tool()
def retake_darks(session: MeasurementSession, reason: str) -> dict[str, Any]:
    """Block the beam and take a fresh dark set with the current background.

    Use it when ``background_changed`` fires, e.g. after room light was
    switched on: the new background becomes part of the dark frames. Costs
    one position from the budget.

    Parameters
    ----------
    session : MeasurementSession
        The run.
    reason : str
        Why.

    Returns
    -------
    dict
        Number of dark frames, dark sets taken so far, mean of the new dark
        in counts, and positions left.
    """
    if session.positions_left <= 0:
        raise ToolInputError("position budget used up; call evaluate and finish")
    session.retake_darks()
    return {
        "dark_frames": N_DARK_FRAMES,
        "dark_sets_taken": session.dark_sets_taken,
        "mean_dark_counts": _num(session.dark_mean.mean(), 2),
        "positions_left": session.positions_left,
    }


@_tool()
def set_status(
    session: MeasurementSession, measurement_id: str, status: str, reason: str
) -> dict[str, Any]:
    """Accept or reject a measurement. Only accepted ones enter the fit.

    A saturated measurement, or one whose widths could not be measured,
    cannot be accepted. The status may be changed again later.

    Parameters
    ----------
    session : MeasurementSession
        The run.
    measurement_id : str
        Id returned by ``measure_at``, e.g. ``"m3"``.
    status : str
        ``"accept"`` or ``"reject"``.
    reason : str
        Why.

    Returns
    -------
    dict
        The id, the new status and the number of accepted measurements.
    """
    measurement = session.measurements.get(measurement_id)
    if measurement is None:
        known = ", ".join(session.measurements) or "none yet"
        raise ToolInputError(
            f"unknown measurement_id {measurement_id!r}; known ids: {known}"
        )
    if status not in ("accept", "reject"):
        raise ToolInputError(f"status must be 'accept' or 'reject', got {status!r}")
    if status == "accept" and measurement.flags["saturated"]:
        raise ToolInputError(
            f"{measurement_id} is saturated and cannot be accepted; measure again "
            "with a shorter exposure"
        )
    if (
        status == "accept"
        and not np.isfinite([measurement.d_x_mean_m, measurement.d_y_mean_m]).all()
    ):
        raise ToolInputError(f"{measurement_id} has no valid widths to accept")
    measurement.status = status
    accepted = sum(m.status == "accept" for m in session.measurements.values())
    return {
        "measurement_id": measurement_id,
        "status": status,
        "accepted_count": accepted,
    }


@_tool()
def evaluate(session: MeasurementSession, reason: str) -> dict[str, Any]:
    """Caustic fit and ISO checks on the accepted measurements, per axis.

    Accepted measurements at the same z are averaged into one position. Each
    axis gets :func:`beampilot.caustic.fit_caustic` with relative residuals,
    ISO coverage in units of its own zR, and a waist-in-range check. Also
    reports general astigmatism (over all accepted frames), an M² < 1 sanity
    flag, and a compact table of all measurements.

    Parameters
    ----------
    session : MeasurementSession
        The run.
    reason : str
        Why.

    Returns
    -------
    dict
        ``status`` ``"ok"`` with ``x`` and ``y`` results (d0 in um, z0 and zR
        in mm, theta in mrad, M², each with std; coverage counts and text;
        waist in range; M² < 1 flag), ``iso_coverage_met``,
        ``general_astigmatism``, ``axes_not_aligned`` and ``table``; or
        ``status`` ``"not_enough_data"`` with a message and the table.
    """
    return _evaluation(session)


@_tool(reason_arg="explanation")
def finish(
    session: MeasurementSession, verdict: str, explanation: str
) -> dict[str, Any]:
    """End the run with a verdict; the report numbers come from session state.

    ``iso_compliant`` is refused if, on a fresh evaluation of the accepted
    measurements, ISO coverage is not met on both axes.

    Parameters
    ----------
    session : MeasurementSession
        The run.
    verdict : str
        One of ``iso_compliant``, ``estimate_not_compliant``,
        ``waist_not_accessible``, ``general_astigmatism_out_of_scope``,
        ``measurement_failed``.
    explanation : str
        The agent's explanation, stored in the report.

    Returns
    -------
    dict
        ``{"report": ...}`` with the verdict, the explanation, the evaluation
        built from the session, and measurement counts.
    """
    if verdict not in VERDICTS:
        raise ToolInputError(f"verdict must be one of {', '.join(VERDICTS)}")
    evaluation = _evaluation(session)
    if verdict == "iso_compliant" and not evaluation.get("iso_coverage_met", False):
        coverage_text = [
            evaluation[axis]["coverage"]["summary"]
            for axis in ("x", "y")
            if axis in evaluation and evaluation[axis]["coverage"]
        ]
        detail = "; ".join(coverage_text) or evaluation.get("message", "")
        raise ToolInputError(
            "cannot report iso_compliant: ISO coverage is not met on both axes "
            f"({detail}). Measure more positions or choose another verdict."
        )
    statuses = [m.status for m in session.measurements.values()]
    report = {
        "verdict": verdict,
        "explanation": explanation,
        "evaluation": evaluation,
        "positions_measured": len(statuses),
        "positions_accepted": statuses.count("accept"),
        "positions_rejected": statuses.count("reject"),
        "positions_pending": statuses.count("pending"),
    }
    session.report = report
    return {"report": report}


def _evaluation(session: MeasurementSession) -> dict[str, Any]:
    """Fit and checks on the accepted measurements, in display units."""
    table = [_table_row(m) for m in session.measurements.values()]
    accepted = [m for m in session.measurements.values() if m.status == "accept"]
    z_positions = sorted({m.z_m for m in accepted})
    if len(z_positions) < MIN_FIT_POSITIONS:
        return {
            "status": "not_enough_data",
            "message": (
                f"{len(z_positions)} accepted position(s); the caustic fit needs at "
                f"least {MIN_FIT_POSITIONS}. ISO 11146-1 asks for at least 10: "
                "about 5 within one Rayleigh length of the waist and 5 beyond two."
            ),
            "accepted_positions": len(z_positions),
            "table": table,
        }

    z = np.array(z_positions)
    wavelength_m = session.wavelength_m
    result: dict[str, Any] = {"status": "ok", "accepted_positions": len(z)}
    coverage_met = []
    m2_below_1 = []
    for axis, attribute in (("x", "d_x_mean_m"), ("y", "d_y_mean_m")):
        d = np.array(
            [
                np.mean([getattr(m, attribute) for m in accepted if m.z_m == zi])
                for zi in z_positions
            ]
        )
        fit = fit_caustic(z, d, wavelength_m=wavelength_m)
        axis_result = _axis_result(session, axis, z, fit)
        result[axis] = axis_result
        coverage_met.append(
            axis_result["coverage"] is not None and axis_result["coverage"]["compliant"]
        )
        m2_below_1.append(axis_result["m2_below_1"])

    azimuths = [w.azimuth_rad for m in accepted for w in m.widths]
    ellipticities = [w.ellipticity for m in accepted for w in m.widths]
    result["iso_coverage_met"] = all(coverage_met)
    result["general_astigmatism"] = general_astigmatism(
        azimuths, ellipticities, circular_threshold=CIRCULAR_THRESHOLD
    )
    result["axes_not_aligned"] = any(m.flags["axes_not_aligned"] for m in accepted)
    result["m2_below_1"] = any(m2_below_1)
    result["table"] = table
    return result


def _axis_result(
    session: MeasurementSession, axis: str, z: np.ndarray, fit: CausticFit
) -> dict[str, Any]:
    """One axis of the evaluation in display units, with coverage."""
    scenario = session.scenario
    coverage_result = None
    if fit.success and np.isfinite(fit.z_r_m) and fit.z_r_m > 0:
        cov = coverage(z, fit.z0_m, fit.z_r_m)
        coverage_result = {
            "near": cov.n_near,
            "far": cov.n_far,
            "between": cov.n_between,
            "compliant": cov.compliant,
            "summary": (
                f"{axis}: {cov.n_near} within ±1 zR, need 5; "
                f"{cov.n_far} beyond 2 zR, need 5"
            ),
        }
    return {
        "fit_success": fit.success,
        "d0_um": _num(fit.d0_m * 1e6, 2),
        "d0_std_um": _num(fit.d0_std_m * 1e6, 2),
        "z0_mm": _num(fit.z0_m * 1e3, 3),
        "z0_std_mm": _num(fit.z0_std_m * 1e3, 3),
        "theta_mrad": _num(fit.theta_rad * 1e3, 4),
        "theta_std_mrad": _num(fit.theta_std_rad * 1e3, 4),
        "z_r_mm": _num(fit.z_r_m * 1e3, 3),
        "z_r_std_mm": _num(fit.z_r_std_m * 1e3, 3),
        "m2": _num(fit.m2, 4),
        "m2_std": _num(fit.m2_std, 4),
        "residual_rms_percent": _num(fit.residual_rms * 100, 3),
        "coverage": coverage_result,
        "waist_in_stage_range": bool(
            scenario.stage_min_m <= fit.z0_m <= scenario.stage_max_m
        ),
        "m2_below_1": bool(fit.m2 < 1.0),
    }


def _table_row(m: Measurement) -> dict[str, Any]:
    return {
        "id": m.measurement_id,
        "z_mm": _num(m.z_m * 1e3, 3),
        "d_x_um": _num(m.d_x_mean_m * 1e6, 1),
        "d_y_um": _num(m.d_y_mean_m * 1e6, 1),
        "status": m.status,
        "suggested": m.suggested_verdict,
        "flags": [name for name, value in m.flags.items() if value],
    }


def _median_misalignment_deg(widths: list) -> float:
    """Median angle between principal axes and x/y over non-circular frames.

    Returns 0 if no frame is non-circular (alignment is then meaningless).
    """
    deviations = []
    for w in widths:
        if np.isfinite(w.ellipticity) and w.ellipticity <= CIRCULAR_THRESHOLD:
            folded = np.degrees(w.azimuth_rad) % 90.0
            deviations.append(min(folded, 90.0 - folded))
    return float(np.median(deviations)) if deviations else 0.0


def _input_number(value: Any, name: str) -> float:
    """Tool argument as a finite float, else ToolInputError."""
    if isinstance(value, bool):
        raise ToolInputError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ToolInputError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(number):
        raise ToolInputError(f"{name} must be finite, got {value!r}")
    return number


def _num(value: float, digits: int) -> float | None:
    """Round for display; NaN or inf become None (valid JSON)."""
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None
