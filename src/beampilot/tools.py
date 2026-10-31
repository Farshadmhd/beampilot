"""The seven agent tools (design section 4), as plain functions on a session.

``change_lens`` is an eighth tool, only for setups with a lens kit
(docs/change-lens-design.md).

Every tool takes a required `reason` (``finish`` uses its `explanation`),
returns a JSON-serialisable dict, and never raises: bad input gives
``{"error": "..."}`` with a helpful message, and any unexpected exception
gives ``{"internal_error": "..."}`` (its traceback goes to the log), so a
demo run can never crash on a tool call. Units at this boundary:
z in mm, diameters in um, wavelength in nm, exposure in ms. Inside the
session everything is SI. Every call, including failed ones, is appended to
``session.log`` with its arguments, reason and result.
"""

from __future__ import annotations

import base64
import inspect
import traceback
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
from scipy.special import erf

from beampilot import lens_plan
from beampilot.caustic import CausticFit, coverage, fit_caustic, general_astigmatism
from beampilot.plots import frame_png
from beampilot.session import N_DARK_FRAMES, Measurement, MeasurementSession
from beampilot.widths import (
    DEFAULT_CORNER_FRACTION,
    Roi,
    WidthResult,
    measure_widths,
    subtract_background,
    width_statistics,
)

# Quality-flag thresholds.
LOW_SIGNAL_COUNTS = 500.0  # background-subtracted peak in counts
MIN_PIXELS_ACROSS = 20  # ISO 11146-1: at least 20 px across the beam width
# |residual offset| / std error. A plain statistical test; that ISO/TR 11146-3
# prescribes 4 sigma is unverified: check the standard text.
BACKGROUND_SIGMA_LIMIT = 4.0
ALIGNMENT_LIMIT_DEG = 10.0  # principal axes vs lab x/y
MAX_CLIPPING_LOSS = 0.01  # ISO 11146-1: power outside the sensor below 1 %
M2_STD_FLOOR_REL = 0.01  # smallest M² uncertainty used for the M² < 1 flag
# waist_not_accessible needs an accepted position this close to the stage end
# on the waist's side (fraction of the stage length).
STAGE_END_FRACTION = 0.05
# Modelled beam light in the corner patches, in offset standard errors, from
# which the offset counts as beam light rather than background.
CORNER_LIGHT_SIGMA = 1.0
CIRCULAR_THRESHOLD = 0.87  # ellipticity above this: circular (unverified, ISO)
MIN_FIT_POSITIONS = 4

LENS_SIMPLIFICATIONS = (
    "thin lens (no thickness, no principal-plane offset)",
    "paraxial optics",
    "no aberrations (spherical, chromatic or astigmatic)",
    "lens centred and not tilted; focal length and position exact (real lenses "
    "have about +-1 % focal-length tolerance, which can dominate z0 for short "
    "focal lengths; see the sensitivity per +1 % f)",
    "no clipping at the lens aperture",
    "no Fresnel losses or reflections from the lens",
    "embedded-Gaussian model: second moments transform by ABCD",
    "same spherical lens for x and y",
)

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
    "clipping_loss": f"estimated clipping loss above {MAX_CLIPPING_LOSS:.0%} of the "
    "power (ISO 11146-1): beam too large or too far off-centre for the sensor",
    "too_few_pixels": f"fewer than {MIN_PIXELS_ACROSS} pixels across the beam",
    "not_converged": "width iteration failed or did not converge",
}
WARNING_FLAGS = {
    "roi_clipped": "the 3 x d integration area is cut by the sensor edge "
    "(estimated clipping loss below 1 %)",
    "beam_in_corners": "beam light reaches the corner patches used for the "
    "background offset: it inflates the offset and biases widths low; centre "
    "the beam or measure where it is smaller",
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
            trace = None
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
            except Exception as exc:  # noqa: BLE001 - the demo must never crash
                result = {"internal_error": f"{type(exc).__name__}: {exc}"}
                trace = traceback.format_exc()
            entry = {
                "tool": func.__name__,
                "arguments": arguments,
                "reason": arguments.get(reason_arg),
                "result": result,
            }
            if trace is not None:
                entry["traceback"] = trace
            session.log.append(entry)
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
        position, position budget, the smallest beam width in um that still
        has 20 pixels across it, and the ``view_frame`` budget left (only if
        the run has one).
    """
    scenario, camera = session.scenario, session.camera
    pixel_um = camera.pixel_size_m * 1e6
    setup: dict[str, Any] = {
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
    if scenario.max_views > 0:  # no view budget: the tool is not offered
        setup["views_left"] = session.views_left
    kit = scenario.lens_kit
    if kit is not None:  # no kit: the lens tool is not offered
        setup["lens_kit_mm"] = [_num(f * 1e3, 3) for f in kit.focal_lengths_m]
        setup["lens_rail_mm"] = [
            _num(kit.rail_min_m * 1e3, 3),
            _num(kit.rail_max_m * 1e3, 3),
            _num(kit.rail_step_m * 1e3, 3),
        ]
        setup["source_z_mm"] = _num(kit.source_z_m * 1e3, 3)
        setup["lens_changes_left"] = kit.max_changes - len(session.lens_changes)
    return setup


@_tool()
def measure_at(
    session: MeasurementSession, z_mm: float, exposure_ms: float, reason: str
) -> dict[str, Any]:
    """Move the stage, take the repeated frames and compute ISO widths.

    Each frame gets dark subtraction and the fine corner offset, then ISO
    second-moment widths. Quality flags (True means a problem):

    - ``saturated``: any raw pixel at full scale.
    - ``low_signal``: mean background-subtracted peak below 500 counts.
    - ``clipping_loss``: more than 1 % of the power of a Gaussian with the
      measured centroid and sigmas (d / 4) falls outside the sensor (ISO
      11146-1). The widths of a clipped beam are themselves biased low, so
      this estimate understates the loss somewhat.
    - ``too_few_pixels``: the smaller mean width spans fewer than 20 pixels.
    - ``not_converged``: a frame's width iteration failed or did not settle.
    - ``roi_clipped``: the 3 x d integration area hit the sensor edge.
    - ``beam_in_corners``: in some frame the converged 3 x d rectangle
      overlaps a corner patch used for the fine offset and a Gaussian with
      the measured centroid, widths and signal puts at least one offset
      standard error of light into the corner patches, so the offset is
      inflated by beam light.
    - ``background_changed``: in some frame without beam in the corners,
      |residual offset| exceeds 4 times its standard error, i.e. the
      background is no longer the one in the dark frames. The 4 sigma limit
      is a statistical choice; that ISO/TR 11146-3 asks for it is unverified.
    - ``axes_not_aligned``: for the non-circular frames (ellipticity <= 0.87),
      the median azimuth deviates from the x/y axes by more than 10 degrees
      (azimuth taken modulo 90 degrees).

    The first five make the suggested verdict ``"reject"``; the last four are
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
    in_corners = []
    frames = session.take_frames(z_m, exposure_ms)
    for raw in frames:
        saturation.append(float(np.mean(raw >= camera.full_scale_counts)))
        image, background = subtract_background(raw, session.dark_mean)
        width = measure_widths(image, pixel_m)
        widths.append(width)
        peaks.append(float(image.max()))
        offsets.append(background.offset_counts)
        offset_ratios.append(
            abs(background.offset_counts) / background.offset_std_err_counts
        )
        light = _corner_beam_light_counts(width, camera.nx, camera.ny, pixel_m)
        in_corners.append(
            _roi_in_corners(width.roi, camera.nx, camera.ny)
            and light >= CORNER_LIGHT_SIGMA * background.offset_std_err_counts
        )
    stats = width_statistics(widths)
    clipping_loss = _clipping_loss(
        float(np.mean([w.x_c_m for w in widths])),
        float(np.mean([w.y_c_m for w in widths])),
        stats.d_x_mean_m / 4.0,
        stats.d_y_mean_m / 4.0,
        camera.nx * pixel_m,
        camera.ny * pixel_m,
    )
    pixels_across = min(stats.d_x_mean_m, stats.d_y_mean_m) / pixel_m
    misalignment_deg = _median_misalignment_deg(widths)

    flags = {
        "saturated": max(saturation) > 0,
        "low_signal": float(np.mean(peaks)) < LOW_SIGNAL_COUNTS,
        "clipping_loss": bool(clipping_loss > MAX_CLIPPING_LOSS),
        "too_few_pixels": bool(pixels_across < MIN_PIXELS_ACROSS),
        "not_converged": not all(w.converged for w in widths),
        "roi_clipped": any(w.roi_clipped for w in widths),
        "beam_in_corners": any(in_corners),
        # Frames with beam light in the corners say nothing about the background.
        "background_changed": any(
            ratio > BACKGROUND_SIGMA_LIMIT and not corner
            for ratio, corner in zip(offset_ratios, in_corners, strict=True)
        ),
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
        "clipping_loss_percent": _num(clipping_loss * 100, 3),
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
        frame=frames[0].astype(np.uint16),  # 12-bit counts: half the memory
    )
    session.add_measurement(measurement)
    result = {
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
    if session.scenario.lens_kit is not None:
        result["segment"] = measurement.segment
    return result


@_tool()
def view_frame(
    session: MeasurementSession, measurement_id: str, reason: str
) -> dict[str, Any]:
    """Look at one camera frame of a measurement, as an image.

    Optional. Returns the first of the measurement's repeated frames as a PNG
    on a square-root brightness scale (black = dark level, yellow = full
    scale), which shows structure the second-moment numbers cannot: extra
    spots, odd shapes, background patterns, clipping. Uses a small separate
    budget (``max_views``, default 4), not positions.

    The image travels to the model next to this result (the agent loop
    attaches it by ``image_id``); logs and run files keep only this dict.

    Parameters
    ----------
    session : MeasurementSession
        The run.
    measurement_id : str
        Id returned by ``measure_at``, e.g. ``"m3"``.
    reason : str
        Why the agent wants to look.

    Returns
    -------
    dict
        Measurement id, z in mm, exposure in ms, image size in px, the
        brightness scale, views left, and ``image_id``.
    """
    measurement = session.measurements.get(measurement_id)
    if measurement is None:
        known = ", ".join(session.measurements) or "none yet"
        raise ToolInputError(
            f"unknown measurement_id {measurement_id!r}; known ids: {known}"
        )
    if session.views_left <= 0:
        raise ToolInputError(
            f"view budget used up ({session.scenario.max_views} views per run)"
        )
    if measurement.frame is None:
        raise ToolInputError(f"{measurement_id} has no stored frame")
    camera = session.camera
    png = frame_png(measurement.frame, camera.full_scale_counts, camera.offset_counts)
    session.views_used += 1
    image_id = f"view{session.views_used}-{measurement_id}"
    session.images[image_id] = "data:image/png;base64," + base64.b64encode(png).decode()
    return {
        "measurement_id": measurement_id,
        "z_mm": _num(measurement.z_m * 1e3, 3),
        "exposure_ms": _num(measurement.exposure_ms, 4),
        "image_px": [camera.nx, camera.ny],
        "pixel_size_um": _num(camera.pixel_size_m * 1e6, 3),
        "brightness_scale": "square root: black = dark level, yellow = full scale",
        "orientation": "row 0 at the top; +x to the right, +y down (as centroid_um)",
        "views_left": session.views_left,
        "image_id": image_id,
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
    if measurement.segment != session.segment:
        raise ToolInputError(
            f"{measurement_id} belongs to the beam before the lens change "
            f"(segment {measurement.segment}, now {session.segment}); it no longer "
            "enters any fit"
        )
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
    flag (only for a successful fit with M² + 2 sigma < 1), and a compact
    table of all measurements.

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
        waist accessibility inside / outside / uncertain from z0 +- 2 sigma;
        M² < 1 flag; d0, z0, zR and M² are None with a ``not_determined``
        reason if the fit failed or no accepted position lies within +-1 zR
        of the fitted waist, and ``waist_side`` then says where it seems to
        be), ``iso_coverage_met``,
        ``general_astigmatism``, ``axes_not_aligned`` and ``table``; or
        ``status`` ``"not_enough_data"`` with a message and the table.
    """
    return _evaluation(session)


@_tool()
def change_lens(
    session: MeasurementSession, f_mm: float, lens_z_mm: float, reason: str
) -> dict[str, Any]:
    """Insert a thin lens before the stage to form an artificial waist.

    Only for setups with a lens kit (``get_setup`` lists it). Costs one
    position. A new beam segment starts: earlier measurements no longer enter
    any fit and cannot change status. A later change replaces the lens (the
    original beam through the new lens only). Refused for a focal length
    outside the kit, a position off the rail, no changes or positions left,
    or a beam that would overfill the lens at that position.

    Parameters
    ----------
    session : MeasurementSession
        The run.
    f_mm : float
        Focal length in mm, one of ``lens_kit_mm``.
    lens_z_mm : float
        Lens position in mm (stage frame), on ``lens_rail_mm``.
    reason : str
        Why.

    Returns
    -------
    dict
        The lens (focal length and position in mm), the new segment, positions
        and lens changes left, and a note on the new segment.
    """
    kit = session.scenario.lens_kit
    if kit is None:
        raise ToolInputError("this setup has no lens kit")
    f_m = _input_number(f_mm, "f_mm") * 1e-3
    lens_z_m = _input_number(lens_z_mm, "lens_z_mm") * 1e-3
    if not any(abs(f_m - f) <= 1e-9 for f in kit.focal_lengths_m):
        kit_mm = ", ".join(f"{f * 1e3:g}" for f in kit.focal_lengths_m)
        raise ToolInputError(f"f_mm must be one of the kit: {kit_mm}")
    steps = (lens_z_m - kit.rail_min_m) / kit.rail_step_m
    if not (
        kit.rail_min_m - 1e-9 <= lens_z_m <= kit.rail_max_m + 1e-9
        and abs(steps - round(steps)) < 1e-6
    ):
        raise ToolInputError(
            f"lens_z_mm must be on the rail: {kit.rail_min_m * 1e3:g} to "
            f"{kit.rail_max_m * 1e3:g} mm in steps of {kit.rail_step_m * 1e3:g} mm"
        )
    lens_z_m = kit.rail_min_m + round(steps) * kit.rail_step_m
    if len(session.lens_changes) >= kit.max_changes:
        raise ToolInputError(f"no lens changes left ({kit.max_changes} per run)")
    if session.positions_left <= 0:
        raise ToolInputError("position budget used up; call evaluate and finish")
    if session.lens_would_clip(lens_z_m):
        raise ToolInputError(
            f"the beam overfills the lens at {lens_z_m * 1e3:g} mm (wider than "
            f"{kit.max_beam_at_lens_m * 1e3:g} mm); choose another position or lens"
        )
    listed = any(
        abs(f_m - f) <= 1e-9 and abs(lens_z_m - z) <= 1e-9
        for f, z in session.listed_lens_options
    )
    prediction = _lens_prediction(session, f_m, lens_z_m)
    session.insert_lens(f_m, lens_z_m)
    session.listed_lens_options = []
    result = {
        "f_mm": _num(f_m * 1e3, 3),
        "lens_z_mm": _num(lens_z_m * 1e3, 3),
        "segment": session.segment,
        "positions_left": session.positions_left,
        "lens_changes_left": kit.max_changes - len(session.lens_changes),
        "listed_option": listed,
        "prediction": prediction,
        "note": (
            f"New beam segment {session.segment}: earlier measurements no longer "
            "enter any fit. Scout the new beam again."
        ),
    }
    if not listed:
        result["listed_option_note"] = (
            "This combination was not among the lens_options of the last evaluate."
        )
    return result


def _lens_prediction(
    session: MeasurementSession, f_m: float, lens_z_m: float
) -> dict[str, Any]:
    """Predicted artificial waist from the fit of the current segment only."""
    accepted = _accepted_now(session)
    z_positions = sorted({m.z_m for m in accepted})
    if len(z_positions) < MIN_FIT_POSITIONS:
        return {
            "basis": None,
            "reason": f"fewer than {MIN_FIT_POSITIONS} accepted positions to fit",
        }
    fits = _segment_fits(accepted, z_positions, session.wavelength_m)
    if not all(fit.success for fit in fits.values()):
        return {"basis": None, "reason": "the caustic fit failed"}
    prediction: dict[str, Any] = {
        "basis": (
            f"prediction from the fit of segment {session.segment} (not measured; "
            "the fitted waist may be extrapolated)"
        )
    }
    if any(lens_plan.waist_undetermined(fit) for fit in fits.values()):
        prediction["label"] = lens_plan.UNDETERMINED_LABEL + (
            f" (prediction at M² = {lens_plan.NOMINAL_ASSUMED_M2:g})"
        )
    for axis, fit in fits.items():
        values, stds = lens_plan.predict(
            fit, f_m, lens_z_m, _current_lens(session), session.wavelength_m
        )
        prediction[axis] = _waist_display(values, stds)
    return prediction


@_tool(reason_arg="explanation")
def finish(
    session: MeasurementSession, verdict: str, explanation: str
) -> dict[str, Any]:
    """End the run with a verdict; the report numbers come from session state.

    ``iso_compliant`` is refused if, on a fresh evaluation of the accepted
    measurements, ISO coverage is not met on both axes, general astigmatism
    is flagged, or M² < 1 is flagged. ``waist_not_accessible`` is refused
    unless an axis's waist is ``outside`` and an accepted position lies near
    the stage end on that side.

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
    if verdict == "iso_compliant" and evaluation.get("general_astigmatism"):
        raise ToolInputError(
            "cannot report iso_compliant: general astigmatism is flagged; use "
            "general_astigmatism_out_of_scope"
        )
    if verdict == "iso_compliant" and evaluation.get("m2_below_1"):
        raise ToolInputError(
            "cannot report iso_compliant: M² < 1 is flagged, so the data are not "
            "sound; check the measurements or choose another verdict"
        )
    if verdict == "waist_not_accessible":
        problem = _waist_not_accessible_problem(session, evaluation)
        if problem:
            raise ToolInputError(f"cannot report waist_not_accessible: {problem}")
    return {"report": _write_report(session, verdict, explanation, evaluation)}


def _waist_not_accessible_problem(
    session: MeasurementSession, evaluation: dict[str, Any]
) -> str | None:
    """Why waist_not_accessible is not supported yet, or None if it is.

    Needs an axis with accessibility ``outside`` and an accepted position
    within ``STAGE_END_FRACTION`` of the stage length from the stage end on
    that waist's side: a far-field-only fit can misplace the waist, so the
    end must have been approached first.
    """
    if evaluation.get("status") != "ok":
        return "not enough data to locate the waist"
    scenario = session.scenario
    margin = STAGE_END_FRACTION * (scenario.stage_max_m - scenario.stage_min_m)
    accepted = [m.z_m for m in _accepted_now(session)]
    outside = []
    for axis in ("x", "y"):
        result = evaluation[axis]
        if result["waist_accessibility"] != "outside":
            continue
        below = result["waist_side"] == "below" or (
            result["waist_side"] is None
            and (result["z0_mm"] or 0.0) * 1e-3 < scenario.stage_min_m
        )
        end = scenario.stage_min_m if below else scenario.stage_max_m
        if any(abs(z - end) <= margin for z in accepted):
            return None
        outside.append(f"{axis} (stage {'start' if below else 'end'})")
    if not outside:
        return "no axis has waist_accessibility 'outside'"
    return (
        f"the waist of {', '.join(outside)} looks outside, but no accepted position "
        f"lies within {margin * 1e3:.1f} mm of that stage end yet; measure closer "
        "to it first"
    )


def fallback_report(session: MeasurementSession, reason: str) -> dict[str, Any]:
    """Report written by the controller when a hard limit stops the agent.

    Not an agent tool. The verdict comes from session state:
    ``estimate_not_compliant`` if M² is determined on both axes, otherwise
    ``measurement_failed``. ``stopped_by_limit`` holds the reason.

    Parameters
    ----------
    session : MeasurementSession
        The stopped run (not finished).
    reason : str
        Why the controller stopped the run, e.g. the limit that was hit.

    Returns
    -------
    dict
        The report, also stored as ``session.report``.
    """
    evaluation = _evaluation(session)
    determined = evaluation.get("status") == "ok" and all(
        evaluation[axis]["m2"] is not None for axis in ("x", "y")
    )
    verdict = "estimate_not_compliant" if determined else "measurement_failed"
    explanation = (
        f"Stopped by the controller before the agent finished: {reason}. "
        "Verdict assigned from the measurements taken so far."
    )
    report = _write_report(session, verdict, explanation, evaluation)
    report["stopped_by_limit"] = reason
    return report


def _write_report(
    session: MeasurementSession,
    verdict: str,
    explanation: str,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Report from session state; stored as ``session.report``."""
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
    if session.scenario.lens_kit is not None:
        report.update(_lens_report(session, evaluation))
    session.report = report
    return report


def _lens_report(
    session: MeasurementSession, evaluation: dict[str, Any]
) -> dict[str, Any]:
    """Report fields for a setup with a lens kit (all from session state)."""
    lens = session.lens
    if lens is None:
        return {"method": "direct", "lens": None}
    ok = evaluation.get("status") == "ok"
    keys = ("d0_um", "d0_std_um", "z0_mm", "z0_std_mm", "z_r_mm", "z_r_std_mm")
    return {
        "method": "artificial_waist",
        "lens": {
            "f_mm": _num(lens["f_m"] * 1e3, 3),
            "lens_z_mm": _num(lens["lens_z_m"] * 1e3, 3),
            "segment": lens["segment"],
            "inserted_before_measurement": lens["before_measurement"],
        },
        "artificial_waist": (
            {axis: {k: evaluation[axis][k] for k in keys} for axis in ("x", "y")}
            if ok
            else None
        ),
        "original_beam_back_calculated": (
            evaluation.get("original_beam_back_calculated") if ok else None
        ),
        "m2": (
            {
                axis: {
                    "m2": evaluation[axis]["m2"],
                    "m2_std": evaluation[axis]["m2_std"],
                }
                for axis in ("x", "y")
            }
            if ok
            else None
        ),
        "wording": (
            "ISO 11146-1 procedure applied at an artificial waist (thin-lens model)"
        ),
        "simplifications": list(LENS_SIMPLIFICATIONS),
    }


def _evaluation(session: MeasurementSession) -> dict[str, Any]:
    """Fit and checks on the accepted measurements, in display units."""
    with_segment = session.scenario.lens_kit is not None
    table = [_table_row(m, with_segment) for m in session.measurements.values()]
    accepted = _accepted_now(session)
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
    fits = _segment_fits(accepted, z_positions, wavelength_m)
    for axis, fit in fits.items():
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
    if with_segment:
        _add_lens_results(session, result, fits)
    result["table"] = table
    return result


def _segment_fits(
    accepted: list[Measurement], z_positions: list[float], wavelength_m: float
) -> dict[str, CausticFit]:
    """Caustic fit per axis; repeats at one z are averaged into one position."""
    z = np.array(z_positions)
    fits = {}
    for axis, attribute in (("x", "d_x_mean_m"), ("y", "d_y_mean_m")):
        d = np.array(
            [
                np.mean([getattr(m, attribute) for m in accepted if m.z_m == zi])
                for zi in z_positions
            ]
        )
        fits[axis] = fit_caustic(z, d, wavelength_m=wavelength_m)
    return fits


def _current_lens(session: MeasurementSession) -> tuple[float, float] | None:
    """(f, position) in m of the lens in the beam now, or None."""
    lens = session.lens
    return None if lens is None else (lens["f_m"], lens["lens_z_m"])


def _lens_geometry(session: MeasurementSession) -> lens_plan.Geometry:
    scenario, camera = session.scenario, session.camera
    return lens_plan.Geometry(
        stage_min_m=scenario.stage_min_m,
        stage_max_m=scenario.stage_max_m,
        half_sensor_m=0.5 * min(camera.nx, camera.ny) * camera.pixel_size_m,
        min_d0_m=MIN_PIXELS_ACROSS * camera.pixel_size_m,
        max_beam_at_lens_m=scenario.lens_kit.max_beam_at_lens_m,
    )


def _waist_display(values, stds) -> dict[str, float | None]:
    """d0 in um, z0 and zR in mm (values and std in m)."""
    std = (lambda k: None) if stds is None else (lambda k: stds[k])
    return {
        "d0_um": _num(values[0] * 1e6, 2),
        "d0_std_um": _num(std(0) * 1e6, 2) if std(0) is not None else None,
        "z0_mm": _num(values[1] * 1e3, 3),
        "z0_std_mm": _num(std(1) * 1e3, 3) if std(1) is not None else None,
        "z_r_mm": _num(values[2] * 1e3, 3),
        "z_r_std_mm": _num(std(2) * 1e3, 3) if std(2) is not None else None,
    }


def _add_lens_results(
    session: MeasurementSession, result: dict[str, Any], fits: dict[str, CausticFit]
) -> None:
    """Lens options and the back-calculated beam, from the fits only.

    Lens options only when an axis is ``outside``; for ``uncertain`` the
    evaluation's advice stays to measure closer to that stage end first.
    """
    kit = session.scenario.lens_kit
    fitted = all(fit.success and fit.cov is not None for fit in fits.values())
    waist_out = any(result[axis]["waist_accessibility"] == "outside" for axis in fits)
    if fitted and waist_out:
        options = lens_plan.lens_options(
            fits,
            kit.focal_lengths_m,
            kit.rail_positions_m(),
            _lens_geometry(session),
            _current_lens(session),
            session.wavelength_m,
        )
        session.listed_lens_options = [(o.f_m, o.lens_z_m) for o in options]
        if any(lens_plan.waist_undetermined(fit) for fit in fits.values()):
            result["lens_options_basis"] = lens_plan.UNDETERMINED_LABEL + (
                f" (predicted values at M² = {lens_plan.NOMINAL_ASSUMED_M2:g})"
            )
        result["lens_options"] = [
            {
                "f_mm": _num(o.f_m * 1e3, 3),
                "lens_z_mm": _num(o.lens_z_m * 1e3, 3),
                "feasible_fraction": _num(o.fraction, 3),
                "margin_points": o.margin,
                "predicted": {
                    axis: _waist_display([w.d0_m, w.z0_m, w.z_r_m], None)
                    for axis, w in o.predicted.items()
                },
            }
            for o in options
        ]
        result["lens_options_note"] = (
            "Predicted from the current fit, not measured. feasible_fraction is the "
            f"share of {lens_plan.DRAWS} draws from the fit uncertainty in which the "
            "artificial waist lies inside the stage, has 20 px across, room for 5 "
            "near and 5 far points per axis and no clipping at the lens."
            if options
            else "No kit lens and rail position looks feasible from the current fit."
        )
    lens = session.lens
    if lens is not None:
        back: dict[str, Any] = {
            "basis": (
                f"calculated, not measured: the fit of segment {session.segment} "
                f"through the inverse of the {lens['f_m'] * 1e3:g} mm lens at "
                f"{lens['lens_z_m'] * 1e3:g} mm"
            )
        }
        for axis, fit in fits.items():
            if result[axis]["m2"] is None:
                back[axis] = None
                continue
            values, stds, (dz0, dd0) = lens_plan.back_calculate(
                fit, lens["f_m"], lens["lens_z_m"]
            )
            back[axis] = {
                **_waist_display(values, stds),
                "dz0_mm_per_plus_1pct_f": _num(dz0 * 1e3, 3),
                "dd0_um_per_plus_1pct_f": _num(dd0 * 1e6, 2),
            }
        result["original_beam_back_calculated"] = back


def _axis_result(
    session: MeasurementSession, axis: str, z: np.ndarray, fit: CausticFit
) -> dict[str, Any]:
    """One axis of the evaluation in display units, with coverage.

    d0, z0, zR and M² are reported only if the fit succeeded and at least one
    accepted position lies within +-1 zR of the fitted waist; otherwise they
    are None and ``not_determined`` says why. Without a point near the waist
    the fit extrapolates and can collapse (d0 -> 0, zR -> 0, M² -> 0), so no
    number is shown. ``waist_side`` then says where the waist seems to be:
    ``below`` / ``above`` the measured positions, or ``within`` a gap between
    them (``waist_gap_mm``).

    ``waist_accessibility`` uses z0 +- 2 sigma(z0) against the stage range:
    ``inside``, ``outside`` (the waist cannot be reached) or ``uncertain``
    (measure closer to ``waist_near_stage_end``). None if the fit failed.
    """
    z0, z_r = fit.z0_m, fit.z_r_m
    not_determined = None
    side = None
    gap_mm = None
    if not fit.success:
        not_determined = "caustic fit failed"
    elif not (np.isfinite(z_r) and z_r > 0 and np.any(np.abs(z - z0) <= z_r)):
        z_lo, z_hi = float(z.min()), float(z.max())
        if z0 < z_lo:
            side, where = (
                "below",
                f"below the measured positions (< {z_lo * 1e3:.1f} mm)",
            )
        elif z0 > z_hi:
            side, where = (
                "above",
                f"above the measured positions (> {z_hi * 1e3:.1f} mm)",
            )
        else:
            side = "within"
            lower = float(z[z <= z0].max())
            upper = float(z[z >= z0].min())
            gap_mm = [_num(lower * 1e3, 1), _num(upper * 1e3, 1)]
            where = f"in the gap between {lower * 1e3:.1f} and {upper * 1e3:.1f} mm"
        not_determined = (
            f"no accepted position within ±1 zR of the fitted waist, which lies "
            f"{where}: d0, z0, zR and M² would be extrapolated"
        )
    determined = not_determined is None
    accessibility, near_end = _waist_accessibility(session, fit)

    def shown(value: float, digits: int) -> float | None:
        return _num(value, digits) if determined else None

    coverage_result = None
    if determined and np.isfinite(fit.z_r_m) and fit.z_r_m > 0:
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
    # The far-field slope is measured even when the waist is extrapolated.
    theta_shown = fit.success
    return {
        "fit_success": fit.success,
        "not_determined": not_determined,
        # Where to measure next when the waist is not determined:
        # "below" / "above" the measured positions or "within" a gap.
        "waist_side": side,
        "waist_gap_mm": gap_mm,
        "d0_um": shown(fit.d0_m * 1e6, 2),
        "d0_std_um": shown(fit.d0_std_m * 1e6, 2),
        "z0_mm": shown(fit.z0_m * 1e3, 3),
        "z0_std_mm": shown(fit.z0_std_m * 1e3, 3),
        "theta_mrad": _num(fit.theta_rad * 1e3, 4) if theta_shown else None,
        "theta_std_mrad": _num(fit.theta_std_rad * 1e3, 4) if theta_shown else None,
        "z_r_mm": shown(fit.z_r_m * 1e3, 3),
        "z_r_std_mm": shown(fit.z_r_std_m * 1e3, 3),
        "m2": shown(fit.m2, 4),
        "m2_std": shown(fit.m2_std, 4),
        "residual_rms_percent": _num(fit.residual_rms * 100, 3),
        "coverage": coverage_result,
        # From the fitted waist +- 2 sigma, even when extrapolated: that is how
        # a waist outside the stage shows up. None if the fit failed.
        "waist_accessibility": accessibility,
        "waist_near_stage_end": near_end,
        # M² < 1 is unphysical; flag it only if it holds beyond 2 sigma. The
        # Jacobian sigma misses systematic errors (one-sided data, shared dark),
        # so it is floored at M2_STD_FLOOR_REL of M².
        "m2_below_1": bool(
            determined
            and fit.m2 + 2.0 * max(fit.m2_std, M2_STD_FLOOR_REL * fit.m2) < 1.0
        ),
    }


def _waist_accessibility(
    session: MeasurementSession, fit: CausticFit
) -> tuple[str | None, str | None]:
    """``inside`` / ``outside`` / ``uncertain`` from z0 +- 2 sigma, and for
    ``uncertain`` the stage end (``start`` / ``end``) to measure closer to.
    """
    if not fit.success:
        return None, None
    scenario = session.scenario
    sigma = fit.z0_std_m if np.isfinite(fit.z0_std_m) else np.inf
    low, high = fit.z0_m - 2.0 * sigma, fit.z0_m + 2.0 * sigma
    if low >= scenario.stage_min_m and high <= scenario.stage_max_m:
        return "inside", None
    if high < scenario.stage_min_m or low > scenario.stage_max_m:
        return "outside", None
    centre = 0.5 * (scenario.stage_min_m + scenario.stage_max_m)
    return "uncertain", "start" if fit.z0_m < centre else "end"


def _table_row(m: Measurement, with_segment: bool = False) -> dict[str, Any]:
    row = {
        "id": m.measurement_id,
        "z_mm": _num(m.z_m * 1e3, 3),
        "d_x_um": _num(m.d_x_mean_m * 1e6, 1),
        "d_y_um": _num(m.d_y_mean_m * 1e6, 1),
        "status": m.status,
        "suggested": m.suggested_verdict,
        "flags": [name for name, value in m.flags.items() if value],
    }
    if with_segment:
        row["segment"] = m.segment
    return row


def _accepted_now(session: MeasurementSession) -> list[Measurement]:
    """Accepted measurements of the current beam segment (the only fit data)."""
    return [
        m
        for m in session.measurements.values()
        if m.status == "accept" and m.segment == session.segment
    ]


def _clipping_loss(
    x_c_m: float,
    y_c_m: float,
    sigma_x_m: float,
    sigma_y_m: float,
    width_m: float,
    height_m: float,
) -> float:
    """Fraction of a Gaussian beam's power that falls outside the sensor.

    The beam has centroid (x_c, y_c) relative to the sensor centre and second-
    moment sigmas along x and y; the sensor spans +-width/2 and +-height/2.
    Returns NaN if a sigma is not finite and positive.
    """
    if not (sigma_x_m > 0 and sigma_y_m > 0):
        return float("nan")

    def inside(centre: float, sigma: float, half: float) -> float:
        scale = np.sqrt(2.0) * sigma
        return 0.5 * (erf((half - centre) / scale) + erf((half + centre) / scale))

    p_x = inside(x_c_m, sigma_x_m, width_m / 2.0)
    p_y = inside(y_c_m, sigma_y_m, height_m / 2.0)
    return float(1.0 - p_x * p_y)


def _roi_in_corners(roi: Roi, nx: int, ny: int) -> bool:
    """True if the integration area overlaps one of the four corner patches.

    The patches are the ones :func:`beampilot.widths.subtract_background`
    uses for the fine offset (default corner fraction).
    """
    cx = max(1, round(DEFAULT_CORNER_FRACTION * nx))
    cy = max(1, round(DEFAULT_CORNER_FRACTION * ny))
    # A rectangle that reaches into an edge column band and an edge row band
    # always covers the patch where the two bands cross.
    reaches_x_band = roi.x_start < cx or roi.x_stop > nx - cx
    reaches_y_band = roi.y_start < cy or roi.y_stop > ny - cy
    return reaches_x_band and reaches_y_band


def _corner_beam_light_counts(
    width: WidthResult, nx: int, ny: int, pixel_m: float
) -> float:
    """Mean beam light over the four corner patches, in counts per pixel.

    Models the beam as a Gaussian with the frame's centroid, sigmas (d / 4)
    and summed signal, evaluated at the corner-patch pixel centres, i.e. the
    amount by which beam light raises the fine offset. 0 if the widths are
    not valid.
    """
    sigma_x, sigma_y = width.d_x_m / 4.0, width.d_y_m / 4.0
    if not (sigma_x > 0 and sigma_y > 0 and width.roi_sum_counts > 0):
        return 0.0
    peak = width.roi_sum_counts * pixel_m**2 / (2.0 * np.pi * sigma_x * sigma_y)
    cx = max(1, round(DEFAULT_CORNER_FRACTION * nx))
    cy = max(1, round(DEFAULT_CORNER_FRACTION * ny))
    columns = np.r_[np.arange(cx), np.arange(nx - cx, nx)]
    rows = np.r_[np.arange(cy), np.arange(ny - cy, ny)]
    x = (columns - (nx - 1) / 2.0) * pixel_m - width.x_c_m
    y = (rows - (ny - 1) / 2.0) * pixel_m - width.y_c_m
    # The four patches are all (row, column) pairs of the two edge bands.
    profile_x = np.exp(-0.5 * (x / sigma_x) ** 2)
    profile_y = np.exp(-0.5 * (y / sigma_y) ** 2)
    return float(peak * profile_x.mean() * profile_y.mean())


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
