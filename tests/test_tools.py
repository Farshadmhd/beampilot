"""Tests for the agent tool layer on simulated scenarios."""

import inspect
import json
from dataclasses import replace

import numpy as np
import pytest

from beampilot import tools
from beampilot.session import (
    MeasurementSession,
    background_drift_scenario,
    beam_too_large_scenario,
    clean_scenario,
    different_waists_scenario,
    partial_saturation_scenario,
    reveal_truth,
    waist_outside_range_scenario,
)
from beampilot.simulator import beam_diameter, rayleigh_length
from beampilot.tools import evaluate, finish, get_setup, measure_at, set_status
from beampilot.widths import Roi, WidthResult

REASON = "test"
# Offsets from the waist in units of the x Rayleigh length: 5 near, 5 far.
ISO_OFFSETS = np.array([-3.0, -2.3, -0.8, -0.4, 0.0, 0.4, 0.8, 2.3, 3.0, 3.6])


def _exposure_ms(session: MeasurementSession, z_mm: float, fraction=0.75) -> float:
    """Exposure for a peak at `fraction` of full scale.

    The test plays the operator and uses the hidden truth; the tools never do.
    """
    beam, camera = session.scenario.beam, session.camera
    z_m = z_mm * 1e-3
    d_x = beam_diameter(z_m, beam.d0x_m, beam.z0x_m, beam.m2x, beam.wavelength_m)
    d_y = beam_diameter(z_m, beam.d0y_m, beam.z0y_m, beam.m2y, beam.wavelength_m)
    signal_per_ms = camera.signal_counts_per_ms * session.faults.power_factor
    peak_per_ms = (
        signal_per_ms * camera.pixel_size_m**2 / (2 * np.pi * (d_x / 4) * (d_y / 4))
    )
    return fraction * camera.full_scale_counts / peak_per_ms


def _measure_and_accept(session: MeasurementSession, z_mm_list) -> list[dict]:
    results = []
    for z_mm in z_mm_list:
        result = measure_at(session, z_mm, _exposure_ms(session, z_mm), REASON)
        assert "error" not in result, result
        set_status(
            session, result["measurement_id"], result["suggested_verdict"], REASON
        )
        results.append(result)
    return results


def _iso_positions_mm(session: MeasurementSession) -> np.ndarray:
    beam = session.scenario.beam
    z_r = rayleigh_length(beam.d0x_m, beam.m2x, beam.wavelength_m)
    z_mid = 0.5 * (beam.z0x_m + beam.z0y_m)
    return (z_mid + ISO_OFFSETS * z_r) * 1e3


def _assert_json(result: dict) -> None:
    json.dumps(result, allow_nan=False)


# get_setup


def test_get_setup_reports_setup_in_display_units() -> None:
    setup = get_setup(MeasurementSession(clean_scenario()), "plan the run")
    _assert_json(setup)
    assert setup["wavelength_nm"] == 1064.0
    assert setup["stage_range_mm"] == [0.0, 250.0]
    assert setup["pixel_size_um"] == 5.5
    assert setup["sensor_px"] == [512, 512]
    assert setup["full_scale_counts"] == 4095
    assert setup["positions_left"] == setup["max_positions"] == 20
    assert setup["min_beam_width_um"] == 110.0
    # Nothing about the hidden beam.
    assert not any(key.startswith(("d0", "z0", "m2")) for key in setup)


@pytest.mark.parametrize("reason", ["", "   ", None, 42])
def test_tools_need_a_reason(reason) -> None:
    session = MeasurementSession(clean_scenario())
    assert "reason" in get_setup(session, reason)["error"]
    assert "reason" in measure_at(session, 120.0, 1.0, reason)["error"]
    assert session.measurements == {}


def test_missing_arguments_give_error_not_exception() -> None:
    session = MeasurementSession(clean_scenario())
    result = measure_at(session, 120.0)
    assert "bad arguments" in result["error"]
    assert "bad arguments" in get_setup(session)["error"]


# measure_at


def test_measure_at_waist_happy_path() -> None:
    session = MeasurementSession(clean_scenario())
    result = measure_at(session, 120.0, _exposure_ms(session, 120.0), "waist")
    _assert_json(result)
    assert result["measurement_id"] == "m1"
    assert result["status"] == "pending"
    assert result["positions_left"] == 19
    assert not any(result["flags"].values()), result["flags"]
    assert result["suggested_verdict"] == "accept"
    assert result["reasons"] == [] and result["warnings"] == []
    assert result["d_x_um"]["mean"] == pytest.approx(200.0, rel=0.02)
    assert result["d_y_um"]["mean"] == pytest.approx(210.0, rel=0.02)
    assert 0 < result["d_x_um"]["std"] < 4.0
    assert 2500 < result["details"]["peak_counts"] < 3500


@pytest.mark.parametrize(
    ("z_mm", "exposure_ms", "message"),
    [
        (-1.0, 1.0, "stage range"),
        (251.0, 1.0, "stage range"),
        (120.0, 0.0, "exposure_ms"),
        (120.0, 500.0, "exposure_ms"),
        ("abc", 1.0, "z_mm must be a number"),
        (True, 1.0, "z_mm must be a number"),
        (float("nan"), 1.0, "finite"),
        (120.0, None, "exposure_ms must be a number"),
    ],
)
def test_measure_at_rejects_bad_input(z_mm, exposure_ms, message) -> None:
    session = MeasurementSession(clean_scenario())
    result = measure_at(session, z_mm, exposure_ms, REASON)
    assert message in result["error"]
    assert session.measurements == {}
    assert session.positions_left == 20


def test_measure_at_accepts_numeric_strings() -> None:
    session = MeasurementSession(clean_scenario())
    assert "error" not in measure_at(session, "120", "1.5", REASON)


def test_position_budget_is_enforced() -> None:
    session = MeasurementSession(replace(clean_scenario(), max_positions=1))
    assert "error" not in measure_at(session, 120.0, 1.5, REASON)
    result = measure_at(session, 130.0, 1.5, REASON)
    assert "budget" in result["error"]


def test_low_signal_flag() -> None:
    session = MeasurementSession(clean_scenario())
    result = measure_at(session, 120.0, _exposure_ms(session, 120.0, 0.05), REASON)
    assert result["flags"]["low_signal"]
    assert result["suggested_verdict"] == "reject"


def test_too_few_pixels_flag() -> None:
    scenario = clean_scenario()
    coarse = replace(scenario.camera, pixel_size_m=20e-6)  # 200 um = 10 px
    session = MeasurementSession(replace(scenario, camera=coarse))
    result = measure_at(session, 120.0, _exposure_ms(session, 120.0), REASON)
    assert result["flags"]["too_few_pixels"]
    assert result["details"]["pixels_across"] < 20
    assert result["suggested_verdict"] == "reject"


def test_no_usable_signal_flags_not_converged_and_cannot_be_accepted() -> None:
    session = MeasurementSession(clean_scenario())
    result = measure_at(session, 120.0, 0.01, REASON)  # peak ~ 20 counts
    _assert_json(result)
    assert result["flags"]["not_converged"]
    assert result["flags"]["low_signal"]
    assert result["suggested_verdict"] == "reject"
    refused = set_status(session, "m1", "accept", REASON)
    assert "no valid widths" in refused["error"]


def _width(azimuth_deg: float, ellipticity: float) -> WidthResult:
    return WidthResult(
        d_x_m=1e-3,
        d_y_m=1e-3,
        x_c_m=0.0,
        y_c_m=0.0,
        azimuth_rad=np.radians(azimuth_deg),
        ellipticity=ellipticity,
        iterations=3,
        converged=True,
        roi=Roi(0, 10, 0, 10),
        roi_sum_counts=1.0,
        roi_clipped=False,
    )


@pytest.mark.parametrize(
    ("frames", "expected_deg"),
    [
        ([(45.0, 0.5)], 45.0),
        ([(89.0, 0.5)], 1.0),  # 89 deg is 1 deg from the y axis
        ([(-30.0, 0.5)], 30.0),
        ([(0.0, 0.5), (90.0, 0.5), (5.0, 0.5)], 0.0),  # median
        ([(45.0, 0.95)], 0.0),  # circular frames are ignored
    ],
)
def test_misalignment_folds_azimuth_modulo_90(frames, expected_deg) -> None:
    widths = [_width(azimuth, ellipticity) for azimuth, ellipticity in frames]
    misalignment = tools._median_misalignment_deg(widths)
    assert misalignment == pytest.approx(expected_deg, abs=1e-9)


# set_status


def test_set_status_accept_and_reject() -> None:
    session = MeasurementSession(clean_scenario())
    measure_at(session, 120.0, 1.5, REASON)
    accepted = set_status(session, "m1", "accept", "looks fine")
    assert accepted == {"measurement_id": "m1", "status": "accept", "accepted_count": 1}
    rejected = set_status(session, "m1", "reject", "changed my mind")
    assert rejected["accepted_count"] == 0
    assert session.measurements["m1"].status == "reject"


def test_set_status_errors() -> None:
    session = MeasurementSession(clean_scenario())
    assert "none yet" in set_status(session, "m1", "accept", REASON)["error"]
    measure_at(session, 120.0, 1.5, REASON)
    assert "known ids: m1" in set_status(session, "m9", "accept", REASON)["error"]
    assert "accept' or 'reject" in set_status(session, "m1", "keep", REASON)["error"]


# Faults


def test_room_light_fault_fires_background_changed() -> None:
    """Room light after 4 measurements; the darks are not retaken."""
    session = MeasurementSession(background_drift_scenario())
    z_mm = 120.0
    for _ in range(4):
        before = measure_at(session, z_mm, _exposure_ms(session, z_mm), REASON)
        assert not before["flags"]["background_changed"]
    after = measure_at(session, z_mm, _exposure_ms(session, z_mm), REASON)
    assert after["flags"]["background_changed"]
    assert after["details"]["residual_offset_counts"] == pytest.approx(3.0, abs=0.5)
    assert after["warnings"]
    # The corner offset still removes a uniform background: widths stay right.
    assert after["suggested_verdict"] == "accept"
    assert after["d_x_um"]["mean"] == pytest.approx(200.0, rel=0.02)


def test_controller_can_switch_room_light_on() -> None:
    session = MeasurementSession(clean_scenario())
    session.set_faults(extra_background_counts=3.0)
    result = measure_at(session, 120.0, _exposure_ms(session, 120.0), REASON)
    assert result["flags"]["background_changed"]


def test_partial_saturation_is_flagged_and_cannot_be_accepted() -> None:
    session = MeasurementSession(partial_saturation_scenario())
    # An exposure that suits the far field of the clean beam; 5x power here.
    far = measure_at(session, 40.0, 3.0, REASON)
    waist = measure_at(session, 120.0, 3.0, REASON)
    assert not far["flags"]["saturated"]
    assert waist["flags"]["saturated"]
    assert waist["details"]["saturated_pixel_fraction"] > 0
    assert waist["suggested_verdict"] == "reject"
    refused = set_status(session, waist["measurement_id"], "accept", REASON)
    assert "saturated" in refused["error"]
    assert session.measurements[waist["measurement_id"]].status == "pending"
    assert "error" not in set_status(session, waist["measurement_id"], "reject", REASON)


def test_beam_too_large_far_out_is_roi_clipped() -> None:
    session = MeasurementSession(beam_too_large_scenario())
    beam = session.scenario.beam
    z_r = rayleigh_length(beam.d0x_m, beam.m2x, beam.wavelength_m)
    z_mm = (beam.z0x_m + 3.2 * z_r) * 1e3
    result = measure_at(session, z_mm, _exposure_ms(session, z_mm), REASON)
    assert result["flags"]["roi_clipped"]
    assert result["suggested_verdict"] == "reject"


# evaluate and finish


def test_evaluate_needs_four_accepted_positions() -> None:
    session = MeasurementSession(clean_scenario())
    _measure_and_accept(session, [100.0, 120.0, 140.0])
    measure_at(session, 160.0, 1.5, REASON)  # pending: not used
    result = evaluate(session, "enough yet?")
    _assert_json(result)
    assert result["status"] == "not_enough_data"
    assert "at least 4" in result["message"]
    assert result["accepted_positions"] == 3
    assert len(result["table"]) == 4


def test_repeats_at_one_z_count_as_one_position() -> None:
    session = MeasurementSession(clean_scenario())
    _measure_and_accept(session, [100.0, 100.0, 120.0, 120.0, 140.0, 160.0])
    result = evaluate(session, REASON)
    assert result["status"] == "ok"
    assert result["accepted_positions"] == 4
    assert len(result["table"]) == 6


def test_clean_run_iso_compliant_report() -> None:
    session = MeasurementSession(clean_scenario(seed=3))
    _measure_and_accept(session, _iso_positions_mm(session))
    result = evaluate(session, "check coverage")
    _assert_json(result)
    truth = reveal_truth(session)

    assert result["status"] == "ok"
    assert result["iso_coverage_met"]
    assert not result["general_astigmatism"]
    assert not result["m2_below_1"]
    for axis in ("x", "y"):
        axis_result = result[axis]
        error = axis_result["m2"] / truth[f"m2_{axis}"] - 1
        print(f"\nclean run M²{axis} {axis_result['m2']} (error {error:+.2%})")
        assert axis_result["fit_success"]
        assert axis_result["waist_in_stage_range"]
        assert axis_result["coverage"]["compliant"]
        assert abs(error) < 0.03
        assert axis_result["z0_mm"] == pytest.approx(truth[f"z0_{axis}_mm"], abs=2.0)

    report = finish(session, "iso_compliant", "10 ISO positions, all accepted")
    _assert_json(report)
    assert report["report"]["verdict"] == "iso_compliant"
    assert report["report"]["evaluation"]["x"]["m2"] == result["x"]["m2"]
    assert report["report"]["positions_accepted"] == 10
    assert session.finished
    assert "finished" in measure_at(session, 120.0, 1.5, REASON)["error"]
    assert "finished" in finish(session, "iso_compliant", "again")["error"]


def test_finish_refuses_iso_compliant_without_coverage() -> None:
    session = MeasurementSession(clean_scenario())
    # Near-waist positions only: no far-field coverage.
    _measure_and_accept(session, [105.0, 110.0, 115.0, 120.0, 125.0, 130.0])
    refused = finish(session, "iso_compliant", "trust me")
    assert "cannot report iso_compliant" in refused["error"]
    assert "beyond 2 zR" in refused["error"]
    assert not session.finished
    done = finish(session, "estimate_not_compliant", "no far-field points")
    assert done["report"]["verdict"] == "estimate_not_compliant"


def test_finish_refuses_iso_compliant_without_data() -> None:
    session = MeasurementSession(clean_scenario())
    assert "cannot report" in finish(session, "iso_compliant", "x")["error"]


def test_finish_rejects_unknown_verdict() -> None:
    session = MeasurementSession(clean_scenario())
    result = finish(session, "great", "x")
    assert "iso_compliant" in result["error"]
    assert "explanation" in finish(session, "measurement_failed", "")["error"]


def test_waist_outside_stage_range_is_reported() -> None:
    session = MeasurementSession(waist_outside_range_scenario())
    # Waist at -50 mm: the stage only reaches the far field (2 to 4.5 zR).
    _measure_and_accept(session, [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    result = evaluate(session, "where is the waist?")
    assert result["status"] == "ok"
    for axis in ("x", "y"):
        assert result[axis]["z0_mm"] < 0
        assert not result[axis]["waist_in_stage_range"]
    assert not result["iso_coverage_met"]


def test_different_xy_waists_fitted_per_axis() -> None:
    session = MeasurementSession(different_waists_scenario())
    positions = [45, 60, 80, 90, 100, 110, 120, 130, 140, 150, 160, 175, 195, 210]
    _measure_and_accept(session, positions)
    result = evaluate(session, REASON)
    truth = reveal_truth(session)
    assert result["x"]["z0_mm"] == pytest.approx(truth["z0_x_mm"], abs=2.0)
    assert result["y"]["z0_mm"] == pytest.approx(truth["z0_y_mm"], abs=2.0)
    assert result["x"]["m2"] == pytest.approx(truth["m2_x"], rel=0.03)
    assert result["y"]["m2"] == pytest.approx(truth["m2_y"], rel=0.03)
    assert result["iso_coverage_met"]
    assert not result["general_astigmatism"]


# Log and hidden truth


def test_every_call_is_logged_with_reason_and_result() -> None:
    session = MeasurementSession(clean_scenario())
    get_setup(session, "first look")
    measure_at(session, 999.0, 1.0, "too far")
    finish(session, "measurement_failed", "giving up")
    assert [entry["tool"] for entry in session.log] == [
        "get_setup",
        "measure_at",
        "finish",
    ]
    assert [entry["reason"] for entry in session.log] == [
        "first look",
        "too far",
        "giving up",
    ]
    assert "error" in session.log[1]["result"]
    assert session.log[1]["arguments"]["z_mm"] == 999.0


def test_tools_never_touch_the_hidden_truth() -> None:
    source = inspect.getsource(tools)
    assert "reveal_truth" not in source
    assert ".beam" not in source
