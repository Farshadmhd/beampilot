"""Tests for the scripted baselines on the six simulated scenarios."""

import inspect
from dataclasses import replace

import numpy as np
import pytest

from beampilot import baselines
from beampilot.baselines import naive_uniform, scripted_iso, suggest_verdict
from beampilot.session import (
    SCENARIOS,
    Faults,
    MeasurementSession,
    background_drift_scenario,
    clean_scenario,
    partial_saturation_scenario,
    reveal_truth,
    waist_outside_range_scenario,
)
from beampilot.tools import VERDICTS


def _tool_calls(session: MeasurementSession, tool: str) -> list[dict]:
    return [entry for entry in session.log if entry["tool"] == tool]


def test_scripted_clean_is_iso_compliant_and_accurate() -> None:
    session = MeasurementSession(clean_scenario())
    report = scripted_iso(session)["report"]
    truth = reveal_truth(session)
    assert report["verdict"] == "iso_compliant"
    for axis in ("x", "y"):
        m2 = report["evaluation"][axis]["m2"]
        error = m2 / truth[f"m2_{axis}"] - 1
        print(f"\nscripted clean M²{axis} {m2} (error {error:+.2%})")
        assert abs(error) < 0.03
    assert session.positions_left >= 0


def test_scripted_waist_outside_range() -> None:
    session = MeasurementSession(waist_outside_range_scenario())
    report = scripted_iso(session)["report"]
    assert report["verdict"] == "waist_not_accessible"
    assert not report["evaluation"]["x"]["waist_in_stage_range"]


def test_scripted_lowers_exposure_on_saturation_and_finishes() -> None:
    session = MeasurementSession(partial_saturation_scenario())
    report = scripted_iso(session)["report"]
    measures = _tool_calls(session, "measure_at")
    saturated = [i for i, m in enumerate(measures) if m["result"]["flags"]["saturated"]]
    assert saturated, "the 5x brighter beam should saturate at least once"
    lowered = [
        i
        for i in saturated
        if i + 1 < len(measures)
        and measures[i + 1]["arguments"]["z_mm"] == measures[i]["arguments"]["z_mm"]
        and measures[i + 1]["arguments"]["exposure_ms"]
        <= measures[i]["arguments"]["exposure_ms"] / 2
    ]
    assert lowered, "a saturated position is measured again at <= half the exposure"
    statuses = [
        session.measurements[measures[i]["result"]["measurement_id"]].status
        for i in saturated
    ]
    assert "accept" not in statuses
    assert report["verdict"] == "iso_compliant"


@pytest.mark.parametrize("power_factor", [200.0, 0.02])
def test_scripted_handles_very_bright_and_very_dim_beams(power_factor: float) -> None:
    """200x brighter or 50x dimmer than the clean beam: exposure from the peak."""
    scenario = replace(clean_scenario(), faults=Faults(power_factor=power_factor))
    session = MeasurementSession(scenario)
    report = scripted_iso(session)["report"]
    print(f"\npower x{power_factor}: {report['verdict']}, {session.positions_used} pos")
    assert report["verdict"] != "measurement_failed"
    accepted = report["positions_accepted"]
    assert accepted >= 4


def test_exposure_from_a_saturated_peak_estimate() -> None:
    """Saturated area of a Gaussian gives the true peak: exp(N p² / 2 pi sx sy)."""
    session = MeasurementSession(clean_scenario())
    operator = baselines._ScriptedOperator(session)
    sigma_um = 50.0
    true_peak = 8 * operator.full_scale
    # Pixels above full scale: ellipse with r²/2σ² = ln(8).
    n_saturated = 2 * np.pi * sigma_um**2 * np.log(8) / 5.5**2
    result = {
        "flags": {"saturated": True, "low_signal": False},
        "details": {"saturated_pixel_fraction": n_saturated / 512**2},
        "d_x_um": {"mean": 4 * sigma_um},
        "d_y_um": {"mean": 4 * sigma_um},
    }
    expected = 0.7 * operator.full_scale / true_peak  # factor 0.0875
    assert operator._exposure_after(result, 10.0) == pytest.approx(
        10.0 * expected, rel=1e-3
    )


def test_scripted_retakes_darks_on_room_light() -> None:
    session = MeasurementSession(background_drift_scenario())
    report = scripted_iso(session)["report"]
    retakes = _tool_calls(session, "retake_darks")
    assert len(retakes) == 1
    assert "error" not in retakes[0]["result"]
    measures = _tool_calls(session, "measure_at")
    first_flagged = next(
        i for i, m in enumerate(measures) if m["result"]["flags"]["background_changed"]
    )
    assert not any(
        m["result"]["flags"]["background_changed"]
        for m in measures[first_flagged + 1 :]
    )
    assert report["verdict"] == "iso_compliant"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_naive_runs_to_finish(name: str) -> None:
    session = MeasurementSession(SCENARIOS[name]())
    result = naive_uniform(session)
    assert result["report"]["verdict"] in VERDICTS
    assert session.finished
    assert len(_tool_calls(session, "measure_at")) == 10


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        ({"status": "not_enough_data"}, "measurement_failed"),
        (
            {
                "status": "ok",
                "x": {"fit_success": False, "waist_in_stage_range": None},
                "y": {"fit_success": True, "waist_in_stage_range": True},
                "general_astigmatism": False,
                "iso_coverage_met": False,
            },
            "measurement_failed",
        ),
        (
            {
                "status": "ok",
                "x": {"fit_success": True, "waist_in_stage_range": False},
                "y": {"fit_success": True, "waist_in_stage_range": True},
                "general_astigmatism": True,
                "iso_coverage_met": True,
            },
            "waist_not_accessible",
        ),
        (
            {
                "status": "ok",
                "x": {"fit_success": True, "waist_in_stage_range": True},
                "y": {"fit_success": True, "waist_in_stage_range": True},
                "general_astigmatism": True,
                "iso_coverage_met": True,
            },
            "general_astigmatism_out_of_scope",
        ),
        (
            {
                "status": "ok",
                "x": {"fit_success": True, "waist_in_stage_range": True},
                "y": {"fit_success": True, "waist_in_stage_range": True},
                "general_astigmatism": False,
                "iso_coverage_met": True,
            },
            "iso_compliant",
        ),
        (
            {
                "status": "ok",
                "x": {"fit_success": True, "waist_in_stage_range": True},
                "y": {"fit_success": True, "waist_in_stage_range": True},
                "general_astigmatism": False,
                "iso_coverage_met": False,
            },
            "estimate_not_compliant",
        ),
    ],
)
def test_suggest_verdict_rules(evaluation: dict, expected: str) -> None:
    assert suggest_verdict(evaluation) == expected


def test_collapsed_fit_predicts_nothing() -> None:
    axis = {"fit_success": True, "d0_um": 0.0, "z0_mm": -40.0, "z_r_mm": 0.0}
    assert baselines._predicted_d_um(axis, 100.0) is None
    good = {"fit_success": True, "d0_um": 200.0, "z0_mm": 120.0, "z_r_mm": 25.0}
    assert baselines._predicted_d_um(good, 145.0) == pytest.approx(200.0 * 2**0.5)


def test_scripted_survives_waist_at_stage_start() -> None:
    """Reviewer case: one axis fits to a collapsed caustic (zR rounded to 0)."""
    scenario = replace(
        clean_scenario(),
        beam=replace(clean_scenario().beam, z0x_m=0.0, z0y_m=0.0),
    )
    session = MeasurementSession(scenario)
    result = scripted_iso(session)
    assert result["report"]["verdict"] in VERDICTS


def test_scout_refill_stops_when_tools_fail(monkeypatch) -> None:
    """A tool error must not make the scout refill loop spin forever."""
    real_measure_at = baselines.measure_at
    calls = {"n": 0}

    def flaky_measure_at(session, z_mm, exposure_ms, reason):
        # Two scouts work; every other position (later scouts and all refill
        # candidates) fails, which used to make the refill loop spin.
        calls["n"] += 1
        if z_mm not in (31.25, 93.75):
            return {"error": "stage stuck"}
        return real_measure_at(session, z_mm, exposure_ms, reason)

    monkeypatch.setattr(baselines, "measure_at", flaky_measure_at)
    session = MeasurementSession(clean_scenario())
    result = scripted_iso(session)
    assert calls["n"] < 50
    assert result["report"]["verdict"] in VERDICTS


def test_baselines_use_only_the_public_tools() -> None:
    """Fair comparison: no hidden truth and no session internals."""
    source = inspect.getsource(baselines)
    for forbidden in (
        "reveal_truth",
        ".scenario",
        ".beam",
        ".measurements",
        ".dark_mean",
        ".faults",
        "session.positions_left",
    ):
        assert forbidden not in source, forbidden
