"""Tests for the scripted baselines on the six simulated scenarios."""

import inspect

import pytest

from beampilot import baselines
from beampilot.baselines import naive_uniform, scripted_iso, suggest_verdict
from beampilot.session import (
    SCENARIOS,
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
    halved = [
        i
        for i in saturated
        if i + 1 < len(measures)
        and measures[i + 1]["arguments"]["z_mm"] == measures[i]["arguments"]["z_mm"]
        and measures[i + 1]["arguments"]["exposure_ms"]
        == pytest.approx(measures[i]["arguments"]["exposure_ms"] / 2, rel=1e-3)
    ]
    assert halved, "a saturated position is measured again at half the exposure"
    statuses = [
        session.measurements[measures[i]["result"]["measurement_id"]].status
        for i in saturated
    ]
    assert "accept" not in statuses
    assert report["verdict"] == "iso_compliant"


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
                "x": {"waist_in_stage_range": False},
                "y": {"waist_in_stage_range": True},
                "general_astigmatism": True,
                "iso_coverage_met": True,
            },
            "waist_not_accessible",
        ),
        (
            {
                "status": "ok",
                "x": {"waist_in_stage_range": True},
                "y": {"waist_in_stage_range": True},
                "general_astigmatism": True,
                "iso_coverage_met": True,
            },
            "general_astigmatism_out_of_scope",
        ),
        (
            {
                "status": "ok",
                "x": {"waist_in_stage_range": True},
                "y": {"waist_in_stage_range": True},
                "general_astigmatism": False,
                "iso_coverage_met": True,
            },
            "iso_compliant",
        ),
        (
            {
                "status": "ok",
                "x": {"waist_in_stage_range": True},
                "y": {"waist_in_stage_range": True},
                "general_astigmatism": False,
                "iso_coverage_met": False,
            },
            "estimate_not_compliant",
        ),
    ],
)
def test_suggest_verdict_rules(evaluation: dict, expected: str) -> None:
    assert suggest_verdict(evaluation) == expected


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
