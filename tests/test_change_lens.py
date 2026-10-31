"""Tests for change_lens: segments, guards and the transformed hidden beam."""

from dataclasses import replace

import pytest

from beampilot.lens import through_thin_lens
from beampilot.session import (
    LensKit,
    MeasurementSession,
    _original_waists,
    clean_scenario,
    session_record,
    waist_outside_range_scenario,
)
from beampilot.tools import (
    change_lens,
    evaluate,
    get_setup,
    measure_at,
    set_status,
)


def _lens_session(**kit_changes) -> MeasurementSession:
    """Waist at -50 mm (before the stage) with a lens kit."""
    scenario = waist_outside_range_scenario(seed=11)
    kit = replace(LensKit(), **kit_changes)
    return MeasurementSession(replace(scenario, lens_kit=kit, max_positions=40))


def _measure_and_accept(session, positions_mm, exposure_ms=2.0) -> list[str]:
    ids = []
    for z in positions_mm:
        result = measure_at(session, z, exposure_ms, "test")
        ids.append(result["measurement_id"])
        set_status(session, result["measurement_id"], "accept", "test")
    return ids


def test_get_setup_lists_the_kit_only_when_there_is_one() -> None:
    plain = get_setup(MeasurementSession(clean_scenario()), "test")
    assert not any(key.startswith(("lens", "source")) for key in plain)
    setup = get_setup(_lens_session(), "test")
    assert setup["lens_kit_mm"] == [25.0, 40.0, 300.0, 750.0, -200.0]
    assert setup["lens_rail_mm"] == [-300.0, -20.0, 10.0]
    assert setup["source_z_mm"] == -400.0
    assert setup["lens_changes_left"] == 2


def test_no_lens_kit_means_no_lens_fields_anywhere() -> None:
    session = MeasurementSession(clean_scenario())
    result = measure_at(session, 120.0, 1.5, "test")
    assert "segment" not in result
    assert "error" in change_lens(session, 40.0, -20.0, "test")
    for z in (60.0, 100.0, 140.0, 180.0):
        measure_at(session, z, 2.0, "test")
    for mid in session.measurements:
        set_status(session, mid, "accept", "test")
    assert all("segment" not in row for row in evaluate(session, "test")["table"])


@pytest.mark.parametrize(
    ("f_mm", "lens_z_mm", "message"),
    [
        (50.0, -20.0, "kit"),
        (40.0, -25.0, "rail"),
        (40.0, -310.0, "rail"),
        (40.0, -10.0, "rail"),
        ("forty", -20.0, "number"),
    ],
)
def test_change_lens_enforces_kit_and_rail(f_mm, lens_z_mm, message) -> None:
    session = _lens_session()
    result = change_lens(session, f_mm, lens_z_mm, "test")
    assert message in result["error"]
    assert session.segment == 0 and session.positions_used == 0


def test_change_lens_limits_changes_positions_and_aperture() -> None:
    session = _lens_session(max_changes=1)
    assert "segment" in change_lens(session, 40.0, -20.0, "test")
    assert "no lens changes left" in change_lens(session, 25.0, -20.0, "test")["error"]

    full = MeasurementSession(
        replace(waist_outside_range_scenario(), lens_kit=LensKit(), max_positions=1)
    )
    measure_at(full, 120.0, 1.5, "test")
    assert "budget" in change_lens(full, 40.0, -20.0, "test")["error"]

    narrow = _lens_session(max_beam_at_lens_m=0.1e-3)  # beam always wider
    result = change_lens(narrow, 40.0, -20.0, "test")
    assert "overfills the lens" in result["error"]
    assert narrow.segment == 0


def test_change_lens_starts_a_segment_with_the_true_beam_through_the_lens() -> None:
    session = _lens_session()
    measure_at(session, 120.0, 1.5, "test")
    result = change_lens(session, 40.0, -50.0, "test")
    assert result["segment"] == 1 and result["lens_changes_left"] == 1
    assert session.positions_used == 2
    expected = [
        through_thin_lens(w, 0.040, -0.050)
        for w in _original_waists(session.scenario.beam)
    ]
    assert session.beam.z0x_m == pytest.approx(expected[0].z0_m)
    assert session.beam.d0y_m == pytest.approx(expected[1].d0_m)
    after = measure_at(session, 120.0, 2.0, "test")
    assert after["segment"] == 1
    assert session.lens == {
        "f_m": 0.040,
        "lens_z_m": pytest.approx(-0.050),
        "segment": 1,
        "before_measurement": 2,
    }
    assert session_record(session)["lens_changes"] == session.lens_changes


def test_a_second_change_replaces_the_first_lens() -> None:
    session = _lens_session()
    change_lens(session, 40.0, -50.0, "test")
    change_lens(session, 25.0, -20.0, "test")
    assert session.segment == 2
    x_only_second = through_thin_lens(
        _original_waists(session.scenario.beam)[0], 0.025, -0.020
    )
    assert session.beam.z0x_m == pytest.approx(x_only_second.z0_m)
    assert session.beam.d0x_m == pytest.approx(x_only_second.d0_m)


def test_old_segment_is_closed_and_never_enters_the_fit() -> None:
    session = _lens_session()
    old = _measure_and_accept(session, [0.0, 40.0, 80.0, 120.0, 160.0])
    change_lens(session, 25.0, -20.0, "test")  # artificial waist near +10 mm
    true_after = session.beam.z0x_m * 1e3
    assert 0.0 < true_after < 30.0
    refused = set_status(session, old[0], "reject", "test")
    assert "before the lens change" in refused["error"]
    assert session.measurements[old[0]].status == "accept"

    evaluation = evaluate(session, "test")
    assert evaluation["status"] == "not_enough_data"
    assert evaluation["accepted_positions"] == 0

    _measure_and_accept(session, [0.0, 5.0, 10.0, 15.0, 25.0, 40.0], 1.0)
    evaluation = evaluate(session, "test")
    assert evaluation["accepted_positions"] == 6
    segments = {row["id"]: row["segment"] for row in evaluation["table"]}
    assert all(segments[mid] == 0 for mid in old)
    # The fit follows the artificial waist, not the old beam at -50 mm.
    z0_fit = evaluation["x"]["z0_mm"]
    assert z0_fit is not None and z0_fit == pytest.approx(true_after, abs=2.0)


def test_lens_kit_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        LensKit(focal_lengths_m=(0.0,))
    with pytest.raises(ValueError):
        LensKit(rail_min_m=-0.5, source_z_m=-0.4)
    assert len(LensKit().rail_positions_m()) == 29
