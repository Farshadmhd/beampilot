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


# Phase 2b: lens options, prediction, back-calculation and report fields.

FAR_SCAN = ((0, 8), (5, 9), (10, 10), (40, 15), (80, 25), (120, 35), (160, 45))


def _scanned(session: MeasurementSession) -> MeasurementSession:
    """Far-field scan of a waist before the stage (all points accepted)."""
    for z, exposure in FAR_SCAN:
        result = measure_at(session, z, exposure, "test")
        set_status(session, result["measurement_id"], "accept", "test")
    return session


def _fits(session):
    from beampilot.tools import _accepted_now, _segment_fits

    accepted = _accepted_now(session)
    return _segment_fits(
        accepted, sorted({m.z_m for m in accepted}), session.wavelength_m
    )


def test_lens_options_are_listed_scored_and_deterministic() -> None:
    session = _scanned(_lens_session())
    first = evaluate(session, "test")
    assert first["x"]["waist_accessibility"] == "outside"
    options = first["lens_options"]
    assert 1 <= len(options) <= 5
    fractions = [o["feasible_fraction"] for o in options]
    assert fractions == sorted(fractions, reverse=True)
    assert all(0 < f <= 1 for f in fractions)
    assert "not measured" in first["lens_options_note"]
    again = _scanned(_lens_session())
    assert evaluate(again, "test")["lens_options"] == options


def test_change_lens_prediction_and_listed_flag() -> None:
    from beampilot.lens_plan import predict

    session = _scanned(_lens_session())
    best = evaluate(session, "test")["lens_options"][0]
    fits = _fits(session)
    result = change_lens(session, best["f_mm"], best["lens_z_mm"], "test")
    assert result["listed_option"] is True
    assert "prediction from the fit" in result["prediction"]["basis"]
    values, stds = predict(fits["x"], best["f_mm"] * 1e-3, best["lens_z_mm"] * 1e-3)
    assert result["prediction"]["x"]["z0_mm"] == pytest.approx(
        values[1] * 1e3, abs=1e-3
    )
    assert result["prediction"]["x"]["z0_std_mm"] > 0

    other = _scanned(_lens_session())
    evaluate(other, "test")
    unlisted = change_lens(other, 300.0, -300.0, "test")
    assert unlisted["listed_option"] is False
    assert "not among the lens_options" in unlisted["listed_option_note"]


def test_prediction_without_a_fit_says_why() -> None:
    session = _lens_session()
    result = change_lens(session, 25.0, -20.0, "test")
    assert result["prediction"]["basis"] is None
    assert "accepted positions" in result["prediction"]["reason"]


def test_report_fields_with_a_lens_kit_and_without() -> None:
    from beampilot.tools import finish

    plain = MeasurementSession(clean_scenario())
    report = finish(plain, "measurement_failed", "test")["report"]
    assert not {"method", "lens", "simplifications"} & set(report)

    direct = _lens_session()
    report = finish(direct, "measurement_failed", "test")["report"]
    assert report["method"] == "direct" and report["lens"] is None

    session = _scanned(_lens_session())
    change_lens(session, 25.0, -20.0, "test")
    _measure_and_accept(session, [0.0, 5.0, 10.0, 15.0, 25.0, 40.0], 1.0)
    evaluation = evaluate(session, "test")
    back = evaluation["original_beam_back_calculated"]
    assert "calculated, not measured" in back["basis"]
    for key in ("d0_std_um", "dz0_mm_per_plus_1pct_f", "dd0_um_per_plus_1pct_f"):
        assert back["x"][key] is not None
    report = finish(session, "estimate_not_compliant", "test")["report"]
    assert report["method"] == "artificial_waist"
    assert report["lens"]["f_mm"] == 25.0 and report["lens"]["segment"] == 1
    assert report["artificial_waist"]["x"]["z0_mm"] == evaluation["x"]["z0_mm"]
    assert report["m2"]["y"]["m2"] == evaluation["y"]["m2"]
    assert "artificial waist (thin-lens model)" in report["wording"]
    assert any("1 % focal-length" in s for s in report["simplifications"])


def test_no_truth_leak_everything_follows_the_fit() -> None:
    """A fit shifted from the truth must give predictions that follow the fit."""
    from beampilot.lens_plan import back_calculate, predict

    session = _lens_session()
    truth = session.scenario.beam
    # The camera sees a beam 15 mm further downstream than the hidden truth.
    session.beam = replace(truth, z0x_m=truth.z0x_m + 0.015, z0y_m=truth.z0y_m + 0.015)
    _scanned(session)
    evaluation = evaluate(session, "test")
    fits = _fits(session)
    option = evaluation["lens_options"][0]
    f_m, z_m = option["f_mm"] * 1e-3, option["lens_z_mm"] * 1e-3
    from_fit = predict(fits["x"], f_m, z_m)[0][1] * 1e3
    from_truth = through_thin_lens(_original_waists(truth)[0], f_m, z_m).z0_m * 1e3
    assert option["predicted"]["x"]["z0_mm"] == pytest.approx(from_fit, abs=1e-3)
    assert abs(from_fit - from_truth) > 1.0
    result = change_lens(session, option["f_mm"], option["lens_z_mm"], "test")
    assert result["prediction"]["x"]["z0_mm"] == pytest.approx(from_fit, abs=1e-3)

    # After the lens, shift the camera's beam again; the back-calculation must
    # follow that fit, not the hidden original beam.
    session.beam = replace(session.beam, z0x_m=session.beam.z0x_m + 0.004)
    _measure_and_accept(session, [0.0, 5.0, 10.0, 15.0, 25.0, 40.0], 1.0)
    back = evaluate(session, "test")["original_beam_back_calculated"]
    expected = back_calculate(_fits(session)["x"], f_m, z_m)[0][1] * 1e3
    assert back["x"]["z0_mm"] == pytest.approx(expected, abs=1e-3)
    assert abs(back["x"]["z0_mm"] - truth.z0x_m * 1e3) > 1.0


def test_draws_without_a_real_waist_count_as_infeasible() -> None:
    import numpy as np

    from beampilot.caustic import CausticFit
    from beampilot.lens_plan import fit_draws

    # A very uncertain fit: many draws have c <= 0 or 4ac - b² <= 0.
    cov = ((1e-8, 0.0, 0.0), (0.0, 1e-2, 0.0), (0.0, 0.0, 4e-6))
    fit = CausticFit(
        2e-4, 1e-4, -0.05, 0.1, 2e-3, 2e-3, 0.1, 0.1, 1.2, 1.0, 0.0, 7, True, cov
    )
    d0, z0, z_r, valid = fit_draws(fit, np.random.default_rng(0))
    assert 0 < valid.sum() < valid.size
    assert np.all(np.isnan(d0[~valid])) and np.all(np.isnan(z0[~valid]))
    assert np.all(d0[valid] > 0) and np.all(z_r[valid] > 0)


# Phase 2c: scripted lens rule, end to end (no API calls).


def test_scripted_operator_recovers_the_original_beam_through_a_lens() -> None:
    """Design section 7, item 7: shared tolerances for the end-to-end case."""
    from beampilot.baselines import scripted_iso
    from beampilot.session import reveal_truth

    scenario = replace(
        waist_outside_range_scenario(seed=31), lens_kit=LensKit(), max_positions=24
    )
    session = MeasurementSession(scenario)
    report = scripted_iso(session)["report"]
    truth = reveal_truth(session)
    assert report["verdict"] == "iso_compliant"
    assert report["method"] == "artificial_waist"
    assert len(session.lens_changes) == 1
    back = report["original_beam_back_calculated"]
    for axis in ("x", "y"):
        m2 = report["m2"][axis]["m2"]
        assert m2 == pytest.approx(truth[f"m2_{axis}"], rel=0.03)
        assert back[axis]["d0_um"] == pytest.approx(truth[f"d0_{axis}_um"], rel=0.05)
        z_r = truth[f"z_r_{axis}_mm"]
        assert abs(back[axis]["z0_mm"] - truth[f"z0_{axis}_mm"]) <= 0.1 * z_r


def test_scripted_operator_without_a_kit_is_unchanged() -> None:
    from beampilot.baselines import scripted_iso

    session = MeasurementSession(waist_outside_range_scenario(seed=31))
    report = scripted_iso(session)["report"]
    assert report["verdict"] == "waist_not_accessible"
    assert "method" not in report and session.lens_changes == []


def test_scripted_scouts_come_from_the_prediction_clipped_to_the_stage() -> None:
    from beampilot.baselines import _ScriptedOperator

    operator = _ScriptedOperator(_lens_session())
    near_start = {
        "basis": "prediction",
        "x": {"z0_mm": 10.0, "z_r_mm": 24.0},
        "y": {"z0_mm": 12.0, "z_r_mm": 26.0},
    }
    # z0' 11, zR' 25: 11, -14 -> 0, 36, -51.5 -> 0 (duplicate), 73.5
    assert operator.scouts_from_prediction(near_start) == [11.0, 0.0, 36.0, 73.5]
    middle = {
        "basis": "prediction",
        "x": {"z0_mm": 120.0, "z_r_mm": 10.0},
        "y": {"z0_mm": 120.0, "z_r_mm": 10.0},
    }
    assert operator.scouts_from_prediction(middle) == [120.0, 110.0, 130.0, 95.0, 145.0]
    assert operator.scouts_from_prediction({"basis": None}) is None


def test_lens_options_only_for_outside_not_for_uncertain() -> None:
    from beampilot.tools import _add_lens_results

    session = _scanned(_lens_session())
    fits = _fits(session)
    for accessibility, listed in (("uncertain", False), ("outside", True)):
        result = {
            axis: {"waist_accessibility": accessibility, "m2": None}
            for axis in ("x", "y")
        }
        _add_lens_results(session, result, fits)
        assert ("lens_options" in result) is listed
