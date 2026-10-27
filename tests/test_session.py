"""Tests for the measurement session, scenarios and faults."""

import json
import threading

import numpy as np
import pytest

from beampilot.session import (
    N_DARK_FRAMES,
    SCENARIOS,
    Faults,
    Measurement,
    MeasurementSession,
    background_drift_scenario,
    clean_scenario,
    random_scenario,
    reveal_truth,
    session_record,
)
from beampilot.simulator import beam_diameter, rayleigh_length, render_frame


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_builds_a_session(name: str) -> None:
    session = MeasurementSession(SCENARIOS[name](seed=1))
    camera = session.camera
    assert session.dark_mean.shape == (camera.ny, camera.nx)
    assert session.positions_left == session.scenario.max_positions
    assert not session.finished
    frames = session.take_frames(0.1, 1.0)
    assert len(frames) == session.scenario.repeats == 5


def test_clean_scenario_is_iso_achievable() -> None:
    """5 positions within zR and 5 beyond 2 zR fit on the stage and sensor."""
    scenario = clean_scenario()
    beam, camera = scenario.beam, scenario.camera
    sensor_m = min(camera.nx, camera.ny) * camera.pixel_size_m
    for d0, z0, m2 in (
        (beam.d0x_m, beam.z0x_m, beam.m2x),
        (beam.d0y_m, beam.z0y_m, beam.m2y),
    ):
        assert d0 / camera.pixel_size_m >= 20  # ISO: >= 20 px across the beam
        z_r = rayleigh_length(d0, m2, beam.wavelength_m)
        # Far points from 2 zR out to the farthest that still fits 3 x d.
        far_m = 3.0 * z_r
        assert z0 - far_m >= scenario.stage_min_m
        assert z0 + far_m <= scenario.stage_max_m
        assert 3 * beam_diameter(z0 + far_m, d0, z0, m2, beam.wavelength_m) < sensor_m


def test_darks_include_start_faults_only() -> None:
    scenario = clean_scenario()
    session = MeasurementSession(scenario)
    assert session.dark_mean.mean() == pytest.approx(
        scenario.camera.offset_counts, abs=0.1
    )
    assert N_DARK_FRAMES == 50
    # Averaging 50 frames reduces the pixel noise by sqrt(50).
    assert session.dark_mean.std() == pytest.approx(
        scenario.camera.read_noise_counts / np.sqrt(N_DARK_FRAMES), rel=0.05
    )


def test_set_faults_changes_frames_but_not_darks() -> None:
    session = MeasurementSession(clean_scenario())
    dark_before = session.dark_mean.copy()
    far_z = 0.0  # beam ~1 mm: corners stay dark
    before = np.mean([f[:20, :20].mean() for f in session.take_frames(far_z, 0.1)])
    session.set_faults(extra_background_counts=5.0)
    after = np.mean([f[:20, :20].mean() for f in session.take_frames(far_z, 0.1)])
    assert after - before == pytest.approx(5.0, abs=0.5)
    assert np.array_equal(session.dark_mean, dark_before)
    session.set_faults(power_factor=2.0)
    assert session.faults == Faults(extra_background_counts=5.0, power_factor=2.0)


def test_random_scenarios_are_valid_for_100_seeds() -> None:
    outside, faulty = 0, 0
    for seed in range(100):
        scenario = random_scenario(seed)
        beam, camera = scenario.beam, scenario.camera
        sensor_m = min(camera.nx, camera.ny) * camera.pixel_size_m
        assert scenario.name == "mystery"
        assert beam.m2x >= 1 and beam.m2y >= 1
        for d0, m2 in ((beam.d0x_m, beam.m2x), (beam.d0y_m, beam.m2y)):
            assert d0 / camera.pixel_size_m >= 20  # ISO: >= 20 px across
            assert 3 * d0 < sensor_m  # the waist fits on the sensor
            z_r = rayleigh_length(d0, m2, beam.wavelength_m)
            assert 0.015 <= z_r <= 0.050
        # Some exposure in range gives an unsaturated, bright enough waist.
        peak_per_ms = (
            camera.signal_counts_per_ms
            * scenario.faults.power_factor
            * camera.pixel_size_m**2
            / (2 * np.pi * (beam.d0x_m / 4) * (beam.d0y_m / 4))
        )
        exposure_ms = 0.7 * camera.full_scale_counts / peak_per_ms
        assert scenario.min_exposure_ms <= exposure_ms <= scenario.max_exposure_ms
        frame = render_frame(
            beam, camera, beam.z0x_m, exposure_ms, np.random.default_rng(0)
        )
        assert frame.max() < camera.full_scale_counts
        outside += not (scenario.stage_min_m <= beam.z0x_m <= scenario.stage_max_m)
        faulty += bool(scenario.fault_schedule) or scenario.faults.power_factor != 1
    assert 5 <= outside <= 30  # about 15 % of waists lie outside the stage
    assert 20 <= faulty <= 60  # about 40 % get one fault
    assert random_scenario(7) == random_scenario(7)


def test_random_scenario_builds_a_session() -> None:
    session = MeasurementSession(random_scenario(3))
    assert session.dark_mean.shape == (512, 512)


def test_requested_fault_applies_before_the_next_measurement() -> None:
    session = MeasurementSession(clean_scenario())
    session.take_frames(0.0, 0.1)
    assert session.fault_events == []
    session.request_faults(extra_background_counts=4.0)
    assert session.faults.extra_background_counts == 0.0  # not yet applied
    worker = threading.Thread(
        target=session.request_faults, kwargs={"power_factor": 3.0}
    )
    worker.start()
    worker.join()
    frames = session.take_frames(0.0, 0.1)
    assert session.faults == Faults(extra_background_counts=4.0, power_factor=3.0)
    assert session.last_frame is frames[0]
    assert session.fault_events[-1]["source"] == "operator"
    assert session.fault_events[-1]["faults"]["power_factor"] == 3.0
    with pytest.raises(ValueError):
        session.request_faults(power_factor=0.0)


def test_schedule_does_not_undo_an_operator_fault() -> None:
    """background_drift switches room light on after 4 measurements; an
    operator power change at the same time must survive."""
    session = MeasurementSession(background_drift_scenario())
    for _ in range(4):
        session.add_measurement(_dummy_measurement(session))
    session.request_faults(power_factor=3.0)
    session.take_frames(0.1, 0.1)
    assert session.faults.extra_background_counts == 3.0
    assert session.faults.power_factor == 3.0
    session.take_frames(0.1, 0.1)
    assert session.faults.power_factor == 3.0
    assert [e["source"] for e in session.fault_events] == ["schedule", "operator"]


def test_ghost_scenarios_put_the_ghost_where_promised() -> None:
    everywhere = MeasurementSession(SCENARIOS["ghost_everywhere"]())
    in_range = MeasurementSession(SCENARIOS["ghost_in_range"]())
    assert "everywhere" in reveal_truth(everywhere)["ghost"]
    assert "150-200 mm" in reveal_truth(in_range)["ghost"]
    record = session_record(everywhere)
    json.dumps(record, allow_nan=False)  # an unlimited window is stored as null
    assert record["scenario_params"]["ghost"]["z_min_m"] is None
    # The ghost changes the frame only inside its z window.
    ghost = in_range.scenario.ghost
    assert ghost.present_at(0.175) and not ghost.present_at(0.120)
    assert reveal_truth(MeasurementSession(clean_scenario())).get("ghost") is None


def test_session_record_holds_what_replay_needs() -> None:
    session = MeasurementSession(clean_scenario())
    record = session_record(session)
    assert record["truth"]["m2_x"] == 1.2
    assert record["scenario_params"]["beam"]["d0x_m"] == pytest.approx(0.2e-3)
    assert record["scenario_params"]["stage_max_m"] == 0.25
    json.dumps(record, allow_nan=False)


def _dummy_measurement(session: MeasurementSession) -> Measurement:
    return Measurement(
        measurement_id=session.next_measurement_id(),
        z_m=0.1,
        exposure_ms=1.0,
        widths=[],
        d_x_mean_m=1e-3,
        d_x_std_m=1e-5,
        d_y_mean_m=1e-3,
        d_y_std_m=1e-5,
        flags={},
        details={},
        suggested_verdict="accept",
    )


def test_power_factor_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Faults(power_factor=0.0)


def test_reveal_truth_in_display_units() -> None:
    truth = reveal_truth(MeasurementSession(clean_scenario()))
    assert truth["scenario"] == "clean"
    assert truth["wavelength_nm"] == pytest.approx(1064.0)
    assert truth["d0_x_um"] == pytest.approx(200.0)
    assert truth["z0_x_mm"] == pytest.approx(120.0)
    assert truth["m2_x"] == 1.2
    assert 20 < truth["z_r_x_mm"] < 30
