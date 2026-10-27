"""Tests for the measurement session, scenarios and faults."""

import numpy as np
import pytest

from beampilot.session import (
    N_DARK_FRAMES,
    SCENARIOS,
    Faults,
    MeasurementSession,
    clean_scenario,
    reveal_truth,
)
from beampilot.simulator import beam_diameter, rayleigh_length


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
    assert N_DARK_FRAMES == 10


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
