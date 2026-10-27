"""Measurement session: all state of one simulated M² run.

A :class:`MeasurementSession` is built from a :class:`Scenario`. It owns the
hidden ground truth, the random generator, the dark frames, the current
faults, every measurement and the tool-call log. The agent never touches it
directly; it only sees the dicts returned by :mod:`beampilot.tools`. All
values here are SI (m, rad); the tool layer converts at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from beampilot.simulator import (
    BeamParams,
    CameraConfig,
    dark_frame,
    rayleigh_length,
    render_frame,
)
from beampilot.widths import WidthResult, mean_dark

# The mean dark is shared by every position of a run, so its pixel noise is a
# correlated error that the caustic fit's uncertainty cannot see. 50 frames
# keep it small (see docs/decisions.md); ISO/TR 11146-3 asks for at least 10.
N_DARK_FRAMES = 50


@dataclass(frozen=True)
class Faults:
    """Faults the controller can switch on during a run.

    Parameters
    ----------
    extra_background_counts : float, optional
        Extra uniform background in counts per pixel, e.g. room light switched
        on after the dark frames were taken. Added like a camera offset (no
        shot noise). Default 0.
    power_factor : float, optional
        Factor on the beam signal, e.g. laser power turned up. Default 1.
    """

    extra_background_counts: float = 0.0
    power_factor: float = 1.0

    def __post_init__(self) -> None:
        if not self.power_factor > 0:
            raise ValueError(f"power_factor must be > 0, got {self.power_factor}")


@dataclass(frozen=True)
class Scenario:
    """Everything that defines one simulated run.

    Parameters
    ----------
    name : str
        Short identifier, e.g. ``"clean"``.
    beam : BeamParams
        Hidden ground truth. Only :func:`reveal_truth` exposes it.
    camera : CameraConfig
        Camera sensor.
    stage_min_m, stage_max_m : float
        Stage travel along z in m.
    max_positions : int
        Budget of ``measure_at`` calls.
    repeats : int, optional
        Frames per position. Default 5 (ISO 11146-1 asks for at least 5).
    seed : int, optional
        Seed of the session's random generator. Default 0.
    faults : Faults, optional
        Faults active from the start (before the dark frames). Default none.
    fault_schedule : tuple of (int, Faults), optional
        ``(n, faults)``: switch to `faults` once `n` measurements have been
        taken, e.g. ``((4, Faults(extra_background_counts=3.0)),)``.
    min_exposure_ms, max_exposure_ms : float, optional
        Allowed exposure range in ms. Defaults 0.01 and 200.

    Raises
    ------
    ValueError
        If the stage range is empty, the budget or repeats are below 1, or
        the exposure range is empty or not positive.
    """

    name: str
    beam: BeamParams
    camera: CameraConfig
    stage_min_m: float
    stage_max_m: float
    max_positions: int
    repeats: int = 5
    seed: int = 0
    faults: Faults = field(default_factory=Faults)
    fault_schedule: tuple[tuple[int, Faults], ...] = ()
    min_exposure_ms: float = 0.01
    max_exposure_ms: float = 200.0

    def __post_init__(self) -> None:
        if not self.stage_max_m > self.stage_min_m:
            raise ValueError("stage_max_m must be > stage_min_m")
        if self.max_positions < 1 or self.repeats < 1:
            raise ValueError("max_positions and repeats must be >= 1")
        if not 0 < self.min_exposure_ms < self.max_exposure_ms:
            raise ValueError("need 0 < min_exposure_ms < max_exposure_ms")


@dataclass
class Measurement:
    """One ``measure_at`` result, kept in SI inside the session.

    Parameters
    ----------
    measurement_id : str
        Identifier shown to the agent, e.g. ``"m3"``.
    z_m : float
        Stage position in m.
    exposure_ms : float
        Exposure time in ms.
    widths : list of WidthResult
        Per-frame ISO widths (widths in m).
    d_x_mean_m, d_x_std_m, d_y_mean_m, d_y_std_m : float
        Mean and sample std of the widths over the repeats, in m.
    flags : dict of str to bool
        Quality flags (see :func:`beampilot.tools.measure_at`).
    details : dict
        Supporting numbers for the flags (peak counts, offset ratio, ...).
    suggested_verdict : str
        ``"accept"`` or ``"reject"``.
    status : str
        ``"pending"``, ``"accept"`` or ``"reject"``.
    """

    measurement_id: str
    z_m: float
    exposure_ms: float
    widths: list[WidthResult]
    d_x_mean_m: float
    d_x_std_m: float
    d_y_mean_m: float
    d_y_std_m: float
    flags: dict[str, bool]
    details: dict[str, Any]
    suggested_verdict: str
    status: str = "pending"


class MeasurementSession:
    """State of one run: hidden beam, darks, faults, measurements and log.

    ``N_DARK_FRAMES`` dark frames are taken at construction, with the faults
    that are active at the start, like a lab run that blocks the beam before
    it begins. Until :meth:`retake_darks` is called, a later background
    change shows up as a residual offset.

    Parameters
    ----------
    scenario : Scenario
        The run to simulate.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.faults = scenario.faults
        self.measurements: dict[str, Measurement] = {}
        self.log: list[dict[str, Any]] = []
        self.report: dict[str, Any] | None = None
        self.positions_used = 0
        self.dark_sets_taken = 0
        self._rng = np.random.default_rng(scenario.seed)
        self.dark_mean = self._take_dark_set()

    @property
    def camera(self) -> CameraConfig:
        """Nominal camera (without faults)."""
        return self.scenario.camera

    @property
    def wavelength_m(self) -> float:
        """Laser wavelength in m: known to the experimenter, not hidden."""
        return self.scenario.beam.wavelength_m

    @property
    def positions_left(self) -> int:
        """Remaining budget; each measurement and each dark retake costs one."""
        return self.scenario.max_positions - self.positions_used

    @property
    def finished(self) -> bool:
        """True once ``finish`` has produced a report."""
        return self.report is not None

    def set_faults(
        self,
        extra_background_counts: float | None = None,
        power_factor: float | None = None,
    ) -> None:
        """Switch faults on or off during a run (controller / demo buttons).

        Parameters
        ----------
        extra_background_counts : float, optional
            New extra background in counts; None keeps the current value.
        power_factor : float, optional
            New beam power factor; None keeps the current value.
        """
        changes: dict[str, float] = {}
        if extra_background_counts is not None:
            changes["extra_background_counts"] = extra_background_counts
        if power_factor is not None:
            changes["power_factor"] = power_factor
        self.faults = replace(self.faults, **changes)

    def take_frames(self, z_m: float, exposure_ms: float) -> list[NDArray[np.int32]]:
        """Render the repeated frames at one position with the current faults.

        Applies any scheduled fault first. The caller (the tool layer) must
        have validated `z_m` and `exposure_ms`.

        Parameters
        ----------
        z_m : float
            Stage position in m.
        exposure_ms : float
            Exposure time in ms.

        Returns
        -------
        list of ndarray of int32
            ``scenario.repeats`` raw frames in counts.
        """
        for after_n, faults in self.scenario.fault_schedule:
            if len(self.measurements) == after_n:
                self.faults = faults
        camera = self._camera_with_faults()
        return [
            render_frame(self.scenario.beam, camera, z_m, exposure_ms, self._rng)
            for _ in range(self.scenario.repeats)
        ]

    def retake_darks(self) -> None:
        """Take a fresh dark set with the current faults; costs one position.

        Any extra background active now (e.g. room light) becomes part of the
        dark, so the corner offset of later frames is back near zero.
        """
        self.dark_mean = self._take_dark_set()
        self.positions_used += 1

    def add_measurement(self, measurement: Measurement) -> None:
        """Store a measurement under its id; costs one position."""
        self.measurements[measurement.measurement_id] = measurement
        self.positions_used += 1

    def next_measurement_id(self) -> str:
        """Identifier for the next measurement: ``"m1"``, ``"m2"``, ..."""
        return f"m{len(self.measurements) + 1}"

    def _take_dark_set(self) -> NDArray[np.float64]:
        camera = self._camera_with_faults()
        self.dark_sets_taken += 1
        return mean_dark([dark_frame(camera, self._rng) for _ in range(N_DARK_FRAMES)])

    def _camera_with_faults(self) -> CameraConfig:
        camera = self.scenario.camera
        return replace(
            camera,
            offset_counts=camera.offset_counts + self.faults.extra_background_counts,
            signal_counts_per_ms=camera.signal_counts_per_ms * self.faults.power_factor,
        )


def reveal_truth(session: MeasurementSession) -> dict[str, float | str]:
    """Hidden ground truth, for the report after a run and for tests.

    The agent tools never call this.

    Parameters
    ----------
    session : MeasurementSession
        The session.

    Returns
    -------
    dict
        Scenario name, wavelength in nm, and per axis the waist diameter in
        um, waist position in mm, Rayleigh length in mm and M².
    """
    beam = session.scenario.beam
    truth: dict[str, float | str] = {
        "scenario": session.scenario.name,
        "wavelength_nm": beam.wavelength_m * 1e9,
    }
    for axis, d0, z0, m2 in (
        ("x", beam.d0x_m, beam.z0x_m, beam.m2x),
        ("y", beam.d0y_m, beam.z0y_m, beam.m2y),
    ):
        truth[f"d0_{axis}_um"] = d0 * 1e6
        truth[f"z0_{axis}_mm"] = z0 * 1e3
        truth[f"z_r_{axis}_mm"] = rayleigh_length(d0, m2, beam.wavelength_m) * 1e3
        truth[f"m2_{axis}"] = m2
    return truth


# Scenario factories (design section 7). Shared camera: 512 x 512 px of
# 5.5 um (2.8 mm sensor), 12 bit. The clean beam has d0 = 200 um (36 px) and
# zR ~ 25 mm, so ISO positions out to ~3.6 zR fit on the sensor and inside
# the 0-250 mm stage.

WAVELENGTH_M = 1064e-9


def _camera() -> CameraConfig:
    return CameraConfig(
        pixel_size_m=5.5e-6,
        nx=512,
        ny=512,
        bit_depth=12,
        offset_counts=100.0,
        read_noise_counts=5.0,
        signal_counts_per_ms=1e6,
    )


def _beam(
    d0x_m: float = 0.20e-3,
    d0y_m: float = 0.21e-3,
    z0x_m: float = 0.120,
    z0y_m: float = 0.120,
    m2x: float = 1.2,
    m2y: float = 1.25,
) -> BeamParams:
    return BeamParams(WAVELENGTH_M, d0x_m, d0y_m, z0x_m, z0y_m, m2x, m2y)


def _scenario(name: str, seed: int, **kwargs: Any) -> Scenario:
    settings: dict[str, Any] = {
        "beam": _beam(),
        "camera": _camera(),
        "stage_min_m": 0.0,
        "stage_max_m": 0.250,
        "max_positions": 20,
    }
    settings.update(kwargs)
    return Scenario(name=name, seed=seed, **settings)


def clean_scenario(seed: int = 0) -> Scenario:
    """Clean, nearly round beam with M² 1.2 / 1.25, waist mid-stage."""
    return _scenario("clean", seed)


def partial_saturation_scenario(seed: int = 0) -> Scenario:
    """Laser 5x brighter: exposures that suit the far field saturate the waist."""
    return _scenario("partial_saturation", seed, faults=Faults(power_factor=5.0))


def background_drift_scenario(seed: int = 0) -> Scenario:
    """Room light switches on (+3 counts) after 4 measurements; no new darks."""
    room_light = Faults(extra_background_counts=3.0)
    return _scenario("background_drift", seed, fault_schedule=((4, room_light),))


def beam_too_large_scenario(seed: int = 0) -> Scenario:
    """d0 300 um, M² 1.5, zR ~ 44 mm: far-field points must stay in a narrow window.

    The 3 x d area leaves the 2.8 mm sensor from about 3 zR (d ~ 0.94 mm),
    and the stage reaches 4.5 zR, so naive far points are clipped. Positions
    between 2 and ~2.8 zR on both sides of the waist still give 5 near + 5
    far without clipping, so the expected verdict is ``iso_compliant``
    through that narrow window, not a failure.
    """
    beam = _beam(d0x_m=0.30e-3, d0y_m=0.30e-3, z0x_m=0.200, z0y_m=0.200)
    beam = replace(beam, m2x=1.5, m2y=1.5)
    return _scenario("beam_too_large", seed, beam=beam, stage_max_m=0.400)


def waist_outside_range_scenario(seed: int = 0) -> Scenario:
    """Waist at -50 mm, before the 0-250 mm stage starts."""
    beam = _beam(z0x_m=-0.050, z0y_m=-0.050)
    return _scenario("waist_outside_range", seed, beam=beam)


def different_waists_scenario(seed: int = 0) -> Scenario:
    """Simple astigmatism: x waist at 100 mm, y waist at 140 mm, M² 1.2 / 1.4."""
    beam = _beam(d0y_m=0.20e-3, z0x_m=0.100, z0y_m=0.140, m2y=1.4)
    return _scenario("different_waists", seed, beam=beam)


SCENARIOS = {
    "clean": clean_scenario,
    "partial_saturation": partial_saturation_scenario,
    "background_drift": background_drift_scenario,
    "beam_too_large": beam_too_large_scenario,
    "waist_outside_range": waist_outside_range_scenario,
    "different_waists": different_waists_scenario,
}
