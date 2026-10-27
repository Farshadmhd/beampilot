"""Measurement session: all state of one simulated M² run.

A :class:`MeasurementSession` is built from a :class:`Scenario`. It owns the
hidden ground truth, the random generator, the dark frames, the current
faults, every measurement and the tool-call log. The agent never touches it
directly; it only sees the dicts returned by :mod:`beampilot.tools`. All
values here are SI (m, rad); the tool layer converts at the boundary.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from beampilot.simulator import (
    Aperture,
    BeamParams,
    CameraConfig,
    Ghost,
    HotPixels,
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
        Position budget: each ``measure_at`` call and each dark retake
        costs one.
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
    ghost : Ghost, optional
        Hidden second spot (e.g. a reflection) in every light frame where it
        is present. Default None.
    max_views : int, optional
        Budget of ``view_frame`` calls (separate from positions). Default 4.
    aperture : Aperture, optional
        Hidden iris that truncates the beam over a z range. Default None.
    hot_pixels : HotPixels, optional
        Defective pixels in every light frame (not in darks). Default None.
    drift_x_per_measurement_m : float, optional
        Slow pointing drift: the beam centre moves along x by this many m per
        measurement taken so far (the first measurement is at 0). Default 0.

    Raises
    ------
    ValueError
        If the stage range is empty, the budget or repeats are below 1, the
        view budget is negative, or the exposure range is empty or not
        positive.
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
    ghost: Ghost | None = None
    max_views: int = 4
    aperture: Aperture | None = None
    hot_pixels: HotPixels | None = None
    drift_x_per_measurement_m: float = 0.0

    def __post_init__(self) -> None:
        if not self.stage_max_m > self.stage_min_m:
            raise ValueError("stage_max_m must be > stage_min_m")
        if self.max_positions < 1 or self.repeats < 1:
            raise ValueError("max_positions and repeats must be >= 1")
        if self.max_views < 0:
            raise ValueError("max_views must be >= 0")
        if not 0 < self.min_exposure_ms < self.max_exposure_ms:
            raise ValueError("need 0 < min_exposure_ms < max_exposure_ms")
        if not np.isfinite(self.drift_x_per_measurement_m):
            raise ValueError("drift_x_per_measurement_m must be finite")


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
    frame : ndarray of uint16, optional
        First raw frame of the repeats, in counts (for ``view_frame``).
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
    frame: NDArray[np.uint16] | None = None


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
        self.views_used = 0
        # PNG data URLs from view_frame, handed to the agent loop by id and
        # never written to logs or run files.
        self.images: dict[str, str] = {}
        self.dark_sets_taken = 0
        self.last_frame: NDArray[np.int32] | None = None
        # Each switch of the faults, for the run file: when and what.
        self.fault_events: list[dict[str, Any]] = []
        self._pending_faults: dict[str, float] = {}
        self._fault_lock = threading.Lock()
        self._schedule_done: set[int] = set()
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
    def views_left(self) -> int:
        """Remaining ``view_frame`` budget."""
        return self.scenario.max_views - self.views_used

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

    def request_faults(
        self,
        extra_background_counts: float | None = None,
        power_factor: float | None = None,
    ) -> None:
        """Queue a fault change from another thread (e.g. a demo button).

        Thread-safe. The change is applied before the next ``measure_at``
        (in :meth:`take_frames`), never in the middle of one.

        Parameters
        ----------
        extra_background_counts : float, optional
            New extra background in counts; None keeps the current value.
        power_factor : float, optional
            New beam power factor (> 0); None keeps the current value.

        Raises
        ------
        ValueError
            If `power_factor` is not positive.
        """
        if power_factor is not None and not power_factor > 0:
            raise ValueError(f"power_factor must be > 0, got {power_factor}")
        with self._fault_lock:
            if extra_background_counts is not None:
                self._pending_faults["extra_background_counts"] = (
                    extra_background_counts
                )
            if power_factor is not None:
                self._pending_faults["power_factor"] = power_factor

    def take_frames(self, z_m: float, exposure_ms: float) -> list[NDArray[np.int32]]:
        """Render the repeated frames at one position with the current faults.

        Applies any scheduled fault and any fault requested with
        :meth:`request_faults` first, and keeps the first frame as
        ``last_frame``. The caller (the tool layer) must have validated `z_m`
        and `exposure_ms`.

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
        self._apply_fault_schedule()
        with self._fault_lock:
            pending, self._pending_faults = self._pending_faults, {}
        if pending:
            self.set_faults(**pending)
            self._record_faults("operator")
        camera = self._camera_with_faults()
        x0_m = self.scenario.drift_x_per_measurement_m * len(self.measurements)
        frames = [
            render_frame(
                self.scenario.beam,
                camera,
                z_m,
                exposure_ms,
                self._rng,
                x0_m=x0_m,
                ghost=self.scenario.ghost,
                aperture=self.scenario.aperture,
                hot_pixels=self.scenario.hot_pixels,
            )
            for _ in range(self.scenario.repeats)
        ]
        self.last_frame = frames[0]
        return frames

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

    def _apply_fault_schedule(self) -> None:
        """Switch to a scheduled fault once its measurement count is reached.

        Each entry applies once, so a later operator request is not undone.
        """
        for index, (after_n, faults) in enumerate(self.scenario.fault_schedule):
            if len(self.measurements) >= after_n and index not in self._schedule_done:
                self._schedule_done.add(index)
                self.faults = faults
                self._record_faults("schedule")

    def _record_faults(self, source: str) -> None:
        self.fault_events.append(
            {
                "before_measurement": len(self.measurements) + 1,
                "source": source,
                "faults": asdict(self.faults),
            }
        )

    def _take_dark_set(self) -> NDArray[np.float64]:
        # Darks taken at the switch-over must already see the new background.
        self._apply_fault_schedule()
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
    ghost = session.scenario.ghost
    if ghost is not None:
        window = (
            "everywhere"
            if not np.isfinite(ghost.z_min_m) and not np.isfinite(ghost.z_max_m)
            else f"{ghost.z_min_m * 1e3:g}-{ghost.z_max_m * 1e3:g} mm"
        )
        truth["ghost"] = (
            f"{ghost.relative_power:.0%} power at ({ghost.offset_x_m * 1e3:+.2f}, "
            f"{ghost.offset_y_m * 1e3:+.2f}) mm from the beam, {window}"
        )
    aperture = session.scenario.aperture
    if aperture is not None:
        truth["aperture"] = (
            f"iris of {aperture.radius_m * 1e3:g} mm radius for "
            f"{aperture.z_min_m * 1e3:g}-{aperture.z_max_m * 1e3:g} mm"
        )
    hot = session.scenario.hot_pixels
    if hot is not None:
        truth["hot_pixels"] = (
            f"{len(hot.pixels)} pixels (row, column) {list(hot.pixels)} with "
            f"+{hot.extra_counts:g} counts in light frames"
        )
    drift = session.scenario.drift_x_per_measurement_m
    if drift:
        truth["drift"] = f"beam centre x moves {drift * 1e6:+g} um per measurement"
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
    """d0 300 um, M² 1.5, zR ~ 44 mm: far-field beams outgrow the 3 x d area.

    The 3 x d area leaves the 2.8 mm sensor from about 3 zR (d ~ 0.94 mm) and
    the stage reaches 4.5 zR (d ~ 1.4 mm). The estimated clipping loss stays
    far below 1 % over the whole stage, so far points beyond 3 zR only get
    the ``roi_clipped`` warning, not a reject. An operator that avoids that
    warning still finds 5 + 5 positions in the window between 2 and ~2.8 zR
    on both sides. Either way the expected verdict is ``iso_compliant``, not
    a failure.
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


def ghost_everywhere_scenario(seed: int = 0) -> Scenario:
    """Clean beam with a reflection (15 % power, 0.54 mm away) at every z.

    The widths include the ghost wherever it falls inside the iterated 3·d
    integration area: always away from the waist; near the waist it depends
    on the frame, because 15 % is close to the 1/e² start threshold. No
    caustic of true ISO widths can be built. Expected verdict:
    measurement_failed, with the ghost named as the cause.
    """
    ghost = Ghost(relative_power=0.15, offset_x_m=0.45e-3, offset_y_m=-0.30e-3)
    return _scenario("ghost_everywhere", seed, ghost=ghost)


def ghost_in_range_scenario(seed: int = 0) -> Scenario:
    """Clean beam with a reflection (15 % power, 0.5 mm away) for 150-200 mm.

    Measurements in that range are wrong; ISO coverage is still possible with
    near points around the 120 mm waist and far points below ~70 mm and above
    200 mm. Expected verdict: iso_compliant, with the ghost positions rejected.
    """
    ghost = Ghost(
        relative_power=0.15,
        offset_x_m=0.5e-3,
        offset_y_m=0.0,
        z_min_m=0.150,
        z_max_m=0.200,
    )
    return _scenario("ghost_in_range", seed, ghost=ghost)


SCENARIOS = {
    "clean": clean_scenario,
    "partial_saturation": partial_saturation_scenario,
    "background_drift": background_drift_scenario,
    "beam_too_large": beam_too_large_scenario,
    "waist_outside_range": waist_outside_range_scenario,
    "different_waists": different_waists_scenario,
    "ghost_everywhere": ghost_everywhere_scenario,
    "ghost_in_range": ghost_in_range_scenario,
}

# Ranges of the mystery beam. zR and M² are drawn and d0 follows from
# zR = pi d0² / (4 M² lambda), which keeps >= 20 px across the waist
# (d0 >= 127 um) and the ISO far field (2-3 zR) on the 250 mm stage.
MYSTERY_WAVELENGTHS_M = (632.8e-9, 1064e-9)
MYSTERY_ZR_RANGE_M = (0.020, 0.045)
MYSTERY_M2_RANGE = (1.0, 2.0)
MYSTERY_WAIST_INSIDE_PROBABILITY = 0.85
MYSTERY_FAULT_PROBABILITY = 0.4


def random_scenario(seed: int = 0) -> Scenario:
    """A "mystery beam": random but physically sensible, maybe one fault.

    Draws per axis M² in [1, 2] and zR in [20, 45] mm and derives the waist
    diameter d0 = sqrt(4 M² lambda zR / pi), at 633 or 1064 nm. The x waist
    lies inside the 0-250 mm stage with probability 0.85 (else 15-60 mm
    before or after it); the y waist is within a few mm of it (simple
    astigmatism). Brightness varies tenfold. With probability 0.4 there is
    one fault: room light (+2 to 5 counts) after 3-8 measurements, or the
    laser 3x brighter from the start.

    Parameters
    ----------
    seed : int, optional
        Seed for both the drawn parameters and the session noise.

    Returns
    -------
    Scenario
        Named ``"mystery"``.
    """
    rng = np.random.default_rng([seed, 2026])  # independent of the noise stream
    wavelength = float(rng.choice(MYSTERY_WAVELENGTHS_M))

    def axis(m2: float, z_r: float) -> float:
        return float(np.sqrt(4.0 * m2 * wavelength * z_r / np.pi))

    m2x = float(rng.uniform(*MYSTERY_M2_RANGE))
    m2y = float(np.clip(m2x * rng.uniform(0.9, 1.15), 1.0, 2.2))
    z_rx = float(rng.uniform(*MYSTERY_ZR_RANGE_M))
    z_ry = float(np.clip(z_rx * rng.uniform(0.85, 1.15), *MYSTERY_ZR_RANGE_M))
    if rng.random() < MYSTERY_WAIST_INSIDE_PROBABILITY:
        z0x = float(rng.uniform(0.040, 0.210))
    else:
        z0x = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.015, 0.060))
        z0x = z0x if z0x < 0 else 0.250 + z0x
    z0y = z0x + float(rng.normal(0.0, 0.008))
    beam = BeamParams(wavelength, axis(m2x, z_rx), axis(m2y, z_ry), z0x, z0y, m2x, m2y)
    camera = replace(_camera(), signal_counts_per_ms=float(10 ** rng.uniform(5.5, 6.5)))

    settings: dict[str, Any] = {"beam": beam, "camera": camera}
    if rng.random() < MYSTERY_FAULT_PROBABILITY:
        if rng.random() < 0.5:
            room_light = Faults(extra_background_counts=float(rng.uniform(2.0, 5.0)))
            settings["fault_schedule"] = ((int(rng.integers(3, 9)), room_light),)
        else:
            settings["faults"] = Faults(power_factor=3.0)
    return _scenario("mystery", seed, **settings)


def _ghost_record(ghost: Ghost | None) -> dict[str, Any] | None:
    if ghost is None:
        return None
    return {k: (v if np.isfinite(v) else None) for k, v in asdict(ghost).items()}


def session_record(session: MeasurementSession) -> dict[str, Any]:
    """What a run file keeps besides the agent's steps (for replay and review).

    Parameters
    ----------
    session : MeasurementSession
        The finished (or stopped) session.

    Returns
    -------
    dict
        Positions used, revealed truth, scenario parameters (beam and camera
        in SI, stage range), fault events and the tool log.
    """
    scenario = session.scenario
    return {
        "positions_used": session.positions_used,
        "truth": reveal_truth(session),
        "scenario_params": {
            "beam": asdict(scenario.beam),
            "camera": asdict(scenario.camera),
            "stage_min_m": scenario.stage_min_m,
            "stage_max_m": scenario.stage_max_m,
            # asdict keeps +-inf for an unlimited z window; JSON needs null.
            "ghost": _ghost_record(scenario.ghost),
            "aperture": None
            if scenario.aperture is None
            else asdict(scenario.aperture),
            "hot_pixels": (
                None if scenario.hot_pixels is None else asdict(scenario.hot_pixels)
            ),
            "drift_x_per_measurement_m": scenario.drift_x_per_measurement_m,
            "faults": asdict(scenario.faults),
            "fault_schedule": [
                {"after_measurements": n, "faults": asdict(f)}
                for n, f in scenario.fault_schedule
            ],
        },
        "fault_events": session.fault_events,
        "tool_log": session.log,
    }
