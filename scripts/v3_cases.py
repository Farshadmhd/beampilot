"""Evaluation v3 cases (change_lens) and the oracle with lenses.

Pure definitions, no API calls. The oracle judges the **true** beam:

1. ``measurement_failed`` if fewer than ``MIN_USABLE_POSITIONS`` stage positions
   (5 mm grid) give a usable beam size: diameter at least 20 pixels and at
   most half the smaller sensor side.
2. Waist inside the stage: the v1 oracle rule (room for 5 near and 5 far
   points per axis) gives ``iso_compliant`` or ``estimate_not_compliant``.
3. Waist outside: ``iso_compliant`` if any kit x rail lens makes an
   artificial waist that passes the same rule on both axes (inside the stage,
   >= 20 px, no clipping at the lens), otherwise ``waist_not_accessible``.

Run this file to print the case list, the oracle verdicts and how L8 was
chosen::

    .venv\\Scripts\\python.exe scripts\\v3_cases.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

from beampilot.lens import Waist
from beampilot.lens_plan import Geometry, _feasible
from beampilot.session import (
    LensKit,
    Scenario,
    _beam,
    _scenario,
)
from beampilot.simulator import rayleigh_length

POSITIONS = 28  # v3 position budget for every contender
MIN_USABLE_POSITIONS = 5
L8_MIN_USABLE_POSITIONS = 10
GRID_STEP_M = 0.005  # stage positions counted by the oracle
PIXELS_ACROSS = 20


@dataclass(frozen=True)
class LensCase:
    """One v3 beam: waist position, diameter and M² (both axes equal)."""

    case_id: str
    z0_m: float
    d0_m: float
    m2: float

    def scenario(self, seed: int) -> Scenario:
        beam = _beam(
            d0x_m=self.d0_m,
            d0y_m=self.d0_m,
            z0x_m=self.z0_m,
            z0y_m=self.z0_m,
            m2x=self.m2,
            m2y=self.m2,
        )
        return _scenario(
            self.case_id, seed, beam=beam, lens_kit=LensKit(), max_positions=POSITIONS
        )


def _geometry(scenario: Scenario) -> Geometry:
    camera = scenario.camera
    return Geometry(
        stage_min_m=scenario.stage_min_m,
        stage_max_m=scenario.stage_max_m,
        half_sensor_m=0.5 * min(camera.nx, camera.ny) * camera.pixel_size_m,
        min_d0_m=PIXELS_ACROSS * camera.pixel_size_m,
        max_beam_at_lens_m=scenario.lens_kit.max_beam_at_lens_m,
    )


def _true_waists(scenario: Scenario) -> list[Waist]:
    beam = scenario.beam
    return [
        Waist(d0, z0, rayleigh_length(d0, m2, beam.wavelength_m))
        for d0, z0, m2 in (
            (beam.d0x_m, beam.z0x_m, beam.m2x),
            (beam.d0y_m, beam.z0y_m, beam.m2y),
        )
    ]


def usable_positions(scenario: Scenario) -> int:
    """Stage positions (5 mm grid) where both axes have a usable diameter."""
    geometry = _geometry(scenario)
    grid = np.arange(scenario.stage_min_m, scenario.stage_max_m + 1e-9, GRID_STEP_M)
    ok = np.ones(grid.size, dtype=bool)
    for waist in _true_waists(scenario):
        d = np.array([waist.diameter_at(z) for z in grid])
        ok &= (d >= geometry.min_d0_m) & (d <= geometry.half_sensor_m)
    return int(ok.sum())


def _arrays(waist: Waist):
    return (
        np.array([waist.d0_m]),
        np.array([waist.z0_m]),
        np.array([waist.z_r_m]),
        np.array([True]),
    )


def feasible_lenses(scenario: Scenario) -> list[tuple[float, float]]:
    """Kit x rail lenses (f, position in m) that work for the true beam."""
    geometry, kit = _geometry(scenario), scenario.lens_kit
    waists = _true_waists(scenario)
    return [
        (f, z)
        for f in kit.focal_lengths_m
        for z in kit.rail_positions_m()
        if all(_feasible(geometry, _arrays(w), f, z)[0][0] for w in waists)
    ]


def _direct_ok(scenario: Scenario) -> bool:
    """v1 oracle rule for a waist inside the stage (room for 5 + 5 points)."""
    geometry = _geometry(scenario)
    for waist in _true_waists(scenario):
        # A "lens" of infinite focal length at the waist leaves it unchanged.
        if not _feasible(geometry, _arrays(waist), math.inf, waist.z0_m)[0][0]:
            return False
    return True


def oracle_verdict(scenario: Scenario) -> str:
    """Expected verdict for a v3 scenario (see the module docstring)."""
    if usable_positions(scenario) < MIN_USABLE_POSITIONS:
        return "measurement_failed"
    lo, hi = scenario.stage_min_m, scenario.stage_max_m
    if all(lo <= w.z0_m <= hi for w in _true_waists(scenario)):
        return "iso_compliant" if _direct_ok(scenario) else "estimate_not_compliant"
    return "iso_compliant" if feasible_lenses(scenario) else "waist_not_accessible"


def choose_l8() -> tuple[LensCase, int, list[LensCase]]:
    """The L8 beam: measurable but with no feasible kit x rail lens.

    Grid: waist -150..-10 mm and 260..500 mm (10 mm steps), d0 150..350 um
    (10 um steps), M² 1.0..1.8 (0.2 steps). Candidates have at least
    ``L8_MIN_USABLE_POSITIONS`` usable stage positions and no feasible lens.
    Chosen: the most usable positions; ties by the waist closest to a stage
    end, then the smaller d0, then the smaller M².

    Returns
    -------
    tuple
        The chosen case, its usable positions, and all candidates.
    """
    z_values = [z / 1000 for z in range(-150, -9, 10)] + [
        z / 1000 for z in range(260, 501, 10)
    ]
    candidates = []
    for z0 in z_values:
        for d0_um in range(150, 351, 10):
            for m2 in (1.0, 1.2, 1.4, 1.6, 1.8):
                case = LensCase("L8", z0, d0_um * 1e-6, m2)
                scenario = case.scenario(0)
                usable = usable_positions(scenario)
                if usable >= L8_MIN_USABLE_POSITIONS and not feasible_lenses(scenario):
                    candidates.append((case, usable))

    def key(item):
        case, usable = item
        distance = min(abs(case.z0_m - 0.0), abs(case.z0_m - 0.25))
        return (-usable, distance, case.d0_m, case.m2)

    candidates.sort(key=key)
    chosen, usable = candidates[0]
    return chosen, usable, [c for c, _ in candidates]


CASES: list[LensCase] = [
    LensCase("L1", -0.030, 200e-6, 1.2),
    LensCase("L2", -0.060, 200e-6, 1.2),
    LensCase("L3", -0.100, 250e-6, 1.4),
    LensCase("L4", -0.080, 160e-6, 1.0),
    LensCase("L5", 0.280, 200e-6, 1.2),
    LensCase("L6", 0.320, 300e-6, 1.6),
    LensCase("L7", 0.380, 250e-6, 1.4),
]


# Chosen by choose_l8() on 31 Oct 2026, before any v3 run (a test checks that
# the search still returns it): 67 candidates; most usable positions (44).
L8 = LensCase("L8", -0.020, 230e-6, 1.0)


def all_cases() -> list[LensCase]:
    """L1-L7 plus L8 (chosen by :func:`choose_l8`)."""
    return [*CASES, L8]


def main() -> int:
    chosen, usable, candidates = choose_l8()
    print(
        f"L8 chosen from {len(candidates)} candidates: waist {chosen.z0_m * 1e3:g} mm, "
        f"d0 {chosen.d0_m * 1e6:g} um, M2 {chosen.m2:g}, {usable} usable positions"
    )
    for case in all_cases():
        scenario = case.scenario(0)
        lenses = feasible_lenses(scenario)
        print(
            f"{case.case_id}: waist {case.z0_m * 1e3:+g} mm, "
            f"d0 {case.d0_m * 1e6:g} um, M2 {case.m2:g}: "
            f"usable {usable_positions(scenario)}, feasible lenses "
            f"{len(lenses)}, oracle {oracle_verdict(scenario)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
