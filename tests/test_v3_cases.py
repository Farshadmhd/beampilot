"""The v3 case definitions and the oracle with lenses (scripts/v3_cases.py)."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v3():
    spec = importlib.util.spec_from_file_location(
        "v3_cases", REPO / "scripts" / "v3_cases.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["v3_cases"] = module
    spec.loader.exec_module(module)
    return module


def test_oracle_verdicts_of_the_v3_cases(v3) -> None:
    verdicts = {c.case_id: v3.oracle_verdict(c.scenario(0)) for c in v3.all_cases()}
    assert verdicts == {
        **{f"L{i}": "iso_compliant" for i in range(1, 8)},
        "L8": "waist_not_accessible",
    }
    assert all(c.scenario(0).max_positions == 28 for c in v3.all_cases())


def test_l8_is_measurable_without_a_feasible_lens_even_with_margin(v3) -> None:
    scenario = v3.L8.scenario(0)
    assert v3.usable_positions(scenario) >= 10
    corners = v3.perturbed_waists(scenario)
    assert len(corners) == 27
    assert all(v3.feasible_lenses(scenario, corner) == [] for corner in corners)


def test_l8_is_what_the_recorded_search_returns(v3) -> None:
    chosen, usable, candidates = v3.choose_l8()
    assert chosen.z0_m == pytest.approx(v3.L8.z0_m)
    assert chosen.d0_m == pytest.approx(v3.L8.d0_m)
    assert chosen.m2 == pytest.approx(v3.L8.m2)
    assert usable == 25 and len(candidates) == 87


def test_too_few_usable_positions_means_measurement_failed(v3) -> None:
    far = v3.LensCase("X", 0.450, 160e-6, 1.0).scenario(0)
    assert v3.usable_positions(far) < 5
    assert v3.oracle_verdict(far) == "measurement_failed"
