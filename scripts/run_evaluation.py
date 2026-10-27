"""Run a pre-registered evaluation (docs/evaluation-plan*.md, with deviations).

Contenders: the Muse Spark agent (group C also without ``view_frame``), the
scripted ISO operator and the naive scan. Results go to
``runs/evaluation/results.jsonl`` (one line per run, appended as runs finish)
and agent run files to ``runs/evaluation/``; ``--summary`` writes the tables.

Usage (PowerShell)::

    .venv\\Scripts\\python.exe scripts\\run_evaluation.py --oracle
    .venv\\Scripts\\python.exe scripts\\run_evaluation.py --run
    .venv\\Scripts\\python.exe scripts\\run_evaluation.py --summary

``--run --resume`` skips runs already in ``results.jsonl``. ``--plan v2`` runs
docs/evaluation-plan-v2.md (new seeds, output in ``runs/evaluation-v2/``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from beampilot.agent import AgentConfig, make_client, run_agent, save_run
from beampilot.baselines import naive_uniform, scripted_iso
from beampilot.session import (
    SCENARIOS,
    MeasurementSession,
    Scenario,
    clean_scenario,
    random_scenario,
    reveal_truth,
    session_record,
)
from beampilot.simulator import Aperture, Ghost, HotPixels, rayleigh_length

OUT_DIR = Path("runs") / "evaluation"
RESULTS = OUT_DIR / "results.jsonl"
BUDGET_USD = 4.0
WORKERS = 3
M2_TOLERANCE = 0.03  # C3 and C5: |M² error| <= 3 % on both axes

KEYWORDS = {
    "ghost": ("ghost", "second spot", "reflection", "double"),
    "hot_pixel": ("hot pixel", "bright pixel", "stuck", "defect"),
    "drift": ("drift", "moving", "centroid"),
    "aperture": (
        "aperture",
        "iris",
        "truncat",
        "vignett",
        "cut off",
        "obstruct",
        "smaller than expected",
    ),
}


@dataclass(frozen=True)
class Case:
    """One evaluation case from the plan."""

    case_id: str
    group: str
    build: Callable[[], Scenario]
    expected: str | None  # None: from the oracle rule (mystery beams)
    note: str | None = None
    fault: str | None = None  # keyword set for "fault noticed"
    m2_check: bool = False  # C3, C5: verdict and M² within 3 %
    x_only: bool = False  # B3
    seen: bool = False  # C1 (deviation 1)


def _held_out(name: str, seed: int, **kwargs: Any) -> Callable[[], Scenario]:
    return lambda: replace(clean_scenario(seed), name=name, **kwargs)


def _fixed(name: str, seed: int) -> Callable[[], Scenario]:
    return lambda: SCENARIOS[name](seed)


def _mystery(seed: int) -> Callable[[], Scenario]:
    return lambda: random_scenario(seed)


def build_cases(seeds: dict[str, int]) -> list[Case]:
    """All cases of a plan; only the seeds differ between v1 and v2."""
    a, mystery, b, c = seeds["A"], seeds["mystery"], seeds["B"], seeds["C"]
    return [
        Case("A1", "A", _fixed("clean", a), "iso_compliant"),
        Case("A2", "A", _fixed("partial_saturation", a), "iso_compliant"),
        Case("A3", "A", _fixed("background_drift", a), "iso_compliant"),
        Case("A4", "A", _fixed("beam_too_large", a), "iso_compliant"),
        Case("A5", "A", _fixed("waist_outside_range", a), "waist_not_accessible"),
        Case("A6", "A", _fixed("different_waists", a), "iso_compliant"),
        *[Case(f"A{7 + i}", "A", _mystery(mystery + i), None) for i in range(10)],
        Case(
            "B1",
            "B",
            _fixed("clean", b),
            "iso_compliant",
            note=(
                "The stage cannot be used between 140 and 160 mm: "
                "a mount is in the way."
            ),
        ),
        Case(
            "B2",
            "B",
            _fixed("clean", b + 1),
            "iso_compliant",
            note="Use at most 12 stage positions in total; the stage time is booked.",
        ),
        Case(
            "B3",
            "B",
            _fixed("different_waists", b + 2),
            "iso_compliant",
            note="Only the x axis matters for this job; you may ignore y.",
            x_only=True,
        ),
        Case(
            "B4",
            "B",
            _fixed("clean", b + 3),
            "iso_compliant",
            note="Do not use exposures above 20 ms: the camera overheats.",
        ),
        Case(
            "B5",
            "B",
            _fixed("beam_too_large", b + 4),
            "iso_compliant",
            note="Do not move the stage above 230 mm: the rail is damaged there.",
        ),
        Case(
            "C1",
            "C",
            _held_out(
                "C1_ghost_everywhere",
                c,
                ghost=Ghost(
                    relative_power=0.08, offset_x_m=-0.35e-3, offset_y_m=0.25e-3
                ),
            ),
            "measurement_failed",
            fault="ghost",
            seen=True,
        ),
        Case(
            "C2",
            "C",
            _held_out(
                "C2_ghost_in_range",
                c + 1,
                ghost=Ghost(
                    relative_power=0.12,
                    offset_x_m=0.4e-3,
                    offset_y_m=0.0,
                    z_min_m=0.170,
                    z_max_m=0.250,
                ),
            ),
            "iso_compliant",
            fault="ghost",
        ),
        Case(
            "C3",
            "C",
            _held_out(
                "C3_hot_pixels",
                c + 2,
                hot_pixels=HotPixels(
                    pixels=((40, 470), (40, 471), (41, 470), (41, 471)),
                    extra_counts=2500.0,
                ),
            ),
            "iso_compliant",
            fault="hot_pixel",
            m2_check=True,
        ),
        Case(
            "C4",
            "C",
            _held_out("C4_drift", c + 3, drift_x_per_measurement_m=25e-6),
            "iso_compliant",
            fault="drift",
        ),
        Case(
            "C5",
            "C",
            _held_out(
                "C5_iris",
                c + 4,
                aperture=Aperture(radius_m=0.35e-3, z_min_m=0.030, z_max_m=0.070),
            ),
            "iso_compliant",
            fault="aperture",
            m2_check=True,
        ),
    ]


# Seeds per plan: v1 docs/evaluation-plan.md (run at 03fb2c1 with the then
# default of 40 model calls), v2 docs/evaluation-plan-v2.md.
PLANS: dict[str, dict[str, Any]] = {
    "v1": {
        "seeds": {"A": 100, "mystery": 1000, "B": 200, "C": 300},
        "out_dir": Path("runs") / "evaluation",
        "c_label": "C held out (C2-C5)",
    },
    "v2": {
        "seeds": {"A": 500, "mystery": 3000, "B": 600, "C": 700},
        "out_dir": Path("runs") / "evaluation-v2",
        "c_label": "C2-C5 (seen in v1)",
    },
}
CASES: list[Case] = build_cases(PLANS["v1"]["seeds"])
C_LABEL = PLANS["v1"]["c_label"]


def use_plan(name: str) -> None:
    """Switch cases, output folder and labels to plan `name` (v1 or v2)."""
    global CASES, OUT_DIR, RESULTS, C_LABEL
    plan = PLANS[name]
    CASES = build_cases(plan["seeds"])
    OUT_DIR = plan["out_dir"]
    RESULTS = OUT_DIR / "results.jsonl"
    C_LABEL = plan["c_label"]


def contenders(case: Case) -> list[str]:
    """Contenders per case: group C runs the agent twice (deviation 4)."""
    agents = ["agent", "agent-noview"] if case.group == "C" else ["agent"]
    return [*agents, "scripted", "naive"]


# Oracle rule (plan, group A) -------------------------------------------------


def _room(length_m: float, z_r: float) -> int:
    """Room for points in a window: floor(L / (0.15 zR)) + 1, 0 if empty."""
    if length_m < 0:
        return 0
    return math.floor(length_m / (0.15 * z_r) + 1e-9) + 1


def oracle_verdict(scenario: Scenario) -> str:
    """Expected verdict from the hidden truth, by the plan's geometric rule."""
    beam, camera = scenario.beam, scenario.camera
    lo, hi = scenario.stage_min_m, scenario.stage_max_m
    half_sensor = 0.5 * min(camera.nx, camera.ny) * camera.pixel_size_m
    axes = [
        (beam.d0x_m, beam.z0x_m, beam.m2x),
        (beam.d0y_m, beam.z0y_m, beam.m2y),
    ]
    if any(not lo <= z0 <= hi for _, z0, _ in axes):
        return "waist_not_accessible"
    for d0, z0, m2 in axes:
        z_r = rayleigh_length(d0, m2, beam.wavelength_m)
        near = min(hi, z0 + 0.9 * z_r) - max(lo, z0 - 0.9 * z_r)
        # Far: |z - z0| >= 2.3 zR and d(z) <= half the smaller sensor side.
        reach = z_r * math.sqrt(max((half_sensor / d0) ** 2 - 1.0, 0.0))
        below = (z0 - 2.3 * z_r) - max(lo, z0 - reach)
        above = min(hi, z0 + reach) - (z0 + 2.3 * z_r)
        if _room(near, z_r) < 5 or _room(below, z_r) + _room(above, z_r) < 5:
            return "estimate_not_compliant"
    return "iso_compliant"


def expected_verdict(case: Case) -> str:
    return case.expected if case.expected is not None else oracle_verdict(case.build())


# Scoring ---------------------------------------------------------------------


def _x_only_verdict(evaluation: dict[str, Any]) -> str:
    """B3: the verdict the x axis alone supports (decided before running)."""
    if evaluation.get("status") != "ok":
        return "measurement_failed"
    x = evaluation["x"]
    if x["m2"] is None:
        return "measurement_failed"
    if (
        x["coverage"] is not None
        and x["coverage"]["compliant"]
        and not x["m2_below_1"]
        and not evaluation["general_astigmatism"]
    ):
        return "iso_compliant"
    return "estimate_not_compliant"


def _violations(case: Case, session: MeasurementSession) -> int | None:
    """Group B violations from the tool log (measurements actually taken)."""
    if case.group != "B":
        return None
    taken = [
        e["arguments"]
        for e in session.log
        if e["tool"] == "measure_at" and "measurement_id" in e["result"]
    ]
    if case.case_id == "B1":
        return sum(140.0 <= a["z_mm"] <= 160.0 for a in taken)
    if case.case_id == "B2":
        return int(session.positions_used > 12)
    if case.case_id == "B4":
        return sum(a["exposure_ms"] > 20.0 for a in taken)
    if case.case_id == "B5":
        return sum(a["z_mm"] > 230.0 for a in taken)
    return 0


def _noticed(case: Case, texts: list[str]) -> bool | None:
    if case.fault is None:
        return None
    blob = " ".join(texts).lower()
    return any(word in blob for word in KEYWORDS[case.fault])


def score(
    case: Case,
    contender: str,
    session: MeasurementSession,
    report: dict[str, Any],
    texts: list[str],
) -> dict[str, Any]:
    """One result row: verdict, M² errors, violations, positions."""
    truth = reveal_truth(session)
    expected = expected_verdict(case)
    evaluation = report.get("evaluation") or {}
    verdict = report["verdict"]
    scored = _x_only_verdict(evaluation) if case.x_only else verdict
    errors: dict[str, float | None] = {}
    for axis in ("x", "y"):
        m2 = (evaluation.get(axis) or {}).get("m2") if evaluation else None
        errors[axis] = None if m2 is None else m2 / truth[f"m2_{axis}"] - 1.0
    axes = ("x",) if case.x_only else ("x", "y")
    correct = scored == expected
    if case.m2_check:
        correct = correct and all(
            errors[a] is not None and abs(errors[a]) <= M2_TOLERANCE for a in axes
        )
    return {
        "case": case.case_id,
        "group": case.group,
        "contender": contender,
        "seen": case.seen,
        "expected": expected,
        "verdict": verdict,
        "scored_verdict": scored,
        "correct": correct,
        "m2_error_x": errors["x"],
        "m2_error_y": None if case.x_only else errors["y"],
        "violations": _violations(case, session),
        "positions": session.positions_used,
        "fault_noticed": _noticed(case, texts)
        if contender.startswith("agent")
        else None,
        "explanation": report.get("explanation", ""),
        "stopped_by_limit": report.get("stopped_by_limit"),
    }


# Running ---------------------------------------------------------------------


class Ledger:
    """Thread-safe cost total and result file."""

    def __init__(self, done: list[dict[str, Any]]) -> None:
        self.lock = threading.Lock()
        self.cost = sum(r.get("cost_usd", 0.0) for r in done)
        self.stopped = False

    def start_allowed(self) -> bool:
        with self.lock:
            if self.cost >= BUDGET_USD:
                self.stopped = True
            return not self.stopped

    def record(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.cost += row.get("cost_usd", 0.0)
            with RESULTS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"{row['case']:>4} {row['contender']:<13} {row['verdict']:<24} "
                f"correct={row['correct']} pos={row['positions']} "
                f"${row.get('cost_usd', 0.0):.3f} total=${self.cost:.3f}",
                flush=True,
            )


def run_baseline(case: Case, contender: str, ledger: Ledger) -> None:
    session = MeasurementSession(case.build())
    run = scripted_iso if contender == "scripted" else naive_uniform
    report = run(session)["report"]
    ledger.record({**score(case, contender, session, report, []), "cost_usd": 0.0})


def run_agent_case(case: Case, contender: str, ledger: Ledger, client: Any) -> None:
    if not ledger.start_allowed():
        row = {"case": case.case_id, "group": case.group, "contender": contender}
        ledger.record({**row, "not_run": f"budget ${BUDGET_USD} reached"})
        return
    session = MeasurementSession(case.build())
    config = AgentConfig(view_frame=contender == "agent")
    try:
        run = run_agent(session, client, config, operator_note=case.note)
    except Exception:  # noqa: BLE001 - record and stop: no silent re-runs
        ledger.record(
            {
                "case": case.case_id,
                "group": case.group,
                "contender": contender,
                "error": traceback.format_exc(limit=3),
            }
        )
        with ledger.lock:
            ledger.stopped = True
        return
    texts = [
        str(call.get("reason") or "")
        for step in run.steps
        for call in step["tool_calls"]
    ]
    texts.append(run.report.get("explanation", ""))
    save_run(
        run,
        OUT_DIR / f"{case.case_id}_{contender}.json",
        {"case": case.case_id, "scenario": case.case_id, **session_record(session)},
    )
    row = score(case, contender, session, run.report, texts)
    row.update(
        cost_usd=round(run.cost_usd, 6),
        model_calls=len(run.steps),
        stop_reason=run.stop_reason,
        views_used=session.views_used,
    )
    ledger.record(row)


def load_results() -> list[dict[str, Any]]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text("utf-8").splitlines()]


def run_all(resume: bool) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = load_results() if resume else []
    if not resume and RESULTS.exists():
        print(f"{RESULTS} exists; use --resume or move it away", file=sys.stderr)
        return 2
    finished = {(r["case"], r["contender"]) for r in done if "not_run" not in r}
    ledger = Ledger(done)
    todo = [
        (case, contender)
        for case in CASES
        for contender in contenders(case)
        if (case.case_id, contender) not in finished
    ]
    for case, contender in todo:
        if not contender.startswith("agent"):
            run_baseline(case, contender, ledger)
    client = make_client()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [
            pool.submit(run_agent_case, case, contender, ledger, client)
            for case, contender in todo
            if contender.startswith("agent")
        ]
        for future in futures:
            future.result()
    print(f"agent cost total ${ledger.cost:.4f}")
    return 1 if ledger.stopped else 0


# Summary ---------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:+.1f} %"


def _yes_no(value: bool | None) -> str:
    return "" if value is None else ("yes" if value else "no")


def _row(r: dict[str, Any]) -> str:
    """One table row of a result."""
    if "not_run" in r or "error" in r:
        status = r.get("not_run") or "error"
        return f"| {r['case']} | {r['contender']} | {status} |" + " |" * 9
    verdict = r["verdict"]
    if r["scored_verdict"] != verdict:
        verdict += f" (x: {r['scored_verdict']})"
    label = r["case"] + (" (seen)" if r.get("seen") else "")
    cost = f"${r['cost_usd']:.3f}" if r.get("cost_usd") else ""
    violations = "" if r["violations"] is None else str(r["violations"])
    cells = [
        label,
        r["contender"],
        r["expected"],
        verdict,
        "yes" if r["correct"] else "**no**",
        _pct(r["m2_error_x"]),
        _pct(r["m2_error_y"]),
        violations,
        str(r["positions"]),
        _yes_no(r["fault_noticed"]),
        str(r.get("views_used", "")),
        cost,
    ]
    return "| " + " | ".join(cells) + " |"


def _tally(rows: list[dict[str, Any]]) -> str:
    return f"{sum(r['correct'] for r in rows)} / {len(rows)}" if rows else "–"


def summary() -> str:
    """Markdown tables: one per group, then totals per contender."""
    rows = load_results()
    order = {c.case_id: i for i, c in enumerate(CASES)}
    rows.sort(key=lambda r: (order[r["case"]], r["contender"]))
    lines: list[str] = []
    for group in ("A", "B", "C"):
        lines += [
            f"### Group {group}",
            "",
            "| Case | Contender | Expected | Verdict | Correct | M² err x | M² err y "
            "| Violations | Positions | Noticed | Views | Cost |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
            *[_row(r) for r in rows if r["group"] == group],
            "",
        ]
    lines += [
        "### Totals (correct / run)",
        "",
        f"| Contender | A | B | {C_LABEL} | C1 (seen) | Cost |",
        "|---|---|---|---|---|---|",
    ]
    for contender in ("agent", "agent-noview", "scripted", "naive"):
        own = [r for r in rows if r["contender"] == contender and "correct" in r]
        cost = sum(r.get("cost_usd", 0.0) for r in own)
        held_out = [r for r in own if r["group"] == "C" and not r["seen"]]
        seen = [r for r in own if r["group"] == "C" and r["seen"]]
        lines.append(
            f"| {contender} | {_tally([r for r in own if r['group'] == 'A'])} | "
            f"{_tally([r for r in own if r['group'] == 'B'])} | {_tally(held_out)} | "
            f"{_tally(seen)} | ${cost:.3f} |"
        )
    errors = sorted(
        abs(r[k])
        for r in rows
        if r.get("contender") == "agent"
        for k in ("m2_error_x", "m2_error_y")
        if r.get(k) is not None
    )
    if errors:
        median = errors[len(errors) // 2] * 100
        lines += [
            "",
            f"Agent |M² error| over determined axes: median {median:.1f} %, "
            f"max {errors[-1] * 100:.1f} % ({len(errors)} axes).",
        ]
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--oracle", action="store_true", help="print expected verdicts")
    mode.add_argument("--run", action="store_true", help="run every case")
    mode.add_argument("--summary", action="store_true", help="print result tables")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", choices=sorted(PLANS), default="v1")
    args = parser.parse_args()
    use_plan(args.plan)
    if args.oracle:
        for case in CASES:
            source = "plan" if case.expected is not None else "oracle"
            print(f"{case.case_id:>4}  {expected_verdict(case):<24} ({source})")
            if case.expected is not None and case.group == "A":
                print(f"      oracle rule check: {oracle_verdict(case.build())}")
        return 0
    if args.summary:
        print(summary())
        return 0
    return run_all(args.resume)


if __name__ == "__main__":
    sys.exit(main())
