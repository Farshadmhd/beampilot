"""Run evaluation v3 (change_lens) as pre-registered in docs/evaluation-plan-v3.md.

Cases L1-L8 from ``scripts/v3_cases.py`` on seeds 900-907 (L1 -> 900, ...).
Contenders: the agent (change_lens and view_frame on), the scripted operator
(lens rule) and the naive scan. Results go to ``runs/evaluation-v3/``; one
line per run in ``results.jsonl``; ``--summary`` prints the tables.

Usage (PowerShell)::

    .venv\\Scripts\\python.exe scripts\\run_evaluation_v3.py --oracle
    .venv\\Scripts\\python.exe scripts\\run_evaluation_v3.py --run
    .venv\\Scripts\\python.exe scripts\\run_evaluation_v3.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_cases import LensCase, all_cases, oracle_verdict  # noqa: E402

from beampilot.agent import AgentConfig, make_client, run_agent, save_run  # noqa: E402
from beampilot.baselines import naive_uniform, scripted_iso  # noqa: E402
from beampilot.session import (  # noqa: E402
    MeasurementSession,
    reveal_truth,
    session_record,
)

OUT_DIR = Path("runs") / "evaluation-v3"
RESULTS = OUT_DIR / "results.jsonl"
FIRST_SEED = 900
BUDGET_USD = 3.0  # plan v3, deviation 1 (was 2.0)
# Agent limits for v3 (plan deviation 1): more model calls and tokens; the
# per-run cost cap stays at $0.50.
AGENT_CONFIG = {
    "change_lens": True,
    "max_model_calls": 80,
    "max_total_tokens": 3_000_000,
    "max_cost_usd": 0.50,
}
CONTENDERS = ("agent", "scripted", "naive")
# Measurement within tolerance (iso_compliant cases): design section 7, item 7.
M2_TOL = 0.03
D0_TOL = 0.05
Z0_TOL_ZR = 0.1


def seed_of(case: LensCase) -> int:
    return FIRST_SEED + int(case.case_id[1:]) - 1


def within_tolerance(report: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """Errors of the back-calculated beam and M² against the truth, per axis."""
    back = report.get("original_beam_back_calculated") or {}
    m2 = report.get("m2") or {}
    errors: dict[str, Any] = {}
    ok = bool(back) and bool(m2)
    for axis in ("x", "y"):
        b, m = back.get(axis), (m2.get(axis) or {}).get("m2")
        if b is None or m is None:
            errors[axis] = None
            ok = False
            continue
        d0 = b["d0_um"] / truth[f"d0_{axis}_um"] - 1
        z0 = (b["z0_mm"] - truth[f"z0_{axis}_mm"]) / truth[f"z_r_{axis}_mm"]
        m2e = m / truth[f"m2_{axis}"] - 1
        errors[axis] = {
            "d0_rel": d0,
            "z0_zr": z0,
            "m2_rel": m2e,
            "dz0_mm_per_plus_1pct_f": b["dz0_mm_per_plus_1pct_f"],
        }
        ok &= abs(d0) <= D0_TOL and abs(z0) <= Z0_TOL_ZR and abs(m2e) <= M2_TOL
    return {"within_tolerance": ok, "errors": errors}


def score(case, contender, session, report, extra) -> dict[str, Any]:
    truth = reveal_truth(session)
    expected = oracle_verdict(case.scenario(seed_of(case)))
    lens = report.get("lens")
    row = {
        "case": case.case_id,
        "seed": seed_of(case),
        "contender": contender,
        "expected": expected,
        "verdict": report["verdict"],
        "correct": report["verdict"] == expected,
        "lens": None if lens is None else [lens["f_mm"], lens["lens_z_mm"]],
        "lens_changes": len(session.lens_changes),
        "positions": session.positions_used,
        "explanation": report.get("explanation", ""),
        "stopped_by_limit": report.get("stopped_by_limit"),
        **extra,
    }
    if expected == "iso_compliant":
        row.update(within_tolerance(report, truth))
    return row


def record(row: dict[str, Any]) -> None:
    with RESULTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"{row['case']} {row['contender']:<9} {row.get('verdict', row.get('not_run'))} "
        f"correct={row.get('correct')} tol={row.get('within_tolerance')} "
        f"pos={row.get('positions')} ${row.get('cost_usd', 0):.3f}",
        flush=True,
    )


def run_all() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS.exists():
        print(f"{RESULTS} exists; move it away first", file=sys.stderr)
        return 2
    cases = all_cases()
    for case in cases:
        for contender, strategy in (
            ("scripted", scripted_iso),
            ("naive", naive_uniform),
        ):
            session = MeasurementSession(case.scenario(seed_of(case)))
            report = strategy(session)["report"]
            record(score(case, contender, session, report, {"cost_usd": 0.0}))
    client = make_client()
    total = 0.0
    for case in cases:
        if total >= BUDGET_USD:
            record({"case": case.case_id, "contender": "agent", "not_run": "budget"})
            continue
        session = MeasurementSession(case.scenario(seed_of(case)))
        try:
            run = run_agent(session, client, AgentConfig(**AGENT_CONFIG))
        except Exception:  # noqa: BLE001 - record and stop: no silent re-runs
            record(
                {
                    "case": case.case_id,
                    "contender": "agent",
                    "error": traceback.format_exc(limit=3),
                }
            )
            return 1
        total += run.cost_usd
        save_run(
            run,
            OUT_DIR / f"{case.case_id}_agent.json",
            {"case": case.case_id, "seed": seed_of(case), **session_record(session)},
        )
        extra = {
            "cost_usd": round(run.cost_usd, 6),
            "model_calls": len(run.steps),
            "stop_reason": run.stop_reason,
            "views_used": session.views_used,
        }
        record(score(case, "agent", session, run.report, extra))
    print(f"agent cost total ${total:.4f}")
    return 0


def summary() -> str:
    rows = [json.loads(line) for line in RESULTS.read_text("utf-8").splitlines()]
    lines = [
        "| Case | Contender | Expected | Verdict | Correct | Within tolerance "
        "| Lens | Positions | Model calls | Cost |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if "not_run" in row or "error" in row:
            status = row.get("not_run") or "error"
            lines.append(
                f"| {row['case']} | {row['contender']} | {status} |" + " |" * 7
            )
            continue
        lens = (
            "–" if row["lens"] is None else f"{row['lens'][0]:g} @ {row['lens'][1]:g}"
        )
        tol = row.get("within_tolerance")
        lines.append(
            f"| {row['case']} | {row['contender']} | {row['expected']} | "
            f"{row['verdict']} | {'yes' if row['correct'] else '**no**'} | "
            f"{'–' if tol is None else ('yes' if tol else '**no**')} | {lens} | "
            f"{row['positions']} | {row.get('model_calls', '–')} | "
            f"${row.get('cost_usd', 0.0):.3f} |"
        )
    for contender in CONTENDERS:
        own = [r for r in rows if r.get("contender") == contender and "correct" in r]
        tol = [r for r in own if "within_tolerance" in r]
        lines.append(
            f"\n**{contender}:** {sum(r['correct'] for r in own)}/{len(own)} correct "
            f"verdicts; {sum(r['within_tolerance'] for r in tol)}/{len(tol)} within "
            f"tolerance; cost ${sum(r.get('cost_usd', 0.0) for r in own):.3f}."
        )
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--oracle", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.oracle:
        for case in all_cases():
            verdict = oracle_verdict(case.scenario(seed_of(case)))
            print(f"{case.case_id} seed {seed_of(case)}: {verdict}")
        return 0
    if args.summary:
        print(summary())
        return 0
    return run_all()


if __name__ == "__main__":
    sys.exit(main())
