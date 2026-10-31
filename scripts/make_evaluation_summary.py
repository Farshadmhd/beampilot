"""Write docs/evaluation-summary.md from the v1 and v2 result files.

Every number in the summary is computed here from ``docs/results/*.jsonl``
(copies of ``runs/evaluation*/results.jsonl`` written by
``scripts/run_evaluation.py``); nothing is typed by hand. Run again after any
change to the result files::

    .venv\\Scripts\\python.exe scripts\\make_evaluation_summary.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "evaluation-summary.md"
VERSIONS = ("v1", "v2")
RESULT_FILES = {
    v: REPO / "docs" / "results" / f"evaluation-{v}.jsonl" for v in VERSIONS
}
CONTENDERS = ("agent", "agent-noview", "scripted", "naive")
# Plans and the commits that fixed them before any run (see the results files).
PLANS = {
    "v1": {
        "plan": "evaluation-plan.md",
        "results": "evaluation-results.md",
        "commits": [
            ("d2f445e", "plan"),
            ("b3025e9", "deviations 1-7"),
            ("9ca92df", "deviation 8 and evaluation script"),
            ("cda0f73", "deviation 9"),
        ],
        "code": "03fb2c1",
    },
    "v2": {
        "plan": "evaluation-plan-v2.md",
        "results": "evaluation-results-v2.md",
        "commits": [("75fe382", "plan"), ("3241612", "deviations 1-2")],
        "code": "3403a76",
    },
}
GROUPS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "A": lambda r: r["group"] == "A",
    "B": lambda r: r["group"] == "B",
    "C2-C5": lambda r: r["group"] == "C" and not r["seen"],
    "C1 (seen)": lambda r: r["group"] == "C" and r["seen"],
}


def load(path: Path) -> list[dict[str, Any]]:
    """Result rows of one evaluation (rows that were run and scored)."""
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    return [r for r in rows if "correct" in r]


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:+.1f} %"


def _tally(rows: list[dict[str, Any]]) -> str:
    return f"{sum(r['correct'] for r in rows)}/{len(rows)}" if rows else "–"


def correct_table(data: dict[str, list[dict[str, Any]]]) -> list[str]:
    header = "| Contender | " + " | ".join(f"{g}: v1 · v2 · both" for g in GROUPS)
    lines = [header + " |", "|---|" + "---|" * len(GROUPS)]
    for contender in CONTENDERS:
        cells = []
        for pick in GROUPS.values():
            per = [
                [r for r in data[v] if r["contender"] == contender and pick(r)]
                for v in VERSIONS
            ]
            both = per[0] + per[1]
            if not both:
                cells.append("–")
                continue
            cells.append(f"{_tally(per[0])} · {_tally(per[1])} · **{_tally(both)}**")
        lines.append(f"| {contender} | " + " | ".join(cells) + " |")
    return lines


def effort_table(data: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = [
        "| Contender | Version | Runs | Mean positions | Mean model calls | "
        "Stopped by a limit | Median / max \\|M² error\\| | Cost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for contender in CONTENDERS:
        for version in VERSIONS:
            rows = [r for r in data[version] if r["contender"] == contender]
            if not rows:
                continue
            calls = [r["model_calls"] for r in rows if "model_calls" in r]
            limited = [
                r["case"] for r in rows if "limit" in (r.get("stop_reason") or "")
            ]
            errors = sorted(
                abs(r[k])
                for r in rows
                for k in ("m2_error_x", "m2_error_y")
                if r.get(k) is not None
            )
            error = (
                f"{statistics.median(errors) * 100:.1f} % / {errors[-1] * 100:.1f} %"
                if errors
                else "–"
            )
            mean_calls = f"{statistics.mean(calls):.1f}" if calls else "–"
            stopped = (", ".join(limited) or "none") if calls else "–"
            lines.append(
                f"| {contender} | {version} | {len(rows)} | "
                f"{statistics.mean(r['positions'] for r in rows):.1f} | "
                f"{mean_calls} | {stopped} | {error} | "
                f"${sum(r.get('cost_usd', 0.0) for r in rows):.2f} |"
            )
    return lines


def violations_table(data: dict[str, list[dict[str, Any]]]) -> list[str]:
    cases = sorted({r["case"] for v in VERSIONS for r in data[v] if r["group"] == "B"})
    lines = [
        "| Contender | " + " | ".join(f"{c} v1 · v2" for c in cases) + " | Total |",
        "|---|" + "---|" * (len(cases) + 1),
    ]
    for contender in ("agent", "scripted", "naive"):
        cells, total = [], 0
        for case in cases:
            counts = []
            for version in VERSIONS:
                row = next(
                    r
                    for r in data[version]
                    if r["case"] == case and r["contender"] == contender
                )
                counts.append(row["violations"])
                total += row["violations"]
            cells.append(f"{counts[0]} · {counts[1]}")
        lines.append(f"| {contender} | " + " | ".join(cells) + f" | **{total}** |")
    return lines


def failure_table(data: dict[str, list[dict[str, Any]]], case: str) -> list[str]:
    lines = [
        "| Version | Contender | Verdict (expected) | Correct | M² err x / y | "
        "Views | Fault named |",
        "|---|---|---|---|---|---|---|",
    ]
    for version in VERSIONS:
        for contender in ("agent", "agent-noview"):
            row = next(
                r
                for r in data[version]
                if r["case"] == case and r["contender"] == contender
            )
            named = "yes" if row["fault_noticed"] else "no"
            lines.append(
                f"| {version} | {contender} | {row['verdict']} ({row['expected']}) | "
                f"{'yes' if row['correct'] else '**no**'} | "
                f"{_pct(row['m2_error_x'])} / {_pct(row['m2_error_y'])} | "
                f"{row.get('views_used', '–')} | {named} |"
            )
    return lines


def summary(data: dict[str, list[dict[str, Any]]]) -> str:
    agent_runs = sum(
        1 for v in VERSIONS for r in data[v] if r["contender"].startswith("agent")
    )
    cost = sum(r.get("cost_usd", 0.0) for v in VERSIONS for r in data[v])
    plan_lines = []
    for version in VERSIONS:
        plan = PLANS[version]
        commits = ", ".join(f"`{h}` ({what})" for h, what in plan["commits"])
        plan_lines.append(
            f"- **{version}:** [{plan['plan']}]({plan['plan']}), pre-registered in "
            f"{commits}; run on `main` at `{plan['code']}`. Full results: "
            f"[{plan['results']}]({plan['results']})."
        )
    lines = [
        "# BeamPilot evaluation summary (v1 + v2)",
        "",
        "Generated by `scripts/make_evaluation_summary.py` from "
        "`docs/results/evaluation-v1.jsonl` and `docs/results/evaluation-v2.jsonl`. "
        "No number on this page is typed by hand.",
        "",
        "## Plans",
        "",
        *plan_lines,
        "",
        "Between v1 and v2 only two things changed: the model-call limit (40 → 60) "
        "and a prompt asking for parallel tool calls. Seeds differ.",
        "",
        "## Read this first",
        "",
        "- **Small N.** Each case ran twice in total: once in v1 and once in v2, with "
        "different seeds. A difference of one or two cases is noise, not a measured "
        "effect.",
        '- **C1 is "seen".** The prompt was changed after the agent failed the same '
        "fault type during development, so C1 is never counted as held out. C2-C5 "
        "were held out in v1 and seen in v2.",
        "- **The scripted baseline cannot read operator notes, by design.** In "
        "group B it runs without the note; its violations are counted below. Group B "
        "is therefore not a fair contest.",
        "- **The naive scan is a reference only** (10 fixed positions, 1 ms).",
        "",
        "## Correct verdicts",
        "",
        "Correct / run, per version and both together. Group C is run twice per "
        "version for the agent: with `view_frame` (`agent`) and without "
        "(`agent-noview`).",
        "",
        *correct_table(data),
        "",
        "## Effort, accuracy and cost",
        "",
        *effort_table(data),
        "",
        "## Constraint violations in group B",
        "",
        "Measurements that broke the operator note (B2: 1 if more than 12 positions "
        "were used; B3 has no constraint).",
        "",
        *violations_table(data),
        "",
        "## Main failure modes",
        "",
        "**C1, ghost reflection at every position (seen): saw the ghost but kept "
        "the points.** In v1 the agent with `view_frame` named the ghost and gave the "
        "right verdict. In v2 it looked at frames only after coverage was met, "
        "described the second spot, and still kept those points. Without "
        "`view_frame` the agent read the constant `axes_not_aligned` flag as a "
        "rotated beam. The wrong runs report a confident, very wrong M².",
        "",
        *failure_table(data, "C1"),
        "",
        "**C5, iris clipping the beam over 30-70 mm (no flag fires): never looked.** "
        "In no run did the agent look at a frame. It accepted far points inside the "
        "iris range and did not act on the rising fit residual, so the verdict was "
        "right but M² missed the 3 % tolerance.",
        "",
        *failure_table(data, "C5"),
        "",
        f"Total: {agent_runs} agent runs, ${cost:.2f} estimated API cost.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    data = {v: load(RESULT_FILES[v]) for v in VERSIONS}
    OUT.write_text(summary(data), encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
