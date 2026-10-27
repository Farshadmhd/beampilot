"""Run one real Muse Spark agent run on a simulated scenario.

Prints each model call with its tool calls and reasons, then the report, the
revealed truth, total tokens and the estimated cost, and saves the run as JSON
in ``runs/`` (git-ignored). Only simulated data is sent to the API.

Usage (from the repository root, PowerShell)::

    .venv\\Scripts\\python.exe scripts\\run_agent.py --scenario clean --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from beampilot.agent import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    PRICES_PER_MTOK,
    AgentConfig,
    make_client,
    run_agent,
    save_run,
)
from beampilot.session import (
    SCENARIOS,
    MeasurementSession,
    random_scenario,
    reveal_truth,
    session_record,
)

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def short(result: dict[str, Any]) -> str:
    """One-line summary of a tool result."""
    if "error" in result or "internal_error" in result:
        return json.dumps(result, ensure_ascii=False)
    if "flags" in result:
        flags = [name for name, value in result["flags"].items() if value]
        return (
            f"{result['measurement_id']} at {result['z_mm']} mm: d_x "
            f"{result['d_x_um']['mean']} um, d_y {result['d_y_um']['mean']} um, peak "
            f"{result['details']['peak_counts']}, flags {flags or 'none'}, "
            f"suggested {result['suggested_verdict']}, "
            f"{result['positions_left']} left"
        )
    if result.get("status") == "ok":
        axes = []
        for axis in ("x", "y"):
            a = result[axis]
            coverage = a["coverage"]["summary"] if a["coverage"] else "no coverage"
            axes.append(
                f"M²{axis} {a['m2']} z0 {a['z0_mm']} zR {a['z_r_mm']} [{coverage}]"
            )
        return "; ".join(axes) + f"; ISO coverage met {result['iso_coverage_met']}"
    if "report" in result:
        return f"report: verdict {result['report']['verdict']}"
    text = json.dumps(result, ensure_ascii=False)
    return text if len(text) < 200 else text[:200] + " ..."


def print_step(step: dict[str, Any]) -> None:
    usage = step["usage"]
    print(
        f"\n-- model call {step['call']} ({step['latency_s']} s, in "
        f"{usage['input_tokens']} [cached {usage['cached_tokens']}], out "
        f"{usage['output_tokens']} [reasoning {usage['reasoning_tokens']}], "
        f"${step['cost_usd']:.4f})"
    )
    if step["text"]:
        print(f"   says: {step['text'].strip()}")
    for call in step["tool_calls"]:
        arguments = {k: v for k, v in call["arguments"].items() if k != "reason"}
        print(f"   {call['name']}({arguments})")
        print(f"      reason: {call.get('reason')}")
        print(f"      -> {short(call['result'])}")


def main() -> int:
    # Model text may contain any Unicode; never crash a paid run on printing,
    # and flush each line so a redirected log shows progress.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario", choices=[*sorted(SCENARIOS), "mystery"], default="clean"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--note", default=None, help="operator note for the agent")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT)
    parser.add_argument("--max-calls", type=int, default=40)
    parser.add_argument("--max-cost", type=float, default=0.50, help="USD per run")
    args = parser.parse_args()

    factory = (
        random_scenario if args.scenario == "mystery" else SCENARIOS[args.scenario]
    )
    session = MeasurementSession(factory(seed=args.seed))
    config = AgentConfig(
        model=args.model,
        reasoning_effort=args.effort,
        max_model_calls=args.max_calls,
        max_cost_usd=args.max_cost,
    )
    print(
        f"== Agent run: {args.scenario}, seed {args.seed}, {args.model}, "
        f"effort {args.effort}"
    )
    if args.note:
        print(f"   operator note: {args.note}")
    run = run_agent(
        session, make_client(), config, on_step=print_step, operator_note=args.note
    )

    print(f"\n== Stop reason: {run.stop_reason}")
    print("\n== Report")
    print(json.dumps(run.report, indent=2, ensure_ascii=False))
    truth = reveal_truth(session)
    print("\n== Revealed truth")
    print(json.dumps(truth, indent=2, ensure_ascii=False))
    usage = run.usage
    print(
        f"\n== Totals: {len(run.steps)} model calls, {session.positions_used} "
        f"positions, input {usage.input_tokens} (cached {usage.cached_tokens}), "
        f"output {usage.output_tokens} (reasoning {usage.reasoning_tokens})"
    )
    print(
        f"   estimated cost ${run.cost_usd:.4f} (prices {PRICES_PER_MTOK} per "
        "million tokens -- check pricing page)"
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNS_DIR / f"{stamp}_{args.scenario}_s{args.seed}_{args.model}.json"
    save_run(
        run,
        path,
        {"scenario": args.scenario, "seed": args.seed, **session_record(session)},
    )
    print(f"   saved {path}")
    return 0 if run.stop_reason == "finished" else 2


if __name__ == "__main__":
    sys.exit(main())
