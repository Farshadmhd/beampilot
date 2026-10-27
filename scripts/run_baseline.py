"""Run one baseline strategy on one scenario and print the run.

Prints the tool log (tool, arguments, reason and a short result), the final
report, and then the revealed ground truth.

Usage::

    .venv\\Scripts\\python.exe scripts\\run_baseline.py clean scripted_iso
    .venv\\Scripts\\python.exe scripts\\run_baseline.py clean naive_uniform -s 2
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from beampilot.baselines import naive_uniform, scripted_iso
from beampilot.session import SCENARIOS, MeasurementSession, reveal_truth

STRATEGIES = {"naive_uniform": naive_uniform, "scripted_iso": scripted_iso}


def short_result(result: dict[str, Any]) -> str:
    """One-line summary of a tool result."""
    if "error" in result or "internal_error" in result:
        return json.dumps(result, ensure_ascii=False)
    if "measurement_id" in result and "flags" in result:
        flags = [name for name, value in result["flags"].items() if value]
        return (
            f"{result['measurement_id']}: d_x {result['d_x_um']['mean']} um, "
            f"d_y {result['d_y_um']['mean']} um, peak "
            f"{result['details']['peak_counts']}, flags {flags or 'none'}, "
            f"suggested {result['suggested_verdict']}"
        )
    if "status" in result and "x" in result:
        return (
            f"M²x {result['x']['m2']}, M²y {result['y']['m2']}, "
            f"z0 {result['x']['z0_mm']}/{result['y']['z0_mm']} mm, "
            f"coverage met {result['iso_coverage_met']}"
        )
    if "report" in result:
        return f"verdict {result['report']['verdict']}"
    return json.dumps(result, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("strategy", choices=sorted(STRATEGIES))
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    session = MeasurementSession(SCENARIOS[args.scenario](seed=args.seed))
    STRATEGIES[args.strategy](session)

    print(f"== Tool log: {args.strategy} on {args.scenario} (seed {args.seed})")
    for i, entry in enumerate(session.log, start=1):
        arguments = {
            k: v for k, v in entry["arguments"].items() if k not in ("reason",)
        }
        print(f"{i:3d}. {entry['tool']}({arguments})")
        print(f"     reason: {entry['reason']}")
        print(f"     result: {short_result(entry['result'])}")
    print("\n== Report")
    print(json.dumps(session.report, indent=2, ensure_ascii=False))
    print("\n== Revealed truth")
    print(json.dumps(reveal_truth(session), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
