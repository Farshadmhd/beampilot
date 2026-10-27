"""Smoke test of tool calling with the Meta Model API (Responses API).

Sends one question that needs ONE dummy tool (``get_beam_width``) at two
positions, runs the returned function calls locally, sends the results back
with ``function_call_output`` and ``previous_response_id``, and prints every
output item, the token usage and the latency of each model call. The tool
returns a made-up number; no real or lab data is sent.

Usage (from the repository root, PowerShell)::

    .venv\\Scripts\\python.exe scripts\\smoke_test_tools.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_CALLS = 4

INSTRUCTIONS = (
    "You are testing a tool interface. Use the tool to answer. Always fill in the "
    "'reason' argument with a short reason. Answer in one sentence at the end."
)
TOOL = {
    "type": "function",
    "name": "get_beam_width",
    "description": "Dummy tool: returns a beam width in um at a stage position in mm.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "z_mm": {"type": "number", "description": "Stage position in mm."},
            "reason": {"type": "string", "description": "Why you call this tool."},
        },
        "required": ["z_mm", "reason"],
        "additionalProperties": False,
    },
}


def get_beam_width(z_mm: float, reason: str) -> dict:
    """Made-up width: 200 um at 120 mm, growing linearly away from it."""
    return {"z_mm": z_mm, "d_um": round(200.0 + 8.0 * abs(z_mm - 120.0), 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="muse-spark-1.2")
    parser.add_argument("--effort", default="low")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        print("MODEL_API_KEY is not set (see .env.example).", file=sys.stderr)
        return 1
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    next_input: list | str = (
        "What are the beam widths at z = 100 mm and z = 140 mm? Use the tool."
    )
    previous_id = None
    for call in range(1, MAX_CALLS + 1):
        start = time.perf_counter()
        response = client.responses.create(
            model=args.model,
            instructions=INSTRUCTIONS,  # not kept across turns: resend every call
            input=next_input,
            tools=[TOOL],
            reasoning={"effort": args.effort},
            max_output_tokens=4000,
            previous_response_id=previous_id,
        )
        latency = time.perf_counter() - start
        usage = response.usage
        print(f"\n== model call {call}: status {response.status}, {latency:.2f} s")
        print(
            f"   tokens: input {usage.input_tokens} "
            f"(cached {usage.input_tokens_details.cached_tokens}), "
            f"output {usage.output_tokens} "
            f"(reasoning {usage.output_tokens_details.reasoning_tokens})"
        )
        outputs = []
        for item in response.output:
            if item.type == "function_call":
                print(
                    f"   function_call: name={item.name} call_id={item.call_id} "
                    f"arguments={item.arguments}"
                )
                result = get_beam_width(**json.loads(item.arguments))
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(result),
                    }
                )
            elif item.type == "message":
                print(f"   message: {response.output_text!r}")
            else:
                print(f"   {item.type} item")
        if not outputs:
            print("\nSmoke test OK: the model answered after the tool calls.")
            return 0
        print(f"   -> {len(outputs)} function call(s) in this turn; sending results")
        next_input = outputs
        previous_id = response.id
    print("Stopped: no final answer within the call limit.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
