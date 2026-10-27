"""Smoke test of image input with the Meta Model API (Responses API).

Renders one simulated camera frame (main spot off-centre to the upper right,
a weak second spot to the lower left), encodes it as PNG and asks the model
what it sees, in two ways:

a) documented: ``input_image`` (base64 data URL) in a user message;
b) inside a ``function_call_output`` of a dummy ``view_frame`` tool, the
   shape a real view_frame tool would use. The docs disagree on whether the
   model processes images outside user messages, so this is checked here.

Only simulated data is sent. Usage::

    .venv\\Scripts\\python.exe scripts\\smoke_test_image.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from beampilot.session import _beam, _camera  # noqa: E402
from beampilot.simulator import render_frame  # noqa: E402

BASE_URL = "https://api.meta.ai/v1"
MODEL = "muse-spark-1.2"
REPO_ROOT = Path(__file__).resolve().parents[1]
QUESTION = (
    "This is a laser beam profile from a camera (false colour, bright = more "
    "light). How many light spots do you see? For each spot: where is it in the "
    "image (e.g. upper right), roughly how large compared with the image, and how "
    "bright relative to the brightest one? Is anything saturated? Answer briefly."
)


def frame_png() -> str:
    """Base64 PNG of a frame with a main spot and a weak ghost spot."""
    camera = _camera()
    rng = np.random.default_rng(0)
    beam = _beam()  # clean beam, waist 200/210 um at 120 mm
    z_m = 0.150  # ~1.2 zR from the waist: d ~ 310 um = 56 px
    main = render_frame(beam, camera, z_m, 3.0, rng, x0_m=0.45e-3, y0_m=-0.45e-3)
    ghost = render_frame(beam, camera, z_m, 0.6, rng, x0_m=-0.6e-3, y0_m=0.5e-3)
    frame = main + (ghost - camera.offset_counts)  # one offset only
    rgb = plt.get_cmap("inferno")(np.clip(frame / camera.full_scale_counts, 0, 1))
    buffer = io.BytesIO()
    plt.imsave(buffer, rgb[..., :3], format="png")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def report(label: str, response, start: float) -> None:
    usage = response.usage
    print(
        f"\n== {label}: status {response.status}, {time.perf_counter() - start:.1f} s"
    )
    print(
        f"   tokens: input {usage.input_tokens}, output {usage.output_tokens} "
        f"(reasoning {usage.output_tokens_details.reasoning_tokens})"
    )
    print(f"   answer: {response.output_text!r}")


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("MODEL_API_KEY"):
        print("MODEL_API_KEY is not set (see .env.example).", file=sys.stderr)
        return 1
    client = OpenAI(api_key=os.environ["MODEL_API_KEY"], base_url=BASE_URL)
    image = f"data:image/png;base64,{frame_png()}"
    print(f"PNG data URL: {len(image) / 1024:.0f} kB")

    # a) documented path: image in a user message.
    start = time.perf_counter()
    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": QUESTION},
                    {"type": "input_image", "image_url": image},
                ],
            }
        ],
        reasoning={"effort": "low"},
        max_output_tokens=2000,
        store=False,
    )
    report("a) image in a user message", response, start)

    # b) image returned by a tool (function_call_output with content parts).
    tool = {
        "type": "function",
        "name": "view_frame",
        "description": "Returns the latest camera frame as an image.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    }
    history: list = [
        {
            "role": "user",
            "content": "Call view_frame once, then answer: " + QUESTION,
        }
    ]
    start = time.perf_counter()
    first = client.responses.create(
        model=MODEL,
        input=history,
        tools=[tool],
        reasoning={"effort": "low"},
        max_output_tokens=2000,
        store=False,
        include=["reasoning.encrypted_content"],
    )
    calls = [item for item in first.output if item.type == "function_call"]
    if not calls:
        print("\n== b) the model did not call view_frame:", first.output_text)
        return 0
    for item in first.output:
        if item.type == "reasoning" and item.encrypted_content:
            history.append(
                {
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": item.encrypted_content,
                }
            )
        elif item.type == "function_call":
            history.append(
                {
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                }
            )
    history.append(
        {
            "type": "function_call_output",
            "call_id": calls[0].call_id,
            "output": [
                {"type": "input_text", "text": json.dumps({"measurement_id": "m1"})},
                {"type": "input_image", "image_url": image},
            ],
        }
    )
    second = client.responses.create(
        model=MODEL,
        input=history,
        tools=[tool],
        reasoning={"effort": "low"},
        max_output_tokens=2000,
        store=False,
    )
    report("b) image inside a function_call_output", second, start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
