"""Smoke test for the Meta Model API (OpenAI-compatible).

Loads ``MODEL_API_KEY`` from the repository's ``.env`` file, lists the
available models (GET /v1/models) and sends one tiny chat request to a
standard Muse Spark model.

Usage (from the repository root, PowerShell)::

    .venv\\Scripts\\python.exe scripts\\smoke_test_meta_api.py
    .venv\\Scripts\\python.exe scripts\\smoke_test_meta_api.py --model muse-spark-1.1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.2"
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Run the smoke test.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on a configuration error.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model ID for the test request (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum number of output tokens (default: 16).",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        print(
            "MODEL_API_KEY is not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    print(f"Available models at {BASE_URL}:")
    model_ids = sorted(model.id for model in client.models.list())
    for model_id in model_ids:
        print(f"  - {model_id}")
    if args.model not in model_ids:
        print(f"Warning: {args.model!r} is not in the model list.", file=sys.stderr)

    print(f"\nSending test request to {args.model} ...")
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Reply with OK"}],
        max_tokens=args.max_tokens,
    )
    print(f"Reply: {response.choices[0].message.content!r}")
    if response.usage is not None:
        print(
            f"Tokens: prompt={response.usage.prompt_tokens}, "
            f"completion={response.usage.completion_tokens}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
