"""Muse Spark agent loop over the six measurement tools (design sections 4, 10).

The model sees only the system prompt, one task message and the JSON results
of the tools; all physics numbers come from :mod:`beampilot.tools`. Verified
against the Meta Model API docs (Responses API, 27 Oct 2026):

- ``POST /v1/responses`` via the openai SDK with ``base_url`` below.
- ``instructions`` is not kept across turns: it is resent on every call.
- Stateless turns: ``store=False`` with
  ``include=["reasoning.encrypted_content"]`` and the full history resent
  each call (reasoning items replayed with ``summary`` and
  ``encrypted_content``). Chaining with ``previous_response_id`` failed with
  "referenced response not found or expired" in parallel runs. Tool results
  go back as ``function_call_output`` items with the matching ``call_id``
  and the output as a JSON string.
- ``max_output_tokens`` caps reasoning plus visible output.
- ``parallel_tool_calls`` defaults to true; this loop runs the calls one
  after another in the order given, like a real stage.
- ``reasoning={"effort": ...}``; ``"none"`` is not supported.
- Usage: ``input_tokens``, ``input_tokens_details.cached_tokens``,
  ``output_tokens``, ``output_tokens_details.reasoning_tokens``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from beampilot import tools
from beampilot.session import MeasurementSession

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.2"
DEFAULT_EFFORT = "low"

# USD per million tokens, from https://dev.meta.ai/docs/pricing-rate-limits on
# 27 Oct 2026 -- check the pricing page before relying on these numbers.
PRICES_PER_MTOK = {
    "standard": {"input": 1.25, "cached_input": 0.15, "output": 4.25},
    "contributor": {"input": 0.10, "cached_input": 0.002, "output": 0.20},
}

TASK = (
    "Measure the beam propagation ratio M² of the laser beam on this bench, "
    "separately for x and y, following ISO 11146-1. Use as few positions as "
    "the standard allows. Start with get_setup. The run ends only when finish "
    "succeeds."
)
OPERATOR_NOTE_PREFIX = (
    "Operator note (from the person at the bench; the tools do not enforce it, "
    "so respect it yourself): "
)
RETRIES = 2
RETRY_DELAY_S = 2.0
TRANSIENT_ERRORS = {
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
}
NUDGE = (
    "Continue by calling tools. The run ends only when finish succeeds; if you "
    "are done, call finish with a verdict and an explanation."
)

_REASON = {"type": "string", "description": "Short reason for this call."}
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_setup",
        "description": "Fixed setup: wavelength (nm), stage range (mm), pixel size "
        "(um), sensor, full scale, exposure range (ms), repeats, position budget.",
        "properties": {"reason": _REASON},
    },
    {
        "name": "measure_at",
        "description": "Move the stage to z_mm and take the repeated frames at "
        "exposure_ms. Returns ISO second-moment widths (um, mean and std), "
        "quality flags, a suggested verdict and positions left. The measurement "
        "starts as pending. Costs one position.",
        "properties": {
            "z_mm": {"type": "number", "description": "Stage position in mm."},
            "exposure_ms": {"type": "number", "description": "Exposure in ms."},
            "reason": _REASON,
        },
    },
    {
        "name": "retake_darks",
        "description": "Block the beam and take a fresh dark set with the current "
        "background. Use after background_changed. Costs one position.",
        "properties": {"reason": _REASON},
    },
    {
        "name": "set_status",
        "description": "Accept or reject a pending measurement. Only accepted "
        "measurements enter the fit. Saturated measurements cannot be accepted.",
        "properties": {
            "measurement_id": {"type": "string", "description": "e.g. m3"},
            "status": {"type": "string", "enum": ["accept", "reject"]},
            "reason": _REASON,
        },
    },
    {
        "name": "evaluate",
        "description": "Caustic fit per axis on the accepted measurements: d0 "
        "(um), z0 and zR (mm), theta (mrad), M² with uncertainties, ISO coverage "
        "per axis, waist in stage range, general astigmatism, M² < 1 flag, and a "
        "table of all measurements.",
        "properties": {"reason": _REASON},
    },
    {
        "name": "finish",
        "description": "End the run with a verdict and an explanation. The report "
        "numbers come from the measurements, not from you. iso_compliant is "
        "refused if ISO coverage is not met on both axes.",
        "properties": {
            "verdict": {"type": "string", "enum": list(tools.VERDICTS)},
            "explanation": {
                "type": "string",
                "description": "Short explanation of the verdict for the report.",
            },
            "reason": _REASON,
        },
    },
]
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": schema["name"],
        "description": schema["description"],
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": schema["properties"],
            "required": list(schema["properties"]),
            "additionalProperties": False,
        },
    }
    for schema in TOOL_SCHEMAS
]
TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_setup": tools.get_setup,
    "measure_at": tools.measure_at,
    "retake_darks": tools.retake_darks,
    "set_status": tools.set_status,
    "evaluate": tools.evaluate,
    "finish": tools.finish,
}


@dataclass
class AgentConfig:
    """Settings and hard limits of one agent run.

    Parameters
    ----------
    model : str
        Model ID. Default ``muse-spark-1.2``.
    reasoning_effort : str
        ``minimal``, ``low``, ``medium``, ``high`` or ``xhigh``. Default low.
    max_model_calls : int
        Hard limit on model calls per run. Default 40.
    max_total_tokens : int
        Token budget per run (input + output, cached input included).
        Default 1,500,000.
    max_cost_usd : float
        Cost budget per run in USD, from :data:`PRICES_PER_MTOK`. Default 0.50.
    max_output_tokens : int
        Output cap per model call (reasoning + visible). Default 8000.
    """

    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_EFFORT
    max_model_calls: int = 40
    max_total_tokens: int = 1_500_000
    max_cost_usd: float = 0.50
    max_output_tokens: int = 8000


@dataclass
class Usage:
    """Token counts of one call or a whole run."""

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.cached_tokens += other.cached_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens


@dataclass
class AgentRun:
    """Everything that happened in one run, ready to save as JSON."""

    config: AgentConfig
    steps: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    stop_reason: str = ""
    report: dict[str, Any] | None = None
    operator_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "operator_note": self.operator_note,
            "stop_reason": self.stop_reason,
            "model_calls": len(self.steps),
            "usage": {**asdict(self.usage), "total_tokens": self.usage.total_tokens},
            "cost_usd": round(self.cost_usd, 6),
            "report": self.report,
            "steps": self.steps,
        }


def system_prompt() -> str:
    """The system prompt from ``beampilot/prompts/system.md``."""
    return (resources.files("beampilot") / "prompts" / "system.md").read_text(
        encoding="utf-8"
    )


def make_client() -> Any:
    """OpenAI SDK client for the Meta Model API, key from ``.env``.

    Raises
    ------
    RuntimeError
        If ``MODEL_API_KEY`` is not set.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path.cwd() / ".env")
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("MODEL_API_KEY is not set (see .env.example)")
    # Retries are handled in run_agent (transient errors only), not by the SDK.
    return OpenAI(api_key=api_key, base_url=BASE_URL, max_retries=0)


def call_cost_usd(usage: Usage, model: str) -> float:
    """Estimated cost of `usage` in USD (check the pricing page)."""
    prices = PRICES_PER_MTOK["contributor" if "contributor" in model else "standard"]
    uncached = usage.input_tokens - usage.cached_tokens
    return (
        uncached * prices["input"]
        + usage.cached_tokens * prices["cached_input"]
        + usage.output_tokens * prices["output"]
    ) / 1e6


def run_agent(
    session: MeasurementSession,
    client: Any,
    config: AgentConfig | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_model_reply: Callable[[dict[str, Any]], None] | None = None,
    on_tool_call: Callable[[dict[str, Any]], None] | None = None,
    operator_note: str | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> AgentRun:
    """Run the agent until ``finish`` succeeds or a hard limit is hit.

    Every function call of a turn is executed through :mod:`beampilot.tools`
    in the order the model gave them, and all results go back in the next
    call. A turn without function calls gets a short nudge. Callbacks run in
    the calling thread; faults for a demo are injected separately with
    :meth:`MeasurementSession.request_faults`.

    Parameters
    ----------
    session : MeasurementSession
        The run's session (hidden truth stays inside it).
    client : object
        OpenAI-compatible client with ``responses.create`` (a fake in tests).
    config : AgentConfig, optional
        Model, effort and limits.
    on_step : callable, optional
        Called with each step's log entry after its tool calls ran.
    on_model_reply : callable, optional
        Called as soon as a model reply arrives, before its tool calls run,
        with the step entry (call number, usage, cost, text).
    on_tool_call : callable, optional
        Called after each tool call with its entry (name, arguments, reason,
        result).
    operator_note : str, optional
        Plain-language note from the person at the bench (e.g. a blocked z
        range). Added to the first message and logged; the tools do not
        enforce it.
    should_stop : callable, optional
        Checked before every model call (e.g. a cancel flag set when the demo
        user closes the tab); True stops the run like a hard limit.

    Returns
    -------
    AgentRun
        Steps, token usage, cost, stop reason and the report. If a hard limit
        or an API error stops the run, the controller writes the report
        (:func:`beampilot.tools.fallback_report`).
    """
    config = config or AgentConfig()
    note = (operator_note or "").strip() or None
    run = AgentRun(config=config, operator_note=note)
    instructions = system_prompt()
    task = TASK if note is None else f"{TASK}\n\n{OPERATOR_NOTE_PREFIX}{note}"
    # Stateless conversation: the full history is resent on every call, with
    # the reasoning replayed from its encrypted content. Server-stored
    # responses (previous_response_id) expired during parallel runs.
    history: list[dict[str, Any]] = [{"role": "user", "content": task}]

    for call_number in range(1, config.max_model_calls + 1):
        if should_stop is not None and should_stop():
            run.stop_reason = "cancelled by the controller"
            break
        if run.usage.total_tokens >= config.max_total_tokens:
            run.stop_reason = f"token budget reached ({run.usage.total_tokens})"
            break
        if run.cost_usd >= config.max_cost_usd:
            run.stop_reason = f"cost budget reached (${run.cost_usd:.4f})"
            break
        start = time.perf_counter()
        try:
            response = _create_with_retry(
                client,
                model=config.model,
                instructions=instructions,  # not kept across turns: resend
                input=list(history),
                tools=TOOL_DEFINITIONS,
                reasoning={"effort": config.reasoning_effort},
                max_output_tokens=config.max_output_tokens,
                store=False,
                include=["reasoning.encrypted_content"],
            )
        except Exception as exc:  # noqa: BLE001 - log and stop cleanly
            run.stop_reason = api_error_text(exc)
            break
        latency = time.perf_counter() - start
        usage = _usage(response)
        cost = call_cost_usd(usage, config.model)
        run.usage.add(usage)
        run.cost_usd += cost

        step: dict[str, Any] = {
            "call": call_number,
            "status": getattr(response, "status", None),
            "latency_s": round(latency, 3),
            "usage": asdict(usage),
            "cost_usd": round(cost, 6),
            "text": getattr(response, "output_text", "") or "",
            "tool_calls": [],
        }
        if on_model_reply is not None:
            on_model_reply(dict(step))
        outputs = []
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            entry, result = _execute(session, item)
            step["tool_calls"].append(entry)
            if on_tool_call is not None:
                on_tool_call(entry)
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )
        run.steps.append(step)
        if on_step is not None:
            on_step(step)

        if session.finished:
            run.stop_reason = "finished"
            run.report = session.report
            break
        history.extend(_replay_items(response.output))
        history.extend(outputs or [{"role": "user", "content": NUDGE}])
    else:
        run.stop_reason = f"model call limit reached ({config.max_model_calls})"
    if not session.finished:
        # The controller never trusts the agent to stop: write the report
        # from session state with a verdict assigned by rule.
        run.report = tools.fallback_report(session, run.stop_reason)
    return run


def save_run(run: AgentRun, path: Path, extra: dict[str, Any]) -> None:
    """Write the run (plus e.g. scenario, seed and truth) as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {**extra, **run.to_dict()}
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), "utf-8")


def api_error_text(exc: BaseException) -> str:
    """Safe description of an API error: type and HTTP status only.

    The message text is left out on purpose: it comes from the server and could
    contain (parts of) the API key, and it ends up in the UI, the report and
    the saved run file.
    """
    status = getattr(exc, "status_code", None)
    suffix = f" (HTTP {status})" if isinstance(status, int) else ""
    return f"API error: {type(exc).__name__}{suffix}"


def _create_with_retry(client: Any, **request: Any) -> Any:
    """``responses.create`` with up to 2 retries on transient errors."""
    for attempt in range(RETRIES + 1):
        try:
            return client.responses.create(**request)
        except Exception as exc:
            transient = type(exc).__name__ in TRANSIENT_ERRORS
            if not transient or attempt == RETRIES:
                raise
            time.sleep(RETRY_DELAY_S * (attempt + 1))
    raise AssertionError("unreachable")


def _replay_items(output: list[Any]) -> list[dict[str, Any]]:
    """Output items of a response as input items for the next request.

    Reasoning is replayed from its encrypted content (``summary`` is required,
    ``[]`` if empty). The API requires a reasoning item to be followed by an
    assistant message or a function call before the next user message, so a
    reasoning-only turn gets a minimal assistant message.
    """
    items: list[dict[str, Any]] = []
    for item in output:
        kind = getattr(item, "type", None)
        if kind == "reasoning":
            encrypted = getattr(item, "encrypted_content", None)
            if encrypted:
                summary = [
                    {"type": "summary_text", "text": part.text}
                    for part in getattr(item, "summary", None) or []
                ]
                items.append(
                    {
                        "type": "reasoning",
                        "summary": summary,
                        "encrypted_content": encrypted,
                    }
                )
        elif kind == "message":
            text = "".join(
                getattr(part, "text", "") or "" for part in getattr(item, "content", [])
            )
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            )
        elif kind == "function_call":
            items.append(
                {
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                }
            )
    if items and items[-1]["type"] == "reasoning":
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "(thinking)"}],
            }
        )
    return items


def _execute(session: MeasurementSession, item: Any) -> tuple[dict, dict]:
    """Run one function call through the tools; never raises."""
    name = item.name
    try:
        arguments = json.loads(item.arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
    except ValueError as exc:
        result = {"error": f"arguments are not valid JSON: {exc}"}
        return {"name": name, "arguments": item.arguments, "result": result}, result
    reason = arguments.get("reason")
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        result = {"error": f"unknown tool {name!r}; tools: {', '.join(TOOL_FUNCTIONS)}"}
    else:
        call_arguments = dict(arguments)
        if name == "finish":
            # finish logs its explanation as the reason; the schema's extra
            # reason is kept in this agent log only.
            call_arguments.pop("reason", None)
        result = function(session, **call_arguments)
    entry = {"name": name, "arguments": arguments, "reason": reason, "result": result}
    return entry, result


def _usage(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return Usage(
        input_tokens=usage.input_tokens or 0,
        cached_tokens=getattr(input_details, "cached_tokens", 0) or 0,
        output_tokens=usage.output_tokens or 0,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) or 0,
    )
