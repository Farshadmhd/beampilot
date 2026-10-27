"""Agent loop tests with a fake client: no API calls."""

import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from beampilot.agent import (
    IMAGE_STUB,
    NUDGE,
    TOOL_DEFINITIONS,
    AgentConfig,
    Usage,
    call_cost_usd,
    run_agent,
    system_prompt,
)
from beampilot.session import MeasurementSession, clean_scenario, reveal_truth
from beampilot.tools import VERDICTS


def _call(name: str, **arguments) -> SimpleNamespace:
    _call.counter += 1
    return SimpleNamespace(
        type="function_call",
        name=name,
        call_id=f"call_{_call.counter}",
        arguments=json.dumps({"reason": "test", **arguments}),
    )


_call.counter = 0


def _response(*items, text="", tokens=(1000, 200, 100, 0)) -> SimpleNamespace:
    input_tokens, output_tokens, reasoning, cached = tokens
    _response.counter += 1
    return SimpleNamespace(
        id=f"resp_{_response.counter}",
        status="completed",
        output=list(items),
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached),
            output_tokens=output_tokens,
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
    )


_response.counter = 0


class FakeClient:
    """Returns scripted responses and records every request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("the loop asked for more responses than scripted")
        response = self._responses.pop(0)
        if callable(response):
            return response()
        return response


def _session() -> MeasurementSession:
    return MeasurementSession(clean_scenario())


def test_loop_runs_tool_calls_and_stops_after_finish() -> None:
    client = FakeClient(
        [
            _response(_call("get_setup")),
            _response(
                _call("measure_at", z_mm=120.0, exposure_ms=1.5),
                _call("set_status", measurement_id="m1", status="accept"),
            ),
            _response(
                _call("finish", verdict="measurement_failed", explanation="test run")
            ),
            _response(_call("get_setup")),  # must never be requested
        ]
    )
    session = _session()
    run = run_agent(session, client)
    assert run.stop_reason == "finished"
    assert len(client.requests) == 3
    assert session.finished
    assert run.report["verdict"] == "measurement_failed"
    assert session.measurements["m1"].status == "accept"
    # The two parallel calls ran in order and were both logged.
    names = [c["name"] for c in run.steps[1]["tool_calls"]]
    assert names == ["measure_at", "set_status"]
    assert run.steps[1]["tool_calls"][0]["reason"] == "test"


def _reasoning(encrypted="enc-abc") -> SimpleNamespace:
    return SimpleNamespace(type="reasoning", summary=[], encrypted_content=encrypted)


def _outputs(request: dict) -> list[dict]:
    return [i for i in request["input"] if i.get("type") == "function_call_output"]


def test_requests_follow_the_stateless_responses_protocol() -> None:
    first = _call("get_setup")
    client = FakeClient(
        [
            _response(_reasoning(), first),
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    run_agent(_session(), client, AgentConfig(reasoning_effort="medium"))
    one, two = client.requests
    instructions = system_prompt()
    assert one["instructions"] == two["instructions"] == instructions
    for request in (one, two):
        assert request["store"] is False
        assert request["include"] == ["reasoning.encrypted_content"]
        assert "previous_response_id" not in request
    assert one["reasoning"] == {"effort": "medium"}
    assert one["model"] == "muse-spark-1.2"
    # Full history: task, replayed reasoning, the call, then its output.
    task, reasoning, call, output = two["input"]
    assert task["role"] == "user"
    assert reasoning == {
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "enc-abc",
    }
    assert call["type"] == "function_call" and call["call_id"] == first.call_id
    assert output["type"] == "function_call_output"
    assert output["call_id"] == first.call_id
    assert json.loads(output["output"])["wavelength_nm"] == 1064.0


def test_tool_error_goes_back_to_the_model() -> None:
    client = FakeClient(
        [
            _response(_call("measure_at", z_mm=999.0, exposure_ms=1.0)),
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    run = run_agent(_session(), client)
    sent = json.loads(_outputs(client.requests[1])[0]["output"])
    assert "stage range" in sent["error"]
    assert run.stop_reason == "finished"


def test_bad_arguments_and_unknown_tools_become_errors() -> None:
    broken = SimpleNamespace(
        type="function_call", name="measure_at", call_id="c1", arguments="{not json"
    )
    unknown = SimpleNamespace(
        type="function_call", name="fire_laser", call_id="c2", arguments="{}"
    )
    client = FakeClient(
        [
            _response(broken, unknown),
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    run_agent(_session(), client)
    outputs = [json.loads(o["output"]) for o in _outputs(client.requests[1])]
    assert "not valid JSON" in outputs[0]["error"]
    assert "unknown tool" in outputs[1]["error"]


def test_refused_finish_keeps_the_loop_going() -> None:
    client = FakeClient(
        [
            _response(
                _call("finish", verdict="iso_compliant", explanation="too early")
            ),
            _response(_call("finish", verdict="measurement_failed", explanation="ok")),
        ]
    )
    run = run_agent(_session(), client)
    assert (
        "cannot report"
        in json.loads(_outputs(client.requests[1])[0]["output"])["error"]
    )
    assert run.stop_reason == "finished"
    assert run.report["verdict"] == "measurement_failed"


def test_text_only_reply_gets_a_nudge() -> None:
    message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text="I think M² is 1.2.")],
    )
    client = FakeClient(
        [
            _response(message, text="I think M² is 1.2."),
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    run_agent(_session(), client)
    _, replayed, nudge = client.requests[1]["input"]
    assert replayed["role"] == "assistant"
    assert replayed["content"][0]["text"] == "I think M² is 1.2."
    assert nudge == {"role": "user", "content": NUDGE}


def test_reasoning_only_turn_is_bridged_before_the_nudge() -> None:
    """The API wants an assistant message between reasoning and a user turn."""
    client = FakeClient(
        [
            _response(_reasoning()),
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    run_agent(_session(), client)
    _, reasoning, bridge, nudge = client.requests[1]["input"]
    assert reasoning["type"] == "reasoning"
    assert bridge["type"] == "message" and bridge["role"] == "assistant"
    assert nudge["content"] == NUDGE


def test_transient_api_error_is_retried(monkeypatch) -> None:
    monkeypatch.setattr("beampilot.agent.RETRY_DELAY_S", 0.0)

    class RateLimitError(Exception):
        pass

    def busy():
        raise RateLimitError("slow down")

    client = FakeClient(
        [
            busy,
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    run = run_agent(_session(), client)
    assert run.stop_reason == "finished"
    assert len(client.requests) == 2


def test_stops_at_the_model_call_limit() -> None:
    client = FakeClient([_response(_call("get_setup")) for _ in range(10)])
    session = _session()
    run = run_agent(session, client, AgentConfig(max_model_calls=5))
    assert len(client.requests) == 5
    assert run.stop_reason == "model call limit reached (5)"
    # The controller writes a fallback report from session state.
    assert session.finished
    assert run.report["verdict"] == "measurement_failed"
    assert run.report["stopped_by_limit"] == "model call limit reached (5)"
    assert "Stopped by the controller" in run.report["explanation"]


def test_fallback_report_uses_determined_m2() -> None:
    """Stopped after enough good data: estimate_not_compliant, not failed."""
    positions = [60.0, 100.0, 120.0, 140.0, 180.0]
    calls = [
        _response(_call("measure_at", z_mm=z, exposure_ms=exposure))
        for z, exposure in zip(positions, [10.0, 1.5, 1.2, 1.5, 9.0], strict=True)
    ]
    accepts = [
        _response(_call("set_status", measurement_id=f"m{i}", status="accept"))
        for i in range(1, 6)
    ]
    client = FakeClient(calls + accepts)
    session = _session()
    run = run_agent(session, client, AgentConfig(max_model_calls=10))
    assert run.report["verdict"] == "estimate_not_compliant"
    assert run.report["evaluation"]["x"]["m2"] is not None
    assert run.report["stopped_by_limit"].startswith("model call limit")


def test_stops_at_the_token_budget() -> None:
    big = (60_000, 1_000, 500, 0)
    client = FakeClient([_response(_call("get_setup"), tokens=big) for _ in range(10)])
    run = run_agent(_session(), client, AgentConfig(max_total_tokens=150_000))
    assert len(client.requests) == 3  # 61k, 122k, 183k tokens -> stop
    assert run.stop_reason.startswith("token budget reached")


def test_stops_at_the_cost_budget() -> None:
    pricey = (100_000, 10_000, 0, 0)  # 0.125 + 0.0425 = $0.1675 per call
    client = FakeClient(
        [_response(_call("get_setup"), tokens=pricey) for _ in range(10)]
    )
    run = run_agent(_session(), client, AgentConfig(max_cost_usd=0.40))
    assert len(client.requests) == 3
    assert run.stop_reason.startswith("cost budget reached")
    assert run.cost_usd == pytest.approx(3 * 0.1675)


def test_api_error_stops_cleanly() -> None:
    def fail():
        raise ConnectionError("network down")

    client = FakeClient([fail])
    run = run_agent(_session(), client)
    # Type only: the message text could carry server text (e.g. a masked key).
    assert run.stop_reason == "API error: ConnectionError"
    assert run.steps == []
    assert run.report["verdict"] == "measurement_failed"
    assert run.report["stopped_by_limit"] == run.stop_reason


def test_hooks_fire_per_reply_and_per_tool_call_in_order() -> None:
    events = []
    client = FakeClient(
        [
            _response(_call("get_setup"), _call("evaluate"), text="looking"),
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    run_agent(
        _session(),
        client,
        on_model_reply=lambda step: events.append(
            ("reply", step["call"], step["text"])
        ),
        on_tool_call=lambda call: events.append(("tool", call["name"])),
        on_step=lambda step: events.append(("step", step["call"])),
    )
    assert events == [
        ("reply", 1, "looking"),
        ("tool", "get_setup"),
        ("tool", "evaluate"),
        ("step", 1),
        ("reply", 2, ""),
        ("tool", "finish"),
        ("step", 2),
    ]


def test_operator_note_reaches_the_agent_and_the_run_file() -> None:
    client = FakeClient(
        [_response(_call("finish", verdict="measurement_failed", explanation="x"))]
    )
    run = run_agent(_session(), client, operator_note="  avoid z 80-100 mm ")
    first = client.requests[0]["input"][0]["content"]
    assert first.endswith("avoid z 80-100 mm")
    assert "do not enforce it" in first
    assert run.operator_note == "avoid z 80-100 mm"
    assert run.to_dict()["operator_note"] == "avoid z 80-100 mm"
    no_note = run_agent(
        _session(),
        FakeClient(
            [_response(_call("finish", verdict="measurement_failed", explanation="x"))]
        ),
    )
    assert no_note.operator_note is None


def test_fault_injected_during_a_run_hits_the_next_measurement() -> None:
    """A demo button calls request_faults from another thread while the loop
    runs; the next measure_at sees it."""
    session = _session()

    def press_button_then_measure():
        threading.Thread(
            target=session.request_faults, kwargs={"extra_background_counts": 3.0}
        ).start()
        time.sleep(0.05)
        return _response(_call("measure_at", z_mm=120.0, exposure_ms=1.5))

    client = FakeClient(
        [
            _response(_call("measure_at", z_mm=120.0, exposure_ms=1.5)),
            press_button_then_measure,
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    run = run_agent(session, client)
    first, second = (s["tool_calls"][0]["result"] for s in run.steps[:2])
    assert not first["flags"]["background_changed"]
    assert second["flags"]["background_changed"]
    assert session.fault_events[0]["source"] == "operator"


def test_api_error_text_has_type_and_status_but_no_message() -> None:
    class AuthenticationError(Exception):
        status_code = 401

    def fail():
        raise AuthenticationError("Incorrect API key provided: sk-SECRET123456")

    run = run_agent(_session(), FakeClient([fail]))
    assert run.stop_reason == "API error: AuthenticationError (HTTP 401)"
    assert "SECRET" not in json.dumps(run.to_dict())


def test_should_stop_cancels_before_the_next_model_call() -> None:
    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    client = FakeClient([_response(_call("get_setup")) for _ in range(5)])
    run = run_agent(_session(), client, should_stop=stop_after_two)
    assert len(client.requests) == 2
    assert run.stop_reason == "cancelled by the controller"
    assert run.report["stopped_by_limit"] == "cancelled by the controller"


def test_view_frame_image_reaches_the_model_but_not_the_log() -> None:
    client = FakeClient(
        [
            _response(_call("measure_at", z_mm=120.0, exposure_ms=1.5)),
            _response(_call("view_frame", measurement_id="m1")),
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    session = _session()
    run = run_agent(session, client)
    output = _outputs(client.requests[2])[-1]["output"]
    assert isinstance(output, list)
    text, image = output
    assert json.loads(text["text"])["image_id"] == "view1-m1"
    assert image["type"] == "input_image"
    assert image["image_url"].startswith("data:image/png;base64,")
    assert session.images == {}  # handed over once, not kept
    logged = json.dumps(run.to_dict()) + json.dumps(session.log)
    assert "base64" not in logged


def test_requests_never_contain_the_hidden_truth() -> None:
    client = FakeClient(
        [
            _response(_call("get_setup")),
            _response(_call("measure_at", z_mm=120.0, exposure_ms=1.5)),
            _response(_call("evaluate")),
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    session = MeasurementSession(replace(clean_scenario(), name="secret-name-7f3a"))
    run_agent(session, client)
    sent = json.dumps(client.requests, default=str, ensure_ascii=False)
    # Measured values are legitimately close to the truth, so check that the
    # scenario identity and the ground-truth fields never leave the session.
    assert "secret-name-7f3a" not in sent
    for key in reveal_truth(session):
        if key not in ("scenario", "wavelength_nm"):  # wavelength is known setup
            assert key not in sent, key


def test_calls_after_a_successful_finish_are_refused() -> None:
    client = FakeClient(
        [
            _response(
                _call("finish", verdict="measurement_failed", explanation="x"),
                _call("get_setup"),
            ),
        ]
    )
    run = run_agent(_session(), client)
    assert run.stop_reason == "finished"
    assert "finished" in run.steps[0]["tool_calls"][1]["result"]["error"]


def test_tool_definitions_are_strict_with_a_required_reason() -> None:
    names = {tool["name"] for tool in TOOL_DEFINITIONS}
    assert names == {
        "get_setup",
        "measure_at",
        "retake_darks",
        "view_frame",
        "set_status",
        "evaluate",
        "finish",
    }
    for tool in TOOL_DEFINITIONS:
        parameters = tool["parameters"]
        assert tool["type"] == "function" and tool["strict"] is True
        assert parameters["additionalProperties"] is False
        assert "reason" in parameters["required"]
        assert set(parameters["required"]) == set(parameters["properties"])
    finish = next(t for t in TOOL_DEFINITIONS if t["name"] == "finish")
    assert finish["parameters"]["properties"]["verdict"]["enum"] == list(VERDICTS)


def test_cost_uses_cached_and_contributor_prices() -> None:
    usage = Usage(input_tokens=1_000_000, cached_tokens=500_000, output_tokens=100_000)
    standard = 0.5 * 1.25 + 0.5 * 0.15 + 0.1 * 4.25
    contributor = 0.5 * 0.10 + 0.5 * 0.002 + 0.1 * 0.20
    assert call_cost_usd(usage, "muse-spark-1.2") == pytest.approx(standard)
    assert call_cost_usd(usage, "muse-spark-1.2-contributor") == pytest.approx(
        contributor
    )


def test_system_prompt_is_short_and_complete() -> None:
    prompt = system_prompt()
    assert len(prompt.split()) < 1500
    for word in ("ISO 11146-1", "reason", "retake_darks", "view_frame", *VERDICTS):
        assert word in prompt


def test_only_the_two_latest_images_are_resent() -> None:
    client = FakeClient(
        [
            _response(
                _call("measure_at", z_mm=100.0, exposure_ms=1.5),
                _call("measure_at", z_mm=120.0, exposure_ms=1.5),
                _call("measure_at", z_mm=140.0, exposure_ms=1.5),
            ),
            _response(_call("view_frame", measurement_id="m1")),
            _response(_call("view_frame", measurement_id="m2")),
            _response(_call("view_frame", measurement_id="m3")),
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    run_agent(_session(), client)
    last = _outputs(client.requests[-1])
    images = [o for o in last if isinstance(o["output"], list)]
    stubs = [
        o for o in last if isinstance(o["output"], str) and IMAGE_STUB in o["output"]
    ]
    assert len(images) == 2
    assert len(stubs) == 1 and '"view1-m1"' in stubs[0]["output"]
    # Earlier requests keep what was actually sent then.
    assert sum(isinstance(o["output"], list) for o in _outputs(client.requests[3])) == 2


def test_view_frame_can_be_switched_off_for_the_ablation() -> None:
    client = FakeClient(
        [
            _response(
                _call("get_setup"), _call("measure_at", z_mm=120.0, exposure_ms=1.5)
            ),
            _response(_call("view_frame", measurement_id="m1")),
            _response(_call("finish", verdict="measurement_failed", explanation="x")),
        ]
    )
    session = _session()
    run = run_agent(session, client, AgentConfig(view_frame=False))
    request = client.requests[0]
    assert "view_frame" not in {t["name"] for t in request["tools"]}
    assert "view_frame" not in request["instructions"]
    assert "# How to work" in request["instructions"]
    assert "views_left" not in run.steps[0]["tool_calls"][0]["result"]
    refused = run.steps[1]["tool_calls"][0]["result"]
    assert (
        "unknown tool" in refused["error"] and "view_frame" not in refused["error"][20:]
    )
    assert session.views_used == 0
    assert "Looking at frames" in system_prompt()
