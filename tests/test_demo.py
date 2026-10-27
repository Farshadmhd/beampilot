"""Tests for the demo pieces: replay, plots and the Gradio app (not launched)."""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure
from test_agent import FakeClient, _call, _response

from beampilot import app as demo_app
from beampilot.agent import run_agent, save_run
from beampilot.plots import caustic_figure, frame_image
from beampilot.replay import RunState, load_run, replay_events
from beampilot.session import MeasurementSession, clean_scenario, session_record

SCRIPT = [
    ("get_setup", {}),
    ("measure_at", {"z_mm": 60.0, "exposure_ms": 10.0}),
    ("measure_at", {"z_mm": 100.0, "exposure_ms": 1.5}),
    ("measure_at", {"z_mm": 120.0, "exposure_ms": 1.2}),
    ("measure_at", {"z_mm": 140.0, "exposure_ms": 1.5}),
    ("measure_at", {"z_mm": 180.0, "exposure_ms": 9.0}),
    *[
        ("set_status", {"measurement_id": f"m{i}", "status": "accept"})
        for i in range(1, 6)
    ],
    ("evaluate", {}),
    ("finish", {"verdict": "estimate_not_compliant", "explanation": "test run"}),
]


def _scripted_client() -> FakeClient:
    return FakeClient([_response(_call(name, **args)) for name, args in SCRIPT])


@pytest.fixture
def saved_run(tmp_path):
    session = MeasurementSession(clean_scenario())
    run = run_agent(session, _scripted_client(), operator_note="avoid 60-80 mm")
    path = tmp_path / "run.json"
    save_run(run, path, {"scenario": "clean", "seed": 0, **session_record(session)})
    return path


def _play(run) -> RunState:
    state = RunState()
    for kind, item in replay_events(run):
        state.apply_model_reply(item) if kind == "reply" else state.apply_tool_call(
            item
        )
    return state


def test_replay_rebuilds_the_run(saved_run) -> None:
    run = load_run(saved_run)
    assert "estimate_not_compliant" in run.title
    state = _play(run)
    assert state.model_calls == len(SCRIPT)
    assert len(state.points) == 5
    assert all(p.status == "accept" for p in state.points.values())
    assert state.report["verdict"] == "estimate_not_compliant"
    assert state.evaluation["x"]["m2"] is not None
    assert state.positions_left == 15
    assert any("**measure_at**" in line for line in state.log)
    assert run.data["operator_note"] == "avoid 60-80 mm"


def test_load_run_rejects_other_files(tmp_path) -> None:
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a BeamPilot run"):
        load_run(other)


def test_caustic_figure_and_frame_image(saved_run) -> None:
    state = _play(load_run(saved_run))
    fig = caustic_figure(state, (0.0, 250.0))
    assert isinstance(fig, Figure)
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert any(label.startswith("fit x") for label in labels)
    assert isinstance(caustic_figure(RunState()), Figure)  # empty run
    image = frame_image(np.array([[0, 2000], [4095, 9000]]), 4095)
    assert image.shape == (2, 2, 3) and image.dtype == np.uint8


def test_replay_generator_ends_with_the_report(saved_run) -> None:
    updates = list(demo_app.replay(None, str(saved_run), 0.0))
    title, plot, image, budget, log, report, truth, run = updates[-1]
    assert "Replay" in title
    assert report["verdict"] == "estimate_not_compliant"
    assert truth is None  # only shown after "Reveal truth"
    assert run.truth["m2_x"] == 1.2
    assert image is not None and image.shape[2] == 3  # re-simulated frame


def test_replay_logs_fault_events() -> None:
    path = demo_app.DEMO_RUNS_DIR / "clean_room_light_live.json"
    if not path.exists():
        pytest.skip("demo run not shipped")
    *_, log, _report, _truth, _run = list(demo_app.replay(None, str(path), 0.0))[-1]
    assert "💡 **Fault (operator button)** from measurement 6" in log


def test_shipped_demo_runs_load() -> None:
    for path in demo_app.demo_runs():
        run = load_run(path)
        assert run.truth and run.data["report"]


def test_live_run_streams_and_reveals_after_the_end(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(demo_app, "_client", _scripted_client)
    monkeypatch.setattr(demo_app, "LIVE_RUNS_DIR", tmp_path)
    request = SimpleNamespace(session_hash="test-live")
    demo_app._BROWSERS.pop("test-live", None)
    updates = list(demo_app.live_run("clean", 0, "", request))
    plot, image, budget, log, report, truth = updates[-1]
    assert report["verdict"] == "estimate_not_compliant"
    assert image is not None
    assert "Positions left:** 15" in budget
    assert "Saved as" in log
    assert list(tmp_path.glob("*.json"))
    revealed = demo_app.reveal_live(request)
    assert revealed["m2_x"] == 1.2
    assert "No run in progress" in demo_app.inject_fault(request, "room_light")


def test_fault_button_reaches_a_running_session() -> None:
    request = SimpleNamespace(session_hash="test-fault")
    browser = demo_app._browser(request)
    browser.session = MeasurementSession(clean_scenario())
    browser.active = True  # as during a live run
    message = demo_app.inject_fault(request, "laser_power")
    assert "x3" in message
    browser.session.take_frames(0.1, 0.1)
    assert browser.session.faults.power_factor == 3.0
    assert "truth is revealed after" in demo_app.reveal_live(request)["message"]


def test_live_runs_are_limited_per_browser_session(monkeypatch) -> None:
    request = SimpleNamespace(session_hash="test-limit")
    browser = demo_app._browser(request)
    browser.runs = demo_app.MAX_RUNS_PER_SESSION
    update = next(demo_app.live_run("clean", 0, "", request))
    assert "Run limit reached" in update[2]
    browser.runs, browser.tokens = 0, demo_app.SESSION_TOKEN_BUDGET
    assert "Token budget" in next(demo_app.live_run("clean", 0, "", request))[2]


def test_build_app_without_launching() -> None:
    pytest.importorskip("gradio")
    app = demo_app.build_app()
    assert app is not None


# Robustness and limits of the live path (review findings)


@pytest.fixture(autouse=True)
def fresh_global_budget(monkeypatch):
    monkeypatch.setattr(demo_app, "_GLOBAL", demo_app.GlobalBudget())


def _slow_client(n=20, delay=0.15) -> FakeClient:
    def slow():
        time.sleep(delay)
        return _response(_call("get_setup"))

    return FakeClient([slow] * n)


def _wait_until_idle(browser, timeout=15.0) -> None:
    end = time.monotonic() + timeout
    while browser.active and time.monotonic() < end:
        time.sleep(0.05)
    assert not browser.active, "worker did not finish"


def test_closing_the_tab_cancels_and_still_accounts(monkeypatch, tmp_path) -> None:
    client = _slow_client()
    monkeypatch.setattr(demo_app, "_client", lambda: client)
    monkeypatch.setattr(demo_app, "LIVE_RUNS_DIR", tmp_path)
    request = SimpleNamespace(session_hash="test-close")
    demo_app._BROWSERS.pop("test-close", None)
    stream = demo_app.live_run("clean", 0, "", request)
    next(stream)  # the first model reply arrived
    stream.close()  # the browser went away
    browser = demo_app._browser(request)
    _wait_until_idle(browser)
    assert len(client.requests) < 20  # stopped early, not run to the end
    assert browser.tokens > 0  # counted in the worker, per reply
    assert browser.session.report["stopped_by_limit"].startswith(
        "cancelled by the controller"
    )
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert demo_app._GLOBAL.active == 0


def test_parallel_starts_in_one_tab_run_only_once(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(demo_app, "_client", lambda: _slow_client(n=6, delay=0.2))
    monkeypatch.setattr(demo_app, "LIVE_RUNS_DIR", tmp_path)
    request = SimpleNamespace(session_hash="test-race")
    demo_app._BROWSERS.pop("test-race", None)
    results = []

    def start():
        results.append(list(demo_app.live_run("clean", 0, "", request)))

    threads = [threading.Thread(target=start) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    refused = [r for r in results if "already in progress" in r[-1][2]]
    assert len(refused) == 3
    assert demo_app._browser(request).runs == 1


def test_process_wide_cap_holds_for_new_session_ids(monkeypatch) -> None:
    monkeypatch.setattr(demo_app._GLOBAL, "runs", demo_app.GLOBAL_MAX_RUNS)
    for session_id in ("invented-1", "invented-2"):
        request = SimpleNamespace(session_hash=session_id)
        update = next(demo_app.live_run("clean", 0, "", request))
        assert "used up its live runs" in update[2]
    monkeypatch.setattr(demo_app._GLOBAL, "runs", 0)
    monkeypatch.setattr(demo_app._GLOBAL, "active", demo_app.MAX_CONCURRENT_RUNS)
    update = next(demo_app.live_run("clean", 0, "", SimpleNamespace(session_hash="x")))
    assert "busy" in update[2]


def test_exception_outside_the_agent_loop_still_reports(monkeypatch, tmp_path) -> None:
    broken = SimpleNamespace(
        id="r", status="completed", output=None, output_text="", usage=None
    )
    client = FakeClient([_response(_call("get_setup")), broken])
    monkeypatch.setattr(demo_app, "_client", lambda: client)
    monkeypatch.setattr(demo_app, "LIVE_RUNS_DIR", tmp_path)
    request = SimpleNamespace(session_hash="test-broken")
    demo_app._BROWSERS.pop("test-broken", None)
    *_, (plot, image, budget, log, report, truth) = demo_app.live_run(
        "clean", 0, "", request
    )
    assert report["verdict"] == "measurement_failed"
    assert report["stopped_by_limit"] == "controller error (TypeError)"
    browser = demo_app._browser(request)
    assert browser.tokens > 0 and not browser.active
    assert list(tmp_path.glob("*.json"))
    assert demo_app.reveal_live(request)["m2_x"] == 1.2


def test_api_error_text_never_reaches_ui_report_or_file(monkeypatch, tmp_path) -> None:
    class AuthenticationError(Exception):
        status_code = 401

    def fail():
        raise AuthenticationError("Incorrect API key provided: sk-SECRET**1234")

    monkeypatch.setattr(demo_app, "_client", lambda: FakeClient([fail]))
    monkeypatch.setattr(demo_app, "LIVE_RUNS_DIR", tmp_path)
    request = SimpleNamespace(session_hash="test-secret")
    demo_app._BROWSERS.pop("test-secret", None)
    updates = list(demo_app.live_run("clean", 0, "", request))
    shown = json.dumps(updates[-1][2:], default=str)
    assert "SECRET" not in shown
    assert "AuthenticationError (HTTP 401)" in shown
    for path in tmp_path.glob("*.json"):
        assert "SECRET" not in path.read_text(encoding="utf-8")


def test_bad_uploads_give_a_message_not_a_crash(saved_run, tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"steps": [{"tool_calls": [{"x": 1}]}]}), "utf-8")
    title = list(demo_app.replay(None, str(broken), 0.0))[-1][0]
    assert "Cannot replay this file" in title

    huge = json.loads(saved_run.read_text(encoding="utf-8"))
    huge["scenario_params"]["camera"]["nx"] = 20000  # would need GBs
    huge_path = tmp_path / "huge.json"
    huge_path.write_text(json.dumps(huge), "utf-8")
    last = list(demo_app.replay(None, str(huge_path), 0.0))[-1]
    assert last[2] is None  # no frame rendered
    assert last[5]["verdict"] == "estimate_not_compliant"


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.__setitem__("report", "x"),
        lambda d: d["scenario_params"]["camera"].__setitem__("bit_depth", 4_000_000),
    ],
    ids=["report-not-a-dict", "absurd-bit-depth"],
)
def test_odd_uploads_never_raise(saved_run, tmp_path, change) -> None:
    data = json.loads(saved_run.read_text(encoding="utf-8"))
    change(data)
    path = tmp_path / "odd.json"
    path.write_text(json.dumps(data), "utf-8")
    updates = list(demo_app.replay(None, str(path), 0.0))  # must not raise
    assert updates


def test_failed_starts_do_not_use_up_the_run_limits(monkeypatch) -> None:
    def no_key():
        raise RuntimeError("MODEL_API_KEY is not set")

    monkeypatch.setattr(demo_app, "_client", no_key)
    request = SimpleNamespace(session_hash="test-nokey")
    demo_app._BROWSERS.pop("test-nokey", None)
    for _ in range(5):
        assert "MODEL_API_KEY" in next(demo_app.live_run("clean", 0, "", request))[2]
    assert demo_app._browser(request).runs == 0
    assert demo_app._GLOBAL.runs == 0 and demo_app._GLOBAL.active == 0


def test_closing_the_tab_forgets_the_browser_state() -> None:
    request = SimpleNamespace(session_hash="test-unload")
    browser = demo_app._browser(request)
    demo_app._forget_browser(request)
    assert browser.cancel.is_set()
    assert "test-unload" not in demo_app._BROWSERS


def test_bad_seed_or_scenario_is_refused() -> None:
    request = SimpleNamespace(session_hash="test-input")
    assert "between 0 and" in next(demo_app.live_run("clean", -3, "", request))[2]
    assert "Unknown scenario" in next(demo_app.live_run("nope", 0, "", request))[2]
