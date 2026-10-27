"""Tests for the demo pieces: replay, plots and the Gradio app (not launched)."""

import json
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
