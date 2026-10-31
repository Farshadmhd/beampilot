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
    title, plot, image, budget, log, card, report, truth, run = updates[-1]
    assert "Recorded run" in title and "not a live run" in title
    assert report["verdict"] == "estimate_not_compliant"
    assert "estimate_not_compliant" in card
    assert truth == ""  # only shown after "Reveal truth"
    assert run.truth["m2_x"] == 1.2
    assert image is not None and image.shape[2] == 3  # re-simulated frame


def test_replay_logs_fault_events() -> None:
    path = demo_app.DEMO_RUNS_DIR / "clean_room_light_live.json"
    if not path.exists():
        pytest.skip("demo run not shipped")
    *_, log, _card, _report, _truth, _run = list(demo_app.replay(None, str(path), 0.0))[
        -1
    ]
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
    plot, image, budget, log, card, report, truth = updates[-1]
    assert report["verdict"] == "estimate_not_compliant"
    assert "M²" in card and "Positions:** 5 used" in card
    assert image is not None
    assert "Positions left:** 15" in budget
    assert "Saved as" in log
    assert list(tmp_path.glob("*.json"))
    revealed = demo_app.reveal_live(request)
    assert revealed["m2_x"] == 1.2
    table = demo_app.reveal_live_table(request)
    assert "| M² | x |" in table and "| 1.200 |" in table
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
    for name in ("BEAMPILOT_LIVE_RUNS", "BEAMPILOT_SPEND_CAP_USD"):
        monkeypatch.delenv(name, raising=False)
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
    *_, (plot, image, budget, log, card, report, truth) = demo_app.live_run(
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
    assert last[6]["verdict"] == "estimate_not_compliant"


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


# Video polish: grouped status lines, coverage, results card, truth table, cap


def test_consecutive_set_status_calls_share_one_log_line(saved_run) -> None:
    state = _play(load_run(saved_run))
    status_lines = [line for line in state.log if line.startswith("**set_status**")]
    assert len(status_lines) == 1
    assert status_lines[0].count("✅") == 5 and "m5 ✅" in status_lines[0]
    # A different call in between starts a new line.
    state.apply_tool_call({"name": "get_setup", "arguments": {}, "result": {}})
    state.apply_tool_call(
        {
            "name": "set_status",
            "arguments": {"measurement_id": "m1", "reason": "bad"},
            "result": {"measurement_id": "m1", "status": "reject"},
        }
    )
    assert state.log[-1] == "**set_status**: m1 ❌ bad"


def test_coverage_counts_accepted_points_against_the_latest_fit() -> None:
    state = RunState()
    assert state.coverage() is None
    assert "after the first fit" in demo_app.coverage_text(state)
    for i, z in enumerate((100.0, 120.0, 130.0, 60.0, 190.0, 200.0), start=1):
        state.apply_tool_call(
            {
                "name": "measure_at",
                "arguments": {},
                "result": {
                    "measurement_id": f"m{i}",
                    "z_mm": z,
                    "exposure_ms": 1.0,
                    "d_x_um": {"mean": 300.0},
                    "d_y_um": {"mean": 300.0},
                },
            }
        )
    fit = {"z0_mm": 120.0, "z_r_mm": 25.0, "m2": 1.2}
    state.evaluation = {"status": "ok", "x": fit, "y": dict(fit, z_r_mm=40.0)}
    for i in (1, 2, 3, 4, 5):
        state.points[f"m{i}"].status = "accept"
    # x: near 100, 120, 130; far 60, 190 (>= 50 mm away). y (zR 40): none >= 80.
    assert state.coverage() == {"x": (3, 2), "y": (3, 0)}
    text = demo_app.coverage_text(state)
    assert "**x** near ⏳ 3/5, far ⏳ 2/5" in text


def test_results_card_and_truth_table_handle_missing_values() -> None:
    state = RunState(max_positions=20, positions_left=8, cost_usd=0.1234)
    state.report = {
        "verdict": "estimate_not_compliant",
        "explanation": "budget ran out",
        "positions_accepted": 10,
        "positions_rejected": 2,
        "evaluation": {
            "status": "ok",
            "x": {"m2": 1.21, "m2_std": 0.004, "d0_um": 201.0, "z0_mm": 119.5},
            "y": {"m2": None},
        },
    }
    card = demo_app.results_card(state)
    assert "⚠️ estimate_not_compliant" in card
    assert "**1.210 ± 0.004**" in card and "| y | not determined" in card
    assert "**Positions:** 12 used (10 accepted, 2 rejected)" in card
    assert "$0.123" in card
    truth = {"scenario": "clean", "m2_x": 1.2, "m2_y": 1.25, "z0_x_mm": 120.0}
    truth["ghost"] = "15% power"
    table = demo_app.truth_table(state.report, truth)
    assert "| M² | x | 1.210 | 1.200 | +0.8 % |" in table
    assert "| M² | y | not determined | 1.250 |" in table
    assert "| Waist position z0 | x | 119.500 mm | 120.000 mm | -0.50 mm |" in table
    assert "- **ghost:** 15% power" in table
    assert demo_app.results_card(RunState()) == ""
    assert "No ground truth" in demo_app.truth_table(None, None)


def test_token_cap_is_configurable_and_shown_when_reached(monkeypatch) -> None:
    monkeypatch.setenv("BEAMPILOT_TOKEN_CAP", "1_000")
    assert demo_app.token_cap_from_env() == 1000
    for bad in ("", "lots", "-5", "0"):
        monkeypatch.setenv("BEAMPILOT_TOKEN_CAP", bad)
        assert demo_app.token_cap_from_env() == demo_app.DEFAULT_TOKEN_CAP
    assert demo_app.cap_text() == ""
    demo_app._GLOBAL.add_tokens(demo_app.GLOBAL_TOKEN_BUDGET)
    assert "Live runs are paused" in demo_app.cap_text()
    assert "Live runs are paused" in demo_app.budget_text(RunState())
    update = next(demo_app.live_run("clean", 0, "", SimpleNamespace(session_hash="c")))
    assert "used up its token budget" in update[2]


# Lens kit (change_lens): live runs, replay and old run files.


def _lens_script_client() -> FakeClient:
    script = [
        ("get_setup", {}),
        ("measure_at", {"z_mm": 60.0, "exposure_ms": 8.0}),
        ("change_lens", {"f_mm": 25.0, "lens_z_mm": -20.0}),
        ("measure_at", {"z_mm": 10.0, "exposure_ms": 1.0}),
        ("finish", {"verdict": "measurement_failed", "explanation": "test run"}),
    ]
    return FakeClient([_response(_call(name, **args)) for name, args in script])


def test_lens_kit_checkbox_offers_change_lens_in_live_runs(
    monkeypatch, tmp_path
) -> None:
    client = _lens_script_client()
    monkeypatch.setattr(demo_app, "_client", lambda: client)
    monkeypatch.setattr(demo_app, "LIVE_RUNS_DIR", tmp_path)
    request = SimpleNamespace(session_hash="test-lens")
    demo_app._BROWSERS.pop("test-lens", None)
    updates = list(
        demo_app.live_run("waist_outside_range", 0, "", request, lens_kit=True)
    )
    assert "change_lens" in {t["name"] for t in client.requests[0]["tools"]}
    log = updates[-1][3]
    assert "**change_lens**(f=25.0 mm, z=-20.0 mm)" in log
    assert "new beam segment 1" in log
    session = demo_app._browser(request).session
    assert session.scenario.max_positions == demo_app.LENS_POSITIONS


def test_lens_kit_is_off_by_default_in_live_runs(monkeypatch, tmp_path) -> None:
    client = _lens_script_client()
    monkeypatch.setattr(demo_app, "_client", lambda: client)
    monkeypatch.setattr(demo_app, "LIVE_RUNS_DIR", tmp_path)
    request = SimpleNamespace(session_hash="test-nolens")
    demo_app._BROWSERS.pop("test-nolens", None)
    list(demo_app.live_run("waist_outside_range", 0, "", request))
    assert "change_lens" not in {t["name"] for t in client.requests[0]["tools"]}


def test_replay_of_a_lens_run_shows_only_the_current_segment(tmp_path) -> None:
    from dataclasses import replace

    from beampilot.agent import AgentConfig
    from beampilot.session import LensKit, waist_outside_range_scenario

    scenario = replace(waist_outside_range_scenario(), lens_kit=LensKit())
    session = MeasurementSession(scenario)
    run = run_agent(session, _lens_script_client(), AgentConfig(change_lens=True))
    path = tmp_path / "lens.json"
    save_run(run, path, {"scenario": "lens", "seed": 0, **session_record(session)})
    state = _play(load_run(path))
    assert state.segment == 1
    assert {p.segment for p in state.points.values()} == {0, 1}
    assert any("**change_lens**" in line for line in state.log)
    fig = caustic_figure(state, (0.0, 250.0))
    plotted = [
        x for coll in fig.axes[0].collections for x, _ in coll.get_offsets().tolist()
    ]
    assert plotted and all(abs(z - 10.0) < 1e-9 for z in plotted)


OLD_RUNS = ("clean_operator_note", "clean_room_light_live", "mystery_s7")


def test_old_run_files_without_segments_still_replay() -> None:
    for path in (demo_app.DEMO_RUNS_DIR / f"{name}.json" for name in OLD_RUNS):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "segment" not in json.dumps(data["steps"])  # written before lenses
        state = _play(load_run(path))
        assert state.segment == 0 and state.points
        assert all(p.segment == 0 for p in state.points.values())
        assert isinstance(caustic_figure(state, (0.0, 250.0)), Figure)
