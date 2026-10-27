"""BeamPilot demo UI (Gradio): live agent runs, fault buttons and replay.

Start locally (needs the optional ``demo`` extra)::

    .venv\\Scripts\\python.exe -m pip install -e ".[demo]"
    .venv\\Scripts\\python.exe -m beampilot.app

Live runs call the Meta Model API with ``MODEL_API_KEY`` from the environment
(or ``.env``); the key is never entered in or shown by the UI. Replay mode
plays saved runs (``docs/runs/*.json`` or an uploaded file) without any API
call. Only simulated data is ever sent.
"""

# No "from __future__ import annotations" here: Gradio inspects the handlers'
# annotations to inject gr.Request, which needs real classes, not strings.
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from beampilot.agent import AgentConfig, make_client, run_agent, save_run
from beampilot.plots import caustic_figure, frame_image
from beampilot.replay import RunState, SavedRun, load_run, replay_events
from beampilot.session import (
    SCENARIOS,
    BeamParams,
    MeasurementSession,
    _camera,
    random_scenario,
    reveal_truth,
    session_record,
)
from beampilot.simulator import render_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_RUNS_DIR = REPO_ROOT / "docs" / "runs"
LIVE_RUNS_DIR = REPO_ROOT / "runs"
MYSTERY = "Mystery beam"

# Limits per browser session (one person's tab).
MAX_RUNS_PER_SESSION = 3
SESSION_TOKEN_BUDGET = 2_000_000
EVENT_POLL_S = 0.25


@dataclass
class BrowserSession:
    """State of one browser tab: its runs, token use and the live session."""

    runs: int = 0
    tokens: int = 0
    session: MeasurementSession | None = None
    truth: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_BROWSERS: dict[str, BrowserSession] = {}
_BROWSERS_LOCK = threading.Lock()
_CLIENT: Any = None


def _browser(request: Any) -> BrowserSession:
    key = getattr(request, "session_hash", None) or "local"
    with _BROWSERS_LOCK:
        return _BROWSERS.setdefault(key, BrowserSession())


def _client() -> Any:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = make_client()  # reads MODEL_API_KEY from env / .env only
    return _CLIENT


def demo_runs() -> list[Path]:
    """Saved runs shipped for replay mode (``docs/runs/*.json``)."""
    return sorted(DEMO_RUNS_DIR.glob("*.json"))


def scenario_choices() -> list[str]:
    return [*SCENARIOS, MYSTERY]


def budget_text(state: RunState) -> str:
    left = "?" if state.positions_left is None else state.positions_left
    total = "?" if state.max_positions is None else state.max_positions
    return (
        f"**Positions left:** {left} / {total}  ·  **model calls:** "
        f"{state.model_calls}  ·  **tokens:** {state.total_tokens:,}  ·  "
        f"**est. cost:** ${state.cost_usd:.3f}"
    )


def fault_line(event: dict[str, Any]) -> str:
    """Log line for a fault switch (from ``session.fault_events``)."""
    faults = event["faults"]
    source = "operator button" if event["source"] == "operator" else "scenario"
    return (
        f"💡 **Fault ({source})** from measurement {event['before_measurement']}: "
        f"extra background {faults['extra_background_counts']:g} counts, "
        f"laser power x{faults['power_factor']:g}"
    )


def log_text(state: RunState, last: int = 60) -> str:
    lines = state.log[-last:]
    return "\n\n".join(lines) if lines else "_No decisions yet._"


def replay_frame(run: SavedRun, state: RunState) -> np.ndarray | None:
    """Frame for the latest measurement of a replay, re-rendered from truth.

    Saved runs keep no camera frames; the image is simulated again from the
    run's ground truth (same beam, new noise, no faults).
    """
    point, truth = state.last_measurement, run.truth
    if point is None or not truth:
        return None
    params = run.data.get("scenario_params")
    if params:
        beam = BeamParams(**params["beam"])
        camera = _camera_from(params["camera"])
    else:
        beam = BeamParams(
            wavelength_m=truth["wavelength_nm"] * 1e-9,
            d0x_m=truth["d0_x_um"] * 1e-6,
            d0y_m=truth["d0_y_um"] * 1e-6,
            z0x_m=truth["z0_x_mm"] * 1e-3,
            z0y_m=truth["z0_y_mm"] * 1e-3,
            m2x=truth["m2_x"],
            m2y=truth["m2_y"],
        )
        camera = _camera()
    frame = render_frame(
        beam, camera, point.z_mm * 1e-3, point.exposure_ms, np.random.default_rng(0)
    )
    return frame_image(frame, camera.full_scale_counts)


def _camera_from(params: dict[str, Any]):
    from beampilot.simulator import CameraConfig

    return CameraConfig(**params)


def build_app():
    """Build the Gradio interface (does not launch it)."""
    import gradio as gr

    with gr.Blocks(title="BeamPilot") as app:
        gr.Markdown(
            "# BeamPilot\nAn agent (Muse Spark) measures laser beam M² to "
            "ISO 11146 on a simulated camera and stage. All data is simulated."
        )
        with gr.Tab("Live run"):
            with gr.Row():
                scenario = gr.Dropdown(
                    scenario_choices(), value="clean", label="Scenario"
                )
                seed = gr.Number(value=0, precision=0, label="Seed")
                start = gr.Button("Start", variant="primary")
            note = gr.Textbox(
                label="Operator note (plain language, e.g. 'avoid z between 80 "
                "and 100 mm')",
                lines=2,
            )
            with gr.Row():
                room_light = gr.Button("Room light on")
                laser_power = gr.Button("Laser power x3")
                fault_status = gr.Markdown()
            live_budget = gr.Markdown()
            with gr.Row():
                live_plot = gr.Plot(label="Caustic")
                live_frame = gr.Image(label="Latest frame", height=300)
            live_log = gr.Markdown(label="Decision log")
            live_report = gr.JSON(label="Report")
            live_reveal = gr.Button("Reveal truth")
            live_truth = gr.JSON(label="Ground truth")

        with gr.Tab("Replay"):
            with gr.Row():
                saved = gr.Dropdown(
                    [p.relative_to(REPO_ROOT).as_posix() for p in demo_runs()],
                    label="Saved run",
                )
                upload = gr.File(label="...or load a run JSON", file_types=[".json"])
                speed = gr.Slider(
                    0.0, 2.0, value=0.4, step=0.1, label="Seconds per step"
                )
                play = gr.Button("Play", variant="primary")
            replay_title = gr.Markdown()
            replay_budget = gr.Markdown()
            with gr.Row():
                replay_plot = gr.Plot(label="Caustic")
                replay_image = gr.Image(label="Latest frame (re-simulated)", height=300)
            replay_log = gr.Markdown(label="Decision log")
            replay_report = gr.JSON(label="Report")
            replay_reveal = gr.Button("Reveal truth")
            replay_truth = gr.JSON(label="Ground truth")
            replay_run = gr.State(None)

        # Gradio passes the browser session only to parameters typed gr.Request.
        def start_run(scenario_name, seed_value, note_text, request: gr.Request):
            yield from live_run(scenario_name, seed_value, note_text, request)

        def room_light_on(request: gr.Request) -> str:
            return inject_fault(request, "room_light")

        def laser_power_x3(request: gr.Request) -> str:
            return inject_fault(request, "laser_power")

        def reveal(request: gr.Request):
            return reveal_live(request)

        start.click(
            start_run,
            inputs=[scenario, seed, note],
            outputs=[
                live_plot,
                live_frame,
                live_budget,
                live_log,
                live_report,
                live_truth,
            ],
        )
        room_light.click(room_light_on, None, fault_status)
        laser_power.click(laser_power_x3, None, fault_status)
        live_reveal.click(reveal, None, live_truth)
        play.click(
            replay,
            inputs=[saved, upload, speed],
            outputs=[
                replay_title,
                replay_plot,
                replay_image,
                replay_budget,
                replay_log,
                replay_report,
                replay_truth,
                replay_run,
            ],
        )
        replay_reveal.click(
            lambda run: run.truth if run else None, replay_run, replay_truth
        )
    return app


def live_run(scenario_name: str, seed: float, note: str, request: Any = None):
    """Run the agent in a thread and stream the UI updates (generator)."""
    browser = _browser(request)
    blank = (None, None, "", "", None, None)
    if browser.runs >= MAX_RUNS_PER_SESSION:
        yield (
            *blank[:2],
            f"Run limit reached ({MAX_RUNS_PER_SESSION} per session); use Replay.",
            *blank[3:],
        )
        return
    if browser.tokens >= SESSION_TOKEN_BUDGET:
        yield (
            *blank[:2],
            "Token budget of this session used up; use Replay.",
            *blank[3:],
        )
        return
    try:
        client = _client()
    except RuntimeError as exc:
        yield (*blank[:2], f"Live runs need an API key: {exc}. Use Replay.", *blank[3:])
        return

    seed = int(seed or 0)
    scenario = (
        random_scenario(seed)
        if scenario_name == MYSTERY
        else SCENARIOS[scenario_name](seed)
    )
    session = MeasurementSession(scenario)
    with browser.lock:
        browser.runs += 1
        browser.session, browser.truth = session, None
    config = AgentConfig(
        max_total_tokens=min(
            AgentConfig().max_total_tokens, SESSION_TOKEN_BUDGET - browser.tokens
        )
    )
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            run = run_agent(
                session,
                client,
                config,
                on_model_reply=lambda step: events.put(("reply", step)),
                on_tool_call=lambda call: events.put(("tool", call)),
                operator_note=note or None,
            )
            events.put(("done", run))
        except Exception as exc:  # noqa: BLE001 - shown in the UI, never crash
            events.put(("error", exc))

    threading.Thread(target=work, daemon=True).start()
    state = RunState()
    stage = (scenario.stage_min_m * 1e3, scenario.stage_max_m * 1e3)
    faults_logged = 0
    while True:
        try:
            kind, payload = events.get(timeout=EVENT_POLL_S)
        except queue.Empty:
            continue
        if kind == "reply":
            state.apply_model_reply(payload)
        elif kind == "tool":
            # Faults switched on before this measurement, logged in order.
            for event in session.fault_events[faults_logged:]:
                state.log.append(fault_line(event))
            faults_logged = len(session.fault_events)
            state.apply_tool_call(payload)
        elif kind == "error":
            state.log.append(f"❌ {type(payload).__name__}: {payload}")
            yield _live_update(state, session, stage)
            return
        else:  # done
            run = payload
            state.set_report(run.report)
            browser.tokens += run.usage.total_tokens
            path = _save_live_run(run, session, scenario_name, seed)
            state.log.append(f"Run ended: {run.stop_reason}. Saved as {path.name}.")
            yield _live_update(state, session, stage)
            return
        yield _live_update(state, session, stage)


def _live_update(state: RunState, session: MeasurementSession, stage):
    frame = session.last_frame
    image = (
        frame_image(frame, session.camera.full_scale_counts)
        if frame is not None
        else None
    )
    return (
        caustic_figure(state, stage),
        image,
        budget_text(state),
        log_text(state),
        state.report,
        None,  # truth stays hidden until "Reveal truth"
    )


def inject_fault(request: Any, fault: str) -> str:
    """Queue a fault for the running session; applied before the next measure_at."""
    browser = _browser(request)
    session = browser.session
    if session is None or session.finished:
        return "No run in progress."
    if fault == "room_light":
        session.request_faults(
            extra_background_counts=session.faults.extra_background_counts + 3.0
        )
        return "💡 Room light on (+3 counts) from the next measurement."
    session.request_faults(power_factor=session.faults.power_factor * 3.0)
    return "🔆 Laser power x3 from the next measurement."


def reveal_live(request: Any = None) -> dict[str, Any] | str:
    browser = _browser(request)
    session = browser.session
    if session is None:
        return {"message": "No run yet."}
    if not session.finished:
        return {"message": "The truth is revealed after the run has finished."}
    return reveal_truth(session)


def replay(saved: str | None, upload: Any, seconds_per_step: float):
    """Play a saved run step by step (generator); no API calls."""
    path = getattr(upload, "name", upload) if upload else None
    path = path or (REPO_ROOT / saved if saved else None)
    if not path:
        yield (
            "Choose a saved run or upload one.",
            None,
            None,
            "",
            "",
            None,
            None,
            None,
        )
        return
    try:
        run = load_run(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        yield (f"Cannot load the run: {exc}", None, None, "", "", None, None, None)
        return
    state = RunState()
    stage = (0.0, 250.0)
    params = run.data.get("scenario_params")
    if params:
        stage = (params["stage_min_m"] * 1e3, params["stage_max_m"] * 1e3)
    title = f"**Replay:** {run.title}"
    note = run.data.get("operator_note")
    if note:
        title += f"\n\n**Operator note:** {note}"
    fault_events = list(run.data.get("fault_events") or [])
    measured = 0
    for kind, item in replay_events(run):
        if kind == "reply":
            state.apply_model_reply(item)
            continue
        if item["name"] == "measure_at" and "measurement_id" in (item["result"] or {}):
            measured += 1
            for event in fault_events:
                if event["before_measurement"] == measured:
                    state.log.append(fault_line(event))
        state.apply_tool_call(item)
        yield (
            title,
            caustic_figure(state, stage),
            replay_frame(run, state),
            budget_text(state),
            log_text(state),
            state.report,
            None,
            run,
        )
        time.sleep(max(0.0, float(seconds_per_step)))
    state.set_report(run.data.get("report"))
    if run.data.get("stop_reason"):
        state.log.append(f"Run ended: {run.data['stop_reason']}.")
    yield (
        title,
        caustic_figure(state, stage),
        replay_frame(run, state),
        budget_text(state),
        log_text(state),
        state.report,
        None,
        run,
    )


def _save_live_run(run, session: MeasurementSession, scenario_name, seed) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = "mystery" if scenario_name == MYSTERY else scenario_name
    path = LIVE_RUNS_DIR / f"{stamp}_demo_{name}_s{seed}.json"
    save_run(run, path, {"scenario": name, "seed": seed, **session_record(session)})
    return path


def main() -> None:
    build_app().queue(default_concurrency_limit=2).launch()


if __name__ == "__main__":
    main()
