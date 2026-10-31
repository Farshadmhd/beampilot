"""BeamPilot demo UI (Gradio): live agent runs, fault buttons and replay.

Start locally (needs the optional ``demo`` extra)::

    .venv\\Scripts\\python.exe -m pip install -e ".[demo]"
    .venv\\Scripts\\python.exe -m beampilot.app

Live runs call the Meta Model API with ``MODEL_API_KEY`` from the environment
(or ``.env``); the key is never entered in or shown by the UI, and API errors
are reported by type and HTTP status only, never by their message text.
Replay mode plays saved runs (``docs/runs/*.json`` or an uploaded file) without
any API call. Only simulated data is ever sent.

Limits: per browser tab (runs, tokens, one run at a time) and process-wide
(one live run at a time, total runs, tokens and spend since start). Per-tab
limits reset when the page is reloaded; the process-wide caps are the real
guard. Environment variables for public hosting (see docs/hosting-plan.md):

- ``BEAMPILOT_LIVE_RUNS``: ``off`` switches live runs off (default ``on``).
- ``BEAMPILOT_SPEND_CAP_USD``: live-run spend cap per process start, in USD
  (default 5). A run starts only if the spend so far plus a worst-case run
  ($0.55) stays within the cap.
- ``BEAMPILOT_TOKEN_CAP``: token cap per process start (default 60,000,000,
  about $20 at the token mix of our agent runs).

When live runs are off, have no key or are out of budget, the page says so when
it loads and the Start button is disabled; replay keeps working.
"""

# No "from __future__ import annotations" here: Gradio inspects the handlers'
# annotations to inject gr.Request, which needs real classes, not strings.
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from beampilot import tools
from beampilot.agent import AgentConfig, AgentRun, make_client, run_agent, save_run
from beampilot.plots import caustic_figure, frame_image
from beampilot.replay import RunState, SavedRun, load_run, replay_events
from beampilot.session import (
    SCENARIOS,
    BeamParams,
    LensKit,
    MeasurementSession,
    _camera,
    random_scenario,
    reveal_truth,
    session_record,
)
from beampilot.simulator import CameraConfig, render_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_RUNS_DIR = REPO_ROOT / "docs" / "runs"
LIVE_RUNS_DIR = REPO_ROOT / "runs"
MYSTERY = "Mystery beam"

# Recording dates of the shipped runs (the run files hold no date).
RECORDED_MANIFEST = DEMO_RUNS_DIR / "recorded.csv"

# Limits per browser tab (reset on page reload: a convenience, not a guard).
MAX_RUNS_PER_SESSION = 3
# One lens run may use its full v3 token limit (LENS_MAX_TOKENS).
SESSION_TOKEN_BUDGET = 3_000_000
# Process-wide caps: they hold across reloads, tabs and invented session ids.
MAX_CONCURRENT_RUNS = 1
GLOBAL_MAX_RUNS = 50
# About $20: agent runs cost ~$0.34 per million tokens (input + output) with
# ~90 % of the input cached; without caching it would be up to ~$75.
DEFAULT_TOKEN_CAP = 60_000_000
TOKEN_CAP_ENV = "BEAMPILOT_TOKEN_CAP"
LIVE_SWITCH_ENV = "BEAMPILOT_LIVE_RUNS"
SPEND_CAP_ENV = "BEAMPILOT_SPEND_CAP_USD"
DEFAULT_SPEND_CAP_USD = 5.0
# Worst case of one run in USD: the per-run cost cap ($0.50) plus the overshoot
# of the one model call that may still run when the cap is reached.
RUN_COST_RESERVE_USD = 0.55
RUN_TIMEOUT_S = 15 * 60
# "Lens kit" live runs (change_lens): the limits of evaluation v3.
LENS_POSITIONS = 28
LENS_MODEL_CALLS = 80
LENS_MAX_TOKENS = 3_000_000
LENS_MAX_COST_USD = 0.50
# Replays run in parallel; live runs have their own queue of one.
REPLAY_CONCURRENCY = 4

LIVE_OFF_TEXT = "Live runs are switched off on this demo right now. Replay works."
NO_KEY_TEXT = (
    "Live runs are not set up on this demo (no MODEL_API_KEY). "
    "Replay works without one."
)
BUDGET_TEXT = "This demo has used its live-run budget for now; use Replay."
EVENT_POLL_S = 0.25
MAX_SEED = 1_000_000
MAX_UPLOAD = "5mb"
# Uploaded runs may only describe a camera up to this size (replay frames).
MAX_REPLAY_PIXELS = 1024


def token_cap_from_env() -> int:
    """Process-wide token cap from ``BEAMPILOT_TOKEN_CAP``, else the default."""
    value = os.environ.get(TOKEN_CAP_ENV, "").strip().replace("_", "")
    try:
        cap = int(value)
    except ValueError:
        return DEFAULT_TOKEN_CAP
    return cap if cap > 0 else DEFAULT_TOKEN_CAP


GLOBAL_TOKEN_BUDGET = token_cap_from_env()
ISO_POINTS = 5  # per axis: near (within 1 zR) and far (beyond 2 zR)


def spend_cap_from_env() -> float:
    """Spend cap per process start in USD from ``BEAMPILOT_SPEND_CAP_USD``.

    Returns
    -------
    float
        The cap in USD; the default (5) if the variable is unset, not a number
        or not positive.
    """
    try:
        cap = float(os.environ.get(SPEND_CAP_ENV, "").strip())
    except ValueError:
        return DEFAULT_SPEND_CAP_USD
    return cap if cap > 0 and cap != float("inf") else DEFAULT_SPEND_CAP_USD


def live_switch_on() -> bool:
    """False if ``BEAMPILOT_LIVE_RUNS`` switches live runs off (default on)."""
    value = os.environ.get(LIVE_SWITCH_ENV, "").strip().lower()
    return value not in {"off", "0", "false", "no"}


def api_key_present() -> bool:
    """Whether ``MODEL_API_KEY`` is set (environment or ``.env``, as the client).

    Only the presence is checked; the value is never returned or shown.
    """
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    return bool(os.environ.get("MODEL_API_KEY", "").strip())


class GlobalBudget:
    """Process-wide caps on runs, tokens and spend (USD), thread-safe.

    Each started run reserves a worst-case cost (``RUN_COST_RESERVE_USD``)
    until it ends, so the spend cap holds even for the last run.
    """

    def __init__(self, spend_cap_usd: float | None = None) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.runs = 0
        self.tokens = 0
        self.spent_usd = 0.0
        self.reserved_usd = 0.0
        self.spend_cap_usd = (
            spend_cap_from_env() if spend_cap_usd is None else spend_cap_usd
        )

    def _out_of_budget(self) -> str | None:
        # Caller holds the lock.
        if self.runs >= GLOBAL_MAX_RUNS:
            return "The demo has used up its live runs; use Replay."
        if self.tokens >= GLOBAL_TOKEN_BUDGET:
            return "The demo has used up its token budget; use Replay."
        worst = self.spent_usd + self.reserved_usd + RUN_COST_RESERVE_USD
        if worst > self.spend_cap_usd:
            return BUDGET_TEXT
        return None

    def out_of_budget(self) -> str | None:
        """Why no new run could start for budget reasons, else None."""
        with self._lock:
            return self._out_of_budget()

    def reserve(self) -> str | None:
        """Reserve a run slot; returns why not, or None if reserved."""
        with self._lock:
            if self.active >= MAX_CONCURRENT_RUNS:
                return "The server is busy with another live run; try again soon."
            refusal = self._out_of_budget()
            if refusal is not None:
                return refusal
            self.active += 1
            self.runs += 1
            self.reserved_usd += RUN_COST_RESERVE_USD
            return None

    def release(self, started: bool = True) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            self.reserved_usd = max(0.0, self.reserved_usd - RUN_COST_RESERVE_USD)
            if not started:
                self.runs = max(0, self.runs - 1)

    def add_tokens(self, tokens: int) -> None:
        with self._lock:
            self.tokens += tokens

    def add_cost(self, cost_usd: float) -> None:
        """Add the estimated cost of one model reply, in USD."""
        with self._lock:
            self.spent_usd += cost_usd

    def tokens_left(self) -> int:
        with self._lock:
            return max(0, GLOBAL_TOKEN_BUDGET - self.tokens)


@dataclass
class BrowserSession:
    """State of one browser tab: its runs, token use and the live session."""

    runs: int = 0
    tokens: int = 0
    active: bool = False
    session: MeasurementSession | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


_GLOBAL = GlobalBudget()
_BROWSERS: dict[str, BrowserSession] = {}
_BROWSERS_LOCK = threading.Lock()
_CLIENT: Any = None
_CLIENT_LOCK = threading.Lock()


def _browser(request: Any) -> BrowserSession:
    key = getattr(request, "session_hash", None) or "local"
    with _BROWSERS_LOCK:
        return _BROWSERS.setdefault(key, BrowserSession())


def _forget_browser(request: Any) -> None:
    """Tab closed: stop its run (if any) and drop its state."""
    key = getattr(request, "session_hash", None) or "local"
    with _BROWSERS_LOCK:
        browser = _BROWSERS.pop(key, None)
    if browser is not None:
        browser.cancel.set()


def _client() -> Any:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = make_client()  # reads MODEL_API_KEY from env / .env only
        return _CLIENT


def demo_runs() -> list[Path]:
    """Saved runs shipped for replay mode (``docs/runs/*.json``)."""
    return sorted(DEMO_RUNS_DIR.glob("*.json"))


def recorded_dates() -> dict[str, str]:
    """Recording date (``YYYY-MM-DD``) per shipped run file name, from the manifest."""
    import csv

    try:
        with RECORDED_MANIFEST.open(encoding="utf-8", newline="") as handle:
            return {row["file"]: row["recorded"] for row in csv.DictReader(handle)}
    except (OSError, KeyError):
        return {}


def recorded_label(path: Path | None, data: Any) -> str:
    """'Recorded run, <date>, <model>': a replay is never taken for a live run."""
    date = recorded_dates().get(path.name) if path is not None else None
    config = data.get("config") if isinstance(data, dict) else None
    model = config.get("model") if isinstance(config, dict) else None
    return f"Recorded run, {date or 'date unknown'}, {model or 'model unknown'}"


def replay_choices() -> list[tuple[str, str]]:
    """(label, value) of each shipped run for the replay dropdown."""
    import json

    choices = []
    for path in demo_runs():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        label = f"{recorded_label(path, data)}: {path.stem}"
        choices.append((label, path.relative_to(REPO_ROOT).as_posix()))
    return choices


def live_unavailable() -> str | None:
    """Why live runs can't start on this server now, or None if they can.

    Checked in this order: the live switch, the API key, the process budget.
    """
    if not live_switch_on():
        return LIVE_OFF_TEXT
    if not api_key_present():
        return NO_KEY_TEXT
    return _GLOBAL.out_of_budget()


def live_banner() -> tuple[str, bool]:
    """Banner text and whether the Start button is enabled, at page load."""
    reason = live_unavailable()
    return (f"⛔ **{reason}**" if reason else ""), reason is None


def scenario_choices() -> list[str]:
    return [*SCENARIOS, MYSTERY]


def budget_text(state: RunState) -> str:
    left = "?" if state.positions_left is None else state.positions_left
    total = "?" if state.max_positions is None else state.max_positions
    text = (
        f"**Positions left:** {left} / {total}  ·  **model calls:** "
        f"{state.model_calls}  ·  **tokens:** {state.total_tokens:,}  ·  "
        f"**est. cost:** ${state.cost_usd:.3f}\n\n{coverage_text(state)}"
    )
    cap = cap_text()
    return f"{text}\n\n{cap}" if cap else text


def coverage_text(state: RunState) -> str:
    """Live ISO point count per axis against the 5 near + 5 far target."""
    counts = state.coverage()
    if counts is None:
        return "**ISO coverage:** counted after the first fit (`evaluate`)."

    def count(n: int) -> str:
        return f"{'✅' if n >= ISO_POINTS else '⏳'} {n}/{ISO_POINTS}"

    parts = [
        f"**{axis}** near {count(near)}, far {count(far)}"
        for axis, (near, far) in counts.items()
    ]
    return "**ISO coverage** (accepted points, latest fit): " + "  ·  ".join(parts)


def cap_text() -> str:
    """Notice shown when the process-wide token cap is used up, else ""."""
    if _GLOBAL.tokens_left() > 0:
        return ""
    return (
        f"⛔ **Live runs are paused:** this server's token cap "
        f"({GLOBAL_TOKEN_BUDGET:,} tokens) is used up. Replay still works."
    )


VERDICT_ICONS = {
    "iso_compliant": "✅",
    "estimate_not_compliant": "⚠️",
    "waist_not_accessible": "🚫",
    "general_astigmatism_out_of_scope": "🚫",
    "measurement_failed": "❌",
}


def results_card(state: RunState) -> str:
    """Verdict, M² ± sigma per axis, positions and cost of the finished run."""
    report = state.report
    if not isinstance(report, dict) or "verdict" not in report:
        return ""
    verdict = str(report["verdict"])
    icon = VERDICT_ICONS.get(verdict, "")
    lines = [f"### {icon} {verdict}" if icon else f"### {verdict}"]
    evaluation = report.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    rows = []
    for axis in ("x", "y"):
        fit = evaluation.get(axis)
        if not isinstance(fit, dict):
            continue
        if not _is_number(fit.get("m2")):
            rows.append(f"| {axis} | not determined | – | – |")
            continue
        sigma = fit.get("m2_std")
        m2 = f"{fit['m2']:.3f}" + (f" ± {sigma:.3f}" if _is_number(sigma) else "")
        rows.append(
            f"| {axis} | **{m2}** | {_shown(fit.get('d0_um'), '.1f')} µm | "
            f"{_shown(fit.get('z0_mm'), '.1f')} mm |"
        )
    if rows:
        lines += ["", "| Axis | M² | Waist d0 | Waist z0 |", "|---|---|---|---|", *rows]
    used = (
        state.max_positions - state.positions_left
        if state.max_positions is not None and state.positions_left is not None
        else report.get("positions_measured", "?")
    )
    lines += [
        "",
        f"**Positions:** {used} used ({report.get('positions_accepted', '?')} "
        f"accepted, {report.get('positions_rejected', '?')} rejected)  ·  "
        f"**model calls:** {state.model_calls}  ·  "
        f"**est. cost:** ${state.cost_usd:.3f}",
    ]
    if report.get("stopped_by_limit"):
        lines.append(f"\n_Stopped by the controller: {report['stopped_by_limit']}_")
    if report.get("explanation"):
        lines.append(f"\n> {report['explanation']}")
    return "\n".join(lines)


# (label, truth key, unit, error kind) per row of the truth table.
TRUTH_ROWS = (
    ("Waist diameter d0", "d0_{}_um", "µm", "pct"),
    ("Waist position z0", "z0_{}_mm", "mm", "diff"),
    ("Rayleigh length zR", "z_r_{}_mm", "mm", "pct"),
    ("M²", "m2_{}", "", "pct"),
)
MEASURED_KEYS = {
    "d0_{}_um": "d0_um",
    "z0_{}_mm": "z0_mm",
    "z_r_{}_mm": "z_r_mm",
    "m2_{}": "m2",
}


def truth_table(report: Any, truth: Any) -> str:
    """Measured against true values per axis, with the error in %.

    The waist position error is given in mm (a % of a stage position means
    nothing). Hidden faults of the scenario are listed below the table.
    """
    if not isinstance(truth, dict):
        return "_No ground truth for this run._"
    evaluation = report.get("evaluation") if isinstance(report, dict) else None
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    lines = [
        f"**Ground truth: {truth.get('scenario', '?')}**",
        "",
        "| Quantity | Axis | Measured | True | Error |",
        "|---|---|---|---|---|",
    ]
    for label, key, unit, kind in TRUTH_ROWS:
        for axis in ("x", "y"):
            true = truth.get(key.format(axis))
            fit = evaluation.get(axis)
            measured = fit.get(MEASURED_KEYS[key]) if isinstance(fit, dict) else None
            error = ""
            if _is_number(measured) and _is_number(true) and true:
                error = (
                    f"{(measured / true - 1) * 100:+.1f} %"
                    if kind == "pct"
                    else f"{measured - true:+.2f} mm"
                )
            shown = (
                f"{measured:.3f} {unit}".strip()
                if _is_number(measured)
                else "not determined"
            )
            true_shown = f"{_shown(true, '.3f')} {unit}".strip()
            lines.append(f"| {label} | {axis} | {shown} | {true_shown} | {error} |")
    standard = {"scenario", "wavelength_nm"} | {
        key.format(axis) for _, key, _, _ in TRUTH_ROWS for axis in ("x", "y")
    }
    extras = [f"- **{k}:** {v}" for k, v in truth.items() if k not in standard]
    if extras:
        lines += ["", "Hidden in this scenario:", *extras]
    return "\n".join(lines)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _shown(value: Any, spec: str) -> str:
    return format(value, spec) if _is_number(value) else "–"


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
    run's ground truth (same beam, new noise, no faults). Returns None if the
    file's parameters are unusable or describe an oversized camera.
    """
    point, truth = state.last_measurement, run.truth
    if point is None or not truth:
        return None
    try:
        params = run.data.get("scenario_params")
        if params:
            beam = BeamParams(**params["beam"])
            camera = CameraConfig(**params["camera"])
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
        if not (
            0 < camera.nx <= MAX_REPLAY_PIXELS
            and 0 < camera.ny <= MAX_REPLAY_PIXELS
            and 1 <= camera.bit_depth <= 16
        ):
            return None
        exposure = float(np.clip(point.exposure_ms, 1e-3, 1e3))
        frame = render_frame(
            beam, camera, point.z_mm * 1e-3, exposure, np.random.default_rng(0)
        )
        return frame_image(frame, camera.full_scale_counts)
    except Exception:  # noqa: BLE001 - an uploaded file may contain anything
        return None


def build_app():
    """Build the Gradio interface (does not launch it)."""
    import gradio as gr

    with gr.Blocks(title="BeamPilot") as app:
        gr.Markdown(
            "# BeamPilot\nAn agent (Muse Spark) measures laser beam M² to "
            "ISO 11146 on a simulated camera and stage. All data is simulated."
        )
        cap_banner = gr.Markdown()
        with gr.Tab("Live run"):
            with gr.Row():
                scenario = gr.Dropdown(
                    scenario_choices(), value="clean", label="Scenario"
                )
                seed = gr.Number(
                    value=0, precision=0, minimum=0, maximum=MAX_SEED, label="Seed"
                )
                start = gr.Button("Start", variant="primary")
            note = gr.Textbox(
                label="Operator note (plain language, e.g. 'avoid z between 80 "
                "and 100 mm')",
                lines=2,
                max_length=500,
            )
            lens_kit = gr.Checkbox(
                value=False,
                label="Lens kit: the agent may insert a thin lens to form an "
                "artificial waist when the waist is outside the stage "
                f"({LENS_POSITIONS} positions)",
            )
            with gr.Row():
                room_light = gr.Button("Room light on")
                laser_power = gr.Button("Laser power x3")
                fault_status = gr.Markdown()
            live_budget = gr.Markdown()
            with gr.Row():
                live_plot = gr.Plot(label="Caustic", scale=3)
                live_frame = gr.Image(label="Latest frame", height=420, scale=2)
            live_log = gr.Markdown(label="Decision log")
            live_card = gr.Markdown()
            live_report = gr.JSON(label="Report")
            live_reveal = gr.Button("Reveal truth")
            live_truth = gr.Markdown()

        with gr.Tab("Replay (recorded runs)"):
            with gr.Row():
                saved = gr.Dropdown(replay_choices(), label="Recorded run")
                upload = gr.File(label="...or load a run JSON", file_types=[".json"])
                speed = gr.Slider(
                    0.0, 2.0, value=0.4, step=0.1, label="Seconds per step"
                )
                play = gr.Button("Play", variant="primary")
            replay_title = gr.Markdown()
            replay_budget = gr.Markdown()
            with gr.Row():
                replay_plot = gr.Plot(label="Caustic", scale=3)
                replay_image = gr.Image(
                    label="Latest frame (re-simulated)", height=420, scale=2
                )
            replay_log = gr.Markdown(label="Decision log")
            replay_card = gr.Markdown()
            replay_report = gr.JSON(label="Report")
            replay_reveal = gr.Button("Reveal truth")
            replay_truth = gr.Markdown()
            replay_run = gr.State(None)

        # Gradio passes the browser session only to parameters typed gr.Request.
        def start_run(
            scenario_name, seed_value, note_text, lens_on, request: gr.Request
        ):
            yield from live_run(
                scenario_name, seed_value, note_text, request, lens_kit=bool(lens_on)
            )

        def room_light_on(request: gr.Request) -> str:
            return inject_fault(request, "room_light")

        def laser_power_x3(request: gr.Request) -> str:
            return inject_fault(request, "laser_power")

        def reveal(request: gr.Request):
            return reveal_live_table(request)

        def tab_closed(request: gr.Request) -> None:
            _forget_browser(request)

        def page_load():
            banner, enabled = live_banner()
            return banner, gr.update(interactive=enabled)

        # UI only: none of these endpoints is offered as a public API.
        private = {"api_visibility": "private"}
        start.click(
            start_run,
            inputs=[scenario, seed, note, lens_kit],
            outputs=[
                live_plot,
                live_frame,
                live_budget,
                live_log,
                live_card,
                live_report,
                live_truth,
            ],
            # One live run at a time in its own queue; replays are not blocked.
            concurrency_limit=MAX_CONCURRENT_RUNS,
            concurrency_id="live",
            **private,
        )
        room_light.click(room_light_on, None, fault_status, **private)
        laser_power.click(laser_power_x3, None, fault_status, **private)
        live_reveal.click(reveal, None, live_truth, **private)
        play.click(
            replay,
            inputs=[saved, upload, speed],
            outputs=[
                replay_title,
                replay_plot,
                replay_image,
                replay_budget,
                replay_log,
                replay_card,
                replay_report,
                replay_truth,
                replay_run,
            ],
            **private,
        )
        replay_reveal.click(
            lambda run: truth_table(run.data.get("report"), run.truth) if run else "",
            replay_run,
            replay_truth,
            **private,
        )
        app.load(page_load, None, [cap_banner, start], **private)
        app.unload(tab_closed)
    return app


def _message(text: str) -> tuple:
    return (None, None, text, "", "", None, "")


def live_run(
    scenario_name: str,
    seed: float,
    note: str,
    request: Any = None,
    lens_kit: bool = False,
):
    """Run the agent in a thread and stream the UI updates (generator).

    With `lens_kit` the scenario gets the lens kit and the agent may use
    ``change_lens`` (artificial waist), with the budget and call limit of
    evaluation v3.

    The worker thread does all bookkeeping (tokens per model reply, fallback
    report, saving, freeing the slots), so it is complete even if the browser
    disconnects. Closing the generator (tab closed) cancels the run before its
    next model call.
    """
    if scenario_name not in scenario_choices():
        yield _message("Unknown scenario.")
        return
    try:
        seed = int(seed or 0)
    except (TypeError, ValueError, OverflowError):
        seed = -1
    if not 0 <= seed <= MAX_SEED:
        yield _message(f"The seed must be between 0 and {MAX_SEED}.")
        return
    if not live_switch_on():
        yield _message(LIVE_OFF_TEXT)
        return

    browser = _browser(request)
    with browser.lock:
        if browser.active:
            refusal = "A run is already in progress in this tab."
        elif browser.runs >= MAX_RUNS_PER_SESSION:
            refusal = (
                f"Run limit reached ({MAX_RUNS_PER_SESSION} per session); use Replay."
            )
        elif browser.tokens >= SESSION_TOKEN_BUDGET:
            refusal = "Token budget of this session used up; use Replay."
        else:
            refusal = _GLOBAL.reserve()
        if refusal is None:
            browser.runs += 1
            browser.active = True
            browser.cancel = threading.Event()
    if refusal is not None:
        yield _message(refusal)
        return

    cancel = browser.cancel
    try:
        client = _client()
        scenario = (
            random_scenario(seed)
            if scenario_name == MYSTERY
            else SCENARIOS[scenario_name](seed)
        )
        if lens_kit:
            scenario = replace(
                scenario,
                lens_kit=LensKit(),
                max_positions=max(scenario.max_positions, LENS_POSITIONS),
            )
        session = MeasurementSession(scenario)
    except Exception as exc:  # noqa: BLE001 - e.g. no API key: free the slots
        _release(browser, started=False)
        text = (
            NO_KEY_TEXT
            if isinstance(exc, RuntimeError)
            else f"Could not start the run ({type(exc).__name__})."
        )
        yield _message(text)
        return
    with browser.lock:
        browser.session = session

    tokens_left = min(SESSION_TOKEN_BUDGET - browser.tokens, _GLOBAL.tokens_left())
    if lens_kit:  # the v3 limits
        config = AgentConfig(
            max_model_calls=LENS_MODEL_CALLS,
            max_total_tokens=min(LENS_MAX_TOKENS, tokens_left),
            max_cost_usd=LENS_MAX_COST_USD,
            change_lens=True,
        )
    else:
        config = AgentConfig(
            max_total_tokens=min(AgentConfig().max_total_tokens, tokens_left)
        )
    events: queue.Queue = queue.Queue()
    worker = threading.Thread(
        target=_work,
        args=(browser, session, client, config, note, scenario_name, seed, events),
        daemon=True,
    )
    worker.start()

    state = RunState()
    stage = (scenario.stage_min_m * 1e3, scenario.stage_max_m * 1e3)
    faults_logged = 0
    deadline = time.monotonic() + RUN_TIMEOUT_S
    try:
        while True:
            if time.monotonic() > deadline and not cancel.is_set():
                cancel.set()
                state.log.append("⏱️ Time limit reached; stopping the run.")
            try:
                kind, payload = events.get(timeout=EVENT_POLL_S)
            except queue.Empty:
                if not worker.is_alive() and events.empty():
                    state.log.append("❌ The run stopped unexpectedly.")
                    yield _live_update(state, session, stage)
                    return
                continue
            if kind == "reply":
                state.apply_model_reply(payload)
            elif kind == "tool":
                # Faults switched on before this measurement, logged in order.
                for event in session.fault_events[faults_logged:]:
                    state.log.append(fault_line(event))
                faults_logged = len(session.fault_events)
                state.apply_tool_call(payload)
            else:  # "end"
                stop_reason, saved_as = payload
                state.set_report(session.report)
                saved = f" Saved as {saved_as}." if saved_as else " (not saved)"
                state.log.append(f"Run ended: {stop_reason}.{saved}")
                yield _live_update(state, session, stage)
                return
            yield _live_update(state, session, stage)
    finally:
        # Finished, or the browser went away: never let a run continue alone.
        cancel.set()


def _work(browser, session, client, config, note, scenario_name, seed, events):
    """Worker thread: run the agent and do all bookkeeping, whatever happens."""
    run: AgentRun | None = None
    stop_reason = "unknown"

    def on_reply(step: dict[str, Any]) -> None:
        usage = step.get("usage", {})
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        with browser.lock:
            browser.tokens += tokens
        _GLOBAL.add_tokens(tokens)
        _GLOBAL.add_cost(float(step.get("cost_usd") or 0.0))
        events.put(("reply", step))

    def should_stop() -> str | None:
        if browser.cancel.is_set():
            return "cancelled by the controller (tab closed or time limit)"
        if browser.tokens >= SESSION_TOKEN_BUDGET:
            return "token budget of this browser session reached"
        if _GLOBAL.tokens_left() <= 0:
            return "token budget of the demo reached"
        return None

    try:
        run = run_agent(
            session,
            client,
            config,
            on_model_reply=on_reply,
            on_tool_call=lambda call: events.put(("tool", call)),
            operator_note=note or None,
            should_stop=should_stop,
        )
        stop_reason = run.stop_reason
    except BaseException as exc:  # noqa: BLE001 - never leave a dead run behind
        # Type only: the message could carry text from the API server.
        stop_reason = f"controller error ({type(exc).__name__})"
    saved_as = None
    try:
        if not session.finished:
            tools.fallback_report(session, stop_reason)
        if run is None:
            run = AgentRun(
                config=config, stop_reason=stop_reason, report=session.report
            )
        saved_as = _save_live_run(run, session, scenario_name, seed).name
    except Exception:  # noqa: BLE001 - saving must not hide the report
        saved_as = None
    finally:
        _release(browser)
        events.put(("end", (stop_reason, saved_as)))


def _release(browser: BrowserSession, started: bool = True) -> None:
    """Free the tab and the global slot; a run that never started is uncounted."""
    with browser.lock:
        browser.active = False
        if not started:
            browser.runs = max(0, browser.runs - 1)
    _GLOBAL.release(started)


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
        results_card(state),
        state.report,
        "",  # truth stays hidden until "Reveal truth"
    )


def inject_fault(request: Any, fault: str) -> str:
    """Queue a fault for the running session; applied before the next measure_at."""
    browser = _browser(request)
    session = browser.session
    if not browser.active or session is None or session.finished:
        return "No run in progress."
    if fault == "room_light":
        session.request_faults(
            extra_background_counts=session.faults.extra_background_counts + 3.0
        )
        return "💡 Room light on (+3 counts) from the next measurement."
    session.request_faults(power_factor=session.faults.power_factor * 3.0)
    return "🔆 Laser power x3 from the next measurement."


def reveal_live(request: Any = None) -> dict[str, Any]:
    browser = _browser(request)
    session = browser.session
    if session is None:
        return {"message": "No run yet."}
    if browser.active or not session.finished:
        return {"message": "The truth is revealed after the run has finished."}
    return reveal_truth(session)


def reveal_live_table(request: Any = None) -> str:
    """The ground truth as a measured-vs-true table, after the run."""
    truth = reveal_live(request)
    if "message" in truth:
        return f"_{truth['message']}_"
    return truth_table(_browser(request).session.report, truth)


def replay(saved: str | None, upload: Any, seconds_per_step: float):
    """Play a saved run step by step (generator); no API calls."""
    uploaded = getattr(upload, "name", upload) if upload else None
    path = uploaded or (REPO_ROOT / saved if saved else None)
    if not path:
        yield _replay_message("Choose a recorded run or upload one.")
        return
    try:
        run = load_run(path)
        # An uploaded file is never matched to the manifest of shipped runs.
        label = recorded_label(None if uploaded else Path(path), run.data)
        yield from _replay(run, seconds_per_step, label)
    except Exception as exc:  # noqa: BLE001 - an uploaded file may contain anything
        yield _replay_message(f"Cannot replay this file ({type(exc).__name__}).")


def _replay_message(text: str) -> tuple:
    return (text, None, None, "", "", "", None, "", None)


def _replay(run: SavedRun, seconds_per_step: float, label: str):
    state = RunState()
    stage = (0.0, 250.0)
    params = run.data.get("scenario_params")
    if params:
        stage = (float(params["stage_min_m"]) * 1e3, float(params["stage_max_m"]) * 1e3)
    title = f"📼 **{label}** (a replay, not a live run): {run.title}"
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
            results_card(state),
            state.report,
            "",
            run,
        )
        time.sleep(float(np.clip(seconds_per_step, 0.0, 2.0)))
    state.set_report(run.data.get("report"))
    if run.data.get("stop_reason"):
        state.log.append(f"Run ended: {run.data['stop_reason']}.")
    yield (
        title,
        caustic_figure(state, stage),
        replay_frame(run, state),
        budget_text(state),
        log_text(state),
        results_card(state),
        state.report,
        "",
        run,
    )


def _save_live_run(run, session: MeasurementSession, scenario_name, seed) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = "mystery" if scenario_name == MYSTERY else scenario_name
    unique = uuid.uuid4().hex[:6]  # runs ending in the same second never collide
    path = LIVE_RUNS_DIR / f"{stamp}_demo_{name}_s{seed}_{unique}.json"
    save_run(run, path, {"scenario": name, "seed": seed, **session_record(session)})
    return path


def main() -> None:
    build_app().queue(default_concurrency_limit=REPLAY_CONCURRENCY).launch(
        max_file_size=MAX_UPLOAD
    )


if __name__ == "__main__":
    main()
