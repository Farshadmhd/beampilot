"""Run state for the demo: built from tool calls, live or from a saved run.

:class:`RunState` applies one tool call after another (as logged by the agent
loop) and keeps what the UI shows: the measurements with their status, the
latest fit, the position budget, a decision log and the report. Live runs and
replays use the same class, so both look identical. No Gradio here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MeasurementPoint:
    """One measurement as shown in the caustic plot (z in mm, d in um)."""

    measurement_id: str
    z_mm: float
    exposure_ms: float
    d_x_um: float | None
    d_y_um: float | None
    status: str = "pending"
    segment: int = 0  # beam segment (lens changes); 0 in old run files


@dataclass
class RunState:
    """What the demo displays, updated tool call by tool call."""

    points: dict[str, MeasurementPoint] = field(default_factory=dict)
    evaluation: dict[str, Any] | None = None
    positions_left: int | None = None
    max_positions: int | None = None
    log: list[str] = field(default_factory=list)
    last_measurement: MeasurementPoint | None = None
    report: dict[str, Any] | None = None
    model_calls: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    segment: int = 0  # current beam segment; a lens change starts a new one
    # Index of the log line that collects consecutive set_status calls.
    _status_line: int | None = field(default=None, repr=False)

    def apply_model_reply(self, step: dict[str, Any]) -> None:
        """Count a model reply (usage and cost) and log its text, if any."""
        self.model_calls += 1
        usage = step.get("usage", {})
        self.total_tokens += usage.get("input_tokens", 0) + usage.get(
            "output_tokens", 0
        )
        self.cost_usd += step.get("cost_usd", 0.0)
        if step.get("text"):
            self.log.append(f"💬 {step['text'].strip()}")

    def apply_tool_call(self, call: dict[str, Any]) -> None:
        """Update the state with one logged tool call (name, arguments, result)."""
        name, result = call["name"], call.get("result") or {}
        arguments = call.get("arguments") or {}
        reason = call.get("reason") or arguments.get("reason") or ""
        if name == "set_status":
            self._log_status(arguments, reason, result)
        else:
            self.log.append(_log_line(name, arguments, reason, result))
        if "error" in result or "internal_error" in result:
            return
        if name == "get_setup":
            self.positions_left = result.get("positions_left")
            self.max_positions = result.get("max_positions")
        elif name == "measure_at":
            point = MeasurementPoint(
                measurement_id=result["measurement_id"],
                z_mm=result["z_mm"],
                exposure_ms=result["exposure_ms"],
                d_x_um=result["d_x_um"]["mean"],
                d_y_um=result["d_y_um"]["mean"],
                segment=result.get("segment", 0),
            )
            self.points[point.measurement_id] = point
            self.last_measurement = point
            self.positions_left = result.get("positions_left")
        elif name == "retake_darks":
            self.positions_left = result.get("positions_left")
        elif name == "change_lens":
            # A new beam: the old fit and points no longer describe it.
            self.segment = result.get("segment", self.segment + 1)
            self.positions_left = result.get("positions_left")
            self.evaluation = None
        elif name == "set_status":
            point = self.points.get(result.get("measurement_id"))
            if point is not None:
                point.status = result["status"]
        elif name == "evaluate" and result.get("status") == "ok":
            self.evaluation = result
        elif name == "finish" and "report" in result:
            self.set_report(result["report"])

    def coverage(self) -> dict[str, tuple[int, int]] | None:
        """ISO point count per axis: (near, far) distinct accepted positions.

        Uses z0 and zR of the latest fit and the measurements accepted so far,
        so the count moves as soon as a point is accepted. Near is
        |z - z0| <= zR, far |z - z0| >= 2 zR, as in ``evaluate``. None before
        the first fit with z0 and zR on both axes.
        """
        evaluation = self.evaluation
        if not evaluation:
            return None
        accepted = {
            p.z_mm
            for p in self.points.values()
            if p.status == "accept" and p.segment == self.segment
        }
        counts = {}
        for axis in ("x", "y"):
            fit = evaluation.get(axis) or {}
            z0, z_r = fit.get("z0_mm"), fit.get("z_r_mm")
            if z0 is None or not z_r:
                return None
            near = sum(abs(z - z0) <= z_r for z in accepted)
            far = sum(abs(z - z0) >= 2 * z_r for z in accepted)
            counts[axis] = (near, far)
        return counts

    def _log_status(self, arguments: dict, reason: str, result: dict) -> None:
        """Consecutive set_status calls share one log line."""
        mid = result.get("measurement_id") or arguments.get("measurement_id", "?")
        if "error" in result or "internal_error" in result:
            item = f"{mid} ⚠️ {result.get('error') or result.get('internal_error')}"
        else:
            mark = "✅" if result.get("status") == "accept" else "❌"
            item = f"{mid} {mark} {reason}".rstrip()
        last = len(self.log) - 1
        if self._status_line is not None and self._status_line == last:
            self.log[last] += f" · {item}"
        else:
            self.log.append(f"**set_status**: {item}")
            self._status_line = len(self.log) - 1

    def set_report(self, report: dict[str, Any] | None) -> None:
        """Store the final report (from finish or the controller's fallback)."""
        if report is None:
            return
        self.report = report
        evaluation = report.get("evaluation")
        if evaluation and evaluation.get("status") == "ok":
            self.evaluation = evaluation


@dataclass
class SavedRun:
    """A run saved by ``scripts/run_agent.py`` (or the demo), for replay."""

    path: Path
    data: dict[str, Any]

    @property
    def steps(self) -> list[dict[str, Any]]:
        return self.data["steps"]

    @property
    def truth(self) -> dict[str, Any] | None:
        return self.data.get("truth")

    @property
    def title(self) -> str:
        data = self.data
        verdict = (data.get("report") or {}).get("verdict", "no report")
        name = data.get("scenario") or data.get("case") or "?"  # evaluation runs
        return (
            f"{name} (seed {data.get('seed', '?')}): {verdict}, "
            f"{data.get('model_calls', len(self.steps))} model calls"
        )


def load_run(path: str | Path) -> SavedRun:
    """Load a saved run JSON and check that it can be replayed.

    Parameters
    ----------
    path : str or Path
        Run file written by :func:`beampilot.agent.save_run`.

    Returns
    -------
    SavedRun

    Raises
    ------
    ValueError
        If the file is not a run file (no ``steps`` list of model replies with
        ``tool_calls``).
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list) or not all(
        isinstance(step, dict) and isinstance(step.get("tool_calls"), list)
        for step in steps
    ):
        raise ValueError(f"{path.name} is not a BeamPilot run file (no steps)")
    return SavedRun(path=path, data=data)


def replay_events(run: SavedRun):
    """Yield ``("reply", step)`` and ``("tool", call)`` events in run order."""
    for step in run.steps:
        yield "reply", step
        for call in step["tool_calls"]:
            yield "tool", call


def _log_line(name: str, arguments: dict, reason: str, result: dict) -> str:
    shown = {k: v for k, v in arguments.items() if k not in ("reason", "explanation")}
    args = ", ".join(f"{k}={v}" for k, v in shown.items())
    line = f"**{name}**({args}) — {reason}"
    if "error" in result:
        return f"{line}\n  ⚠️ {result['error']}"
    if "internal_error" in result:
        return f"{line}\n  ❌ {result['internal_error']}"
    if name == "measure_at":
        flags = [k for k, v in result.get("flags", {}).items() if v]
        d_x, d_y = result["d_x_um"]["mean"], result["d_y_um"]["mean"]
        return (
            f"{line}\n  → {result['measurement_id']}: d_x {d_x} µm, d_y {d_y} µm, "
            f"flags {', '.join(flags) or 'none'}, "
            f"suggested {result.get('suggested_verdict')}"
        )
    if name == "change_lens":
        prediction = result.get("prediction") or {}
        guess = prediction.get("x") or {}
        predicted = (
            f", predicted waist {guess['z0_mm']} mm (from the fit)"
            if guess.get("z0_mm") is not None
            else ""
        )
        return (
            f"🔎 **change_lens**(f={arguments.get('f_mm')} mm, "
            f"z={arguments.get('lens_z_mm')} mm) — {reason}\n  → new beam segment "
            f"{result.get('segment')}{predicted}; earlier points no longer count"
        )
    if name == "evaluate" and result.get("status") == "ok":
        parts = []
        for axis in ("x", "y"):
            a = result[axis]
            m2 = a["m2"] if a["m2"] is not None else "not determined"
            parts.append(f"M²{axis} {m2}")
        met = result["iso_coverage_met"]
        return f"{line}\n  → {', '.join(parts)}, ISO coverage met: {met}"
    if name == "finish" and "report" in result:
        return f"{line}\n  → verdict **{result['report']['verdict']}**"
    return line
