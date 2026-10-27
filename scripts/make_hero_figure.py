"""Hero figure for the README: caustic of a real agent run next to a camera frame.

Uses the saved run ``docs/runs/mystery_s7.json`` (a real Muse Spark run on a
"mystery beam"): the caustic plot is rebuilt from its logged tool calls exactly
as the demo's replay does. Saved runs keep no camera frames, so the frame is
re-simulated from the run's ground truth (same beam, camera, position and
exposure; noise from a fixed seed). The output is the same on every run::

    .venv\\Scripts\\python.exe scripts\\make_hero_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from beampilot.plots import caustic_figure  # noqa: E402
from beampilot.replay import RunState, load_run, replay_events  # noqa: E402
from beampilot.simulator import BeamParams, CameraConfig, render_frame  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "docs" / "runs" / "mystery_s7.json"
FRAME_ID = "m16"  # a far-field point, large enough to see on the full sensor
OUT = REPO / "docs" / "figures" / "hero.png"
FRAME_SEED = 0


def make_figure(run_path: Path, frame_id: str) -> plt.Figure:
    """Caustic (left) and one re-simulated camera frame (right)."""
    run = load_run(run_path)
    state = RunState()
    for kind, item in replay_events(run):
        if kind == "reply":
            state.apply_model_reply(item)
        else:
            state.apply_tool_call(item)
    state.set_report(run.data.get("report"))
    params = run.data["scenario_params"]
    stage = (params["stage_min_m"] * 1e3, params["stage_max_m"] * 1e3)

    fig = caustic_figure(state, stage)
    fig.set_size_inches(13.0, 4.8)
    caustic_ax = fig.axes[0]
    caustic_ax.set_position([0.06, 0.12, 0.54, 0.76])
    report = run.data["report"]
    m2 = [report["evaluation"][axis]["m2"] for axis in ("x", "y")]
    caustic_ax.set_title(
        f"Agent run on a mystery beam: {report['verdict']}, M²x {m2[0]:.3f}, "
        f"M²y {m2[1]:.3f} ({state.model_calls} model calls)",
        fontsize=11,
    )

    point = state.points[frame_id]
    beam = BeamParams(**params["beam"])
    camera = CameraConfig(**params["camera"])
    frame = render_frame(
        beam,
        camera,
        point.z_mm * 1e-3,
        point.exposure_ms,
        np.random.default_rng(FRAME_SEED),
    )
    span = camera.full_scale_counts - camera.offset_counts
    scaled = np.clip((frame - camera.offset_counts) / span, 0.0, 1.0)
    half_x = camera.nx * camera.pixel_size_m * 1e3 / 2
    half_y = camera.ny * camera.pixel_size_m * 1e3 / 2
    frame_ax = fig.add_axes([0.68, 0.12, 0.30, 0.76])
    frame_ax.imshow(
        np.sqrt(scaled),
        cmap="inferno",
        extent=(-half_x, half_x, half_y, -half_y),
        interpolation="nearest",
    )
    frame_ax.set_xlabel("x (mm)")
    frame_ax.set_ylabel("y (mm)")
    frame_ax.set_title(
        f"Camera frame {frame_id}: z = {point.z_mm:g} mm, {point.exposure_ms:g} ms\n"
        "(re-simulated from the run's ground truth, √ scale)",
        fontsize=10,
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--frame", default=FRAME_ID)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    fig = make_figure(args.run, args.frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # No software or date metadata, so the PNG is byte-identical on every run.
    fig.savefig(args.out, dpi=110, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
