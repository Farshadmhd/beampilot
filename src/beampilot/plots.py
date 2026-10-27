"""Plots for the demo: caustic d(z) per axis and camera frame images.

Uses matplotlib's non-interactive Agg backend, so it works in a server thread.
Units as at the tool boundary: z in mm, diameters in um.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from beampilot.replay import RunState  # noqa: E402

AXIS_COLOURS = {"x": "#1f77b4", "y": "#d62728"}


def caustic_figure(
    state: RunState, stage_range_mm: tuple[float, float] | None = None
) -> Figure:
    """d_x and d_y against z with status markers, fits and ISO zones.

    Accepted measurements are filled circles, rejected ones crosses, pending
    ones open circles. After an ``evaluate`` with determined values, the
    fitted caustic d(z) = d0 sqrt(1 + ((z - z0)/zR)²) is drawn per axis, with
    the ±1 zR zone shaded and the zone beyond 2 zR hatched lightly.

    Parameters
    ----------
    state : RunState
        Current run state.
    stage_range_mm : tuple of float, optional
        Stage travel in mm, used for the z limits and the fit curves.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=100)
    points = list(state.points.values())
    z_values = [p.z_mm for p in points]
    if stage_range_mm is not None:
        z_lo, z_hi = stage_range_mm
    elif z_values:
        z_lo, z_hi = min(z_values) - 10.0, max(z_values) + 10.0
    else:
        z_lo, z_hi = 0.0, 1.0

    evaluation = state.evaluation
    for axis, colour in AXIS_COLOURS.items():
        fit = evaluation.get(axis) if evaluation else None
        if fit and fit.get("m2") is not None:
            z0, z_r, d0 = fit["z0_mm"], fit["z_r_mm"], fit["d0_um"]
            z = np.linspace(z_lo, z_hi, 400)
            ax.plot(
                z,
                d0 * np.sqrt(1 + ((z - z0) / z_r) ** 2),
                color=colour,
                lw=1.2,
                label=f"fit {axis}: M² {fit['m2']}",
            )
            ax.axvspan(z0 - z_r, z0 + z_r, color=colour, alpha=0.08)
            for edge in (z0 - 2 * z_r, z0 + 2 * z_r):
                ax.axvline(edge, color=colour, ls=":", lw=0.8)
        for status, marker, fill in (
            ("accept", "o", True),
            ("pending", "o", False),
            ("reject", "x", True),
        ):
            selected = [
                p for p in points if p.status == status and getattr(p, f"d_{axis}_um")
            ]
            if not selected:
                continue
            ax.scatter(
                [p.z_mm for p in selected],
                [getattr(p, f"d_{axis}_um") for p in selected],
                marker=marker,
                s=36,
                color=colour if fill else "none",
                edgecolors=colour if marker == "o" else None,
                label=f"d_{axis} {status}",
            )
    ax.set_xlim(z_lo, z_hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("stage position z (mm)")
    ax.set_ylabel("second-moment diameter (µm)")
    ax.set_title("Caustic (shaded: ±1 zR, dotted: 2 zR)", fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, ncol=2, loc="upper center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.close(fig)  # the caller (Gradio) renders it; free pyplot's reference
    return fig


def frame_image(frame: NDArray[Any], full_scale: int) -> NDArray[np.uint8]:
    """Camera frame as an RGB image (inferno colour map, 0 to full scale).

    Parameters
    ----------
    frame : 2D array
        Raw frame in counts.
    full_scale : int
        ADC full scale in counts; saturated pixels show as the top colour.

    Returns
    -------
    ndarray of uint8, shape (ny, nx, 3)
    """
    scaled = np.clip(np.asarray(frame, dtype=np.float64) / full_scale, 0.0, 1.0)
    rgba = plt.get_cmap("inferno")(scaled)
    return (rgba[..., :3] * 255).astype(np.uint8)
