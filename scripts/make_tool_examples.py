"""Write real example tool outputs to docs/examples/tool-outputs.json.

Runs one clean-scenario session with a fixed seed and records the tool calls
(arguments, reason and result) exactly as the session log holds them. The
operator here is this script: it places the 10 ISO positions and sets the
exposure from the scenario's known truth, which an agent never sees.

Usage::

    .venv\\Scripts\\python.exe scripts\\make_tool_examples.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from beampilot.session import MeasurementSession, clean_scenario
from beampilot.simulator import beam_diameter, rayleigh_length
from beampilot.tools import (
    evaluate,
    finish,
    get_setup,
    measure_at,
    set_status,
)

SEED = 3
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "examples"
ISO_OFFSETS = (-3.0, -2.3, -0.8, -0.4, 0.0, 0.4, 0.8, 2.3, 3.0, 3.6)  # in zR


def exposure_ms(session: MeasurementSession, z_mm: float) -> float:
    """Exposure for a peak of ~75 % of full scale, from the known truth."""
    beam, camera = session.scenario.beam, session.camera
    z_m = z_mm * 1e-3
    d_x = beam_diameter(z_m, beam.d0x_m, beam.z0x_m, beam.m2x, beam.wavelength_m)
    d_y = beam_diameter(z_m, beam.d0y_m, beam.z0y_m, beam.m2y, beam.wavelength_m)
    peak_per_ms = (
        camera.signal_counts_per_ms
        * camera.pixel_size_m**2
        / (2 * np.pi * (d_x / 4) * (d_y / 4))
    )
    return round(0.75 * camera.full_scale_counts / peak_per_ms, 3)


def main() -> None:
    session = MeasurementSession(clean_scenario(seed=SEED))
    beam = session.scenario.beam
    z_r_mm = rayleigh_length(beam.d0x_m, beam.m2x, beam.wavelength_m) * 1e3
    waist_mm = beam.z0x_m * 1e3

    get_setup(session, "Read the setup before planning positions.")
    examples = {"get_setup": session.log[-1]}

    too_long = round(4 * exposure_ms(session, waist_mm), 3)
    saturated = measure_at(
        session, waist_mm, too_long, "First look at the waist with a long exposure."
    )
    examples["measure_at_saturated"] = session.log[-1]
    set_status(session, saturated["measurement_id"], "accept", "Try to keep it.")
    examples["set_status_refused"] = session.log[-1]
    set_status(session, saturated["measurement_id"], "reject", "Saturated.")

    for offset in ISO_OFFSETS:
        z_mm = round(waist_mm + offset * z_r_mm, 1)
        result = measure_at(
            session,
            z_mm,
            exposure_ms(session, z_mm),
            f"ISO position at {offset:+.1f} zR.",
        )
        if offset == 0.0:
            examples["measure_at_clean"] = session.log[-1]
        set_status(
            session,
            result["measurement_id"],
            result["suggested_verdict"],
            "Follow the suggested verdict.",
        )

    evaluate(session, "Check the fit and ISO coverage.")
    examples["evaluate"] = session.log[-1]
    finish(
        session,
        "iso_compliant",
        "10 accepted positions, 5 within zR and 5 beyond 2 zR on both axes.",
    )
    examples["finish"] = session.log[-1]

    document = {
        "_about": (
            "Real tool calls from scripts/make_tool_examples.py: clean scenario, "
            f"seed {SEED}. Positions and exposures were set by the script from "
            "the known truth, not by an agent. Units: z in mm, diameters in um, "
            "wavelength in nm, exposure in ms."
        ),
        **examples,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "tool-outputs.json"
    text = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
