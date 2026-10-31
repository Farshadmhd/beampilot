---
title: BeamPilot
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: scripts/space_app.py
short_description: Muse Spark agent measures laser M² to ISO 11146
license: mit
pinned: false
suggested_hardware: cpu-basic
---

# BeamPilot

An agent (Muse Spark) measures the beam quality M² of a laser to ISO 11146: it
moves a camera along a stage, sets the exposure, judges each frame and fits the
caustic. All data is simulated, with a known ground truth.

Replays of recorded runs need no API key. Live runs are capped and may be
switched off.

Code, evaluation and documentation: https://github.com/Farshadmhd/beampilot
