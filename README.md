# BeamPilot

Agentic laser beam characterisation: an AI agent that plans and checks
ISO 11146 beam measurements (beam widths, caustic fit, M²) and suggests the next
measurement step.

> **Status:** work in progress, built during the Meta Global AI Developer
> Hackathon (Oct 26 – Nov 4, 2026).

## Hackathon provenance
All code in this repository was written during the hackathon, starting
26 October 2026. There is no prior work. The commit history shows the progress.

## Planned components
- Meta Muse Spark as the agent (planning, tool calls, quality checks)
- SAM 3.1 for beam segmentation
- Python implementation of ISO 11146 second-moment beam widths and M² fit
- Demo on simulated beam images with known ground truth

## Licence
MIT
