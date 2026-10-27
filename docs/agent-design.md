# BeamPilot agent design (v1)

Status: v1, agreed 26 Oct 2026.
Updated 27 Oct 2026 (added `retake_darks`). All data is simulated. No lab data, no code from other projects.

## 1. Goal and pitch

BeamPilot is an agent that measures the M² of a laser beam to ISO 11146 with a camera
on a moving stage (simulated). It decides where to measure, judges whether each
measurement can be trusted, recovers from faults, and gives an honest final verdict,
including "this cannot be measured to the standard with this setup".

Why an LLM and not a fixed script: commercial M² instruments already place ISO points
automatically on clean beams. BeamPilot's value is **judgement when things go wrong**
(saturation, background drift, beam too large for the sensor, waist out of reach,
different x/y waists) and **explaining every decision**. We test this claim against a
smart scripted baseline and report the result honestly, even if the agent only wins on
fault scenarios.

## 2. Who does what

| Part | Job | Never does |
|---|---|---|
| Muse Spark (agent) | Chooses positions and exposure, accepts or rejects measurements, decides when to finish, explains each action | Compute or type physics numbers. The report numbers come from Python state. |
| Python tools | Simulator, ISO 11146 widths, weighted caustic fit, coverage and sanity checks | Decide what to measure next or discard data on their own |
| Python controller | Runs the loop, holds all state, enforces hard limits, logs everything | Trust the agent to stop or to report correct numbers |

The phases in section 5 are guidance in the system prompt, not controller code.
The model chooses the order of actions; the controller only enforces limits.

## 3. Units

- Python computes **internally in SI** (metres, radians).
- The tool interface shows z in mm, diameters in µm and wavelength in nm, because that is
  easier for the model to read. Conversion happens only at the tool boundary.
- A known-answer test checks M² end to end, so a unit error (for example a factor 10³) fails.

## 4. Tools (7 core)

Every tool has a required `reason` string. Tool schemas are strict, and verdicts are enums.

1. `get_setup(reason)`
   - Wavelength, stage range, pixel size, sensor size, exposure limits, position budget left.
2. `measure_at(z_mm, exposure_ms, reason)`
   - Moves the stage, takes **5 repeated frames** (ISO 11146-1 asks for at least 5 per position),
     applies background correction, computes ISO second-moment diameters.
   - Returns: `measurement_id`, mean ± std of d_x and d_y, quality flags (saturation fraction,
     peak / full scale, edge clipping, pixels per beam width, residual background, ellipse azimuth),
     and a *suggested* verdict. The measurement starts as `pending`.
   - Several `measure_at` calls may come in one turn (parallel tool calls); the controller runs
     them one after another, like a real stage.
3. `set_status(measurement_id, status, reason)`
   - `status` is `accept` or `reject`. Only accepted measurements go into the fit.
   - Guardrail: the controller refuses to accept a saturated measurement.
4. `evaluate(reason)`
   - Weighted caustic fit **per axis**, per-axis ISO coverage in units of zR
     ("x: 3 within ±1 zR, need about 5; 1 beyond 2 zR, need about 5"),
     waist accessibility (inside / outside / uncertain), general-astigmatism check, M² < 1
     sanity flag,
     and a compact table of all measurements.
   - If an axis fit fails, or no accepted position lies within ±1 zR of its fitted waist,
     d0, z0, zR and M² of that axis are reported as null with a `not_determined` reason
     (and `waist_side`: below/above the measured positions or within a gap,
     `waist_gap_mm`), never as an extrapolated number. The same applies to the `finish`
     report.
   - Waist accessibility per axis from z0 ± 2σ against the stage range: `inside`,
     `outside` or `uncertain` (then `waist_near_stage_end`: measure closer to that end).
     Only `outside` supports the `waist_not_accessible` verdict.
5. `finish(verdict, explanation)`
   - `verdict` is one of: `iso_compliant`, `estimate_not_compliant`, `waist_not_accessible`,
     `general_astigmatism_out_of_scope`, `measurement_failed`.
   - Python builds the report numbers from stored state. The model writes only the verdict and the text.
   - Guardrail: if the model claims `iso_compliant` but `evaluate` says coverage is not met,
     or flags general astigmatism or M² < 1, the controller returns an error and the agent
     must continue or choose another verdict. `waist_not_accessible` is refused unless an
     axis is `outside` and an accepted position lies within 5 % of the stage length from
     that stage end. A waist up to about 1 zR outside the stage may still give an accurate
     M²; the verdict is then `waist_not_accessible` by definition (no near-waist points on
     both sides), not because the number is unusable.
   - If a hard limit (model calls, tokens, cost) or an API error stops the agent first,
     Python writes a fallback report from session state: `estimate_not_compliant` if M²
     is determined on both axes, otherwise `measurement_failed`, with `stopped_by_limit`
     holding the reason.
6. `retake_darks(reason)`
   - Blocks the beam and takes a fresh dark set (50 frames) with the current background, so a
     background change (e.g. room light switched on) becomes part of the dark.
   - Costs one position from the budget. Use it when `measure_at` reports `background_changed`.
7. `view_frame(measurement_id, reason)`
   - Optional. Returns the first frame of a measurement as a PNG image for the model (Muse
     Spark image input), on a square-root brightness scale, so the agent can see what the
     second moments hide (e.g. a ghost reflection).
   - Separate small budget (4 views per run), no position cost. The image goes to the model
     inside the tool output and is not stored in logs or run files.

Stretch tool (only after the core works): `change_lens(f_mm, reason)` for an inaccessible
waist (ISO uses an artificial waist made with an aberration-free lens, and M² stays the same).

## 5. Loop guidance (in the system prompt)

```
Scout:    get_setup, a few measurements spread over the stage range, fix exposure if needed
Locate:   evaluate → first estimates of z0 and zR for x and y
Fill:     measure where coverage is missing, accept/reject each result, evaluate again
Finish:   when coverage is met on both axes, or when the setup cannot meet it
```

Hard limits in the controller: maximum positions, maximum agent turns, token budget per run.
Exact numbers are set after the first test runs.

## 6. Physics (ISO 11146-1)

- Second-moment diameter d = 4σ, per principal axis (x and y, not rotated).
- Integration area: centred on the centroid, 3 × the beam widths, iterated until it converges
  (2005 wording; the 2021 edition relaxed this, exact text unverified).
- Background: subtract a dark frame, then a fine offset from corner patches (2–5 % of the image),
  and **keep negative pixel values** (ISO/TR 11146-3).
- Caustic fit: d(z)² = a + b·z + c·z², fitted by minimising squared **relative** diameter
  deviations, weighted by the repeat variance (`scipy.optimize.least_squares`, not plain `polyfit`).
- Results: z0 = −b/(2c), d0 = √(a − b²/(4c)), Θ = √c (full angle), zR = d0/Θ,
  M² = (π/(8λ)) · √(4ac − b²).
- Point rule: at least 10 positions, about half within one zR of the waist and about half beyond
  2 zR, applied **separately for x and y**.
- Detector rules: at least 20 pixels across the beam (spatial resolution ≥ 1/20 of the beam width),
  clipping loss under 1 % of the power. These create real trade-offs: far from the waist the beam
  can outgrow the sensor; near the waist it can get too small for the pixels.
- General astigmatism: if the azimuth of two non-circular profiles differs by more than 10°,
  ISO 11146-2 applies. That is out of scope: the agent detects it and stops.
- M² > 1 in the simulator: Gaussian profile with an enlarged width ("embedded Gaussian").
  Hermite-Gaussian mode mixes are a later option.

Source of these points: the review read the ISO 11146-1:2005 and ISO/TR 11146-3 texts from
third-party copies. The maintainer should check the key rules once before claiming compliance.

## 7. Scenarios (all simulated)

Core:
1. Clean beam, M² close to 1.
2. Saturation at some positions only.
3. Background drift during the run ("room light turned on").
4. Beam too large for the sensor at far positions.
5. Waist outside the stage range → `waist_not_accessible` with explanation.
6. Different waist positions in x and y (simple astigmatism). All code is per-axis from day one.

Ghost reflection (with `view_frame`): `ghost_everywhere` (expected `measurement_failed`) and
`ghost_in_range` (expected `iso_compliant` after rejecting the ghost positions).

Stretch: `change_lens`, general astigmatism detection demo.

Demo features: fault buttons the judge can press during a run, a "mystery beam" button with
random hidden parameters (ground truth shown at the end), and a replay mode for recorded runs.

## 8. Evaluation

Three contenders on the same set of randomised hidden scenarios (fixed seed):
1. Naive uniform scan
2. Smart scripted ISO rule (scout, fit, place points, refit, simple fault rules)
3. Muse Spark agent

Metrics: ISO-compliance rate, M² error versus ground truth, faults noticed, positions used.
The scripted baseline must be a fair, honest attempt, not built to fail.
Batch size is decided after measuring the cost of one agent run.

## 9. Testing

- Known-answer tests: ideal Gaussian gives M² = 1; a known M² > 1 is recovered; constant offset
  is removed; near-waist-only sampling gives a biased result (divergence badly constrained).
- Optional test oracle: the open-source `laserbeamsize` package, as a dev dependency only
  (licence and use to be checked and agreed first).
- The scripted baseline also tests the tools and controller without spending API credits.

## 10. Muse Spark settings

- Responses API with function calling. Output cap is `max_output_tokens` (reasoning tokens count
  toward it). Resend the system instructions on every call (`previous_response_id` does not keep them).
  Tool results go back as `function_call_output` with the matching `call_id`.
- `reasoning_effort` low or medium for normal steps. Measure latency and tokens; don't guess.
- Development batches may use the cheaper contributor model (data used for training; we only
  send simulated data). Whether hackathon credits cover it: unverified.
- Log tokens and the `reason` field for every step.

## 11. Schedule (Nuremberg time; adjust to real available hours)

| Day | Goal |
|---|---|
| Tue 27 Oct | Simulator (per-axis Gaussian, M², noise, offset, saturation, dark frame) + tests. Workshop 17:00. |
| Wed 28 Oct | ISO widths (corner offset, iterative 3× area) + tests. Workshop 17:00. |
| Thu 29 Oct | Weighted fit and `evaluate` (coverage, checks) + tests |
| Fri 30 Oct | Tool wrappers, controller, scripted baseline from the command line. Check HF Spaces plan. |
| Sat 31 Oct – Sun 1 Nov | Muse Spark loop, logging, first agent runs |
| Mon 2 Nov | Gradio UI (caustic plot, decision log, fault buttons, replay), deploy. **Feature freeze.** |
| Tue 3 Nov | Small honest evaluation, README, record the video |
| Wed 4 Nov | Buffer. Submit in the evening. |

Drop in this order if behind: SAM 3.1 → ghost scenario and `view_frame` → `change_lens` →
polished x/y-waist demo (keep the per-axis code). Never drop: known-answer tests, scripted
baseline, replay mode, the video.

## 12. Open decisions

1. Hosting: Gradio on Hugging Face Spaces is the likely choice. Check the current free/paid
   rules for Spaces before deciding (the review reported changes; unverified).
2. `laserbeamsize` as a dev-only test oracle: yes or no.
3. Impact framing for the video: who benefits (for example small labs and teaching labs without a
   commercial M² instrument). To be written without claims we cannot back up.
4. Whether SAM 3.1 gets its one-hour test at all.
