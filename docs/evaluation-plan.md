# BeamPilot evaluation plan

Status: pre-registered on 27 Oct 2026, **before any evaluation run**. Committed to Git
so the plan, seeds and expected verdicts cannot drift towards the results. Any later
change is listed under "Deviations" at the end, with date and reason.

## Question

Does the Muse Spark agent measure M² to ISO 11146-1 at least as well as a fair
scripted operator on known scenarios, and does it do better where judgement is needed:
plain-language constraints from the operator, and faults it was not developed against?
We report all runs, including failures; there is no cherry-picking and no re-running a
case until it passes.

## Contenders

| Name | What it is |
|---|---|
| `agent` | `beampilot.agent.run_agent`, model `muse-spark-1.2`, reasoning effort `low`, limits 40 model calls, 1.5 M tokens, $0.50 per run (the defaults at commit time) |
| `scripted` | `beampilot.baselines.scripted_iso`: good-faith rule-based operator |
| `naive` | `beampilot.baselines.naive_uniform`: reference only (10 evenly spaced positions, fixed 1 ms) |

All three use only the public tools. One run per case (cost); the scripted and naive
contenders are deterministic for a given seed.

## Group A: known scenarios and mystery beams (expect a tie)

| Case | Scenario | Seed | Expected verdict |
|---|---|---|---|
| A1 | clean | 100 | iso_compliant |
| A2 | partial_saturation | 100 | iso_compliant |
| A3 | background_drift | 100 | iso_compliant |
| A4 | beam_too_large | 100 | iso_compliant |
| A5 | waist_outside_range | 100 | waist_not_accessible |
| A6 | different_waists | 100 | iso_compliant |
| A7-A16 | mystery (`random_scenario`) | 1000-1009 | by the oracle rule below |

Contenders: agent, scripted, naive. The scripted operator was tuned on these six
scenarios in good faith, so we **expect a tie** between agent and scripted on A1-A6.
The mystery beams use seeds never run during development.

**Oracle rule for expected verdicts** (from the hidden truth, decided now): for each
axis, compute zR and the windows on the stage: near = [z0 − 0.9 zR, z0 + 0.9 zR],
far = |z − z0| ≥ 2.3 zR with predicted diameter ≤ half the smaller sensor side. Room
for k points in a window of length L is floor(L / (0.15 zR)) + 1.
- If the waist of any axis lies outside the stage range → `waist_not_accessible`.
- Else if, on both axes, the near window holds room for ≥ 5 points and the far
  windows together hold room for ≥ 5 points → `iso_compliant`.
- Else → `estimate_not_compliant`.

Faults built into a mystery beam (room light, 3x power) do not change its expected
verdict. The rule is implemented as a function in the evaluation script before any
run, and the expected verdicts of A7-A16 are printed and committed before the runs.

## Group B: operator-note constraints

Plain-language notes in the agent's first message; the tools do **not** enforce them.
**The scripted baseline cannot read notes.** It runs the same case without the note,
and we report its violations openly; that is a limitation of scripts, not a fair
contest on this group.

| Case | Scenario | Seed | Operator note | Violation if | Expected verdict |
|---|---|---|---|---|---|
| B1 | clean | 200 | "The stage cannot be used between 140 and 160 mm: a mount is in the way." | any `measure_at` with 140 ≤ z ≤ 160 mm | iso_compliant |
| B2 | clean | 201 | "Use at most 12 stage positions in total; the stage time is booked." | positions used > 12 | iso_compliant |
| B3 | different_waists | 202 | "Only the x axis matters for this job; you may ignore y." | none (scored on x only; positions used compared) | iso_compliant (x) |
| B4 | clean | 203 | "Do not use exposures above 20 ms: the camera overheats." | any exposure > 20 ms | iso_compliant |
| B5 | beam_too_large | 204 | "Do not move the stage above 230 mm: the rail is damaged there." | any `measure_at` with z > 230 mm | iso_compliant |

For B3 the verdict and M² error are scored on x only.

## Group C: held-out faults (defined now, before running)

None of these are in the system prompt by name. The two ghost scenarios built during
development (`ghost_everywhere`, `ghost_in_range`) are **not** used here; group C uses
other parameters and seeds. Hot pixel and drift are implemented after this commit,
exactly as specified, with tests, and without changing the agent's prompt for them.

| Case | Base | Seed | Fault (exact definition) | Expected verdict |
|---|---|---|---|---|
| C1 | clean | 300 | Ghost everywhere: `Ghost(relative_power=0.08, offset_x_m=-0.35e-3, offset_y_m=0.25e-3)` | measurement_failed, with the ghost named as the cause |
| C2 | clean | 301 | Ghost only for 170 ≤ z ≤ 250 mm: `Ghost(relative_power=0.12, offset_x_m=0.4e-3, offset_y_m=0.0, z_min_m=0.170, z_max_m=0.250)` | iso_compliant (ghost positions rejected, far points taken below the waist) |
| C3 | clean | 302 | Hot pixel cluster appearing after the dark frames: 2 × 2 pixels at rows 40-41, columns 470-471, +2500 counts in every light frame (not in darks) | iso_compliant, M² within 3 % (affected frames rejected or the defect explained) |
| C4 | clean | 303 | Slow pointing drift: beam centre x0 = +25 µm × (number of measurements taken so far), y0 = 0 | iso_compliant (drift noticed and named) |

## Metrics

Per run:
- **Correct verdict**: equals the expected verdict (group B3: x only).
- **M² error**: (M²_fit / M²_true − 1) per axis, when M² is determined; reported, and
  summarised as the median and maximum of |error| over cases with a determined M².
- **Constraint violations** (group B): count per the table, from the tool log.
- **Positions used**: measurements plus dark retakes.
- **Cost**: model calls, input/cached/output tokens, USD (prices at commit time;
  check the pricing page).
- **Fault noticed** (group C, agent): the fault is named in a `reason` or in the
  explanation. Keywords decided now: ghost → "ghost", "second spot", "reflection",
  "double"; hot pixel → "hot pixel", "bright pixel", "stuck", "defect"; drift →
  "drift", "moving", "centroid".

## Budget and stopping

Estimated agent cost: about $0.08 per run, 31 agent runs → about $2.50. Hard stop if
the agent runs of this evaluation pass **$4** in total; the remaining cases are then
reported as not run.

## Reporting

One table per group with all contenders, the metrics above, and a short note for every
wrong verdict explaining what went wrong (from the reasons log). Saved runs of the
agent go to `runs/evaluation/` (not committed); a summary table goes to
`docs/evaluation-results.md`.

## Known limitations of this plan

- One run per case: agent results vary from run to run; differences of one or two
  cases are not significant.
- The scripted baseline was developed against the six known scenarios and the
  developer saw the mystery-beam generator; group A favours it, by design.
- Group B is not a contest the scripted baseline can win; it shows what notes change.
- The oracle rule is a simple geometric judgement, not the standard's text.

## Deviations

All of these are dated 27 Oct 2026 and committed **before any evaluation run**.

1. **C1 is no longer held out.** The development scenario `ghost_everywhere` is the
   same fault type, and after seeing the agent fail on it (it never looked at a frame)
   the system prompt was changed (deviation 6). C1 is still run with its seed and
   definition, but it is reported separately as **"seen"** and left out of the
   held-out totals. C2 is the same fault type as the development scenario
   `ghost_in_range`; that was known when the plan was written, and the prompt was
   not changed after that run, so C2 stays held out as planned.
2. **C1 wording fix.** At 8 % power the ghost stays outside the iterated 3·d
   integration area near the waist, so the waist widths come out clean and only the
   far field is inflated. The sentence "no position gives true widths" was wrong for
   C1. The expected verdict (`measurement_failed`, ghost named) is unchanged,
   because no ISO-compliant far field can be measured on either side.
3. **New held-out case C5: an iris that clips the beam in a z range.**

   | Case | Base | Seed | Fault (exact definition) | Expected verdict |
   |---|---|---|---|---|
   | C5 | clean | 304 | Circular aperture in front of the sensor for 30 ≤ z ≤ 70 mm: `Aperture(radius_m=0.35e-3, z_min_m=0.030, z_max_m=0.070)`, centred on the beam axis; the profile is truncated, diffraction ignored | iso_compliant, M² within 3 % (clipped positions rejected or avoided) |

   Widths inside the range come out too small (−21 % at 30 mm, −11 % at 50 mm,
   about −1.5 to −2.5 % at 70 mm depending on exposure and noise). This was checked once on the simulator, without the agent, to
   make sure that **no current flag fires**. Far points are available outside the
   iris range below the waist (below 30 mm) and above it (170–250 mm). Fault-noticed
   keywords: "aperture", "iris", "truncat", "vignett", "cut off", "obstruct",
   "smaller than expected".
4. **Ablation: group C runs twice for the agent.** `agent` is the normal agent with
   `view_frame` (4 views). `agent-noview` has the tool removed from its tool list,
   the "Looking at frames" section removed from its system prompt, and a view
   budget of 0. Same seeds and cases. The scripted and naive contenders run group C
   once (they never look at frames).
5. **Scoring of C3 and C5.** Their expected column includes "M² within 3 %". A run
   counts as correct only if the verdict matches **and** |M² error| ≤ 3 % on both
   axes.
6. **Code changes after pre-registration, before any run.**
   - The system prompt now names general triggers for looking at a frame: a flag at
     every position, an azimuth that stays tilted, widths that do not follow a
     smooth hyperbola. It does not name ghosts, apertures or any group C fault.
   - Only the 2 most recent images stay in the resent history; older ones become a
     short text stub.
   - Hot pixels, pointing drift and the aperture were added to the simulator.
7. **Budget and order.** Agent runs: 16 (A) + 5 (B) + 2 × 5 (C) = 31. The original
   "31" was a miscount of 25. The stop rule stays at **$4** in total for this
   evaluation's agent runs. Order: A1–A16, B1–B5, then C1–C5, each case with and
   without `view_frame`. Up to 3 runs go in parallel. The budget is checked before
   each run starts, so at most the runs already going (≤ 2 × $0.50 per-run cap)
   can pass $4.
8. **Scoring details and oracle output, fixed in `scripts/run_evaluation.py`
   before any agent run.**
   - Expected verdicts of the mystery beams from the oracle rule: A7–A10
     `iso_compliant`, A11 `waist_not_accessible`, A12–A16 `iso_compliant`. As a
     check, the rule gives the same verdicts as the table for A1–A6.
   - B3 (x only): the `finish` guard does not allow `iso_compliant` while y
     coverage is missing, so B3 is scored on the verdict the x axis alone supports.
     It is `iso_compliant` if the final evaluation has x coverage met, x M²
     determined, no M² < 1 flag on x and no general astigmatism;
     `estimate_not_compliant` if x M² is determined; otherwise
     `measurement_failed`.
   - Violations are counted on `measure_at` calls that produced a measurement. B2
     counts 1 if more than 12 positions were used.
   - Fault noticed (agent only): the keywords are searched case-insensitively in
     all tool-call reasons and the `finish` explanation.
   - The scripted and naive contenders were run once, without any API call, while
     the script was being tested. They are deterministic, so the evaluation run
     reproduces those results. No agent run of an evaluation case happened before
     this commit.
9. **Caveats from the physics review, recorded before any agent run.**
   - **C2 and C5 are only partly held out.** No fault is named in the prompt, but
     the new look-triggers were written after seeing a ghost run. "Azimuth stays
     tilted" and "one side of the waist does not match the other" fit C2 (a ghost
     over 170–250 mm gives a tilted azimuth and a lopsided caustic) and C5. C2 and
     C5 stay in the held-out totals as planned, and the report states this caveat
     next to them.
   - **C3 mainly breaks the exposure feedback, not the widths.** Whenever the beam
     peak is below the hot cluster (2,500 counts), the frame maximum is the cluster.
     `peak_counts` then reads about 2,500 whatever the exposure, `low_signal` never
     fires, and at low exposure the width iteration locks onto the cluster
     (`too_few_pixels` or `not_converged`). With a beam peak above 2,500 the widths
     are biased by at most +0.3 %. The expected verdict is achievable, and the
     baselines fail it (deviation 8). The agent may describe it as "beam too small"
     without a fault keyword, so "fault noticed" for C3 is also checked by hand
     against the reasons log.
   - **The C4 keyword "centroid" is weak.** The agent can write "centroid" without
     noticing drift. "Fault noticed" for C4 is also checked by hand; both the
     keyword result and the hand check are reported.
