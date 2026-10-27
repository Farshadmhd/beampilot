# BeamPilot evaluation results

Run on 27 Oct 2026 exactly as pre-registered in
[evaluation-plan.md](evaluation-plan.md), including its Deviations. Code: `main` at
`03fb2c1`, model `muse-spark-1.2`, reasoning effort `low`, limits per run 40 model
calls, 1.5 M tokens, $0.50. Script: `scripts/run_evaluation.py --run`, one run per
case. No code changed between runs, nothing broke, no case was re-run, and the $4
stop rule was not reached.

**Total agent cost: $2.56** ($2.05 for `agent`, $0.51 for `agent-noview`). The
scripted and naive contenders make no API calls.

## Summary

| Contender | A (16) | B (5) | C held out (C2–C5) | C1 (seen) | Mean positions A / B / C | Violations (B) | Cost |
|---|---|---|---|---|---|---|---|
| agent | 15 / 16 | 4 / 5 | 2 / 4 | 1 / 1 | 14.4 / 13.8 / 14.6 | 0 | $2.047 |
| agent-noview | – | – | 2 / 4 | 0 / 1 | – / – / 17.2 | – | $0.512 |
| scripted | 16 / 16 | 5 / 5 | 2 / 4 | 0 / 1 | 15.3 / 17.8 / 15.4 | 8 | $0 |
| naive | 2 / 16 | 0 / 5 | 0 / 4 | 0 / 1 | 10 / 10 / 10 | 4 | $0 |

- **Group A (expected tie):** roughly a tie. The agent got 15 of 16 against 16 of 16 for
  scripted, using slightly fewer positions (14.4 against 15.3 on average). Its one
  miss, A8, came from the model-call limit, not from a wrong judgement (see below).
  Agent |M² error| over all determined axes and groups: median 0.2 %, maximum 6.1 %
  (C5).
- **Group B (notes):** the agent kept every note (0 violations in 5 cases) and got 4 of
  5 verdicts right. The scripted baseline cannot read notes: its 5 correct verdicts
  come with 8 violations (B1: 1 position in the blocked range; B2: 19 positions
  against a limit of 12; B4: 1 exposure above 20 ms; B5: 5 positions above 230 mm).
  As the plan says, this is not a fair contest; it shows what the notes change.
- **Group C (held out, C2–C5):** agent 2 / 4, scripted 2 / 4. Nobody solved C5 (iris)
  within the 3 % tolerance.
- **Ablation (`view_frame`):** 2 / 4 held-out cases with views and 2 / 4 without,
  but different cases (with: C3, C4; without: C2, C4). On the seen case C1, only the
  agent with views got the verdict right. With one run per case this difference is
  not significant (plan, "Known limitations").

## Every wrong verdict, with the reason

From the reasons logs in `runs/evaluation/` (not committed).

**agent (with `view_frame`)**
- **A8:** it stopped at the 40-model-call limit with x at 4 of 5 near points; the
  controller's fallback report gave `estimate_not_compliant`. The agent made one tool
  call per model call (no batching). M² itself was right (+0.3 %).
- **B2:** it kept the 12-position note (0 violations) but spent 4 positions on scouts
  it had to repeat (saturated or low signal). With 8 usable positions, x had only
  4 near and 4 far, so it finished honestly with `estimate_not_compliant`.
- **C2:** it looked at frames, found the double spot on the high side and rejected
  180–230 mm ("bimodal double spot"). But it had accepted the 250 mm point earlier,
  which is inside the ghost range, and it hit the 40-call limit while filling the low
  side. Fallback verdict `estimate_not_compliant`; M² +0.9 / −0.6 %.
- **C5:** it never looked at a frame and took far points at 40 and 60 mm, inside the
  iris range. When those points came in, the fitted M² dropped from 1.19 to 1.12, but
  no trigger fired. It reported `iso_compliant` with M² −5.9 / −6.1 %, outside the
  3 % tolerance.

**agent-noview (ablation)**
- **C1 (seen):** it accepted ghost-inflated widths at every position and read the
  constant `axes_not_aligned` (about 36°) as a rotated beam. It reported
  `iso_compliant` with M² +109 / +60 %. The `finish` guard allows this because
  coverage was met.
- **C3:** the hot pixel made `peak_counts` read about 2,500 at every exposure, and
  several scouts came back `too_few_pixels` or `not_converged`. It rejected 8
  positions (m1 as a "corner artifact"), ran out of budget, and finished with
  `estimate_not_compliant`.
- **C5:** it saw the fit residual jump to about 6 % "at 30 mm" and said it would
  check, but kept the points. It reported `iso_compliant` with M² −7.4 / −7.3 %.

**scripted**
- **C1 (seen):** its rules accept the ghost-inflated widths, giving
  `estimate_not_compliant` with M² +109 / +60 %.
- **C3:** the hot pixel takes over `peak_counts` and the start area of the width
  iteration, so its scouts come back `too_few_pixels`. The rules give up after 4
  positions with `measurement_failed`.
- **C5:** it accepted clipped far points and reported `iso_compliant` with M²
  −3.1 / −3.2 %, just outside the 3 % tolerance.
- Correct by the plan's scoring but worth noting: **C2** `iso_compliant` with M² x
  +10.2 %. Ghost points were accepted, and C2 is scored on the verdict only.

**naive** (reference only): wrong in 24 of 26 cases.
- 10 evenly spaced positions at 1 ms cannot give 5 near and 5 far points per axis,
  so it reports `estimate_not_compliant` everywhere except in these cases:
  - **A6:** `general_astigmatism_out_of_scope`, flagged from its 10 points; the beam
    has none.
  - **C3:** `measurement_failed` (hot pixel).
  - **A5 and A11:** both correct, `waist_not_accessible`.

Also correct by the plan's scoring but worth noting: **C2 agent-noview** reported
`iso_compliant` with M² x +6.7 % because it accepted ghost points.

## Fault noticed (group C, agent)

| Case | agent: keyword / by hand | agent-noview: keyword / by hand |
|---|---|---|
| C1 ghost (seen) | yes / yes ("secondary ghost beam") | no / no |
| C2 ghost | yes / yes ("bimodal double spot") | no / no |
| C3 hot pixel | no / no: viewed m3 ("anomalous tiny beam and off centroid") but never named a defect | no / partly ("corner artifact") |
| C4 drift | no / no | no / no |
| C5 iris | no / no | no / no: saw the residual jump, did not name a cause |

The by-hand checks for C3 and C4 were pre-registered (deviation 9). In C4 the drift
(up to about 0.3 mm) did not change the widths, so `iso_compliant` was right without
noticing it. Views used per run with `view_frame`: C1 4, C2 4, C3 1, C4 0, C5 0. No
view was used in groups A and B.

## Observations not in the plan's metrics

- **The model-call limit decided two cases.** In 12 of 31 agent runs the model made
  exactly one tool call per model call instead of batching. A8 and C2 then hit the
  pre-registered 40-call limit before `finish`. All other runs finished on their own.
  Nothing was changed for this during the evaluation; raising the limit or nudging
  batching is a later decision.
- **The look-triggers did not fire on C5.** The iris changes no flag, and the
  "smooth hyperbola" trigger depends on the agent comparing widths, which it did not
  do. The ablation run noticed the residual jump but did not act on it.
- **Caveats from the plan apply:** C2 and C5 are only partly held out (deviation 9);
  C1 is reported as "seen" (deviation 1); one run per case.

## Full tables

Generated by `scripts/run_evaluation.py --summary` from
`runs/evaluation/results.jsonl`. M² err = M²_fit / M²_true − 1. Noticed = keyword
check (agent only). Views = `view_frame` calls. B3 is scored on x only.

### Group A

| Case | Contender | Expected | Verdict | Correct | M² err x | M² err y | Violations | Positions | Noticed | Views | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | agent | iso_compliant | iso_compliant | yes | +0.3 % | +0.2 % |  | 14 |  | 0 | $0.060 |
| A1 | naive | iso_compliant | estimate_not_compliant | **no** | +1.5 % | +1.1 % |  | 10 |  |  |  |
| A1 | scripted | iso_compliant | iso_compliant | yes | +0.2 % | +0.0 % |  | 19 |  |  |  |
| A2 | agent | iso_compliant | iso_compliant | yes | -0.2 % | -0.2 % |  | 14 |  | 0 | $0.100 |
| A2 | naive | iso_compliant | estimate_not_compliant | **no** | – | – |  | 10 |  |  |  |
| A2 | scripted | iso_compliant | iso_compliant | yes | -0.1 % | -0.1 % |  | 17 |  |  |  |
| A3 | agent | iso_compliant | iso_compliant | yes | -0.2 % | -0.2 % |  | 16 |  | 0 | $0.063 |
| A3 | naive | iso_compliant | estimate_not_compliant | **no** | +1.5 % | +1.1 % |  | 10 |  |  |  |
| A3 | scripted | iso_compliant | iso_compliant | yes | +0.4 % | +0.2 % |  | 20 |  |  |  |
| A4 | agent | iso_compliant | iso_compliant | yes | +0.3 % | +0.1 % |  | 13 |  | 0 | $0.085 |
| A4 | naive | iso_compliant | estimate_not_compliant | **no** | +7.7 % | +7.7 % |  | 10 |  |  |  |
| A4 | scripted | iso_compliant | iso_compliant | yes | -0.0 % | -0.1 % |  | 15 |  |  |  |
| A5 | agent | waist_not_accessible | waist_not_accessible | yes | – | – |  | 14 |  | 0 | $0.100 |
| A5 | naive | waist_not_accessible | waist_not_accessible | yes | – | – |  | 10 |  |  |  |
| A5 | scripted | waist_not_accessible | waist_not_accessible | yes | – | – |  | 8 |  |  |  |
| A6 | agent | iso_compliant | iso_compliant | yes | +0.2 % | +0.3 % |  | 16 |  | 0 | $0.065 |
| A6 | naive | iso_compliant | general_astigmatism_out_of_scope | **no** | +1.1 % | -0.6 % |  | 10 |  |  |  |
| A6 | scripted | iso_compliant | iso_compliant | yes | -0.2 % | -0.3 % |  | 17 |  |  |  |
| A7 | agent | iso_compliant | iso_compliant | yes | +0.0 % | +0.1 % |  | 14 |  | 0 | $0.060 |
| A7 | naive | iso_compliant | estimate_not_compliant | **no** | +17.2 % | +19.0 % |  | 10 |  |  |  |
| A7 | scripted | iso_compliant | iso_compliant | yes | -0.7 % | -0.7 % |  | 14 |  |  |  |
| A8 | agent | iso_compliant | estimate_not_compliant | **no** | +0.3 % | +0.3 % |  | 18 |  | 0 | $0.121 |
| A8 | naive | iso_compliant | estimate_not_compliant | **no** | -0.1 % | -0.2 % |  | 10 |  |  |  |
| A8 | scripted | iso_compliant | iso_compliant | yes | +0.6 % | +0.6 % |  | 16 |  |  |  |
| A9 | agent | iso_compliant | iso_compliant | yes | -0.8 % | -0.7 % |  | 12 |  | 0 | $0.080 |
| A9 | naive | iso_compliant | estimate_not_compliant | **no** | -0.2 % | -0.0 % |  | 10 |  |  |  |
| A9 | scripted | iso_compliant | iso_compliant | yes | -0.2 % | -0.1 % |  | 15 |  |  |  |
| A10 | agent | iso_compliant | iso_compliant | yes | +0.2 % | +0.2 % |  | 16 |  | 0 | $0.069 |
| A10 | naive | iso_compliant | estimate_not_compliant | **no** | +0.1 % | +0.1 % |  | 10 |  |  |  |
| A10 | scripted | iso_compliant | iso_compliant | yes | -0.1 % | -0.2 % |  | 16 |  |  |  |
| A11 | agent | waist_not_accessible | waist_not_accessible | yes | – | – |  | 9 |  | 0 | $0.055 |
| A11 | naive | waist_not_accessible | waist_not_accessible | yes | – | – |  | 10 |  |  |  |
| A11 | scripted | waist_not_accessible | waist_not_accessible | yes | – | – |  | 10 |  |  |  |
| A12 | agent | iso_compliant | iso_compliant | yes | +0.1 % | +0.5 % |  | 15 |  | 0 | $0.084 |
| A12 | naive | iso_compliant | estimate_not_compliant | **no** | -0.3 % | -0.1 % |  | 10 |  |  |  |
| A12 | scripted | iso_compliant | iso_compliant | yes | +0.1 % | +0.2 % |  | 20 |  |  |  |
| A13 | agent | iso_compliant | iso_compliant | yes | -0.0 % | -0.1 % |  | 15 |  | 0 | $0.093 |
| A13 | naive | iso_compliant | estimate_not_compliant | **no** | +2.3 % | +2.1 % |  | 10 |  |  |  |
| A13 | scripted | iso_compliant | iso_compliant | yes | +0.5 % | +0.5 % |  | 16 |  |  |  |
| A14 | agent | iso_compliant | iso_compliant | yes | +0.2 % | +0.0 % |  | 16 |  | 0 | $0.094 |
| A14 | naive | iso_compliant | estimate_not_compliant | **no** | +12.3 % | +11.4 % |  | 10 |  |  |  |
| A14 | scripted | iso_compliant | iso_compliant | yes | +0.9 % | +0.8 % |  | 16 |  |  |  |
| A15 | agent | iso_compliant | iso_compliant | yes | +0.4 % | +0.2 % |  | 14 |  | 0 | $0.058 |
| A15 | naive | iso_compliant | estimate_not_compliant | **no** | +9.8 % | +8.5 % |  | 10 |  |  |  |
| A15 | scripted | iso_compliant | iso_compliant | yes | -0.1 % | -0.4 % |  | 12 |  |  |  |
| A16 | agent | iso_compliant | iso_compliant | yes | +0.3 % | +0.1 % |  | 14 |  | 0 | $0.054 |
| A16 | naive | iso_compliant | estimate_not_compliant | **no** | +7.6 % | +5.9 % |  | 10 |  |  |  |
| A16 | scripted | iso_compliant | iso_compliant | yes | +0.5 % | +0.2 % |  | 14 |  |  |  |

### Group B

| Case | Contender | Expected | Verdict | Correct | M² err x | M² err y | Violations | Positions | Noticed | Views | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | agent | iso_compliant | iso_compliant | yes | +0.5 % | +0.4 % | 0 | 13 |  | 0 | $0.054 |
| B1 | naive | iso_compliant | estimate_not_compliant | **no** | +2.0 % | +1.5 % | 0 | 10 |  |  |  |
| B1 | scripted | iso_compliant | iso_compliant | yes | -0.0 % | -0.1 % | 1 | 18 |  |  |  |
| B2 | agent | iso_compliant | estimate_not_compliant | **no** | -0.3 % | -0.3 % | 0 | 12 |  | 0 | $0.074 |
| B2 | naive | iso_compliant | estimate_not_compliant | **no** | -2.7 % | -2.8 % | 0 | 10 |  |  |  |
| B2 | scripted | iso_compliant | iso_compliant | yes | -0.6 % | -0.6 % | 1 | 19 |  |  |  |
| B3 | agent | iso_compliant | iso_compliant | yes | +0.3 % | – | 0 | 17 |  | 0 | $0.087 |
| B3 | naive | iso_compliant | estimate_not_compliant | **no** | +2.1 % | – | 0 | 10 |  |  |  |
| B3 | scripted | iso_compliant | iso_compliant | yes | -0.3 % | – | 0 | 18 |  |  |  |
| B4 | agent | iso_compliant | iso_compliant | yes | +0.0 % | +0.1 % | 0 | 14 |  | 0 | $0.067 |
| B4 | naive | iso_compliant | estimate_not_compliant | **no** | +2.4 % | +2.8 % | 0 | 10 |  |  |  |
| B4 | scripted | iso_compliant | iso_compliant | yes | +0.1 % | +0.1 % | 1 | 19 |  |  |  |
| B5 | agent | iso_compliant | iso_compliant | yes | +0.0 % | -0.0 % | 0 | 13 |  | 0 | $0.072 |
| B5 | naive | iso_compliant | estimate_not_compliant | **no** | +6.0 % | +6.2 % | 4 | 10 |  |  |  |
| B5 | scripted | iso_compliant | iso_compliant | yes | +0.5 % | +0.5 % | 5 | 15 |  |  |  |

### Group C

| Case | Contender | Expected | Verdict | Correct | M² err x | M² err y | Violations | Positions | Noticed | Views | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 (seen) | agent | measurement_failed | measurement_failed | yes | – | – |  | 16 | yes | 4 | $0.119 |
| C1 (seen) | agent-noview | measurement_failed | iso_compliant | **no** | +109.1 % | +59.6 % |  | 17 | no | 0 | $0.122 |
| C1 (seen) | naive | measurement_failed | estimate_not_compliant | **no** | +104.3 % | +56.2 % |  | 10 |  |  |  |
| C1 (seen) | scripted | measurement_failed | estimate_not_compliant | **no** | +108.7 % | +59.9 % |  | 17 |  |  |  |
| C2 | agent | iso_compliant | estimate_not_compliant | **no** | +0.9 % | -0.6 % |  | 18 | yes | 4 | $0.136 |
| C2 | agent-noview | iso_compliant | iso_compliant | yes | +6.7 % | -0.5 % |  | 19 | no | 0 | $0.103 |
| C2 | naive | iso_compliant | estimate_not_compliant | **no** | +3.0 % | -0.3 % |  | 10 |  |  |  |
| C2 | scripted | iso_compliant | iso_compliant | yes | +10.2 % | +0.0 % |  | 19 |  |  |  |
| C3 | agent | iso_compliant | iso_compliant | yes | +0.1 % | -0.0 % |  | 13 | no | 1 | $0.084 |
| C3 | agent-noview | iso_compliant | estimate_not_compliant | **no** | -0.2 % | -0.2 % |  | 20 | no | 0 | $0.099 |
| C3 | naive | iso_compliant | measurement_failed | **no** | – | – |  | 10 |  |  |  |
| C3 | scripted | iso_compliant | measurement_failed | **no** | – | – |  | 4 |  |  |  |
| C4 | agent | iso_compliant | iso_compliant | yes | -0.3 % | -0.3 % |  | 13 | no | 0 | $0.050 |
| C4 | agent-noview | iso_compliant | iso_compliant | yes | -0.2 % | -0.2 % |  | 14 | no | 0 | $0.093 |
| C4 | naive | iso_compliant | estimate_not_compliant | **no** | -0.5 % | -0.6 % |  | 10 |  |  |  |
| C4 | scripted | iso_compliant | iso_compliant | yes | -0.6 % | -0.6 % |  | 19 |  |  |  |
| C5 | agent | iso_compliant | iso_compliant | **no** | -5.9 % | -6.1 % |  | 13 | no | 0 | $0.060 |
| C5 | agent-noview | iso_compliant | iso_compliant | **no** | -7.4 % | -7.3 % |  | 16 | no | 0 | $0.095 |
| C5 | naive | iso_compliant | estimate_not_compliant | **no** | +3.2 % | +4.3 % |  | 10 |  |  |  |
| C5 | scripted | iso_compliant | iso_compliant | **no** | -3.1 % | -3.2 % |  | 18 |  |  |  |

### Totals (correct / run)

| Contender | A | B | C held out (C2-C5) | C1 (seen) | Cost |
|---|---|---|---|---|---|
| agent | 15 / 16 | 4 / 5 | 2 / 4 | 1 / 1 | $2.047 |
| agent-noview | – | – | 2 / 4 | 0 / 1 | $0.512 |
| scripted | 16 / 16 | 5 / 5 | 2 / 4 | 0 / 1 | $0.000 |
| naive | 2 / 16 | 0 / 5 | 0 / 4 | 0 / 1 | $0.000 |

Agent |M² error| over determined axes: median 0.2 %, max 6.1 % (45 axes).
