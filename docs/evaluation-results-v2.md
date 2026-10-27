# BeamPilot evaluation results v2

Run on 27 Oct 2026 exactly as pre-registered in
[evaluation-plan-v2.md](evaluation-plan-v2.md), including its two Deviations. Code:
`main` at `3403a76`, model `muse-spark-1.2`, reasoning effort `low`, limits per run
**60 model calls**, 1.5 M tokens, $0.50. Script:
`scripts/run_evaluation.py --run --plan v2`, one run per case.
- No code changed between runs, nothing broke, and no case was re-run.
- The $4 stop rule was not reached.
- v1 ([evaluation-results.md](evaluation-results.md)) is unchanged.

**Total agent cost: $2.37** ($2.05 for `agent`, $0.32 for `agent-noview`).

**Pre-registration, all on 27 Oct 2026 and before any v2 run** (the first v2 run file
is from 21:49):

| Commit | Time | What |
|---|---|---|
| `75fe382` | 21:32 | Plan v2: new seeds, same groups and metrics, the two changes, stop rule |
| `3241612` | 21:39 | Deviations 1–2: extra prompt shortening recorded; "accept only after seeing results" wording |
| `3403a76` | 21:46 | Merge of the v2 code (call limit 60, batching prompt) into `main` |

**Read the results with these limits in mind:**
- One run per case, with new seeds. A difference of one or two cases between v1 and
  v2, or between contenders, is noise.
- Nothing in group C is held out in v2: all five fault types were seen in v1.
- C1 is "seen", as in v1.
- The scripted baseline cannot read operator notes, by design (group B).
- v2 tests two changes at once (call limit and prompt), so it cannot say which one
  caused a difference.

## Agent: v1 against v2

| `agent` | v1 (40 calls) | v2 (60 calls, batching prompt) |
|---|---|---|
| Correct, group A | 15 / 16 | **16 / 16** |
| Correct, group B | 4 / 5 | 4 / 5 |
| Correct, C2–C5 | 2 / 4 | 3 / 4 |
| Correct, C1 (seen) | 1 / 1 | **0 / 1** |
| Runs stopped by the call limit | 2 (A8, C2) | **0** |
| Mean model calls per run (max) | 21.4 (40) | **16.2** (33) |
| Tool calls per model call | 1.57 | **2.19** |
| Runs with one tool call per model call (all 31 agent runs) | 12 | **3** |
| Mean positions A / B / C | 14.4 / 13.8 / 14.6 | 15.7 / 14.6 / 15.8 |
| \|M² error\|, median | 0.2 % | 0.3 % |
| \|M² error\|, max (without C1) | 6.1 % (C5) | 6.3 % (C5) |
| Constraint violations (B) | 0 | 0 |
| `set_status` in the same call as its `measure_at` | 0 | 0 |
| Cost, `agent` (31 runs incl. noview) | $2.05 ($2.56) | $2.05 ($2.37) |

**What changed:**
- **Efficiency improved clearly.** No run hit the call limit, model calls per run fell
  by about a quarter, and batching rose from 1.57 to 2.19 tool calls per call. The
  cost per run stayed the same.
- **Both v1 failures caused by the limit are gone.** A8 and C2 are new cases with new
  seeds, so this is a tendency, not proof.
- **Accuracy is unchanged** on everything except C1 (median M² error 0.2–0.3 %).
- **Positions rose slightly** (about +1 per run). Batched near-waist measurements are
  planned before the previous result is seen, and a few came back saturated.
- **No blind accepts.** The model never accepted a measurement in the same model call
  that took it (deviation 2).

## Summary of all contenders (v2)

| Contender | A (16) | B (5) | C2–C5 (seen in v1) | C1 (seen) | Mean positions A / B / C | Violations (B) | Cost |
|---|---|---|---|---|---|---|---|
| agent | 16 / 16 | 4 / 5 | 3 / 4 | 0 / 1 | 15.7 / 14.6 / 15.8 | 0 | $2.050 |
| agent-noview | – | – | 3 / 4 | 0 / 1 | – / – / 15.0 | – | $0.319 |
| scripted | 14 / 16 | 5 / 5 | 3 / 4 | 0 / 1 | 15.9 / 17.8 / 17.0 | 8 | $0 |
| naive | 1 / 16 | 0 / 5 | 0 / 4 | 0 / 1 | 10 / 10 / 10 | 4 | $0 |

The scripted baseline's 8 violations in B are the same as in v1: B1 1, B2 1 (19
positions), B4 1, B5 5. The `view_frame` ablation is a tie on C2–C5 (3 / 4 each);
both contenders failed C1. With one run per case that is noise.

## Every wrong verdict, with the reason

From the reasons logs in `runs/evaluation-v2/` (not committed). The scripted cases
were re-run locally to see why; they are deterministic and make no API calls.

**agent**
- **B2:** it kept the 12-position note (0 violations), but 4 of its 12 positions were
  repeated scouts (saturated or low signal). It ended with 5 near and 3 far per axis
  and honestly finished `estimate_not_compliant`. This is the same pattern as v1.
- **C1 (seen):** it reached coverage first and only then looked at 3 frames. It
  described the "ghost second lobe" in its explanation, but kept those points
  because "the fit remains hyperbolic". It reported `iso_compliant` with M²
  +108 / +59 %. The fault was noticed but not acted on, which is a judgement failure.
- **C5:** it never looked at a frame and accepted far points at 30 and 60 mm, inside
  the iris range. Its own explanation mentions a 5.5 % fit residual, but it did not
  act on it. It reported `iso_compliant` with M² −6.3 / −6.3 % (tolerance 3 %).

**agent-noview**
- **C1 (seen):** it accepted ghost-inflated widths everywhere and reported
  `iso_compliant` with M² +109 / +60 %, as in v1.
- **C5:** it accepted far points at 30 and 55 mm inside the iris range and reported
  `iso_compliant` with M² −6.0 / −6.1 %.

**scripted**
- **A15:** the waist lies 29 mm before the stage. Its point at 7.8 mm was rejected,
  so `finish` refused `waist_not_accessible` (no accepted point within 12.5 mm of
  the stage end), and it ended with `estimate_not_compliant`.
- **A16:** the waist is at 172 mm with zR about 41 mm, so all far points must lie
  below about 90 mm. It reached x 4 / 5 far and its planner placed no fifth, so it
  ended with `estimate_not_compliant` with 3 positions left.
- **C1 (seen):** it accepted ghost-inflated widths: `estimate_not_compliant`, M²
  +111 / +62 %.
- **C3:** the hot pixel broke its scouts (`too_few_pixels`, `not_converged`); there
  was no accepted point near the waist, and it ended with `estimate_not_compliant`.

**naive** (reference only): wrong in 25 of 26 cases. 10 fixed positions at 1 ms
cannot reach 5 + 5 points per axis. Only A5 (`waist_not_accessible`) is right.

Also correct by the plan's scoring but worth noting: **C2** `agent` and
`agent-noview` reported `iso_compliant` with M² x +4.5 % and +4.7 %. Neither looked
at a frame, both accepted ghost points, and C2 is scored on the verdict only.

## Fault noticed (group C)

| Case | agent: keyword / by hand | agent-noview: keyword / by hand |
|---|---|---|
| C1 ghost (seen) | yes / yes, but kept the points | no / no |
| C2 ghost | no / no | no / no |
| C3 hot pixel | no / no: viewed m1–m3 ("why tiny beam at 0 mm") but named no defect | no / no |
| C4 drift | no / no | no / no |
| C5 iris | no / no | no / no |

Views used by `agent`: C1 3, C3 3, others 0. No view was used in groups A and B.
Fewer views than in v1 (C1 4, C2 4, C3 1); possibly the batching prompt makes the
agent move on faster, but one run per case cannot show that.

## Erratum to v1 deviation 2 (wording only)

v1 deviation 2 said that at 8 % power the C1 ghost stays outside the integration area
near the waist, so the waist widths stay clean. **That is wrong for C1's offset.**
The measured waist widths are inflated too (d_x about 414 µm at 125 mm against 200 µm
true, in v1 and v2). The iterated 3·d area grows until it includes the ghost. The
expected verdict (`measurement_failed`) is unaffected. v1's files are left unchanged
as committed; this note corrects the record.

## Full tables

Generated by `scripts/run_evaluation.py --summary --plan v2` from
`runs/evaluation-v2/results.jsonl`. Columns as in v1.

### Group A

| Case | Contender | Expected | Verdict | Correct | M² err x | M² err y | Violations | Positions | Noticed | Views | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | agent | iso_compliant | iso_compliant | yes | +0.0 % | +0.0 % |  | 17 |  | 0 | $0.092 |
| A1 | naive | iso_compliant | estimate_not_compliant | **no** | +1.9 % | +2.5 % |  | 10 |  |  |  |
| A1 | scripted | iso_compliant | iso_compliant | yes | +0.0 % | -0.0 % |  | 19 |  |  |  |
| A2 | agent | iso_compliant | iso_compliant | yes | +0.2 % | +0.3 % |  | 14 |  | 0 | $0.060 |
| A2 | naive | iso_compliant | estimate_not_compliant | **no** | – | – |  | 10 |  |  |  |
| A2 | scripted | iso_compliant | iso_compliant | yes | +0.1 % | +0.1 % |  | 17 |  |  |  |
| A3 | agent | iso_compliant | iso_compliant | yes | -0.0 % | -0.0 % |  | 16 |  | 0 | $0.079 |
| A3 | naive | iso_compliant | estimate_not_compliant | **no** | +1.9 % | +2.5 % |  | 10 |  |  |  |
| A3 | scripted | iso_compliant | iso_compliant | yes | -0.3 % | -0.3 % |  | 20 |  |  |  |
| A4 | agent | iso_compliant | iso_compliant | yes | +1.4 % | +1.1 % |  | 14 |  | 0 | $0.057 |
| A4 | naive | iso_compliant | estimate_not_compliant | **no** | +2.1 % | +2.8 % |  | 10 |  |  |  |
| A4 | scripted | iso_compliant | iso_compliant | yes | +0.4 % | +0.4 % |  | 15 |  |  |  |
| A5 | agent | waist_not_accessible | waist_not_accessible | yes | – | – |  | 13 |  | 0 | $0.059 |
| A5 | naive | waist_not_accessible | waist_not_accessible | yes | – | – |  | 10 |  |  |  |
| A5 | scripted | waist_not_accessible | waist_not_accessible | yes | – | – |  | 8 |  |  |  |
| A6 | agent | iso_compliant | iso_compliant | yes | -0.2 % | -0.1 % |  | 15 |  | 0 | $0.085 |
| A6 | naive | iso_compliant | general_astigmatism_out_of_scope | **no** | +2.8 % | +1.2 % |  | 10 |  |  |  |
| A6 | scripted | iso_compliant | iso_compliant | yes | +0.3 % | +0.4 % |  | 18 |  |  |  |
| A7 | agent | iso_compliant | iso_compliant | yes | +0.6 % | +0.5 % |  | 18 |  | 0 | $0.080 |
| A7 | naive | iso_compliant | estimate_not_compliant | **no** | +5.4 % | +5.8 % |  | 10 |  |  |  |
| A7 | scripted | iso_compliant | iso_compliant | yes | -0.1 % | -0.1 % |  | 18 |  |  |  |
| A8 | agent | iso_compliant | iso_compliant | yes | +0.0 % | +0.1 % |  | 14 |  | 0 | $0.103 |
| A8 | naive | iso_compliant | estimate_not_compliant | **no** | +0.3 % | +0.5 % |  | 10 |  |  |  |
| A8 | scripted | iso_compliant | iso_compliant | yes | +0.1 % | +0.2 % |  | 16 |  |  |  |
| A9 | agent | iso_compliant | iso_compliant | yes | -0.2 % | -0.1 % |  | 17 |  | 0 | $0.074 |
| A9 | naive | iso_compliant | estimate_not_compliant | **no** | – | – |  | 10 |  |  |  |
| A9 | scripted | iso_compliant | iso_compliant | yes | -0.1 % | +0.0 % |  | 17 |  |  |  |
| A10 | agent | iso_compliant | iso_compliant | yes | +0.4 % | +0.5 % |  | 18 |  | 0 | $0.078 |
| A10 | naive | iso_compliant | estimate_not_compliant | **no** | – | – |  | 10 |  |  |  |
| A10 | scripted | iso_compliant | iso_compliant | yes | +0.5 % | +0.6 % |  | 15 |  |  |  |
| A11 | agent | iso_compliant | iso_compliant | yes | +0.0 % | +0.0 % |  | 13 |  | 0 | $0.092 |
| A11 | naive | iso_compliant | estimate_not_compliant | **no** | +7.4 % | +9.5 % |  | 10 |  |  |  |
| A11 | scripted | iso_compliant | iso_compliant | yes | +0.2 % | +0.3 % |  | 16 |  |  |  |
| A12 | agent | iso_compliant | iso_compliant | yes | -0.0 % | -0.1 % |  | 14 |  | 0 | $0.095 |
| A12 | naive | iso_compliant | estimate_not_compliant | **no** | – | -0.3 % |  | 10 |  |  |  |
| A12 | scripted | iso_compliant | iso_compliant | yes | +0.1 % | +0.0 % |  | 17 |  |  |  |
| A13 | agent | iso_compliant | iso_compliant | yes | -0.0 % | -0.0 % |  | 16 |  | 0 | $0.092 |
| A13 | naive | iso_compliant | estimate_not_compliant | **no** | +2.0 % | +1.8 % |  | 10 |  |  |  |
| A13 | scripted | iso_compliant | iso_compliant | yes | +0.0 % | -0.2 % |  | 16 |  |  |  |
| A14 | agent | iso_compliant | iso_compliant | yes | +0.5 % | +0.5 % |  | 17 |  | 0 | $0.119 |
| A14 | naive | iso_compliant | estimate_not_compliant | **no** | – | – |  | 10 |  |  |  |
| A14 | scripted | iso_compliant | iso_compliant | yes | +0.1 % | +0.2 % |  | 15 |  |  |  |
| A15 | agent | waist_not_accessible | waist_not_accessible | yes | – | -2.0 % |  | 16 |  | 0 | $0.080 |
| A15 | naive | waist_not_accessible | estimate_not_compliant | **no** | – | – |  | 10 |  |  |  |
| A15 | scripted | waist_not_accessible | estimate_not_compliant | **no** | – | – |  | 10 |  |  |  |
| A16 | agent | iso_compliant | iso_compliant | yes | -0.3 % | -0.3 % |  | 19 |  | 0 | $0.093 |
| A16 | naive | iso_compliant | estimate_not_compliant | **no** | +3.0 % | +2.5 % |  | 10 |  |  |  |
| A16 | scripted | iso_compliant | estimate_not_compliant | **no** | -0.2 % | -0.3 % |  | 17 |  |  |  |

### Group B

| Case | Contender | Expected | Verdict | Correct | M² err x | M² err y | Violations | Positions | Noticed | Views | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | agent | iso_compliant | iso_compliant | yes | -0.1 % | -0.3 % | 0 | 16 |  | 0 | $0.073 |
| B1 | naive | iso_compliant | estimate_not_compliant | **no** | +3.7 % | +3.9 % | 0 | 10 |  |  |  |
| B1 | scripted | iso_compliant | iso_compliant | yes | -0.2 % | -0.3 % | 1 | 18 |  |  |  |
| B2 | agent | iso_compliant | estimate_not_compliant | **no** | +0.2 % | +0.1 % | 0 | 12 |  | 0 | $0.050 |
| B2 | naive | iso_compliant | estimate_not_compliant | **no** | +7.3 % | +7.7 % | 0 | 10 |  |  |  |
| B2 | scripted | iso_compliant | iso_compliant | yes | +0.1 % | +0.0 % | 1 | 19 |  |  |  |
| B3 | agent | iso_compliant | iso_compliant | yes | +0.6 % | – | 0 | 16 |  | 0 | $0.071 |
| B3 | naive | iso_compliant | general_astigmatism_out_of_scope (x: estimate_not_compliant) | **no** | -1.2 % | – | 0 | 10 |  |  |  |
| B3 | scripted | iso_compliant | iso_compliant | yes | -0.0 % | – | 0 | 18 |  |  |  |
| B4 | agent | iso_compliant | iso_compliant | yes | +0.0 % | -0.1 % | 0 | 14 |  | 0 | $0.064 |
| B4 | naive | iso_compliant | estimate_not_compliant | **no** | +2.8 % | +2.7 % | 0 | 10 |  |  |  |
| B4 | scripted | iso_compliant | iso_compliant | yes | -0.2 % | -0.4 % | 1 | 19 |  |  |  |
| B5 | agent | iso_compliant | iso_compliant | yes | +0.5 % | +0.5 % | 0 | 15 |  | 0 | $0.062 |
| B5 | naive | iso_compliant | estimate_not_compliant | **no** | +3.7 % | +3.9 % | 4 | 10 |  |  |  |
| B5 | scripted | iso_compliant | iso_compliant | yes | +0.1 % | +0.1 % | 5 | 15 |  |  |  |

### Group C

| Case | Contender | Expected | Verdict | Correct | M² err x | M² err y | Violations | Positions | Noticed | Views | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 (seen) | agent | measurement_failed | iso_compliant | **no** | +107.8 % | +59.1 % |  | 18 | yes | 3 | $0.116 |
| C1 (seen) | agent-noview | measurement_failed | iso_compliant | **no** | +109.0 % | +59.8 % |  | 14 | no | 0 | $0.052 |
| C1 (seen) | naive | measurement_failed | estimate_not_compliant | **no** | +123.6 % | +70.3 % |  | 10 |  |  |  |
| C1 (seen) | scripted | measurement_failed | estimate_not_compliant | **no** | +110.7 % | +61.5 % |  | 17 |  |  |  |
| C2 | agent | iso_compliant | iso_compliant | yes | +4.5 % | +0.5 % |  | 15 | no | 0 | $0.071 |
| C2 | agent-noview | iso_compliant | iso_compliant | yes | +4.7 % | +0.1 % |  | 14 | no | 0 | $0.053 |
| C2 | naive | iso_compliant | estimate_not_compliant | **no** | +6.7 % | +4.8 % |  | 10 |  |  |  |
| C2 | scripted | iso_compliant | iso_compliant | yes | +10.5 % | +0.5 % |  | 19 |  |  |  |
| C3 | agent | iso_compliant | iso_compliant | yes | +0.6 % | +0.6 % |  | 14 | no | 3 | $0.069 |
| C3 | agent-noview | iso_compliant | iso_compliant | yes | -0.3 % | -0.2 % |  | 19 | no | 0 | $0.072 |
| C3 | naive | iso_compliant | measurement_failed | **no** | – | – |  | 10 |  |  |  |
| C3 | scripted | iso_compliant | estimate_not_compliant | **no** | – | – |  | 11 |  |  |  |
| C4 | agent | iso_compliant | iso_compliant | yes | -0.1 % | -0.1 % |  | 14 | no | 0 | $0.066 |
| C4 | agent-noview | iso_compliant | iso_compliant | yes | -0.4 % | -0.5 % |  | 14 | no | 0 | $0.060 |
| C4 | naive | iso_compliant | estimate_not_compliant | **no** | +1.7 % | +1.6 % |  | 10 |  |  |  |
| C4 | scripted | iso_compliant | iso_compliant | yes | -0.7 % | -0.7 % |  | 19 |  |  |  |
| C5 | agent | iso_compliant | iso_compliant | **no** | -6.3 % | -6.3 % |  | 18 | no | 0 | $0.072 |
| C5 | agent-noview | iso_compliant | iso_compliant | **no** | -6.0 % | -6.1 % |  | 14 | no | 0 | $0.082 |
| C5 | naive | iso_compliant | estimate_not_compliant | **no** | +0.7 % | +0.8 % |  | 10 |  |  |  |
| C5 | scripted | iso_compliant | iso_compliant | yes | -2.6 % | -2.7 % |  | 19 |  |  |  |

### Totals (correct / run)

| Contender | A | B | C2-C5 (seen in v1) | C1 (seen) | Cost |
|---|---|---|---|---|---|
| agent | 16 / 16 | 4 / 5 | 3 / 4 | 0 / 1 | $2.050 |
| agent-noview | – | – | 3 / 4 | 0 / 1 | $0.319 |
| scripted | 14 / 16 | 5 / 5 | 3 / 4 | 0 / 1 | $0.000 |
| naive | 1 / 16 | 0 / 5 | 0 / 4 | 0 / 1 | $0.000 |

Agent |M² error| over determined axes: median 0.3 %, max 107.8 % (48 axes).
