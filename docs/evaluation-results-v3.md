# BeamPilot evaluation results v3: inaccessible waist with change_lens

Run on 31 Oct 2026, exactly as pre-registered in
[evaluation-plan-v3.md](evaluation-plan-v3.md), including deviation 1.
- **Code:** branch `add-change-lens` at `58e1b29` (the plan commit recording it is
  `f74badf`), model `muse-spark-1.2`, effort `low`.
- **Agent limits:** 80 model calls, 3 M tokens, $0.50 per run; `change_lens` and
  `view_frame` on.
- **Every contender:** 28 positions and the lens kit.
- **Seeds:** 900–907 (L1 → 900 … L8 → 907).
- **Order:** scripted and naive first, then the agent L1 to L8, one at a time.
- **No code changed during the runs, nothing broke, no case was re-run, and no run
  hit any limit.** Every agent run finished on its own.

**Total agent cost: $0.97** (hard stop $3, not reached).

**Pre-registration:** plan `070c0fc` (31 Oct 2026), deviation 1 `58e1b29` and
`f74badf`, all pushed before the first v3 run.

**Read the results with these limits in mind:**
- **One run per case.** Small N: a difference of one case is noise.
- **Not held out.** The cases, the lens kit, the scripted lens rule and the agent's
  lens prompt were developed together, with dry runs on development seeds. v3
  shows that the feature works end to end; it says nothing about unseen faults.
- **L1 and L4 are borderline.** Their best lens fails at some of the 27 corners
  (z0 ±10 mm, d0 ±10 %, M² ±10 %).
- **L8 lies close to the stage end:** its waist is 10 mm past the end, about
  0.8 zR.
- **Thin-lens model with exact f.** The simulator uses the exact focal length. A
  real lens (±1 % focal length) adds about minus the "dz0 per +1 % f" value below
  to the reported z0. The wording is "ISO 11146-1 procedure applied at an
  artificial waist (thin-lens model)", never "ISO compliant for the original beam".

## Summary

| Contender | Correct verdicts | Within tolerance (L1–L7) | Mean positions (L1–L7) | Cost |
|---|---|---|---|---|
| agent | **8 / 8** | **7 / 7** | 21.3 | $0.972 |
| scripted | 8 / 8 | 7 / 7 | 21.9 | $0 |
| naive | 1 / 8 | 0 / 7 | 10 | $0 |

Tolerance (both axes): |M² error| ≤ 3 %, back-calculated d0 within 5 %, z0 within
0.1 × the true zR.
- **Agent and scripted tie**, as the plan expected for a feature built with a
  fair scripted rule.
- **The agent chose the same lens as the scripted operator in 6 of 7 cases.** The
  exception is L7: +750 mm at −130 mm instead of −50 mm.
- **The agent took the first listed option every time.**
- **The naive scan never uses a lens,** so it cannot measure any waist outside
  the stage.

## Agent runs

| Case | Expected | Verdict | Lens (f @ position, mm) | Listed? fraction, rank | Lens changes | Positions | Model calls | Views | Cost |
|---|---|---|---|---|---|---|---|---|---|
| L1 | iso_compliant | iso_compliant | −200 @ −110 | yes, 0.52, 1st of 3 | 1 | 19 | 14 | 0 | $0.093 |
| L2 | iso_compliant | iso_compliant | +25 @ −20 | yes, 1.0, 1st of 4 | 1 | 21 | 19 | 0 | $0.113 |
| L3 | iso_compliant | iso_compliant | +40 @ −30 | yes, 0.95, 1st of 5 \* | 1 | 20 | 16 | 0 | $0.109 |
| L4 | iso_compliant | iso_compliant | +40 @ −20 | yes, 1.0, 1st of 5 \* | 1 | 23 | 50 | 0 | $0.184 |
| L5 | iso_compliant | iso_compliant | +750 @ −20 | yes, 0.65, 1st of 5 | 1 | 23 | 18 | 0 | $0.116 |
| L6 | iso_compliant | iso_compliant | +750 @ −20 | yes, 1.0, 1st of 5 \* | 1 | 21 | 16 | 0 | $0.096 |
| L7 | iso_compliant | iso_compliant | +750 @ −130 | yes, 0.945, 1st of 5 \* | 1 | 22 | 48 | 0 | $0.163 |
| L8 | waist_not_accessible | waist_not_accessible | none (no option listed) | – | 0 | 19 | 15 | 0 | $0.098 |

\* The pre-lens fit did not determine the waist diameter; the options carried the
label "waist diameter not determined; options assume M² between 1 and 2".

- **L8:** all three `evaluate` calls reported "No kit lens and rail position looks
  feasible from the current fit". The agent finished `waist_not_accessible`:
  "Waist fitted at ~261 mm beyond the 250 mm stage end (zR ~12 mm) …".
- **L4 and L7 took 50 and 48 model calls**, close to the old 60-call limit, so
  deviation 1 was needed. No run used a view of a frame.

## Accuracy of the lens cases (worst axis)

| Case | Contender | d0 error | z0 error | M² error | dz0 per +1 % f |
|---|---|---|---|---|---|
| L1 | agent | 0.1 % | 0.004 zR | 0.3 % | +0.29 mm |
| L1 | scripted | 0.4 % | 0.007 zR | 0.2 % | +0.29 mm |
| L2 | agent | 0.2 % | 0.006 zR | 0.0 % | −0.40 mm |
| L2 | scripted | 0.3 % | 0.001 zR | 0.2 % | −0.39 mm |
| L3 | agent | 0.3 % | 0.004 zR | 0.9 % | −0.94 mm |
| L3 | scripted | 0.3 % | 0.004 zR | 0.3 % | −0.95 mm |
| L4 | agent | 0.1 % | 0.003 zR | 0.1 % | −0.81 mm |
| L4 | scripted | 0.1 % | 0.005 zR | 0.1 % | −0.81 mm |
| L5 | agent | 0.3 % | 0.001 zR | 0.1 % | −1.18 mm |
| L5 | scripted | 0.2 % | 0.003 zR | 0.1 % | −1.18 mm |
| L6 | agent | 0.1 % | 0.012 zR | 1.2 % | −1.50 mm |
| L6 | scripted | 0.8 % | 0.013 zR | 0.3 % | −1.49 mm |
| L7 | agent | 0.1 % | 0.002 zR | 0.1 % | −3.40 mm |
| L7 | scripted | 0.2 % | 0.003 zR | 0.2 % | −2.41 mm |

All within tolerance. The largest focal-length sensitivity is L7 with the agent's
lens (+750 mm at −130 mm): −3.4 mm per +1 % f, which is 0.10 zR. A real ±1 % lens
could therefore move L7's z0 to the edge of the tolerance. The simulator uses the
exact f, so this error does not appear in the results above.

## Every wrong verdict or out-of-tolerance result

Only the naive scan was wrong (no reasons log; its fixed rules):
- **L1–L6:** 10 evenly spaced positions at 1 ms, no lens. The fit puts the waist
  `outside`, so `waist_not_accessible`. Wrong, because a kit lens works for these
  beams.
- **L7:** the waist was fitted as `uncertain` near the stage end (z0 not
  determined), so its rules gave `estimate_not_compliant`.
- **L8:** correct (`waist_not_accessible`).

The agent and the scripted operator had no wrong verdict and no result out of
tolerance.

## Full table

Generated by `scripts/run_evaluation_v3.py --summary` from
`runs/evaluation-v3/results.jsonl` (run files not committed).

| Case | Contender | Expected | Verdict | Correct | Within tolerance | Lens | Positions | Model calls | Cost |
|---|---|---|---|---|---|---|---|---|---|
| L1 | scripted | iso_compliant | iso_compliant | yes | yes | -200 @ -110 | 24 | – | $0.000 |
| L1 | naive | iso_compliant | waist_not_accessible | **no** | **no** | – | 10 | – | $0.000 |
| L2 | scripted | iso_compliant | iso_compliant | yes | yes | 25 @ -20 | 18 | – | $0.000 |
| L2 | naive | iso_compliant | waist_not_accessible | **no** | **no** | – | 10 | – | $0.000 |
| L3 | scripted | iso_compliant | iso_compliant | yes | yes | 40 @ -30 | 20 | – | $0.000 |
| L3 | naive | iso_compliant | waist_not_accessible | **no** | **no** | – | 10 | – | $0.000 |
| L4 | scripted | iso_compliant | iso_compliant | yes | yes | 40 @ -20 | 18 | – | $0.000 |
| L4 | naive | iso_compliant | waist_not_accessible | **no** | **no** | – | 10 | – | $0.000 |
| L5 | scripted | iso_compliant | iso_compliant | yes | yes | 750 @ -20 | 26 | – | $0.000 |
| L5 | naive | iso_compliant | waist_not_accessible | **no** | **no** | – | 10 | – | $0.000 |
| L6 | scripted | iso_compliant | iso_compliant | yes | yes | 750 @ -20 | 24 | – | $0.000 |
| L6 | naive | iso_compliant | waist_not_accessible | **no** | **no** | – | 10 | – | $0.000 |
| L7 | scripted | iso_compliant | iso_compliant | yes | yes | 750 @ -50 | 23 | – | $0.000 |
| L7 | naive | iso_compliant | estimate_not_compliant | **no** | **no** | – | 10 | – | $0.000 |
| L8 | scripted | waist_not_accessible | waist_not_accessible | yes | – | – | 13 | – | $0.000 |
| L8 | naive | waist_not_accessible | waist_not_accessible | yes | – | – | 10 | – | $0.000 |
| L1 | agent | iso_compliant | iso_compliant | yes | yes | -200 @ -110 | 19 | 14 | $0.093 |
| L2 | agent | iso_compliant | iso_compliant | yes | yes | 25 @ -20 | 21 | 19 | $0.113 |
| L3 | agent | iso_compliant | iso_compliant | yes | yes | 40 @ -30 | 20 | 16 | $0.109 |
| L4 | agent | iso_compliant | iso_compliant | yes | yes | 40 @ -20 | 23 | 50 | $0.184 |
| L5 | agent | iso_compliant | iso_compliant | yes | yes | 750 @ -20 | 23 | 18 | $0.116 |
| L6 | agent | iso_compliant | iso_compliant | yes | yes | 750 @ -20 | 21 | 16 | $0.096 |
| L7 | agent | iso_compliant | iso_compliant | yes | yes | 750 @ -130 | 22 | 48 | $0.163 |
| L8 | agent | waist_not_accessible | waist_not_accessible | yes | – | – | 19 | 15 | $0.098 |
