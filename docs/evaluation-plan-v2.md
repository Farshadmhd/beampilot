# BeamPilot evaluation plan v2

Status: pre-registered on 27 Oct 2026, **before any v2 run and before the code
changes it tests**. This file is the first commit on branch `add-agent-efficiency`.
Any later change goes under "Deviations" at the end, with date and reason.

v1 ([evaluation-plan.md](evaluation-plan.md), results in
[evaluation-results.md](evaluation-results.md)) **stays unchanged**. v2 does not
replace it; v2 results go to `docs/evaluation-results-v2.md`.

## Why v2

v1 showed an efficiency problem, not a judgement problem:
- **A8 and C2 were stopped by the 40-model-call limit** before `finish`; the
  controller's fallback verdict made both wrong.
- **In 12 of 31 agent runs the model made one tool call per model call** instead of
  requesting independent calls together.

v2 makes exactly two changes to the agent and nothing else (tools, simulator,
scoring, scripted and naive contenders all stay as in v1):

1. **Model-call limit 40 → 60.** The token budget (1.5 M per run) and the cost cap
   ($0.50 per run) stay unchanged.
2. **Prompt:** it says clearly that independent measurements are to be requested
   together in one turn (parallel tool calls), and that after each `evaluate` the
   agent plans several positions at once. The prompt stays under 1,500 words and
   still names no fault.

The question: with these changes, does the agent finish on its own in every run,
with fewer model calls, and without losing accuracy?

## Contenders

| Name | What it is |
|---|---|
| `agent` | `run_agent`, model `muse-spark-1.2`, reasoning effort `low`, limits **60 model calls**, 1.5 M tokens, $0.50 per run |
| `agent-noview` | as `agent`, with `view_frame` switched off (group C only, as v1 deviation 4) |
| `scripted` | `scripted_iso`, unchanged |
| `naive` | `naive_uniform`, unchanged (reference only) |

## Cases: same groups and definitions as v1, new seeds

**New seeds.** None of these seeds was used before. Used so far: 0–99 (tests and
development runs), 100, 200–204, 300–304 and 1000–1009 (v1).

| Group | Cases | Definition | Seeds (v1 → v2) |
|---|---|---|---|
| A | A1–A6 | the six known scenarios | 100 → **500** |
| A | A7–A16 | mystery beams (`random_scenario`) | 1000–1009 → **3000–3009** |
| B | B1–B5 | operator notes, word for word as in v1 | 200–204 → **600–604** |
| C | C1–C5 | the five faults exactly as defined in v1 (incl. C5 from deviation 3) | 300–304 → **700–704** |

The seed sets only the noise for A1–A6, B and C, so their expected verdicts are
the same as in v1. Expected verdicts of the new mystery beams, from v1's oracle
rule (computed before this commit, without any API call):

| Case | Seed | Expected | Case | Seed | Expected |
|---|---|---|---|---|---|
| A7 | 3000 | iso_compliant | A12 | 3005 | iso_compliant |
| A8 | 3001 | iso_compliant | A13 | 3006 | iso_compliant |
| A9 | 3002 | iso_compliant | A14 | 3007 | iso_compliant |
| A10 | 3003 | iso_compliant | A15 | 3008 | waist_not_accessible |
| A11 | 3004 | iso_compliant | A16 | 3009 | iso_compliant |

**Nothing in group C is held out in v2.** All five fault types were seen in the v1
results (C1 also during development). v2 measures efficiency and repeatability on
new noise and new mystery beams, not generalisation to new faults. Group C totals
are reported as "seen in v1".

## Metrics

The same as v1, with the same scoring rules (v1 deviations 5 and 8):
- correct verdict, with B3 scored on x only;
- C3 and C5 count as correct only if M² is within 3 %;
- M² error;
- constraint violations in group B;
- positions used;
- cost;
- fault noticed (keywords, plus the hand check for C3 and C4).

Also reported, not scored:
- model calls per run;
- tool calls per model call (batching);
- the number of runs stopped by a limit.

## Comparison with v1

`docs/evaluation-results-v2.md` gets, next to the v2 tables, a short table for
`agent` v1 against v2:
- correct verdicts per group;
- mean positions;
- mean model calls;
- tool calls per model call;
- runs stopped by the call limit;
- median and maximum |M² error|;
- cost.

The seeds differ between v1 and v2 and each case runs once. A difference of one or
two cases is therefore noise, not a measured effect.

## Budget and stopping

Agent runs: 16 (A) + 5 (B) + 2 × 5 (C) = 31.
- **Hard stop at $4** in total for the v2 agent runs; the remaining cases are then
  reported as not run.
- Order: A1–A16, B1–B5, then C1–C5, each case with and without `view_frame`.
- Up to 3 runs go in parallel. The budget is checked before each run starts.
- No code changes between runs. If something breaks, the evaluation stops and is
  reported, and no case is re-run.

## Known limitations

- One run per case, as in v1.
- Two changes at once (call limit and prompt). If v2 improves, this plan cannot say
  which change caused it. The model-call and batching numbers give a hint.
- A higher call limit can raise the cost per run; the per-run cap stays at $0.50.

## Deviations

None yet.
