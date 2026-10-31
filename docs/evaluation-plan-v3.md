# BeamPilot evaluation plan v3: inaccessible waist with change_lens

Status: pre-registered on 31 Oct 2026, **before any run on the v3 seeds 900–907**.
- **Code:** branch `add-change-lens` at `2a7dfd5` (not merged into main).
- **Runner:** `scripts/run_evaluation_v3.py`.
- **Cases and oracle:** `scripts/v3_cases.py`.
- **No code changes between this commit and the runs.** If something breaks, the
  evaluation stops and is reported; no case is re-run.
- **v1 and v2 stay unchanged.** Later changes go under "Deviations" at the end.

Design: [change-lens-design.md](change-lens-design.md). Progress, margin table and
open issues: [change-lens-progress.md](change-lens-progress.md).

## Question

When the waist lies outside the stage, does the agent use a thin lens to form an
artificial waist, measure it to ISO 11146-1, and give back the original beam (d0,
z0) and M² within tolerance? And does it say `waist_not_accessible` honestly when
no kit lens can help? It is compared with a fair scripted operator that has a
fixed lens rule.

## Contenders

| Name | What it is |
|---|---|
| `agent` | `run_agent`, `muse-spark-1.2`, effort `low`, limits 60 model calls, 1.5 M tokens, $0.50 per run; `AgentConfig(change_lens=True)`, `view_frame` on |
| `scripted` | `scripted_iso` with the lens rule (design section 6): first listed lens option, then scouts from the prediction (at least 4 positions) and its normal fill |
| `naive` | `naive_uniform`, reference only (never uses a lens) |

Every case has the lens kit (+25, +40, +300, +750, −200 mm; rail −300…−20 mm in
10 mm steps) and **28 positions**, for every contender. One run per case.

## Cases (seeds fixed now)

| Case | Seed | Waist z0 (mm) | d0 (µm) | M² | Expected (oracle) | Lens margin |
|---|---|---|---|---|---|---|
| L1 | 900 | −30 | 200 | 1.2 | iso_compliant | **borderline** (1 lens; best works at 12/27 corners) |
| L2 | 901 | −60 | 200 | 1.2 | iso_compliant | ok (2 lenses; 27/27) |
| L3 | 902 | −100 | 250 | 1.4 | iso_compliant | ok (5; 27/27) |
| L4 | 903 | −80 | 160 | 1.0 | iso_compliant | **borderline** (6; best 23/27) |
| L5 | 904 | +280 | 200 | 1.2 | iso_compliant | ok (29; 27/27) |
| L6 | 905 | +320 | 300 | 1.6 | iso_compliant | ok (46; 27/27) |
| L7 | 906 | +380 | 250 | 1.4 | iso_compliant | ok (29; 27/27) |
| L8 | 907 | +260 | 130 | 1.0 | waist_not_accessible | no lens at any of the 27 corners |

Both axes have the same d0, z0 and M². Wavelength 1064 nm; the camera and stage are
the same as in v1 and v2.

**How each column was set:**
- **Expected verdict:** the oracle below, applied to the true beam.
- **Lens margin:** the best feasible lens for the true beam, checked at the 27
  corners z0 ±10 mm, d0 ±10 %, M² ±10 %.
- **L8:** chosen by a recorded search (`choose_l8`):
  - grid: waist −150…−10 mm and 260…500 mm, d0 100…350 µm, M² 1.0…1.8;
  - at least 10 usable stage positions;
  - no feasible lens for the true beam or for any of the 27 corners;
  - the most usable positions wins (25); 87 qualifying candidates.

**Oracle with lenses** (`scripts/v3_cases.py`):
1. `measurement_failed` if fewer than 5 stage positions (5 mm grid) give a usable
   beam size: at least 20 px across and at most half the smaller sensor side.
2. Waist inside the stage: the v1 rule (room for 5 near and 5 far points per
   axis).
3. Waist outside: `iso_compliant` if any kit × rail lens gives an artificial
   waist that passes that rule on both axes (inside the stage, ≥ 20 px, beam
   ≤ 8 mm at the lens); otherwise `waist_not_accessible`.

## Metrics

- **Correct verdict:** equals the oracle's expected verdict.
- **Within tolerance** (cases expected `iso_compliant`), on both axes:
  - |M² error| ≤ 3 %;
  - back-calculated original d0 within 5 %;
  - back-calculated z0 within 0.1 × the true zR.
- **Reported, not scored:**
  - the lens chosen and whether it was a listed option;
  - lens changes;
  - positions;
  - model calls, views and cost;
  - the "dz0 per +1 % f" sensitivity.

## Calibration and development runs (none on 900–907)

**Scripted calibration, seeds 950–957** (L1 → 950 … L8 → 957), 28 positions:

| Case | Lens | Positions | Verdict | Worst error d0 / z0 / M² | dz0 per +1 % f |
|---|---|---|---|---|---|
| L1 | −200 @ −110 | 24 | iso_compliant | 0.1 % / 0.004 zR / 0.1 % | +0.29 mm |
| L2 | +25 @ −20 | 19 | iso_compliant | 0.2 % / 0.002 zR / 0.3 % | −0.40 mm |
| L3 | +40 @ −40 | 19 | iso_compliant | 0.3 % / 0.000 zR / 0.5 % | −0.62 mm |
| L4 | +40 @ −20 | 19 | iso_compliant | 0.1 % / 0.003 zR / 0.5 % | −0.81 mm |
| L5 | +750 @ −20 | 26 | iso_compliant | 0.2 % / 0.005 zR / 0.2 % | −1.18 mm |
| L6 | +750 @ −20 | 24 | iso_compliant | 0.3 % / 0.001 zR / 0.4 % | −1.50 mm |
| L7 | +750 @ −50 | 23 | iso_compliant | 0.2 % / 0.002 zR / 0.5 % | −2.41 mm |
| L8 | none | 13 | waist_not_accessible | – | – |

- **Spread over the 7 lens cases:** d0 ≤ 0.3 %, z0 ≤ 0.005 zR, M² ≤ 0.5 %, well
  inside the tolerances.
- **Development dry run (seeds 60–75, two per case):** 16 of 16 runs reached the
  expected verdict, with worst errors d0 0.6 %, z0 0.012 zR, M² 1.0 %.

**One live agent smoke run** (`waist_outside_range`, waist −50 mm, seed 920, 28
positions; not a v3 case):
- **Outcome:** `iso_compliant`, method `artificial_waist`.
- **Lens:** +25 mm at −20 mm, the listed option with fraction 1.0.
- **Accuracy:** back-calculated d0 +0.2 %, z0 +0.002 zR, M² +0.3 % on both axes.
- **Effort and cost:** 41 model calls, 21 positions, $0.133.

## Budget and stopping

- **Estimate:** 8 agent runs at about $0.10–0.15 → about $1.
- **Hard stop at $2** for the agent runs; the remaining cases are then reported as
  not run.
- **Order:** scripted and naive for all cases first (no API), then the agent L1 to
  L8, one at a time.

## Known limitations

- **Small N.** One run per case, so a difference of one or two cases is noise.
- **Not held out.** The cases, the kit, the scripted lens rule and the agent's lens
  prompt were developed together, with dry runs on development seeds. v3 measures
  whether the feature works end to end, not generalisation to unseen faults.
- **Fixes during development.** After the first dry run L8 was re-selected twice
  and the scripted scout rule was fixed; no fix was aimed at a single case (see
  the progress note).
- **Borderline cases.** L1 and L4 are borderline: their best lens fails at some of
  the ±10 % corners. L8 lies 10 mm past the stage end (about 0.8 zR).
- **Thin-lens model, exact f.** Real lenses (±1 % focal length) would add the
  listed "per +1 % f" error to z0. The wording stays "ISO 11146-1 procedure
  applied at an artificial waist (thin-lens model)", never "ISO compliant for the
  original beam" (design section 10).

## Deviations

1. **31 Oct 2026, before any v3 run: higher agent limits and budget.**
   - **Reason:** the smoke run used 41 of 60 model calls for 21 positions, and
     L5–L7 need 23–26 positions, so the 60-call limit would probably decide some
     runs (as A8 and C2 in v1).
   - **For v3 only:** the agent limits are **80 model calls** and **3 M tokens**;
     the per-run cost cap stays at **$0.50**.
   - **The total hard stop rises from $2 to $3.**
   - **Change:** only the settings in `scripts/run_evaluation_v3.py`
     (`AGENT_CONFIG`, `BUDGET_USD`); no other code changed. Commit `58e1b29`.
   - **The v3 runs use the code at `58e1b29`.**
