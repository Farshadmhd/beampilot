# change_lens: progress

Branch `add-change-lens` (not merged). Design: [change-lens-design.md](change-lens-design.md).

## Done

| Phase | Commit | What |
|---|---|---|
| Design | `5decdbf` | Approved design with the corrections folded in |
| 1 | `091e2c2` | `lens.py`: thin-lens ABCD forward and inverse, Self (1983) cross-check, focal-length sensitivity, σ propagation by numerical Jacobian, a, b, c helpers. 33 known-answer tests (`tests/test_lens.py`). |
| 2a | `cbb3b94` | `LensKit` on `Scenario` (default None); session segments, the true beam through the current lens, `change_lens` with kit, rail, change-limit, budget and 8 mm aperture guards (the aperture check is the only truth-based check, a yes/no). `set_status` refuses earlier segments; `evaluate` fits only the current segment. Lens fields appear only with a kit. |
| 2b | `a90eff0` | `lens_plan.py`: `lens_options` from 200 seeded draws in a, b, c (derived from the fit's d0, z0, θ covariance; `CausticFit.cov` added); fit-based prediction in `change_lens` with `listed_option`; back-calculated original beam with σ and "per +1 % f" sensitivity; report fields (method, lens, artificial_waist, original_beam_back_calculated, m2, wording, simplifications). Truth-leak test. |
| 2c | `2697025`, `6b96414` | Scripted lens rule (first listed option on `waist_not_accessible`) and the end-to-end test. |
| Fix 1 | `7d0de04` | Fairness: after a lens, the scripted operator scouts from the prediction `change_lens` returns (the same the agent gets): at z0', ±1 zR' and ±2.5 zR', clipped to the stage, then its normal fill. |
| Fix 2 | `2b4b829` | `lens_options` only for an axis `outside`; for `uncertain`, `evaluate` keeps its advice to measure closer to that stage end. |
| Review E1 | `ebc1cd0` | A second lens is planned from the original beam (the fit is mapped back through the current lens first), as the simulator renders it. |
| E2 | `bc6def7` | Collapsed far-field fit: planned from θ and z0 with M² in [1, 2], labelled. |
| R1 | `4389598` | A clip refusal (truth-based) costs one position. |
| v3 setup | `6fda336` | `scripts/v3_cases.py`: L1–L8 with 28 positions, the oracle with lenses (incl. `measurement_failed`), the recorded L8 choice. |
| Scout fix | `bf54e93` | After a lens, at least 4 distinct scouts (missing ones step into the stage from a clipped end). |
| L8 margin | `4336058` | L8 re-selected with a 27-corner margin: waist +260 mm, d0 130 µm, M² 1.0. |
| Docs | `545044c` | Design sections 6 and 8 and the margin table updated. |
| Dry run | – | Seeds 60–75, 28 positions: 16/16 runs reach the oracle's verdict; worst errors d0 0.6 %, z0 0.012 zR, M² 1.0 %. |
| Phase 3 | `396e2e3` | `AgentConfig(change_lens=False)`; byte-identical tools and prompt with the switch off (hash test); `prompts/change_lens.md` (1,631 words with lenses); `run_agent.py --lens --max-positions`. Live smoke run (seed 920, $0.133): `iso_compliant`, +25 mm @ −20 mm, d0 +0.2 %, z0 +0.002 zR, M² +0.3 %. |
| Phase 4 | `2a7dfd5` + plan | Calibration on 950–957 (8/8 expected verdicts); `scripts/run_evaluation_v3.py`; `docs/evaluation-plan-v3.md`. |
| v3 | `58e1b29`, `f74badf`, `3f7b01a` | Deviation 1 (80 calls, 3 M tokens, $3 stop), then the v3 runs: agent 8/8 correct and 7/7 within tolerance ($0.97), scripted 8/8 and 7/7, naive 1/8. `docs/evaluation-results-v3.md`. |
| Phase 5 | `ac055de` + docs | "Lens kit" checkbox (off by default, 28 positions, 80 calls), log line per lens change, caustic of the current segment only; replay of lens runs and of old run files (tested). README section, CLAUDE.md note. |

Tests: see the latest commit (`.venv\Scripts\python.exe -m pytest`).

## Next

Phases 1–5 are done. Next: the maintainer's merge of `add-change-lens` into main.

## Margin of the v3 cases (31 Oct 2026)

Computed by `scripts/v3_cases.py` (`margin_report`), from the true beam:
- **Feasible lenses:** the kit × rail lenses that work.
- **Best lens:** the one with the largest spare margin; ties go to the larger |f|,
  then to the lens closer to the stage.
- **Corners:** at how many of the 27 corners (z0 ±10 mm, d0 ±10 %, M² ±10 %) the
  best lens still works.

| Case | Waist (mm) | d0 (µm) | M² | Feasible lenses | Best lens | Corners | Margin |
|---|---|---|---|---|---|---|---|
| L1 | −30 | 200 | 1.2 | 1 | −200 mm @ −110 mm | 12/27 | **borderline** |
| L2 | −60 | 200 | 1.2 | 2 | +25 mm @ −20 mm | 27/27 | ok |
| L3 | −100 | 250 | 1.4 | 5 | +40 mm @ −20 mm | 27/27 | ok |
| L4 | −80 | 160 | 1.0 | 6 | +40 mm @ −20 mm | 23/27 | **borderline** |
| L5 | +280 | 200 | 1.2 | 29 | +750 mm @ −20 mm | 27/27 | ok |
| L6 | +320 | 300 | 1.6 | 46 | +750 mm @ −20 mm | 27/27 | ok |
| L7 | +380 | 250 | 1.4 | 29 | +750 mm @ −30 mm | 27/27 | ok |
| L8 | +260 | 130 | 1.0 | 0 | – | – | no lens at any corner (by selection) |

## Open issues

1. **Waist after the stage needed more positions.** With 24 positions the scripted
   operator ran out at 280 mm (development seeds 41–43), although its lens choice
   and back-calculation were accurate. Resolved: the v3 budget is 28 positions for
   every contender (`6fda336`); L5–L7 then succeeded in the dry run, the
   calibration and v3.
2. (Resolved, `2b4b829`.) Lens options are listed only for `outside`.
3. (Resolved, `bc6def7`.) **Collapsed far-field-only fit (review E2).** Inside lens
   planning only, an undetermined waist diameter is planned from θ and z0, with M²
   drawn in [1, 2], and labelled. `fit_caustic` is unchanged.
4. (Resolved, `4389598`.) **Truth probing via the 8 mm refusal (review R1).** A
   refusal because the beam really clips at the lens costs one position; kit,
   rail, change-limit and finished-run refusals stay free.
5. (Resolved, `6fda336` and `4336058`.) **L8.**
   - The first L8 (waist at 450 mm) had no usable stage position. The oracle now
     expects `measurement_failed` for fewer than 5 usable positions.
   - The second choice (waist −20 mm, d0 230 µm) had no margin.
   - The current L8 is the waist at **+260 mm, d0 130 µm, M² 1.0**. It was chosen
     by the recorded search in `scripts/v3_cases.py`: at least 10 usable stage
     positions, and no feasible kit × rail lens for the true beam or for any of
     the 27 corners (z0 ±10 mm, d0 ±10 %, M² ±10 %).
   - The rule picks the most usable positions: 25 of 87 candidates, with d0 from
     100 µm. Its waist lies 10 mm past the stage end (about 0.8 zR).
6. (Resolved.) README sentence on the focal-length tolerance (review R3).
7. (Resolved.) Design doc: sorting by the larger spare margin, linearised a, b, c
   draws, and the collapsed-fit planning rule.
8. (Resolved, `bf54e93` and `4336058`.) **Dry-run failure on the second L8 (dev
   seed 74).** A barely-listed lens put the predicted waist at the stage start, and
   the clipped post-lens scouts collapsed to 3 positions. Fixed in general: after
   a lens there are always at least 4 distinct scouts, stepping into the stage from
   a clipped end. L8 was re-selected with the margin. The rerun reached the
   expected verdict in 16 of 16 runs.
9. **Focal-length sensitivity (open, documented).** For waists after the stage it
   reaches 1.2–3.4 mm in z0 per +1 % f; for the agent's L7 lens, 3.4 mm equals the
   0.1 zR tolerance. It is not simulated (the simulator uses the exact f), so the
   v3 tolerances do not include it. This is stated in the README and the v3
   results.
