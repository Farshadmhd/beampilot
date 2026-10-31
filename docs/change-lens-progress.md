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

Tests: see the latest commit (`.venv\Scripts\python.exe -m pytest`).

## Next

- **STOP 1**: report to the maintainer (true vs back-calculated, lens, positions).
- **Phase 3**:
  - `AgentConfig(change_lens=False)`; the tool is offered only with the switch on
    and a lens kit; schema and prompt section (≤ 1,650 words with lenses);
  - byte-identical test with the switch off;
  - one live smoke run: `waist_outside_range`, seed 920, cost cap $0.30.
- **Phase 4**:
  - calibration (scripted, seeds 950–957, the 8 v3 beams);
  - `docs/evaluation-plan-v3.md`, committed and pushed before any run on seeds
    900–907 (STOP 2).
- **Phase 5**: UI and replay, README section, CLAUDE.md note (STOP 3).

## Open issues

1. **Waist after the stage, scripted, 24 positions.** With the waist at 280 mm
   (development seeds 41–43, after fix 1), the scripted operator chooses +750 mm
   at −20 mm, and the back-calculation is accurate (d0 within 0.6 %, z0 within
   0.006 zR, M² within 1.1 %).
   - It still uses all 24 positions and ends 4 of 5 far points short per axis
     (`estimate_not_compliant`).
   - Positions: 11 measurements before the lens (4 of them retries after
     saturation or low signal), 1 for the lens, 12 after it.
   - Decided: the v3 budget is 28 positions for every contender (`6fda336`).
2. (Resolved in fix 2.) Lens options are listed only for `outside`.
3. (Resolved in `bc6def7`.) **Collapsed far-field-only fit (review E2).** Inside
   lens planning only, an undetermined waist diameter is planned from θ and z0
   with M² drawn in [1, 2], and labelled. `fit_caustic` is unchanged.
4. (Resolved in `4389598`.) **Truth probing via the 8 mm refusal (review R1).** A
   refusal because the beam really clips at the lens now costs one position;
   kit, rail, change-limit and finished-run refusals stay free.
5. (Resolved in `6fda336`.) **L8 replaced.** The old L8 (waist at 450 mm) has no
   usable stage position; the oracle now expects `measurement_failed` for fewer
   than 5 usable positions. The new L8 (waist −20 mm, d0 230 µm, M² 1.0) was
   chosen by a recorded search in `scripts/v3_cases.py`: 67 candidates with
   ≥ 10 usable positions and no feasible lens; the one with the most usable
   positions (44). It is a borderline case: the waist is about 0.5 zR before the
   stage.
6. (Resolved.) README sentence on the focal-length tolerance (review R3).
7. (Resolved.) Design doc: sorting by the larger spare margin, linearised a, b, c
   draws, and the collapsed-fit planning rule.
8. For waists after the stage, the focal-length sensitivity is about 1.2 mm in z0
   per +1 % f (+750 mm lens), half of 0.1 zR. It is not simulated (the simulator
   uses the exact f), so the v3 tolerances do not include it.
