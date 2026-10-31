# change_lens: design (artificial waist for an inaccessible waist)

Status: design only, 31 Oct 2026, branch `add-change-lens`. No code yet. This is the
stretch tool from [agent-design.md](agent-design.md) (section 4). Implementing it
changes the agent, the prompt and the tools, which are frozen since 27 Oct 2026. So
coding starts only after the maintainer approves this design and lifts the freeze
for this feature. Everything new sits behind a switch that is **off by default**, so
v1 and v2 stay reproducible.

## Goal

When a waist lies outside the stage, the agent may insert one thin, aberration-free
lens before the stage. The lens forms an **artificial waist** inside the stage. The
agent measures that waist to ISO 11146-1 and reports two things:
- M² (unchanged by an ideal lens);
- the original beam (d0, z0, zR per axis), **calculated back** through the inverse
  ABCD matrix.

All numbers come from Python; the model only chooses the lens, where to measure,
and the verdict.

## 1. Physics (per axis, x and y independently)

**Embedded Gaussian.** A beam with M² is treated as a Gaussian of "effective
wavelength" M²·λ (as in the simulator). With the second-moment waist radius
W0 = d0/2:

- zR = π W0² / (M² λ)
- q(z) = (z − z0) + i·zR (complex beam parameter; the real part is the distance
  from the waist).
- Diameter anywhere: d(z) = d0 · |q(z)| / zR.

**ABCD matrices** (paraxial, SI inside Python):

- Free space over a length L: [[1, L], [0, 1]]
- Thin lens of focal length f (f > 0 converging, f < 0 diverging): [[1, 0], [−1/f, 1]]
- Transformation: q' = (A·q + B) / (C·q + D). For the thin lens alone this is
  1/q' = 1/q − 1/f.

**Forward: original beam to artificial waist.** Lens at z_L:
1. Take the original q just before the lens: q_L = (z_L − z0) + i·zR.
2. Apply the lens: q_L' = 1 / (1/q_L − 1/f).
3. The new waist is at z0' = z_L − Re(q_L'), with zR' = Im(q_L') and
   W0' = sqrt(zR' · M² λ / π).

M² is unchanged, because the ABCD matrices are real with determinant 1 and the
embedded-Gaussian picture holds exactly for them. The textbook form (Self 1983)
gives the same result. With s = z_L − z0 (the original waist s before the lens):
- s' = f + (s − f) f² / ((s − f)² + zR²)
- magnification m = f / sqrt((s − f)² + zR²)
- W0' = m·W0, zR' = m²·zR

**Back-propagation: measured artificial waist to original beam.** The fit after the
lens gives z0', zR' and M² per axis.
1. q_L' = (z_L − z0') + i·zR' at the lens, just after it.
2. Invert the lens: q_L = 1 / (1/q_L' + 1/f).
3. The original waist is at z0 = z_L − Re(q_L), with zR = Im(q_L) and
   W0 = sqrt(zR · M² λ / π).

For a general ABCD this is the inverse matrix [[D, −B], [−C, A]] (det = 1).

**Uncertainties.** σ of the back-calculated d0, z0 and zR are propagated from the
caustic fit's covariance (first order, numerical Jacobian of the function above).
They are statistical only: the lens focal length and position are assumed exact
(see the simplifications).

**Units.** SI (m) in Python. At the tool boundary, as for every other tool:
f and positions in mm, diameters in µm, wavelength in nm.

**Simplifications (stated in the report and the README):**
1. Thin lens (no thickness, no principal-plane offset).
2. Paraxial optics.
3. No aberrations (spherical, chromatic or astigmatic).
4. The lens is perfectly centred and not tilted, and its focal length and
   position are exact (no tolerance).
5. The lens aperture does not clip the beam. The tool refuses a placement where
   the beam diameter at the lens exceeds 8 mm (a 25.4 mm lens: more than 3 beam
   radii of margin).
6. No Fresnel losses or back-reflections, so no ghost from the lens.
7. Embedded-Gaussian model: a real multimode beam is not exactly Gaussian, but
   second moments still transform by ABCD. That is the property ISO relies on;
   to be verified (section 10).
8. Same lens for x and y (spherical). Simple astigmatism is handled per axis;
   general astigmatism stays out of scope.

## 2. Geometry

- **Axis:** z along the beam, increasing downstream. **z = 0 is the stage start**
  (the camera's closest position); the stage runs 0–250 mm, the same coordinates
  the agent already uses.
- **Laser exit (source) at z = −400 mm**, fixed and public (in `get_setup`). An
  original waist may lie anywhere, even upstream of the source (a virtual waist of
  a diverging laser); the algebra does not care.
- **Lens rail:** the lens can only go between the source and the stage start:
  `lens_z_mm` from −300 to −20 mm in 10 mm steps (29 positions). There are at least
  100 mm to the laser and 20 mm of clearance to the camera.

**Two cases:**
- **Waist before the stage start (z0 < 0).** The beam is already diverging at the
  stage. Two options:
  - a short converging lens after the waist (f smaller than the waist-to-lens
    distance, here 25–40 mm) re-images it into the stage;
  - a diverging lens before the waist (−200 mm) pushes it downstream.

  There are few workable options (1–5 per beam in the exploration below), so the
  choice must be computed, not guessed.
- **Waist after the stage end (z0 > 250 mm).** The beam converges through the
  stage. A long converging lens (300–750 mm) pulls the waist upstream into the
  stage. It also shrinks it, so the 20-pixel minimum (d0' ≥ 110 µm) is what limits
  this case.

**Lens kit** (standard catalogue focal lengths): **+25, +40, +300, +750, −200 mm**.

A design-time exploration (a scratch script, not committed; the implementation
redoes it with the real lens module and a test) used the existing camera (512 px of
5.5 µm), the 0–250 mm stage, 1064 nm, and the plan's coverage oracle (room for 5
near and 5 far points, far diameter ≤ half the sensor). It searched 99 grid beams:
- waists at −150…−20 mm and 260…450 mm;
- d0 160–300 µm;
- M² 1.0–1.8.

Results:
- **88 of the 99 beams** can be solved by some catalogue lens (25–1000 mm, ±).
- +40, +750 and −200 mm alone cover 87 of them; adding +25 mm covers all 88.
- +300 mm is in the kit for robustness (several large-M² cases).
- The other 11 beams cannot be solved with this camera and stage: they make good
  "honest `waist_not_accessible`" cases.

## 3. Tool interface

```
change_lens(f_mm, lens_z_mm, reason)
```

- **`f_mm`** must be one of the kit values. **`lens_z_mm`** must be a rail
  position.
- **Cost:** 1 position from the budget (mounting and realignment). **At most 2
  lens changes per run**; a second change replaces the first lens. There is no
  "remove" call.
- **Refused** (error result, nothing changes) for: a value outside the kit or the
  rail, a third change, the beam at the lens over 8 mm, or a finished run.
- **Effect:** a new **beam segment** starts (section 4). Darks stay valid (the
  lens does not change the background).
- **Returns:**
  - the lens (f_mm, lens_z_mm), `segment` (1 or 2), `positions_left`,
    `lens_changes_left`;
  - `prediction`: per axis, the predicted d0' (µm), z0' (mm) and zR' (mm), with σ,
    computed from the **current fit** of the old segment. It is **marked as a
    prediction** (`"basis": "fit of segment 0 (extrapolated waist)"`). If no fit
    exists, it is null with a reason.
  - a reminder that measurements of the old segment no longer enter any fit.

**Choosing a lens without guessing.** When `evaluate` reports an axis as `outside`
(or `uncertain`) and the scenario has a lens kit, its result gains a
`lens_options` list. Python computes every kit × rail combination from the current
fit and keeps the feasible ones: predicted z0' inside the stage, d0' ≥ 110 µm on
both axes, room for 5 near and 5 far points, and no clipping at the lens. They are
sorted by a fixed robustness score: feasible across z0 ± 2σ of the current fit,
then by the smallest margin. The agent picks from these numbers; it never computes
optics itself.

`get_setup` shows `lens_kit_mm`, `lens_rail_mm` (min, max, step), `source_z_mm`
and `lens_changes_left`, and only when the scenario enables lenses. The system
prompt gets a short "Inaccessible waist: change_lens" section, also only when
enabled (like the `view_frame` section).

**Open decision: prompt length.** The base prompt is at 1,495 of 1,500 words, and
the lens section needs about 120 more. Two options:
- (a) allow up to 1,650 words when lenses are enabled (the base test keeps 1,500);
- (b) shorten elsewhere.

## 4. Session: before and after the lens are different beams

- `MeasurementSession.segment` starts at 0. Each `change_lens` increments it and
  stores the lens (f, z_L) and the **transformed hidden beam**: per axis d0', z0',
  same M². The simulator stays unchanged, because the session simply renders with
  the transformed `BeamParams`.
- Every `Measurement` records its `segment`. Ids keep counting (m1, m2, …), so the
  log stays readable.
- **`evaluate` fits only the accepted measurements of the current segment.** The
  table still lists all measurements, with a `segment` column.
- **Old points never enter a new fit.** `set_status` on a measurement from an
  earlier segment is refused ("belongs to the beam before the lens change"). The
  fit filters by segment as well, so the protection does not rely on statuses.
- Coverage, the near-point rule, waist accessibility, the general-astigmatism check
  and the M² < 1 flag apply to the current segment only.
- The run file records the segment of every measurement and each lens change
  (when, f, z_L), so replay can draw the segments separately.

## 5. evaluate and finish

**`evaluate`** (current segment) reports, per axis:
- the usual fit: d0, z0, zR and M², with σ;
- `lens_options` when the waist is outside (section 3);
- after a lens change, also `original_beam_back_calculated`: d0, z0 and zR with σ,
  computed by Python as in section 1 and marked `"calculated, not measured"`.

**`finish`** writes the report from session state as today, plus:
- `method`: `"direct"` or `"artificial_waist"`;
- `lens`: f_mm, lens_z_mm, segment, and before which measurement it was inserted;
- `artificial_waist`: per axis, the measured d0', z0' and zR', with σ;
- `original_beam_back_calculated`: per axis, d0, z0 and zR, with σ;
- `m2`: per axis, from the artificial-waist fit (the ISO quantity);
- the list of simplifications from section 1.

**Verdicts stay the same five.** `iso_compliant` applies when the **measured
artificial-waist caustic** meets the ISO rules on both axes (coverage, no general
astigmatism, no M² < 1 flag), with `method = "artificial_waist"`. No new verdict is
needed: the method field says how it was obtained. Whether ISO calls this compliant
for the original beam is the question in section 10; the report wording stays "ISO
11146-1 caustic measured at an artificial waist" until that is verified.

**Guards:**
- The existing `finish` guards apply to the current segment.
- `waist_not_accessible` is still allowed when a lens kit exists (honest when no
  option is feasible). The agent is **not** forced to use a lens; that stays a
  judgement scored by the evaluation.

The numbers in the report never come from the model; the free-text explanation
remains unchecked, as today.

## 6. Scripted baseline (fair comparison)

The scripted operator gets the same information as the agent and a fixed rule:
1. It runs its normal procedure.
2. If it would end with an axis `outside` (after approaching that stage end, as the
   guard requires) and the scenario has a lens kit, it takes `evaluate`'s
   `lens_options`. These are the same Python numbers the agent sees.
3. It picks the first option by a fixed order: robust over ±2σ first, then the
   largest minimum near/far room over both axes, then the larger |f|, then the
   lens closer to the stage.
4. It calls `change_lens` and restarts its scout and fill procedure on the new
   segment, with the exposure scaled from the predicted d0'.
5. If no option is feasible, it finishes `waist_not_accessible`, as today.
6. At most one lens change; it never tries a second.

The naive scan never uses a lens (reference only).

## 7. Known-answer tests

1. **Textbook focus:** an ideal Gaussian (M² = 1) through a thin lens matches the
   Self (1983) s', m and W0' to 1e-9 relative, for several s/f ratios. That
   includes s = f, where s' = f, and both signs of f.
2. **M² unchanged:**
   - analytically, q algebra keeps M², i.e. zR'·π/(W0'²λ) is the same;
   - numerically, a noiseless simulated caustic after the lens fits the input M²
     to 0.1 %.
3. **Round trip:** forward then inverse gives back d0, z0 and zR to 1e-9 relative
   for random beams, lenses and positions, including a lens upstream of the waist.
4. **Unit error caught:** feeding f in mm where m is expected (a factor 1e3) must
   make the textbook test and the round trip fail. A test asserts that the wrong
   scale changes z0' by much more than the tolerance.
5. **Thin-lens limits:**
   - a lens at the original waist gives zR' = zR·f²/(f² + zR²) (the known
     special case);
   - f → ±∞ leaves the beam unchanged.
6. **Session separation:**
   - after `change_lens`, `evaluate` uses only the new segment;
   - `set_status` on old ids is refused;
   - points from the old segment never appear in the fit, even if they were
     accepted.
7. **End to end:** a waist-outside scenario (for example z0 = −60 mm) plus the
   chosen kit lens and the simulated measurements gives:
   - the back-calculated original d0 within 3 % and z0 within 2 mm;
   - M² within 3 % of the truth.

   This runs with the scripted baseline, so there are no API calls.
8. **Tool contract:**
   - kit and rail are enforced, as are the change limit and the 8 mm clipping
     refusal;
   - the prediction is labelled as such;
   - nothing changes when the switch is off: the v1 and v2 tool list and prompt
     are unchanged.

## 8. Mini-evaluation v3 (to be pre-registered before any run)

`docs/evaluation-plan-v3.md` will be committed **before any v3 run and after the
code is merged**, the same way as v1 and v2. v1 and v2 stay unchanged.

**Cases:** 8 waist-outside beams, each with a lens kit and 24 positions (4 more
than usual, for the lens change and a second scout). Seeds **900–907** (never used
so far: 0–99, 100, 200–204, 300–304, 500, 600–604, 700–704, 1000–1009 and
3000–3009 are taken).

| Case | Waist z0 (mm) | d0 (µm) | M² | Preliminary expected verdict |
|---|---|---|---|---|
| L1 | −30 | 200 | 1.2 | iso_compliant (1 option: −200 mm) |
| L2 | −60 | 200 | 1.2 | iso_compliant (2 options: +25, +40 mm) |
| L3 | −100 | 250 | 1.4 | iso_compliant (+25, +40 mm) |
| L4 | −80 | 160 | 1.0 | iso_compliant (−200, +25, +40 mm) |
| L5 | 280 | 200 | 1.2 | iso_compliant (+750 mm) |
| L6 | 320 | 300 | 1.6 | iso_compliant (+300, +750 mm) |
| L7 | 380 | 250 | 1.4 | iso_compliant (+750 mm) |
| L8 | 450 | 160 | 1.0 | waist_not_accessible (no kit option) |

The final expected verdicts come from an **oracle with lenses**, applied to the
true beam and committed in the v3 plan: `iso_compliant` if any kit × rail option
passes the v1 coverage rule, otherwise `waist_not_accessible`.

**Metrics:**
- correct verdict;
- for `iso_compliant` cases, a "measurement within tolerance" flag: |M² error| ≤ 3 %
  on both axes, back-calculated d0 within 5 % and z0 within 3 mm;
- lens choices, positions, model calls and cost.

**Contenders, in this order:**
1. `agent` (lens on, `view_frame` on);
2. `scripted` (lens rule);
3. `naive` (reference);
4. if the budget allows, `agent-nolens` as a control (the v2 configuration).

**Cost:** an estimated $0.10–0.15 per agent run (lens runs need more calls), about
$1 for agent and $2 with the control. **Hard stop at $2 in total**: cases not run
by then are reported as not run. One run per case; small N, stated as such.

## 9. Work estimate and what to cut

| Part | Hours |
|---|---|
| Lens physics module and known-answer tests (section 7, items 1–5) | 2 |
| Session segments, transformed hidden beam, tests | 2 |
| `change_lens` tool, `get_setup`, `lens_options` in `evaluate`, `finish` report, guards, tests | 3.5 |
| Agent switch, schema, prompt section, tests | 1.5 |
| Scripted lens rule and end-to-end test | 2 |
| v3 plan, oracle with lenses, script support, run, results | 2.5 |
| Replay and UI: show segments | 1 |
| Physics review, docs | 1.5 |
| **Total** | **≈ 16 h** |

The deadline is **Wed 4 Nov, 21:00 ET**. The hosted demo and the video script are
still missing and matter more for the submission, so they should be scheduled
first.

**Cut in this order if behind:**
1. UI segment display (the log line is enough);
2. the `agent-nolens` control runs in v3;
3. the second lens change (allow one only);
4. σ propagation for the back-calculated values (report them without σ, labelled);
5. the scripted lens rule (v3 becomes agent-only, stated openly);
6. v3 itself (ship the feature with its tests only, no evaluation claims).

## 10. ISO 11146-1 points to verify before calling this "compliant"

From the official text (the project only had third-party copies; see the README),
check whether 11146-1:
1. Allows or requires an **artificial waist** when the waist is not accessible, and
   in which words: for example the conditions under which a focusing element must
   be used.
2. Says what **"aberration-free" focusing element** means: any criterion or
   tolerance, lens type, or f-number relative to the beam.
3. Says whether the **transformed caustic must meet the same rules**: at least 10
   positions, about half within one zR and half beyond 2 zR of the **artificial**
   waist, and the same 3·d integration area and detector rules.
4. Gives **formulas or a procedure for the original beam**: whether d0, z0 and
   θ of the original beam are to be calculated from the artificial waist, and
   whether the result must be labelled as calculated.
5. States what the **test report** must contain for this method: lens focal
   length, lens position, distances, and any uncertainty contribution from the
   focal-length tolerance.
6. Allows the per-axis treatment of **simple astigmatic** beams through a spherical
   lens, and how lens misalignment is handled. Misalignment could create general
   astigmatism.
7. Changed any of this in the **2021 edition**; the project's wording basis is the
   2005 edition, and 2021 is unverified.

Until these are checked, the report and README say "ISO 11146-1 procedure applied
at an artificial waist (thin-lens model)" and not "ISO compliant for the original
beam".
