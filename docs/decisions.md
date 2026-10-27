# Design decisions

Short records of choices that are not obvious from the code. Newest last.
`docs/agent-design.md` stays the source of truth; these entries fill in details.

## 2026-10-27: Ellipticity from the principal axes

`measure_widths` reports d_x and d_y along the lab axes (the design assumes beams
aligned to x/y), but `ellipticity` and `azimuth_rad` come from the principal axes of
the full second-moment matrix (sigma_x², sigma_y², sigma_xy). Ellipticity is the
ratio of the minor to the major principal width. Reason: lab-axis widths of a
rotated ellipse can be equal even when the beam is strongly elliptical, so a lab-axis
ratio would hide exactly the case that the general-astigmatism check needs to see.

## 2026-10-27: Stop the width iteration when the rectangle repeats

The integration area is a pixel rectangle of 3 · d around the centroid. With noise,
the rectangle can flip by one pixel each pass, so d alternates by about 1e-4 relative
and never meets `rtol`. `measure_widths` therefore also stops, as converged, when the
next rectangle is one that was already used: further passes would only cycle through
the same areas. The reported widths come from the last pass; the alternatives differ
by about 1e-4 relative, far below the repeat scatter. Before this rule, 20 of 100
simulated frames at a peak of ~430 counts hit `max_iter`.

## 2026-10-27: Centroid relative to the image centre

Centroids and pixel coordinates are in metres relative to the image centre, with the
centre at pixel index (n - 1) / 2. x runs along columns, y along rows (positive y is
towards higher row indices). The simulator uses the same convention, so a beam
rendered at (x0, y0) is measured at (x0, y0). Reason: the stage axis does not care
where the beam sits on the sensor, and a centred origin makes off-centre and clipping
checks symmetric.

## 2026-10-27: Start the width iteration from a 1/e² estimate, not the full image

The first integration area is the 3 · d rectangle around the centroid and 4σ widths
of the pixels above 1/e² of the peak; the ISO iteration then runs unchanged from
there. Reason: on a full 512 × 512 sensor, a residual offset of ~0.2 counts (the
corner estimate's standard error is ~0.15 counts) weighted by r² outweighs the σ² of
a 200 µm beam holding ~3·10⁶ counts, so the full-image first pass gave negative
second moments on several near-waist frames. The thresholded pixels only choose
the starting rectangle; they never enter the reported widths. Whether ISO/TR 11146-3
prescribes a specific starting area: unverified.

Kept after the physics review. For a single-lobe beam the start cannot bias the
result: the 1/e² estimate is ~0.83 d, so the first rectangle spans ±5σ and the
iteration climbs to the same fixed point a full-image start descends to. Two known
limits: (1) with a strong halo or a ghost spot there can be two fixed points, and this
start finds the core while a full-image start would include the ghost; (2) a single
hot pixel brighter than ~7× the beam peak makes the start fall back to the full
image. Neither occurs in the simulator yet; a 3 × 3 median before taking the peak
would address (2).

## 2026-10-27: Compare astigmatism azimuths modulo 90 degrees

`measure_widths` reports the azimuth of the major principal axis. A beam with simple
astigmatism (different x and y waist positions) swaps major and minor axis where
d_x = d_y, so its azimuth jumps from 90° to 0° although the axes never rotate.
`general_astigmatism` therefore compares azimuths modulo 90°: only the pair of axes
matters. Comparing modulo 180° flagged design scenario 6 as general astigmatism.

## 2026-10-27: ISO point coverage means at least 5 near and 5 far, per axis

ISO 11146-1 asks for at least 10 positions, "about half" within one Rayleigh length
of the waist and "about half" beyond two Rayleigh lengths. `coverage` reads this as
at least 5 distinct positions with |z − z0| ≤ zR and at least 5 with |z − z0| ≥ 2 zR,
evaluated separately for x and y with that axis's fitted z0 and zR. Repeated frames
at one z count once. Positions between zR and 2 zR are counted but do not help.
The thresholds are parameters (`min_near`, `min_far`) in case the standard text,
which the maintainer still has to check, allows fewer.

## 2026-10-27: Caustic fit uncertainties are scaled by the residual variance

`fit_caustic` reports 1σ uncertainties from s² (JᵀJ)⁻¹ with s² = Σr² / (n − 3),
like `curve_fit` with `absolute_sigma=False`. The repeat stds (from only 5 frames)
therefore act as relative weights, and scatter beyond them, e.g. from position
errors, still shows up in the uncertainty. With 2 % simulated noise the reported M²
uncertainty matched the actual scatter over 300 seeds within ~10 %. Sampling only
near the waist inflates the M² uncertainty by ~6× at ±0.5 zR, ~3× at ±0.7 zR and
only ~1.8× at ±1 zR, compared with ISO sampling; no bias was seen at 2 % noise.

## 2026-10-27: The agent pipeline fits with relative residuals

`evaluate` calls `fit_caustic` with `d_std_m=None`, i.e. relative residuals
(d_model − d) / d. Reason: the std of only 5 repeats scatters by ~35 %, so a lucky low
std gives one position several times the weight. In a test with 2 % noise per frame,
5 repeats at 10 ISO positions and 300 seeds, the weighted fit's M² scattered by
0.0080 against 0.0068 unweighted (~18 % noisier), with no bias in either. The
weighted option stays in `fit_caustic` for callers with better std estimates.

## 2026-10-27: 50 dark frames per dark set

The mean dark is subtracted from every frame of a run, so its fixed pixel noise is
the same error at every stage position; the caustic fit's residual-based
uncertainty cannot see it. With 10 darks (5 counts read noise) the M² scatter
between 12 simulated clean runs was 0.47 % against a reported 0.17 %. With 50 darks
it is 0.13–0.14 % against a reported 0.15 %, the same as with a perfect dark. Cost:
40 more frames once per dark set. ISO/TR 11146-3's minimum of 10 is still met.

## 2026-10-27: New tool retake_darks (6 core tools)

A background change after the dark frames (room light) was detected by
`background_changed` but the agent could not act on it. `retake_darks` blocks the
beam and takes a fresh 50-frame dark set with the current background, so the change
becomes part of the dark. It costs one position from the budget, like a measurement,
so retaking darks is a real trade-off and cannot be used for free after every frame.
Design section 4 now lists 6 core tools.

## 2026-10-27: Reject on estimated clipping loss, not on a cut integration area

`measure_at` used to suggest "reject" whenever the 3 · d integration area hit the
sensor edge. That cuts the beam at >= 6σ on one side, which biases d by far less than
0.1 %, so it was much stricter than the ISO 11146-1 rule (power outside the sensor
below 1 %). Now `clipping_loss` estimates the power of a Gaussian with the measured
centroid and σ = d/4 that falls outside the sensor (erf), and only a loss above 1 %
suggests "reject"; `roi_clipped` remains as a warning. The widths of a clipped beam
are themselves biased low (σ ~3.6 % low when cut at ±2.6σ), so the estimate
understates the loss by up to ~25 % near the 1 % limit.

## 2026-10-27: Known limitations (open review risks)

1. **Clipping loss is underestimated.** `clipping_loss` uses σ = d/4 from the measured
   widths, but a clipped beam's widths are themselves too small. In noise-free
   tests the estimate is 0.80 × the true loss at 1 % and 0.70 × at 2 %, so the
   reject only starts at a true loss of about 1.4 %, where d is already ~3 % low.
   Possible fix: invert the truncated-Gaussian variance relation, or lower the
   limit on the estimate to ~0.75 %.
2. **`beam_too_large` has no hard trade-off any more.** Since `roi_clipped` is only
   a warning, nothing in that scenario produces a reject: at the far stage end
   (4.5 zR, d ≈ 1.4 mm) the estimated loss is 0.01 %. If the scenario should force
   a decision, the beam must grow: a loss above 1 % needs d above ~2 mm.
3. **Real corner light can hide a room-light change.** When beam light really
   reaches the corner patches (`beam_in_corners`), the frame is not used for
   `background_changed`, so a simultaneous background change goes unreported
   (e.g. waist_outside_range at 180 mm plus 3 counts). Subtracting the modelled
   corner light and testing the rest would catch it, but the model runs ~30 % low
   for clipped beams, so it needs a looser limit; left as is for now.

## 2026-10-27: No extrapolated waist numbers

`evaluate` and the `finish` report show d0, z0, zR and M² of an axis as null with a
`not_determined` reason if the fit failed or its waist lies outside the measured z
range. Reason: a far-field-only fit collapses (d0 → 0, M² → 0) and must never reach
the report as a number. θ stays (the far-field slope is measured), and
`waist_in_stage_range` still uses the fitted z0, which is how a waist outside the
stage is detected. `waist_side` (below/above) tells the operator where to measure
next. Known cost: a waist inside the stage but outside the measured span is also
nulled until a position on its side is measured; the scripted baseline does that
(10/10 seeds compliant for waists 10–30 mm and 235–245 mm from the stage start).
The reviewer's alternative, null only when no accepted point lies within ~1–2 zR of
the fitted waist, is left open for the maintainer.

## 2026-10-27: Agent loop uses the stateless Responses API mode

Verified in the Meta Model API docs: `instructions` is not kept across turns (resent
every call), `max_output_tokens` caps reasoning plus visible output, tool results go
back as `function_call_output` with the matching `call_id`, `parallel_tool_calls`
defaults to true, `reasoning.effort` accepts minimal to xhigh (not "none"). Chaining
turns with `previous_response_id` worked in a single run but failed in all six
parallel runs with "400 referenced response not found or expired". The loop
therefore uses the documented stateless mode: `store=false`,
`include=["reasoning.encrypted_content"]`, and the full history (reasoning items,
assistant messages, function calls and outputs) resent on every call. Prompt
caching keeps this cheap (~90 % of input tokens cached in the real runs). Transient
API errors (connection, timeout, rate limit, 5xx) are retried twice; any other
error stops the run with a logged reason. Hard limits per run: 40 model calls,
1.5 M tokens, $0.50. The first six real runs (seed 0, muse-spark-1.2, effort low)
gave the expected verdict on every scenario with M² within 0.5 %; they used the
prompt before the review fixes in this branch (number rule, background advice).

## 2026-10-27: Waist numbers need a point within ±1 zR; accessibility from ±2σ

Replaces the "waist inside the measured z range" test. d0, z0, zR and M² of an axis
are null only if the fit failed or no accepted position lies within ±1 zR of the
fitted waist. That still catches the collapse (far-field-only data: zR → 0 leaves no
point within ±1 zR) but keeps sound fits whose waist lies just outside the measured
span (e.g. waist 25 mm, nearest scout 31 mm). `waist_side` says below / above the
measured positions or `within` a gap (`waist_gap_mm`).

Waist accessibility is now one of three states from z0 ± 2σ(z0) against the stage:
`inside`, `outside` or `uncertain` (`waist_near_stage_end` says which end). Only
`outside` supports `waist_not_accessible`. A far-field-only fit can misplace the
waist, so operators (scripted baseline, agent prompt) measure as close as possible
to that stage end before concluding `outside`. Scripted sweep, seeds 0-9: waists 2 mm
from either end are now iso_compliant 10/10 (before: 5/10 wrongly
waist_not_accessible); waist_outside_range stays waist_not_accessible 10/10.

## 2026-10-27: Stricter finish guard and a controller fallback report

`finish` also refuses `iso_compliant` when general astigmatism or M² < 1 is flagged.
When a hard limit or an API error stops the agent, the controller writes the report
itself: `estimate_not_compliant` if M² is determined on both axes, otherwise
`measurement_failed`, with `stopped_by_limit` set to the reason, so every run ends
with a report built from session state.

## 2026-10-27: Known limitation: extreme simple astigmatism

The scripted baseline skips grid positions where the predicted beam (either axis) is
wider than half the sensor (`MAX_D_PER_SENSOR = 0.5`), so for x and y waists far apart
the near points of one axis can be blocked by the size of the other axis there,
although such frames are often only `roi_clipped` and acceptable. Since the near-gap
fill (which ignores that size check) was added, x 20 / y 230 mm is iso_compliant 10/10;
but x 5 / y 245 mm still ends `estimate_not_compliant` with budget left (the gap fill
is blocked by the minimum spacing). Conversely the gap fill could plan positions where
the beam really overfills the sensor (not seen with the 2.8 mm sensor). A per-axis
check against the clipping-loss estimate would fix both; left as is for now.

## 2026-10-27: M² < 1 flag with a σ floor; waist_not_accessible needs the stage end

The Jacobian σ(M²) misses systematic errors (one-sided data, the shared dark), so a
true M² = 1.0 beam fitted 0.997 with a tiny σ was flagged as M² < 1 and `finish` then
refused a correct `iso_compliant` (scripted, 1 of 10 seeds). The flag now uses
σ floored at 1 % of M² (`M2_STD_FLOOR_REL`). This deviates from the plain
"M² + 2σ < 1" rule asked for earlier; the maintainer should confirm.

σ(z0) is just as optimistic: after far-field-only scouts a waist 2 mm inside the stage
was reported `outside` at 10σ. So `finish` refuses `waist_not_accessible` unless an
axis is `outside` and an accepted position lies within 5 % of the stage length
(`STAGE_END_FRACTION`) from that stage end; the scripted operator approaches the end
to that margin before concluding. waist_outside_range stays waist_not_accessible
10/10 (8 positions).
