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
