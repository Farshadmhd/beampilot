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
