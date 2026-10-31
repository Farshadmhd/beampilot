
# Inaccessible waist: change_lens

This setup has a lens kit (`get_setup` lists it). If `evaluate` reports a waist
`outside` the stage and lists `lens_options`, you may insert one thin lens before the
stage with `change_lens(f_mm, lens_z_mm)` to form an artificial waist inside it. Pick
from `lens_options`, higher `feasible_fraction` first; never compute optics yourself.
The returned `prediction` comes from your fit, not from a measurement.

A lens change starts a new beam: earlier measurements no longer count. Scout the new
beam again around the predicted waist, then measure it to the ISO rules as usual.
Each change costs one position, and so does a refusal because the beam overfills the
lens. If no option is listed, `waist_not_accessible` stays the right verdict. With a
lens, the report adds the back-calculated original beam; M² is unchanged by the lens.
