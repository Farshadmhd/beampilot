"""Known-answer tests for ISO 11146 beam widths, using the simulator."""

import numpy as np
import pytest

from beampilot.simulator import BeamParams, CameraConfig, dark_frame
from beampilot.widths import mean_dark, subtract_background

# Astigmatic test beam: different waist sizes, positions and M² per axis.
BEAM = BeamParams(
    wavelength_m=1064e-9,
    d0x_m=0.5e-3,
    d0y_m=0.4e-3,
    z0x_m=0.10,
    z0y_m=0.12,
    m2x=1.2,
    m2y=1.5,
)
CAMERA = CameraConfig(
    pixel_size_m=5.5e-6,
    nx=512,
    ny=512,
    bit_depth=12,
    offset_counts=100.0,
    read_noise_counts=5.0,
    signal_counts_per_ms=1e6,
)


# 4 sigma of the corner-patch mean: 4 patches of 18 x 18 px, ~5.2 counts noise each.
CORNER_MEAN_TOL = 4 * 5.2 / np.sqrt(4 * 18 * 18)


def _darks(camera: CameraConfig, n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [dark_frame(camera, rng) for _ in range(n)]


def test_fewer_than_ten_dark_frames_raises() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        mean_dark(_darks(CAMERA, 9, seed=0))


def test_mean_dark_averages_frames() -> None:
    dark = mean_dark(_darks(CAMERA, 10, seed=0))
    assert dark.shape == (CAMERA.ny, CAMERA.nx)
    assert dark.mean() == pytest.approx(CAMERA.offset_counts, abs=0.1)
    # Averaging 10 frames reduces the pixel noise by sqrt(10).
    assert dark.std() == pytest.approx(CAMERA.read_noise_counts / np.sqrt(10), rel=0.05)


def test_negative_values_survive_background_subtraction() -> None:
    dark = mean_dark(_darks(CAMERA, 10, seed=0))
    frame = dark_frame(CAMERA, np.random.default_rng(1))  # no beam: pure noise
    image, _ = subtract_background(frame, dark)
    assert image.dtype == np.float64
    assert (image < 0).sum() > 0.4 * image.size  # about half the pixels
    assert image.mean() == pytest.approx(0.0, abs=CORNER_MEAN_TOL)


def test_fine_offset_removes_residual_background() -> None:
    dark = mean_dark(_darks(CAMERA, 10, seed=0))
    extra_counts = 3.0  # e.g. room light switched on after the dark frames
    frame = dark_frame(CAMERA, np.random.default_rng(1)) + extra_counts
    image, info = subtract_background(frame, dark)
    assert info.offset_counts == pytest.approx(extra_counts, abs=CORNER_MEAN_TOL)
    assert info.std_counts == pytest.approx(
        CAMERA.read_noise_counts * np.sqrt(1 + 1 / 10), rel=0.1
    )
    assert len(info.corner_means_counts) == 4
    assert image.mean() == pytest.approx(0.0, abs=CORNER_MEAN_TOL)
