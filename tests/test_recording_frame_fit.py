"""Manual recording must not crash when a delivered frame size differs from
the pre-allocated recording buffer.

Some camera modes (and the simulator) can report one configured frame size
while delivering a nearby sensor-valid one. Storing such a frame into the
fixed-size buffer raised a NumPy shape-mismatch and crashed the camera
worker mid-recording. image_utils.fit_frame_to_shape pads/crops the spatial
overhang (and skips fundamentally incompatible frames), so recording
degrades gracefully instead of dying.
"""

import numpy as np

from modules.image_utils import fit_frame_to_shape


def test_crops_larger_mono_frame():
    image = np.ones((110, 100), dtype=np.uint8)
    fitted = fit_frame_to_shape(image, (100, 100))
    assert fitted.shape == (100, 100)
    assert fitted.dtype == np.uint8
    assert (fitted == 1).all()  # overlap preserved


def test_pads_smaller_mono_frame_with_black():
    image = np.ones((90, 100), dtype=np.uint8)
    fitted = fit_frame_to_shape(image, (100, 100))
    assert fitted.shape == (100, 100)
    assert (fitted[:90, :] == 1).all()  # delivered pixels preserved
    assert (fitted[90:, :] == 0).all()  # overhang black-padded


def test_fits_color_frame_per_channel():
    image = np.ones((100, 110, 3), dtype=np.uint16)
    fitted = fit_frame_to_shape(image, (100, 100, 3))
    assert fitted.shape == (100, 100, 3)
    assert fitted.dtype == np.uint16


def test_skips_frame_with_mismatched_dimensionality():
    image = np.ones((100, 100), dtype=np.uint8)  # mono
    assert fit_frame_to_shape(image, (100, 100, 3)) is None  # color buffer


def test_skips_color_frame_with_wrong_channel_count():
    image = np.ones((100, 100, 3), dtype=np.uint8)
    assert fit_frame_to_shape(image, (100, 100, 4)) is None
