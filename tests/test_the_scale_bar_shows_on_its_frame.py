# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scale bar shows on the frame it is drawn on: its colour is the frame's.

``image_utils.add_scale_bar`` draws the bar black where the frame inside the
bar's box is bright and white where it is dark, so the bar is seen on whatever
it is drawn over and every capture of one frame gets one bar, whoever made it.
Bright is a mean at or above half the white the frame is shown at, which for a
sum is one frame's white: a sum is rendered and recorded against it. A white
bar is the frame's stored full scale, so it is full white in the file.

A black bar is drawn too: the bar's geometry is masked with a nonzero
sentinel, so a value of 0 still lands (an empty mask once left the black bar
undrawn).

Scale-bar rendering is pure numpy/cv2, so the draw's tests run without the
Kivy app; they need a scale, which the scale_capabilities fixture supplies.
The capture's test runs on the simulated scope, no GUI, so the colour is the
API's.
"""

from __future__ import annotations

import numpy as np
import pytest

import modules.image_utils as image_utils
from tests.objective_records import objective
from tests.scope_fakes import bind_settings_like_a_session

_OBJECTIVE = objective(focal_length_mm=10.0, magnification=20)


_OPTICS = None


@pytest.fixture
def _scale(scale_capabilities):
    """The draw's tests draw a scale bar, which needs a known pixel size."""
    global _OPTICS
    _OPTICS = scale_capabilities
    yield
    _OPTICS = None


def _add(image, *, frames_summed=1, frame_bits=8):
    image_utils._scale_bar_cache = {}  # module-global cache; isolate each call
    return image_utils.add_scale_bar(
        image=image,
        objective=_OBJECTIVE,
        binning_size=1,
        frames_summed=frames_summed,
        frame_bits=frame_bits,
        capabilities=_OPTICS,
    )


def _bar_pixels(shape, dtype, *, frames_summed=1, frame_bits=8):
    """Where the bar and its text land on such a frame: a white bar on a black one.

    The text's antialiased edge enters the bar by the depth's full scale, so
    the pixels are those of a frame of the same shape, type and depth.
    """
    black = np.zeros(shape, dtype=dtype)
    return _add(black, frames_summed=frames_summed, frame_bits=frame_bits) != 0


def _bar_value(image, *, frames_summed=1, frame_bits=8):
    """The one value the bar was drawn in on ``image``."""
    drawn = _add(image, frames_summed=frames_summed, frame_bits=frame_bits)
    bar = _bar_pixels(image.shape, image.dtype, frames_summed=frames_summed, frame_bits=frame_bits)
    values = np.unique(drawn[bar])
    assert len(values) == 1, f'the bar was drawn in {values}, not one value'
    return int(values[0])


@pytest.mark.usefixtures('_scale')
def test_a_bright_frame_gets_a_black_bar_and_a_dark_one_a_white_bar():
    bright = np.full((400, 400), 200, dtype=np.uint8)
    dark = np.full((400, 400), 50, dtype=np.uint8)

    assert _bar_value(bright) == 0
    assert _bar_value(dark) == 255


@pytest.mark.usefixtures('_scale')
def test_a_frame_is_bright_by_half_its_own_white():
    """A 12-bit frame in a 16-bit array: 3000 is bright, 1500 dark, its white 4095."""
    assert _bar_value(np.full((400, 400), 3000, dtype=np.uint16), frame_bits=12) == 0
    assert _bar_value(np.full((400, 400), 1500, dtype=np.uint16), frame_bits=12) == 4095


@pytest.mark.usefixtures('_scale')
def test_a_sum_is_bright_by_one_frames_white_and_white_at_its_own_full_scale():
    """Four 12-bit frames summed: 5000 is past one frame's white, so shown white and given a black bar."""
    assert (
        _bar_value(np.full((400, 400), 5000, dtype=np.uint16), frames_summed=4, frame_bits=12) == 0
    )
    # Dark, the bar is the sum's full scale (14 bits), full white in its file.
    assert (
        _bar_value(np.full((400, 400), 1000, dtype=np.uint16), frames_summed=4, frame_bits=12)
        == 16383
    )
    # Sixteen 12-bit frames reach the 16-bit container.
    assert (
        _bar_value(np.zeros((400, 400), dtype=np.uint16), frames_summed=16, frame_bits=12) == 65535
    )


@pytest.mark.usefixtures('_scale')
def test_the_colour_is_chosen_from_what_is_under_the_bar_not_the_whole_frame():
    """A dim frame bright under the bar gets black; a bright one dark under the bar gets white."""
    under = _bar_pixels((400, 400), np.uint8)
    rows = np.flatnonzero(under.any(axis=1))
    cols = np.flatnonzero(under.any(axis=0))
    box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))

    dim_with_a_bright_patch = np.full((400, 400), 20, dtype=np.uint8)
    dim_with_a_bright_patch[box] = 240
    bright_with_a_dark_patch = np.full((400, 400), 240, dtype=np.uint8)
    bright_with_a_dark_patch[box] = 20

    assert _bar_value(dim_with_a_bright_patch) == 0
    assert _bar_value(bright_with_a_dark_patch) == 255


@pytest.mark.usefixtures('_scale')
def test_a_black_bar_is_drawn():
    image = np.full((400, 400), 200, dtype=np.uint8)
    # A black bar introduces a run of 0-valued pixels on a frame with none.
    assert (_add(image) == 0).sum() > 20


@pytest.mark.usefixtures('_scale')
def test_the_bar_is_drawn_on_a_copy_of_a_read_only_frame():
    # A camera's delivered frame is read-only and shared with every other
    # consumer: the bar is drawn on a copy, and the frame keeps its pixels.
    image = np.zeros((400, 400), dtype=np.uint8)
    image.flags.writeable = False

    result = _add(image)

    assert (result == 255).any(), 'no bar was drawn'
    assert not image.any(), 'the frame the bar was drawn from was changed'


@pytest.mark.parametrize(
    ('pattern', 'bar_is_black'),
    [('white', True), ('black', False)],
)
def test_a_capture_draws_the_bar_its_frame_needs_with_no_gui(sim_scope, pattern, bar_is_black):
    """Through the API alone, a white frame's bar is black and a black frame's bar is white."""
    bind_settings_like_a_session(sim_scope, scale_bar={'enabled': True})
    sim_scope._camera_driver.set_test_pattern(enabled=True, pattern=pattern)

    image = sim_scope.imaging.get_image(force_to_8bit=True, force_new_capture=True)

    frame_value = 255 if pattern == 'white' else 0
    bar = image != frame_value
    assert bar.sum() > 20, 'no bar was drawn'
    assert set(np.unique(image[bar])) == ({0} if bar_is_black else {255})
