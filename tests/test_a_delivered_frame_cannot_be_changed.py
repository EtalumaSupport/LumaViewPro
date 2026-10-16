# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame the camera delivered cannot be changed by whoever holds it.

The camera stores each delivered frame once and hands that one array to
every consumer: the frame listeners (a recording among them), the preview
and a capture. The store marks it read-only before any consumer has it,
so a consumer that draws on a frame -- the preview's scale bar, a video's
timestamp -- draws on its own copy, and every other consumer keeps the
pixels the camera delivered.
"""

import threading

import numpy as np
import pytest

from tests.scope_fakes import bind_settings_like_a_session


def _next_delivered_frame(scope):
    """The next frame the camera delivers, as a frame listener receives it."""
    seen = []
    arrived = threading.Event()

    def _listener(image, timestamp, chunks):
        if not seen:
            seen.append(image)
            arrived.set()

    scope.imaging.add_frame_listener(_listener, name='delivered-frame probe')
    try:
        assert arrived.wait(5.0), 'the simulated camera delivered no frame'
    finally:
        scope.imaging.remove_frame_listener(_listener)
    return seen[0]


def test_a_delivered_frame_and_every_view_of_it_refuse_a_write(sim_scope):
    """A listener's frame refuses a write, and a view of it cannot be made writeable."""
    image = _next_delivered_frame(sim_scope)

    assert not image.flags.writeable
    with pytest.raises(ValueError):
        image[0, 0] = 0
    with pytest.raises(ValueError):
        image.view().flags.writeable = True


def test_the_preview_draws_its_scale_bar_on_its_own_copy(sim_scope):
    """The preview's frame carries the bar; the delivered frame it was drawn from does not."""
    bind_settings_like_a_session(sim_scope, scale_bar={'enabled': True})
    # A fluorescence channel's bar is white, so it shows on a dark frame.
    sim_scope.imaging.set_scale_bar_color('Green')
    delivered = []

    def _listener(image, timestamp, chunks):
        delivered.append(image)

    sim_scope.imaging.add_frame_listener(_listener, name='delivered-frame probe')
    _next_delivered_frame(sim_scope)
    # Stopped, the camera's stored frame is the last one delivered, and it
    # is the frame the preview reads.
    sim_scope.imaging.disable_streaming()
    sim_scope.imaging.remove_frame_listener(_listener)
    stored = delivered[-1]
    as_delivered = stored.copy()

    preview, _ = sim_scope.imaging.get_image_from_buffer(force_to_8bit=False)

    np.testing.assert_array_equal(stored, as_delivered)
    assert np.count_nonzero(preview != as_delivered) > 0, 'the preview drew no bar'
