# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A read of the buffered frame returns the store's record of that frame with its image.

``ImagingAPI.buffered_image`` handed back the image and the frame's
hand-over number and dropped the rest of the record, its arrival among it,
so the live view could not say whether a gap between two frames it drew was
the camera's or its own. It now returns the frame's record, the one the
store built and every listener is handed, beside the image it read from it.
"""

from __future__ import annotations

import pytest

from tests.camera_fakes import stall_with_a_frame_buffered


@pytest.fixture
def heard_then_stalled(sim_scope):
    heard = []
    sim_scope.imaging.add_frame_listener(heard.append, name='heard')
    stall_with_a_frame_buffered(sim_scope)
    yield sim_scope, heard
    sim_scope.simulation.clear()


def test_a_buffered_read_returns_the_record_the_store_built(heard_then_stalled):
    scope, heard = heard_then_stalled

    image, frame = scope.imaging.buffered_image(force_to_8bit=True)

    # The stall holds the last frame stored; the last frame a listener was
    # handed is that one record, its arrival and counts the store's.
    assert image is not None
    assert frame is heard[-1]
