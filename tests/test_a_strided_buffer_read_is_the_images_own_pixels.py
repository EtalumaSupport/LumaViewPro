# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A strided read of the buffered frame is every Nth pixel of the image, converted, with no overlay.

The histogram counts every fourth pixel of the live frame. It used to read
the whole frame through ``get_image_from_buffer``, which drew the scale bar
into a full-frame copy and converted every pixel of a 12-bit frame to 8
bits, and only then took every fourth pixel: 3.67 ms on the Kivy main
thread twice a second, where converting the sample alone is 1.14 ms. It
also counted the bar and its label as image. A read with a stride now takes
the sample from the raw frame and converts only that; a bar drawn on a
sample would be the wrong size, so none is drawn.
"""

from __future__ import annotations

import numpy as np
import pytest

from _lumascope.modules import image_utils
from _lumascope.modules.exceptions import ArgumentRefusedError
from _lumascope.modules.image_mode import ImageMode
from tests.camera_fakes import stall_with_a_frame_buffered


@pytest.fixture
def stalled_12bit(sim_session):
    scope = sim_session.scope
    scope.camera.set_image_mode(ImageMode.SCIENTIFIC)
    at_capture = stall_with_a_frame_buffered(scope)
    yield sim_session, at_capture
    scope.simulation.clear()


def test_a_strided_read_is_the_raw_frames_sample_converted_with_no_bar(stalled_12bit):
    session, at_capture = stalled_12bit
    imaging = session.scope.imaging
    session.set_scale_bar(False)
    raw, seq = imaging.get_image_from_buffer(force_to_8bit=False)
    # Red twice in whole runs and green alone: the message says whether the
    # capture itself was 8-bit, or the buffer later held another frame.
    assert raw.dtype == np.uint16, (
        f'the buffer holds frame {seq} as {raw.dtype}; the 12-bit capture took a'
        f' {at_capture.captured.dtype} frame, after which the buffer held frame {at_capture.seq}'
    )
    expected = image_utils.convert_to_8bit(raw[::4, ::4], 12)

    session.set_scale_bar(True)
    # The bar is drawn on a whole read: the frame is the one sampled.
    whole, whole_seq = imaging.get_image_from_buffer(force_to_8bit=True)
    assert whole_seq == seq
    assert not np.array_equal(whole, image_utils.convert_to_8bit(raw, 12)), 'no bar was drawn'

    sample, sample_seq = imaging.get_image_from_buffer(force_to_8bit=True, stride=4)
    assert sample_seq == seq
    np.testing.assert_array_equal(sample, expected)


def test_a_stride_below_one_is_refused(stalled_12bit):
    session, _at_capture = stalled_12bit
    with pytest.raises(ArgumentRefusedError) as refused:
        session.scope.imaging.get_image_from_buffer(stride=0)
    assert (refused.value.reason, refused.value.argument) == ('wrong_kind', 'stride')
