# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A stored frame is one record: its pixels and every fact about it travel together.

The camera's frame store (``ImageHandlerBase._store_frame``) builds one
``Frame`` per delivered frame -- pixels, arrival number, wall and steady
arrival times, depth and chunks -- and every reader is handed that record.
Before, the pixels, the chunks and the depth sat in separate latest slots,
read one after another, so a capture could return one frame's pixels with a
newer frame's chunks in its saved record.
"""

import datetime
import threading

import numpy as np
import pytest

from drivers.camera import ImageHandlerBase


def _store(handler, value, *, chunks=None, seq_hint=0):
    return handler._store_frame(
        np.full((4, 4), value, dtype=np.uint8),
        datetime.datetime.now(),
        chunks=chunks,
        arrived_ns=1_000 + seq_hint,
        significant_bits=8,
        wire_bytes=16,
    )


class TestTheStoreBuildsOneRecord:
    def test_the_record_carries_the_frames_pixels_number_times_depth_and_chunks(self):
        handler = ImageHandlerBase()
        stored = _store(handler, 7, chunks={'ExposureTime': 14530.0, 'FrameID': 12})

        assert handler.get_last_image() is stored
        assert stored.seq == 1
        assert stored.arrived_ns == 1_000
        assert stored.significant_bits == 8
        assert int(stored.image[0, 0]) == 7
        assert dict(stored.chunks) == {'ExposureTime': 14530.0, 'FrameID': 12}

    def test_the_record_cannot_be_changed(self):
        handler = ImageHandlerBase()
        stored = _store(handler, 7, chunks={'Gain': 1.0})

        with pytest.raises(ValueError):
            stored.image[0, 0] = 0
        with pytest.raises(TypeError):
            stored.chunks['Gain'] = 2.0
        with pytest.raises(AttributeError):
            stored.seq = 5

    def test_a_camera_without_chunks_stores_none(self):
        handler = ImageHandlerBase()
        assert _store(handler, 1).chunks is None

    def test_the_next_frame_replaces_the_record_and_leaves_the_last_one_as_it_was(self):
        handler = ImageHandlerBase()
        first = _store(handler, 1, chunks={'ExposureTime': 14530.0, 'Gain': 1.0})
        second = _store(handler, 2, chunks={'ExposureTime': 30000.0}, seq_hint=1)

        assert handler.get_last_image() is second
        assert dict(second.chunks) == {'ExposureTime': 30000.0}
        assert dict(first.chunks) == {'ExposureTime': 14530.0, 'Gain': 1.0}
        assert int(first.image[0, 0]) == 1
        assert (first.seq, second.seq) == (1, 2)

    def test_a_listener_is_handed_the_stored_record(self):
        handler = ImageHandlerBase()
        heard = []
        handler.register_frame_callback(heard.append)

        stored = _store(handler, 3)

        assert heard == [stored]


class TestNoFrameIsNone:
    def test_before_the_first_frame(self):
        assert ImageHandlerBase().get_last_image() is None

    def test_after_a_failed_grab(self):
        handler = ImageHandlerBase()
        _store(handler, 1, chunks={'ExposureTime': 14530.0})
        handler._record_failure()
        assert handler.get_last_image() is None

    def test_after_a_reset(self):
        handler = ImageHandlerBase()
        _store(handler, 1, chunks={'ExposureTime': 14530.0})
        handler.reset()
        assert handler.get_last_image() is None


def test_a_capture_records_the_frame_it_returns_when_a_newer_one_arrives_during_it(
    sim_scope, monkeypatch
):
    """The saved record names the frame whose pixels the capture returns.

    Each simulated frame is stamped with an id, in its first pixel and in a
    FrameID chunk (the simulator stamps no chunks of its own yet). The
    capture's saturation check is made to wait until the camera has stored
    a newer frame, so a capture that read the chunks of the newest frame
    instead of its own would name that newer frame.
    """
    camera = sim_scope._camera_driver
    handler = camera.cam_image_handler
    store = type(handler)._store_frame
    stamped = iter(range(1, 1_000_000))

    def store_with_an_id(self, image, timestamp, chunks=None, **kwargs):
        frame_id = next(stamped)
        image = image.copy()
        image.flat[0] = frame_id % 200 + 1
        return store(self, image, timestamp, {'FrameID': frame_id}, **kwargs)

    monkeypatch.setattr(type(handler), '_store_frame', store_with_an_id)
    measure = type(sim_scope.imaging).saturated_fraction
    newer = threading.Event()

    def measure_then_wait_for_a_newer_frame(image, significant_bits):
        answer = measure(image, significant_bits)
        assert handler.wait_for_frame_after(handler.frames_delivered, 5.0)
        newer.set()
        return answer

    monkeypatch.setattr(
        type(sim_scope.imaging),
        'saturated_fraction',
        staticmethod(measure_then_wait_for_a_newer_frame),
    )

    image = sim_scope.imaging.capture_and_wait(force_to_8bit=False, all_ones_check=True)

    assert newer.is_set(), 'no newer frame arrived during the capture; this proves nothing'
    record = sim_scope.imaging.last_capture_info['frame_record']
    returned = int(image.flat[0]) - 1  # the stamp is id % 200 + 1
    assert returned == record.frame_id % 200, (
        f'the capture returned frame {returned} (mod 200) but recorded FrameID {record.frame_id}'
    )


def test_a_captured_image_is_read_only(sim_scope):
    """A capture's image is the instrument's account: it refuses a write,
    whichever path made it (here an 8-bit rendering of the camera's frame)."""
    image = sim_scope.imaging.capture_and_wait(force_to_8bit=True)

    with pytest.raises(ValueError):
        image[0, 0] = 0
