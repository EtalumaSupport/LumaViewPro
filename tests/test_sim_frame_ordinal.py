# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated camera's frame fields all describe the same frame.

The simulator free-runs as a real camera does: while it is grabbing, its
acquisition thread stores one frame per interval in the image handler every
driver shares, whether or not anything reads it. The handler writes pixels,
timestamp, depth and arrival ordinal under one lock, so a reader gets all
four from the same frame.

Two consequences, both pinned below:

  - Preview polls see distinct frames, each under its own ordinal, so the
    settle gate (which dedupes on the ordinal) can retire a skip count in a
    sim run -- the gate the capture paths rely on.
  - A field can never belong to a different frame than the pixels it came
    with. An ordinal that runs ahead of its pixels retires a settle count the
    pixels predate, which is a capture accepted under the previous
    gain/exposure/LED state -- the defect the gate exists to prevent.

And the property free-running exists for: the count advances with nobody
reading, so a stream that has stopped is told apart from one nobody watches.
"""

from __future__ import annotations

import itertools
import pathlib
import sys
import threading
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests.camera_fakes import grab_a_frame_made_after_now


def _cam():
    from drivers.simulated_camera import SimulatedCamera

    cam = SimulatedCamera(width=32, height=24)
    cam.start_grabbing()
    return cam


def _poll_across_ten_stores(cam, poll):
    """Call ``poll`` back to back until the camera has stored ten more frames.

    The polls interleave with the acquisition thread's stores for exactly
    as long as ten stores take; the bound is the owner's own frame wait,
    whose timeout is the failure, never a fixed window. ``poll`` runs at
    least ten times.
    """
    handler = cam.cam_image_handler
    tenth = handler.frames_delivered + 9
    stored = threading.Event()
    reached = []

    def wait_for_the_tenth():
        reached.append(handler.wait_for_frame_after(tenth, 5.0))
        stored.set()

    waiter = threading.Thread(target=wait_for_the_tenth, name='ten-stores')
    waiter.start()
    try:
        polls = 0
        while not stored.is_set() or polls < 10:
            poll()
            polls += 1
    finally:
        waiter.join()
    assert reached == [True], 'the camera stopped storing frames before ten more arrived'


class TestTheOrdinalAdvancesWithTheFrame:
    """A stored frame gets a number of its own, and the stream stores frames on its own."""

    def test_the_stream_advances_the_ordinal_with_nobody_reading(self):
        cam = _cam()
        try:
            before = cam.frames_delivered
            deadline = time.monotonic() + 5.0
            while cam.frames_delivered < before + 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            after = cam.frames_delivered
        finally:
            cam.stop_grabbing()
        assert after >= before + 2, 'a grabbing camera stored no frames while nothing read it'

    def test_two_frames_apart_return_two_ordinals(self):
        cam = _cam()
        try:
            _r1, _t1, seq1 = grab_a_frame_made_after_now(cam)
            _r2, _t2, seq2 = grab_a_frame_made_after_now(cam)
        finally:
            cam.stop_grabbing()
        assert seq1 is not None and seq2 is not None
        assert seq2 > seq1, 'a newer frame must not reuse the previous number'

    def test_the_ordinal_a_poll_returns_is_the_one_the_gate_reads(self):
        cam = _cam()
        grab_a_frame_made_after_now(cam)
        cam.stop_grabbing()
        _r, _i, _t, _b, seq = cam.grab_latest()
        assert seq == cam.frames_delivered

    def test_callback_frames_are_the_counted_frames(self):
        cam = _cam()
        seen = []
        second_frame = threading.Event()

        def on_frame(img, ts, chunks):
            seen.append(ts)
            if len(seen) >= 2:
                second_frame.set()

        before = cam.frames_delivered
        cam.register_frame_callback(on_frame)
        try:
            reached = second_frame.wait(5.0)
            after = cam.frames_delivered
        finally:
            cam.stop_grabbing()
        assert reached, 'no second frame reached the callback; the rest of this test proves nothing'
        assert after > before, 'frames the callback saw are invisible to the settle gate'


class TestABufferedFrameKeepsItsOwnFields:
    """With no newer frame stored, a poll hands back the buffered frame, so it
    hands back ITS number and ITS depth -- not whatever the camera looks like
    afterwards."""

    def test_a_repeat_poll_repeats_the_buffered_frames_ordinal(self):
        cam = _cam()
        grab_a_frame_made_after_now(cam)
        cam.stop_grabbing()
        first = cam.grab_latest()
        second = cam.grab_latest()
        assert second[4] == first[4], 'the buffered frame was handed back under a new number'

    def test_a_buffered_frame_keeps_the_depth_it_was_generated_under(self):
        cam = _cam()
        cam.set_pixel_format('Mono12')
        grab_a_frame_made_after_now(cam)
        cam.stop_grabbing()
        first = cam.grab_latest()
        cam.set_pixel_format('Mono8')
        _r, _i, _t, bits, _s = cam.grab_latest()
        assert first[3] == 12
        assert bits == 12, 'the buffered 12-bit frame was labelled with the new format depth'


class TestFieldsSurviveConcurrentStores:
    """The acquisition thread stores on its own thread while readers poll. A
    field read apart from the others can belong to the next frame, which is
    how pixels get a number that outranks them."""

    def test_pixels_and_ordinal_come_from_the_same_frame(self, monkeypatch):
        from drivers.simulated_camera import SimulatedCamera

        cam = SimulatedCamera(width=32, height=24)
        cam.exposure_t(1.0)  # the delivery ceiling: as many stores as it makes
        # Stamp each generated frame's own serial into a pixel, so the PIXELS
        # say which frame they are. The nth frame generated carries stamp n and
        # must be handed out as ordinal n, however reads and stores interleave.
        counter = itertools.count(1)

        def stamped():
            frame = np.zeros((24, 32), dtype=np.uint8)
            frame[0, 0] = next(counter) % 251
            return frame

        monkeypatch.setattr(cam, '_generate_image', stamped)
        cam.start_grabbing()
        polls = 0
        try:
            grab_a_frame_made_after_now(cam)

            def poll():
                nonlocal polls
                _r, img, _ts, _bits, seq = cam.grab_latest()
                assert img is not None
                assert img[0, 0] == seq % 251, (
                    f'frame stamped {img[0, 0]} was handed out as ordinal {seq}'
                )
                polls += 1

            _poll_across_ten_stores(cam, poll)
        finally:
            cam.stop_grabbing()
        assert polls >= 10

    def test_the_ordinal_never_runs_backwards_under_concurrency(self):
        cam = _cam()
        cam.exposure_t(1.0)
        seen = []
        try:
            grab_a_frame_made_after_now(cam)
            _poll_across_ten_stores(cam, lambda: seen.append(cam.grab_latest()[4]))
        finally:
            cam.stop_grabbing()
        assert len(set(seen)) >= 10, 'too few distinct frames for the check to mean anything'
        assert all(b >= a for a, b in itertools.pairwise(seen)), 'an ordinal went backwards'


class TestThePreviewPollCanRetireASettleCount:
    """The end-to-end consequence: preview frames reach the gate as distinct
    frames and a hardware write can actually settle in sim."""

    def test_preview_polls_settle_a_write(self):
        from modules.frame_validity import FrameValidity

        cam = _cam()
        try:
            fv = FrameValidity(lambda: cam.frames_delivered)
            fv.invalidate('gain')
            assert not fv.is_valid
            for _ in range(12):
                grab_a_frame_made_after_now(cam)
                _r, _i, _t, _b, seq = cam.grab_latest()
                fv.count_frame(seq)
                if fv.is_valid:
                    break
        finally:
            cam.stop_grabbing()
        assert fv.is_valid, 'preview frames never retired the skip count'
