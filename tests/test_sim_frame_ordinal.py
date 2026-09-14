# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated camera's frame fields all describe the same frame.

The simulator is the only camera in the tree that mints a frame instead of
receiving one from an SDK callback. Every real driver publishes through one
handler chokepoint that writes pixels, timestamp, depth and arrival ordinal
under a single lock, so none of them can hand back an ordinal belonging to a
different frame. The simulator wrote those fields by hand at four separate
places and two of them never advanced the ordinal at all -- the preview poll
and the callback pump.

Two consequences, both pinned below:

  - A frozen ordinal makes every preview poll look like the SAME frame to the
    settle gate, which dedupes on identity. No preview frame can ever retire a
    skip count, so a sim run cannot exercise the gate the capture paths rely on.
  - Reading a field back after the lock drops lets the pump's next frame
    supply it. An ordinal that runs ahead of its pixels retires a settle count
    the pixels predate, which is a capture accepted under the previous
    gain/exposure/LED state -- the defect the gate exists to prevent.
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


def _cam():
    from drivers.simulated_camera import SimulatedCamera

    cam = SimulatedCamera(width=32, height=24)
    cam.start_grabbing()
    return cam


class TestTheOrdinalAdvancesWithTheFrame:
    """A generated frame gets a number of its own, whichever path generates it."""

    def test_two_polls_return_two_ordinals(self):
        cam = _cam()
        try:
            _r1, _i1, _t1, _b1, seq1 = cam.grab_latest()
            _r2, _i2, _t2, _b2, seq2 = cam.grab_latest()
        finally:
            cam.stop_grabbing()
        assert seq1 is not None and seq2 is not None
        assert seq2 > seq1, 'a freshly generated frame must not reuse the previous number'

    def test_the_ordinal_a_poll_returns_is_the_one_the_gate_reads(self):
        cam = _cam()
        try:
            _r, _i, _t, _b, seq = cam.grab_latest()
            assert seq == cam.frames_delivered
        finally:
            cam.stop_grabbing()

    def test_the_pump_advances_the_ordinal(self):
        cam = _cam()
        seen = []
        cam.register_frame_callback(lambda img, ts, chunks: seen.append(ts))
        try:
            before = cam.frames_delivered
            deadline = time.monotonic() + 5.0
            while len(seen) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            after = cam.frames_delivered
        finally:
            cam.stop_grabbing()
        assert len(seen) >= 2, 'pump did not deliver; the rest of this test proves nothing'
        assert after > before, 'frames the pump generated are invisible to the settle gate'


class TestABufferedFrameKeepsItsOwnFields:
    """The exposure-gated paths hand back the buffered frame, so they hand back
    ITS number and ITS depth -- not whatever the camera looks like afterwards."""

    def _buffered(self, cam):
        """Mint one frame, then poll again inside the same exposure window."""
        cam.exposure_t(5_000.0)  # 5 s: the second poll cannot cross it
        cam.set_test_pattern(enabled=True, pattern='image_cycle')
        first = cam.grab_latest()
        second = cam.grab_latest()
        return first, second

    def test_a_buffered_poll_repeats_the_buffered_frames_ordinal(self):
        cam = _cam()
        try:
            first, second = self._buffered(cam)
        finally:
            cam.stop_grabbing()
        assert second[4] == first[4], 'the buffered frame was handed back under a new number'

    def test_a_buffered_frame_keeps_the_depth_it_was_generated_under(self):
        cam = _cam()
        try:
            cam.set_pixel_format('Mono12')
            first, _ = self._buffered(cam)
            cam.set_pixel_format('Mono8')
            _r, _i, _t, bits, _s = cam.grab_latest()
        finally:
            cam.stop_grabbing()
        assert first[3] == 12
        assert bits == 12, 'the buffered 12-bit frame was labelled with the new format depth'


class TestFieldsSurviveAConcurrentMint:
    """The pump mints on its own thread. A field read after the lock drops can
    belong to the next frame, which is how pixels get a number that outranks
    them."""

    def test_pixels_and_ordinal_come_from_the_same_frame(self, monkeypatch):
        cam = _cam()
        # Stamp each generated frame's own serial into a pixel, so the PIXELS
        # say which frame they are. The nth frame generated carries stamp n and
        # must be handed out as ordinal n, no matter who else is minting.
        counter = itertools.count(1)

        def stamped():
            frame = np.zeros((24, 32), dtype=np.uint8)
            frame[0, 0] = next(counter) % 251
            return frame

        monkeypatch.setattr(cam, '_generate_image', stamped)
        stop = threading.Event()

        def churn():
            while not stop.is_set():
                cam.grab_new_capture(0.0)

        t = threading.Thread(target=churn, daemon=True)
        t.start()
        try:
            for _ in range(200):
                _r, img, _ts, _bits, seq = cam.grab_latest()
                assert img is not None
                assert img[0, 0] == seq % 251, (
                    f'frame stamped {img[0, 0]} was handed out as ordinal {seq}'
                )
        finally:
            stop.set()
            t.join(timeout=5.0)
            cam.stop_grabbing()

    def test_the_ordinal_never_runs_backwards_under_concurrency(self):
        cam = _cam()
        seen = []
        stop = threading.Event()

        def churn():
            while not stop.is_set():
                cam.grab_new_capture(0.0)

        t = threading.Thread(target=churn, daemon=True)
        t.start()
        try:
            for _ in range(200):
                _r, _i, _t2, _b, seq = cam.grab_latest()
                seen.append(seq)
        finally:
            stop.set()
            t.join(timeout=5.0)
            cam.stop_grabbing()
        assert all(b >= a for a, b in itertools.pairwise(seen)), 'an ordinal went backwards'


class TestThePreviewPollCanRetireASettleCount:
    """The end-to-end consequence: with a real ordinal, preview frames reach the
    gate as distinct frames and a hardware write can actually settle in sim."""

    def test_preview_polls_settle_a_write(self):
        from modules.frame_validity import FrameValidity

        cam = _cam()
        try:
            fv = FrameValidity(lambda: cam.frames_delivered)
            fv.invalidate('gain')
            assert not fv.is_valid
            for _ in range(12):
                _r, _i, _t, _b, seq = cam.grab_latest()
                fv.count_frame(seq)
                if fv.is_valid:
                    break
        finally:
            cam.stop_grabbing()
        assert fv.is_valid, 'preview frames never retired the skip count'
