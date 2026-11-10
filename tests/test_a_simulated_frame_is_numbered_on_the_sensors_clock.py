# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated camera numbers each frame on the sensor's own time, whatever the host costs to draw it.

A camera starts, exposes and hands over every frame on its own schedule;
the host's drawing and carrying come after, and a slow host delays when a
frame is stored, or drops it, never when it was numbered. Frame validity
places a hardware change in the stream by that number, so a simulator that
numbered frames behind its own drawing put frames started before a move's
stop after it, and a capture after the move got a frame exposed mid-move.

Pinned here: the hand-over keeps the sensor's time under a drawing slower
than the frame period; a register written while a frame is being drawn
leaves that frame as it was numbered; and a stream stopped and restarted
from a frame callback leaves one stream storing.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from _lumascope.drivers import host_clock
from _lumascope.drivers.simulated_camera import SimulatedCamera
from tests.camera_fakes import grab_a_frame_made_after_now

PERIOD_S = 1.0 / SimulatedCamera._MAX_DELIVERY_FPS

# Six frame periods: a host that cannot draw a frame in the time the sensor
# takes to make one, slow enough that a hand-over held behind the drawing
# lags by far more than a loaded host delays the sensor thread's wake-up.
SLOW_DRAWING_S = 0.150


def _slow_drawing(cam, on_draw=None):
    """Draw every frame ``SLOW_DRAWING_S`` late, calling ``on_draw(exposure)`` first."""
    generate = cam._generate_image

    def slow(exposure, *args):
        if on_draw is not None:
            on_draw(exposure)
        time.sleep(SLOW_DRAWING_S)
        return generate(exposure, *args)

    cam._generate_image = slow


class _Stored:
    """The first ``wanted`` frames stored after it is built, through the camera's frame callback."""

    def __init__(self, cam, wanted):
        self.frames = []
        self._wanted = wanted
        self._collected = threading.Event()
        cam.register_frame_callback(self._collect)

    def _collect(self, frame):
        if len(self.frames) < self._wanted:
            self.frames.append(frame)
            if len(self.frames) == self._wanted:
                self._collected.set()

    def wait(self, timeout_s=5.0):
        assert self._collected.wait(timeout_s), f'only {len(self.frames)} frames were stored'
        return self.frames


def test_a_drawing_slower_than_the_frame_period_never_delays_the_number():
    """Each frame is handed over its own period after it starts, however long the host takes to draw it.

    The hand-over is stamped on the host's clock as the sensor thread wakes
    to it, so it trails the frame's leaving by however late the host wakes
    that thread, tens of milliseconds under a loaded test run. The bound is
    set by the defect, not by the host: a hand-over held behind a drawing
    trails its start by the drawing's time or more, and every later one by
    more again.
    """
    cam = SimulatedCamera(width=48, height=24)
    cam.exposure_t(1.0)  # the delivery ceiling's period sets the frame period
    _slow_drawing(cam)
    cam.open_and_start()
    try:
        grab_a_frame_made_after_now(cam, timeout_s=5.0)
        frames = _Stored(cam, 5).wait()
        clock_offset_ns = cam.camera_clock_ticks() - host_clock.steady_ns()
    finally:
        cam.stop_grabbing()

    leaves_s = PERIOD_S + (SimulatedCamera.SENSOR_PIPELINE_FRAMES - 1) * PERIOD_S
    lags_s = [(f.arrived_ns - (f.chunks['Timestamp'] - clock_offset_ns)) / 1e9 for f in frames]
    assert all(lag < leaves_s + SLOW_DRAWING_S / 2 for lag in lags_s), (
        f'start to hand-over {[round(lag, 4) for lag in lags_s]} s; the camera takes {leaves_s} s'
        f' and the host {SLOW_DRAWING_S} s to draw'
    )


def test_a_pattern_written_during_a_drawing_leaves_that_frame_as_it_was_numbered():
    """A frame numbered before a test-pattern write is drawn in the pattern it was numbered under."""
    cam = SimulatedCamera(width=48, height=24)
    cam.set_test_pattern(enabled=True, pattern='black')
    drawn_through_the_write = []

    def write_once(exposure):
        if not drawn_through_the_write:
            drawn_through_the_write.append(exposure.frame_id)
            cam.set_test_pattern(enabled=True, pattern='white')

    _slow_drawing(cam, on_draw=write_once)
    stored = _Stored(cam, 3)
    cam.open_and_start()
    try:
        frames = stored.wait()
    finally:
        cam.stop_grabbing()

    (frame_id,) = drawn_through_the_write
    (frame,) = [f for f in frames if f.chunks['FrameID'] == frame_id]
    assert not np.any(frame.image), (
        'a frame numbered under the black pattern was drawn in the pattern written after it'
    )


def test_a_stream_restarted_from_a_frame_callback_leaves_one_stream_storing():
    """A listener that changes the pixel format from its callback stops and restarts the stream on
    the delivery thread; the old stream ends, and frames keep coming at one stream's rate.

    The window is timed as it ran, since a loaded host oversleeps it, and
    the bound sits halfway to the two streams' rate the defect stores at.
    """
    cam = SimulatedCamera(width=48, height=24)
    cam.exposure_t(1.0)
    restarted = threading.Event()

    def restart_once(frame):
        if not restarted.is_set():
            restarted.set()
            cam.set_pixel_format('Mono12')

    cam.register_frame_callback(restart_once)
    cam.open_and_start()
    try:
        assert restarted.wait(5.0), 'no frame reached the callback; the rest proves nothing'
        grab_a_frame_made_after_now(cam, timeout_s=5.0)
        opened_ns = host_clock.steady_ns()
        before = cam.delivered_counts[0]
        time.sleep(1.0)
        stored = cam.delivered_counts[0] - before
        window_s = (host_clock.steady_ns() - opened_ns) / 1e9
    finally:
        cam.stop_grabbing()

    assert stored <= window_s * SimulatedCamera._MAX_DELIVERY_FPS * 1.5, (
        f'{stored} frames stored in {window_s:.3f} s: more than one stream is storing'
    )
