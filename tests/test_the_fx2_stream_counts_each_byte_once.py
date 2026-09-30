# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The FX2 stream line's byte count is the bytes that arrived, each counted once.

Before, the grab loop counted every take, and while it waits for a frame's
closing delimiter it puts the frame's bytes back and takes them again, so the
same bytes were counted over and over: on an LS620 the line read 16.4-18.8
MB/s for a stream of 14.8. Now the stream counts bytes where they arrive.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from drivers import fx2driver

W = H = 100
_LAYOUT = fx2driver.frame_layout(W, H)
_BODY = bytes(_LAYOUT.frame_bytes)


def test_a_stream_counts_the_bytes_that_arrived_not_the_bytes_taken():
    stream = fx2driver._ByteStream()
    stream.append(b'0123456789')
    taken = stream.take(0)
    stream.put_back(taken[6:], limit=100, keep=100)
    stream.append(b'abc')
    stream.take(0)
    assert stream.take_arrived_count() == 13
    assert stream.take_arrived_count() == 0


def test_the_grab_loop_counts_a_frame_it_waited_on_once():
    stream = fx2driver._ByteStream()
    cam = object.__new__(fx2driver.FX2Camera)
    cam._fx2 = SimpleNamespace(
        stream=stream, take_gone_report=lambda: False, device_present=lambda: True
    )
    cam._grabbing = True
    cam._width, cam._height = W, H
    cam.stream_stats = fx2driver.StreamStats()
    stored = []
    cam.cam_image_handler = SimpleNamespace(
        _store_frame=lambda image, ts, significant_bits: stored.append(image.shape)
    )

    loop = threading.Thread(target=cam._grab_loop, daemon=True)
    loop.start()
    sent = 0
    for _ in range(3):
        # A frame's bytes, then a wait before the next delimiter: the loop
        # holds a frame's worth with nothing closing it, as on the wire.
        stream.append(fx2driver.FRAME_DELIM + _BODY)
        sent += len(fx2driver.FRAME_DELIM) + len(_BODY)
        time.sleep(0.05)
    stream.append(fx2driver.FRAME_DELIM)
    sent += len(fx2driver.FRAME_DELIM)
    deadline = time.monotonic() + 2.0
    while len(stored) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    cam._grabbing = False
    loop.join(2.0)

    assert len(stored) == 3
    assert cam.stream_stats._total_bytes == sent


def test_a_new_stream_counts_nothing_from_the_last_one():
    stream = fx2driver._ByteStream()
    stream.append(b'left over from the last stream')
    stream.restart()
    assert stream.take_arrived_count() == 0
    assert stream.take(0) is None  # nothing buffered, nothing new to parse


def test_a_stop_counts_what_arrived_after_the_grab_loops_last_take():
    stream = fx2driver._ByteStream()
    cam = object.__new__(fx2driver.FX2Camera)
    cam._fx2 = SimpleNamespace(stream=stream, stop_stream=lambda: None)
    cam._grabbing = True
    cam._grab_thread = None
    cam.stream_stats = fx2driver.StreamStats()
    stream.append(b'tail')
    cam.stop_grabbing()
    assert cam.stream_stats._total_bytes == 4
