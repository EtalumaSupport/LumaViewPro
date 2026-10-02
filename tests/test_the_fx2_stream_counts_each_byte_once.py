# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The FX2 stream line's byte count is the bytes that arrived, each counted once.

Before, the grab loop counted every take, and while it waited for a frame's
closing delimiter it put the frame's bytes back and took them again, so the
same bytes were counted over and over: on an LS620 the line read 16.4-18.8
MB/s for a stream of 14.8. Now the stream counts bytes where they arrive,
once per packet, whatever the grab loop does with the frames they make.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from drivers import fx2driver

W = H = 100
_LAYOUT = fx2driver.frame_layout(W, H)
_BODY = bytes(_LAYOUT.frame_bytes)
_STEP = 2 * fx2driver.ISO_TRANSACTION_SIZE
# The frame as the device sends it: packets of two transactions, the last short.
_PACKETS = [_BODY[i : i + _STEP] for i in range(0, len(_BODY), _STEP)]


def test_a_stream_counts_the_bytes_that_arrived_not_the_bytes_taken():
    stream = fx2driver._ByteStream()
    stream.packet(b'0123456789')
    stream.take_frames()
    stream.packet(b'abc')
    stream.take_frames()
    stream.take_frames()
    assert stream.take_counts().arrived == 13
    assert stream.take_counts().arrived == 0


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
        # A frame's packets one at a time, the loop passing between them, as
        # on the wire: it sees a frame in assembly many times before its end.
        stream.packet(fx2driver.FRAME_DELIM)
        sent += len(fx2driver.FRAME_DELIM)
        for packet in _PACKETS:
            stream.packet(packet)
            sent += len(packet)
            time.sleep(0.01)
    deadline = time.monotonic() + 2.0
    while len(stored) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    cam._grabbing = False
    loop.join(2.0)

    assert len(stored) == 3
    assert cam.stream_stats._total_bytes == sent


def test_a_new_stream_counts_nothing_from_the_last_one():
    stream = fx2driver._ByteStream()
    stream.packet(b'left over from the last stream')
    stream.restart()
    assert stream.take_counts().arrived == 0
    assert stream.take_frames() == []  # nothing assembled from the last stream


def test_a_stop_counts_what_arrived_after_the_grab_loops_last_pass():
    stream = fx2driver._ByteStream()
    cam = object.__new__(fx2driver.FX2Camera)
    cam._fx2 = SimpleNamespace(stream=stream, stop_stream=lambda: None)
    cam._grabbing = True
    cam._grab_thread = None
    cam.stream_stats = fx2driver.StreamStats()
    stream.packet(b'tail!')
    cam.stop_grabbing()
    assert cam.stream_stats._total_bytes == 5
