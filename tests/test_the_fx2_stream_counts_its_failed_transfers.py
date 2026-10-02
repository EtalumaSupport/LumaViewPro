# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The FX2 stream counts the transfers and packets that failed.

The stream-health line and the stop line both print a USB error count. Only
the bulk fallback fed it, and no constructed camera can take that path, so
on every stream that runs the count was 0 whatever happened: the ISO
callback resubmitted a failed transfer, and skipped a failed packet inside a
completed one, without a record. A failed packet is bytes missing from the
stream, the cause of a partial frame, so a count of zero beside partial
frames told a reader the link was clean when it was not.
"""

from __future__ import annotations

from types import SimpleNamespace

from drivers import fx2driver

COMPLETED = fx2driver.usb1.TRANSFER_COMPLETED
CANCELLED = fx2driver.usb1.TRANSFER_CANCELLED
FAILED = object()  # any status that is neither completed nor cancelled


def _camera():
    """The libusb transport's ISO callback, feeding a stream.

    Not streaming, so the callback does not resubmit the transfer.
    """
    transport = fx2driver._LibusbTransport()
    transport._stream = fx2driver._ByteStream()
    return transport


def _received(cam):
    """Each frame the stream ended, with whether a failure fell inside it."""
    return [(bytes(data), damaged) for data, damaged in cam._stream.take_frames()]


def _transfer(status, packets=()):
    return SimpleNamespace(getStatus=lambda: status, iterISO=lambda: iter(packets))


def _errors(cam):
    return cam._stream.take_counts().usb_errors


def test_a_transfer_that_fails_is_counted():
    cam = _camera()
    cam._iso_callback(_transfer(FAILED))
    assert _errors(cam) == 1


def test_each_failed_packet_in_a_completed_transfer_is_counted_and_its_bytes_are_not_kept():
    cam = _camera()
    packets = [(COMPLETED, b'ab'), (FAILED, b'xx'), (COMPLETED, b'cd'), (FAILED, b'')]
    cam._iso_callback(_transfer(COMPLETED, packets))
    assert _errors(cam) == 2
    # Each short packet ends a frame. The failed packet's bytes are in none,
    # and the frame it fell in is marked, so it is never stored.
    assert _received(cam) == [(b'ab', False), (b'cd', True)]


def test_a_clean_transfer_counts_nothing():
    cam = _camera()
    cam._iso_callback(_transfer(COMPLETED, [(COMPLETED, b'ab'), (COMPLETED, b'')]))
    assert _errors(cam) == 0
    assert _received(cam) == [(b'ab', False)]


def test_a_cancelled_transfer_is_the_stream_stopping_not_an_error():
    cam = _camera()
    cam._iso_callback(_transfer(CANCELLED))
    assert _errors(cam) == 0
