# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2 frame ends where the device ends it, not at the delimiter.

The FX2 firmware writes a 4-byte delimiter between frames, and the driver
used to find frames by it. At some windows (1896, 1880, 1844 wide) the
shipped firmware loses the delimiter or writes 4 other bytes in its place,
so two whole frames arrived as one chunk of the wrong length and both were
discarded: 7-15% of the frames at those windows, a live-view stall each
time and a hole in a recording. The device also ends every frame with a
short ISO packet, one that is not a whole number of 1024-byte transactions;
the driver joined the packets and lost that mark. Now each packet reaches
the stream as a packet, a frame ends at the short one, and the delimiter is
only checked and counted.

The packet shapes below are the bench's (an LS620 at 1896x1896, 2026-10-01):
a frame's end in a microframe of 1024 + r bytes with no delimiter after
it, the same followed by 4 bytes that are not the delimiter, and a clean end
of 2048 + r followed by the delimiter, r being the frame's bytes past its
whole transactions. The bench's r was 123, at a Column_Size of w + 1; here it
is the r of the frame the driver now asks for at that window.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from drivers import fx2driver
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

T = fx2driver.ISO_TRANSACTION_SIZE
DELIM = fx2driver.FRAME_DELIM
WRONG = b'\x29' + DELIM[1:]  # a pixel byte where the delimiter's first belongs
FB = fx2driver.frame_layout(1896, 1896).frame_bytes
R = FB % T  # the short end's bytes past the whole transactions


def _frame(end: int, fill: int = 7) -> list[bytes]:
    """One 1896x1896 frame as packets: whole transactions, then an end of ``end`` bytes."""
    body = FB - end
    assert body % T == 0
    return [bytes([fill]) * T] * (body // T) + [bytes([fill]) * end]


def _feed(stream, packets):
    for p in packets:
        stream.packet(p)


def _lengths(stream):
    return [(len(data), damaged) for data, damaged in stream.take_frames()]


# -- the bench's shapes --------------------------------------------------------


def test_a_frame_whose_delimiter_was_lost_is_still_two_whole_frames():
    stream = fx2driver._ByteStream()
    _feed(stream, [DELIM, *_frame(1024 + R), *_frame(2048 + R), DELIM])

    assert _lengths(stream) == [(FB, False), (FB, False)]
    counts = stream.take_counts()
    assert (counts.delimiters_missing, counts.delimiters_wrong) == (1, 0)


def test_a_frame_whose_delimiter_was_overwritten_is_still_two_whole_frames():
    stream = fx2driver._ByteStream()
    _feed(stream, [DELIM, *_frame(1024 + R), WRONG, *_frame(2048 + R), DELIM])

    assert _lengths(stream) == [(FB, False), (FB, False)]
    counts = stream.take_counts()
    assert (counts.delimiters_missing, counts.delimiters_wrong) == (0, 1)


def test_clean_frames_count_no_delimiter_fault():
    stream = fx2driver._ByteStream()
    _feed(stream, [DELIM, *_frame(2048 + R), DELIM, *_frame(1024 + R), DELIM])

    assert _lengths(stream) == [(FB, False), (FB, False)]
    counts = stream.take_counts()
    assert (counts.delimiters_missing, counts.delimiters_wrong) == (0, 0)
    assert counts.arrived == 2 * FB + 3 * len(DELIM)


# -- failures land in the frame they fell in -----------------------------------


def test_a_failed_packet_damages_its_frame_and_not_the_next():
    stream = fx2driver._ByteStream()
    first = _frame(2048 + R)
    stream.packet(DELIM)
    _feed(stream, first[:10])
    stream.fail()
    _feed(stream, first[10:])
    _feed(stream, [DELIM, *_frame(2048 + R)])

    frames = _lengths(stream)
    assert frames[0][1] is True
    assert frames[1] == (FB, False)
    assert stream.take_counts().usb_errors == 1


def _libusb_transport(stream):
    transport = fx2driver._LibusbTransport()
    transport._stream = stream
    return transport


def _transfer(status, packets=()):
    return SimpleNamespace(getStatus=lambda: status, iterISO=lambda: iter(packets))


def test_a_libusb_transfer_that_fails_whole_damages_the_frame_it_fell_in():
    stream = fx2driver._ByteStream()
    transport = _libusb_transport(stream)
    ok = fx2driver.usb1.TRANSFER_COMPLETED
    first = _frame(2048 + R)

    transport._iso_callback(_transfer(ok, [(ok, p) for p in [DELIM, *first[:10]]]))
    transport._iso_callback(_transfer(object()))  # failed whole: its bytes never arrive
    transport._iso_callback(_transfer(ok, [(ok, p) for p in first[10:]]))

    assert _lengths(stream) == [(FB, True)]
    assert stream.take_counts().usb_errors == 1


def test_libusb_hands_each_packet_on_as_a_packet():
    stream = fx2driver._ByteStream()
    transport = _libusb_transport(stream)
    ok = fx2driver.usb1.TRANSFER_COMPLETED
    packets = [DELIM, *_frame(1024 + R), *_frame(2048 + R)]

    transport._iso_callback(_transfer(ok, [(ok, p) for p in packets]))

    # Joined into one run the two frames would be one chunk; as packets they
    # end where the device ended them.
    assert _lengths(stream) == [(FB, False), (FB, False)]


def test_the_windows_reader_feeds_packets_and_failures_to_the_stream(monkeypatch):
    readers = []

    class Reader:
        def __init__(self, vid, pid, **kwargs):
            self.kwargs = kwargs
            self.device = SimpleNamespace(control_transfer=lambda *a, **kw: 0)
            readers.append(self)

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setitem(
        sys.modules, 'drivers.winusb_iso', types.SimpleNamespace(WinUsbIsoReader=Reader)
    )
    monkeypatch.setattr(fx2driver._WinUsbTransport, '_release_idle', lambda self: None)
    stream = fx2driver._ByteStream()
    fx2driver._WinUsbTransport().start_stream(stream, on_gone=lambda: None)
    on_data, on_error = readers[0].kwargs['on_data'], readers[0].kwargs['on_error']

    first = _frame(2048 + R)
    on_data(DELIM)
    for p in first[:10]:
        on_data(p)
    on_error()  # a failed packet or read, from the reader's one thread, in order
    for p in [*first[10:], DELIM, *_frame(2048 + R)]:
        on_data(p)

    assert _lengths(stream) == [(FB, True), (FB, False)]


# -- starts, window changes, and a stream with no frame end --------------------


def test_a_stream_that_opens_on_the_delimiter_keeps_its_first_frame():
    stream = fx2driver._ByteStream()
    _feed(stream, [DELIM, *_frame(2048 + R)])

    assert _lengths(stream) == [(FB, False)]
    assert stream.take_counts().delimiters_missing == 0


def test_a_stream_that_opens_mid_frame_loses_only_that_frame():
    stream = fx2driver._ByteStream()
    _feed(stream, [*_frame(2048 + R)[100:], DELIM, *_frame(2048 + R)])

    frames = _lengths(stream)
    assert frames[0][0] < FB
    assert frames[1] == (FB, False)


def test_a_flush_just_after_a_frame_end_counts_no_missing_delimiter():
    stream = fx2driver._ByteStream()
    _feed(stream, [DELIM, *_frame(2048 + R)])
    stream.flush()  # a window change, before the delimiter after that end came
    _feed(stream, _frame(2048 + R))

    assert _lengths(stream) == [(FB, False)]
    assert stream.take_counts().delimiters_missing == 0


def test_a_flush_mid_frame_drops_what_was_assembled():
    stream = fx2driver._ByteStream()
    stream.packet(DELIM)
    _feed(stream, _frame(2048 + R)[:500])  # half a frame of the old window
    stream.flush()
    _feed(stream, _frame(2048 + R))

    assert _lengths(stream) == [(FB, False)]


def _grab_loop_on(stream, w, h):
    cam = object.__new__(fx2driver.FX2Camera)
    cam._fx2 = SimpleNamespace(
        stream=stream, take_gone_report=lambda: False, device_present=lambda: True
    )
    cam._grabbing = True
    cam._width, cam._height = w, h
    cam.stream_stats = fx2driver.StreamStats()
    stored = []
    cam.cam_image_handler = SimpleNamespace(
        _store_frame=lambda image, ts, significant_bits: stored.append(image.shape)
    )
    return cam, stored


def test_a_frame_a_failure_fell_in_is_not_stored_even_at_its_length():
    stream = fx2driver._ByteStream()
    cam, stored = _grab_loop_on(stream, 1896, 1896)
    first = _frame(2048 + R)
    stream.packet(DELIM)
    _feed(stream, first[:10])
    stream.fail()  # a failed packet: the frame's length can still come out right
    _feed(stream, first[10:])
    _feed(stream, [DELIM, *_frame(2048 + R)])

    loop = threading.Thread(target=cam._grab_loop, daemon=True)
    loop.start()
    deadline = time.monotonic() + 2.0
    while not stored and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    cam._grabbing = False
    loop.join(2.0)

    assert stored == [(1896, 1896)]
    assert cam.stream_stats.summary()['partial_frames'] == 1


def test_a_stream_with_no_frame_end_is_held_to_one_largest_frame():
    stream = fx2driver._ByteStream()
    longest = fx2driver.frame_layout(fx2driver.IMG_WIDTH, fx2driver.IMG_HEIGHT).frame_bytes
    held = 0
    for _ in range(3 * longest // T):
        stream.packet(b'\x00' * T)
        held = max(held, len(stream._frame))

    assert held <= longest + T
    frames = _lengths(stream)
    assert len(frames) == 2
    assert all(n > longest for n, _damaged in frames)


def test_the_unpatched_firmwares_packets_are_never_stored_whole():
    # The original hex commits no short packet: its delimiter rides at the end
    # of a data microframe (1028, 2052 bytes), so each frame ends there short.
    stream = fx2driver._ByteStream()
    body = FB - FB % T
    packets = [b'\x07' * T] * (body // T - 1) + [b'\x07' * T + DELIM]
    _feed(stream, packets * 2)

    assert all(n != FB for n, _damaged in _lengths(stream))


def test_a_window_whose_frame_is_even_is_refused():
    with pytest.raises(ValueError, match='odd-length'):
        fx2driver.frame_layout(1896, 1895)


# -- end to end, through the simulated LS620 -----------------------------------


def _wait_until(condition, timeout_s):
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)
    return condition()


@pytest.fixture
def session():
    settings = complete_settings()
    settings['microscope'] = 'LS620'
    settings['simulator_tier'] = 'fast'
    session = ScopeSession.create(settings, simulate=True)
    session.scope.imaging.start_streaming()
    assert _wait_until(lambda: session.scope.imaging.get_image() is not None, 5.0)
    yield session
    session.shutdown()


@pytest.mark.parametrize('fault', ['lose_delimiters', 'garble_delimiters'])
def test_a_simulated_ls620_stores_every_frame_whatever_its_delimiters(session, fault):
    camera = session.scope._camera_driver
    transport = session.scope._led_driver._fx2._transport
    # Every second frame from here: under delimiter framing each fault glued
    # two whole frames into one shifted chunk.
    getattr(transport, fault)(2)
    before = camera.stream_stats.summary()

    assert _wait_until(
        lambda: camera.stream_stats.summary()['good_frames'] >= before['good_frames'] + 8, 6.0
    )

    after = camera.stream_stats.summary()
    assert after['shifted_frames'] == before['shifted_frames']
    assert after['partial_frames'] == before['partial_frames']
    key = 'delimiters_missing' if fault == 'lose_delimiters' else 'delimiters_wrong'
    assert after[key] > before[key]


def test_the_stream_and_stop_lines_say_how_often_the_delimiter_failed(session, caplog, monkeypatch):
    monkeypatch.setattr(fx2driver, 'logger', logging.getLogger('fx2_under_test'))
    caplog.set_level(logging.INFO, logger='fx2_under_test')
    camera = session.scope._camera_driver
    camera.STATS_LOG_INTERVAL = 0.2
    transport = session.scope._led_driver._fx2._transport
    transport.lose_delimiters(2)
    transport.garble_delimiters(3)
    assert _wait_until(
        lambda: (
            min(
                camera.stream_stats.summary()['delimiters_missing'],
                camera.stream_stats.summary()['delimiters_wrong'],
            )
            > 0
        ),
        6.0,
    )

    camera.stop_grabbing()

    s = camera.stream_stats.summary()
    said = f'delimiters {s["delimiters_missing"]} missing / {s["delimiters_wrong"]} wrong'
    lines = [r.getMessage() for r in caplog.records]
    assert any(
        'stream:' in m and 'delimiters' in m and '0 missing / 0 wrong' not in m for m in lines
    )
    assert any('streaming stopped' in m and said in m for m in lines)


def test_a_simulated_unplugs_failed_transfers_reach_the_stream():
    from drivers.simulated_fx2 import SimulatedFX2

    sim = SimulatedFX2()
    sim.connection.start_stream()
    sim.connection.stream.take_counts()
    sim.connection._transport.unplug(resubmit_failures=2)

    assert sim.connection.stream.take_counts().usb_errors == 2
    sim.connection._transport.close()
