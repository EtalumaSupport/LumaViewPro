# Copyright Etaluma, Inc.
"""FX2 hardware tests -- opt-in via --run-fx2-hardware.

These tests require:
  1. pyusb + libusb1 installed (the conftest usb/usb1 mocks are skipped
     when the flag is set)
  2. A connected FX2 scope (LS620 / LS560 class, MT9P031 sensor)

Skipped by default. Run with:
    pytest tests/test_fx2_hardware.py --run-fx2-hardware

The `fx2_hardware` marker is gated by conftest.pytest_collection_modifyitems --
test bodies do NOT need their own skip dance.

Mirrors the shape of test_pylon_hardware.py / test_ids_hardware.py so the
abstraction is symmetric across camera vendors.
"""

import os
import threading
import time
import unittest
from collections import Counter
from datetime import datetime
from itertools import pairwise

import numpy as np
import pytest

# When --run-fx2-hardware is set, conftest skips installing the usb/usb1
# mocks so the real libusb stack loads here. When the flag is NOT set,
# this import succeeds against the conftest mock and the marker below
# skips the tests at collection time.
from drivers.fx2driver import (
    IMG_WIDTH,
    REG_EXPOSURE,
    REG_COL_SIZE,
    REG_READ_MODE2,
    REG_ROW_BLACK,
    FX2Camera,
    FX2LEDController,
    column_size_for,
    exposure_s,
    frame_layout,
    frame_time_s,
    shutter_width_for,
)
from lvp_logger import logger


@pytest.mark.fx2_hardware
class TestFX2(unittest.TestCase):
    def setUp(self):
        self.camera = FX2Camera()
        self.camera.open_and_start()

    def tearDown(self):
        self.camera.disconnect()
        time.sleep(0.5)

    def test_connected(self):
        self.assertTrue(self.camera.is_connected())

    def test_set_frame_size_returns_delivered_geometry(self):
        # 638/482 are deliberately off the 4-px step grid: the driver rounds
        # down and must return what it actually applied, matching what
        # get_frame_size() then reports -- the no-read-back caching contract.
        delivered = self.camera.set_frame_size(638, 482)
        self.assertEqual(delivered, {'width': 636, 'height': 480})
        self.assertEqual(self.camera.get_frame_size(), delivered)

    def test_grab_frame_matches_window(self):
        # The real proof the sensor window applied: the delivered frame's
        # pixel dimensions match the size just set.
        delivered = self.camera.set_frame_size(800, 600)
        self.assertEqual(delivered, {'width': 800, 'height': 600})
        self.camera.start_grabbing()
        time.sleep(1.0)  # let the ISO stream + frame parser resync
        result, _timestamp, _seq = self.camera.grab()
        self.assertTrue(result)
        self.assertIsNotNone(self.camera.array)
        self.assertEqual(self.camera.array.shape[0], 600)
        self.assertEqual(self.camera.array.shape[1], 800)


# ---------------------------------------------------------------------------
# Characterization: the Stage 3 bench measurements (exposure above the 178 ms
# cap, the exposure and gain sweeps, frame-validity skip counts). Each test
# records what the sensor did through the driver logger; run with
# --driver-log so the lines reach driver_test_<ts>.log:
#
#     pytest tests/test_fx2_hardware.py --run-fx2-hardware --driver-log -k Characterization
#
# Needs light on the sensor for the sweeps and the steps: a fluorescent LED on
# anything that returns a mid-grey frame.
# ---------------------------------------------------------------------------

_BLUE = 0
_SATURATED = 250.0


def _new_frames(camera, count, *, after_seq=None, timeout_s=10.0):
    """The next ``count`` distinct stored frames: ``(seq, mean, max)`` each."""
    frames = []
    last = after_seq
    deadline = time.monotonic() + timeout_s
    while len(frames) < count and time.monotonic() < deadline:
        ok, image, _ts, _bits, seq = camera.grab_latest()
        if ok and seq is not None and (last is None or seq > last):
            frames.append((seq, float(np.mean(image)), int(np.max(image))))
            last = seq
        else:
            time.sleep(0.005)
    return frames


def _settled_mean(camera, *, skip=4, count=3):
    _new_frames(camera, skip)
    frames = _new_frames(camera, count)
    return float(np.mean([m for _s, m, _x in frames])), max(x for _s, _m, x in frames)


class _FX2BenchCase(unittest.TestCase):
    """A streaming camera and its LED controller; the bench classes below share it."""

    def setUp(self):
        self.camera = FX2Camera()
        self.camera.open_and_start()
        self.led = FX2LEDController()
        self.camera.start_grabbing()
        self.assertTrue(_new_frames(self.camera, 1), 'no frame after start')

    def tearDown(self):
        self.led.leds_off()
        self.camera.disconnect()
        time.sleep(0.5)

    def _light_to_mid_grey(self):
        """Light Blue at the first current whose frame is mid-grey; return the mA."""
        self.camera.exposure_t(50)
        self.camera.gain(0)
        for ma in (5, 10, 20, 50, 100, 200, 400):
            self.led.led_on(_BLUE, ma)
            mean, peak = _settled_mean(self.camera)
            logger.info(
                '[FX2 bench] light: Blue %d mA at 50 ms / 0 dB -> mean %.1f max %d', ma, mean, peak
            )
            if mean >= 40:
                return ma
        self.skipTest('no LED current gave a mid-grey frame; nothing to measure')


@pytest.mark.fx2_hardware
class TestFX2Characterization(_FX2BenchCase):
    def test_measure_the_stream_above_the_exposure_cap(self):
        """Row 3: does the parser lose frames above 178 ms? Recorded, not asserted."""
        self.camera.set_frame_size(1900, 1900)
        for ms in (100, 178, 200, 214, 250, 400, 800):
            applied_us = self.camera.exposure_t(ms)
            time.sleep(1.0 + 3 * ms / 1000)
            self.camera.stream_stats.reset()
            time.sleep(6.0 + 5 * ms / 1000)
            s = self.camera.stream_stats.summary()
            logger.info(
                '[FX2 bench] exposure %d ms (applied %.1f ms): %.1f s, %d good / %d partial / '
                '%d shifted, %.2f fps, %.1f MB/s, shifted sizes %s',
                ms,
                applied_us / 1000,
                s['elapsed_s'],
                s['good_frames'],
                s['partial_frames'],
                s['shifted_frames'],
                s['fps_average'],
                s['throughput_MBps'],
                sorted(set(s['shifted_sizes'])),
            )

    def test_measure_long_exposures_with_light(self):
        """Row 3, lit: does the frame mean keep scaling with exposure above 178 ms?"""
        self.camera.set_frame_size(1900, 1900)
        self.camera.gain(0)
        self.led.led_on(_BLUE, 20)
        for ms in (50, 178, 250, 400, 800):
            applied_us = self.camera.exposure_t(ms)
            _new_frames(self.camera, 3, timeout_s=10.0)
            frames = _new_frames(self.camera, 3, timeout_s=10.0)
            means = [round(m, 1) for _s, m, _x in frames]
            logger.info(
                '[FX2 bench] lit exposure (Blue 20 mA, 0 dB): asked %d ms, applied %.1f ms -> '
                'means %s, max %s, mean per ms %.3f',
                ms,
                applied_us / 1000,
                means,
                [x for _s, _m, x in frames],
                float(np.mean(means)) / (applied_us / 1000),
            )

    def test_measure_the_shifted_frames_at_1896(self):
        """Row 6 follow-up: the sizes of the frames the parser discards at 1896x1896."""
        self.led.led_on(_BLUE, 200)
        self.camera.exposure_t(50)
        for w, h in ((1900, 1900), (1896, 1896)):
            self.camera.set_frame_size(w, h)
            time.sleep(2.0)
            self.camera.stream_stats.reset()
            time.sleep(60.0)
            s = self.camera.stream_stats.summary()
            logger.info(
                '[FX2 bench] window %dx%d: %.1f s, %d good / %d partial / %d shifted, '
                '%.1f MB/s, shifted sizes %s, partial sizes %s',
                w,
                h,
                s['elapsed_s'],
                s['good_frames'],
                s['partial_frames'],
                s['shifted_frames'],
                s['throughput_MBps'],
                s['shifted_sizes'],
                s['partial_sizes'],
            )

    def test_measure_the_exposure_and_gain_sweeps(self):
        """Row 4: read-back against the request, and the frame mean rising with each."""
        ma = self._light_to_mid_grey()
        self.camera.gain(0)
        exposure_means = []
        for ms in (5, 10, 20, 30, 50, 75, 100, 150, 178):
            applied_us = self.camera.exposure_t(ms)
            mean, peak = _settled_mean(self.camera)
            exposure_means.append((ms, mean))
            logger.info(
                '[FX2 bench] exposure sweep (Blue %d mA, 0 dB): asked %d ms, applied %.4f ms, '
                'read back %.4f ms -> mean %.1f max %d',
                ma,
                ms,
                applied_us / 1000,
                self.camera.get_exposure_t(),
                mean,
                peak,
            )
        self.camera.exposure_t(20)
        gain_means = []
        for db in (0, 3, 6, 9, 12, 18, 24, 30, 36, 42.1):
            applied = self.camera.gain(db)
            mean, peak = _settled_mean(self.camera)
            gain_means.append((db, mean))
            logger.info(
                '[FX2 bench] gain sweep (Blue %d mA, 20 ms): asked %.1f dB, applied %.4f dB, '
                'read back %.4f dB -> mean %.1f max %d',
                ma,
                db,
                applied,
                self.camera.get_gain(),
                mean,
                peak,
            )
        for name, sweep in (('exposure', exposure_means), ('gain', gain_means)):
            below = [m for _v, m in sweep if m < _SATURATED]
            falls = [(a, b) for a, b in pairwise(below) if b < a - 1.0]
            logger.info(
                '[FX2 bench] %s sweep monotonic: %s', name, 'yes' if not falls else f'no {falls}'
            )
            self.assertEqual(falls, [], f'{name} sweep mean fell between steps')

    def test_measure_frames_until_a_change_shows(self):
        """Row 5: stored frames after a write that still show the old value."""
        ma = self._light_to_mid_grey()
        steps = (
            ('LED off', lambda: self.led.led_off(_BLUE), lambda: self.led.led_on(_BLUE, ma)),
            ('LED on', lambda: self.led.led_on(_BLUE, ma), lambda: self.led.led_off(_BLUE)),
            (
                'exposure 50 -> 100 ms',
                lambda: self.camera.exposure_t(100),
                lambda: self.camera.exposure_t(50),
            ),
            ('gain 0 -> 6 dB', lambda: self.camera.gain(6), lambda: self.camera.gain(0)),
        )
        for name, change, undo in steps:
            for trial in range(5):
                undo()
                before = _new_frames(self.camera, 8)
                base = float(np.mean([m for _s, m, _x in before[-3:]]))
                last_seq = before[-1][0]
                change()
                after = _new_frames(self.camera, 10, after_seq=last_seq)
                settled = float(np.mean([m for _s, m, _x in after[-3:]]))
                half = base + 0.5 * (settled - base)
                stale = next(
                    (i for i, (_s, m, _x) in enumerate(after) if (m - half) * (settled - base) > 0),
                    None,
                )
                logger.info(
                    '[FX2 bench] %s, trial %d: before %.1f, settled %.1f, stale frames %s, means %s',
                    name,
                    trial + 1,
                    base,
                    settled,
                    stale,
                    [round(m, 1) for _s, m, _x in after],
                )


# ---------------------------------------------------------------------------
# Timing: the sensor's row time and the pixel-clock sweep (FX2 plan section
# 21, Phase A). Measurement only: the PLL is written through the driver's own
# register write with the stream stopped, as connect() writes it, and P1 goes
# back to 13 at the end. Run with --driver-log:
#
#     pytest tests/test_fx2_hardware.py --run-fx2-hardware --driver-log -k Timing
# ---------------------------------------------------------------------------

_REG_HBLANK = 0x05
_P1_TODAY = 13
_LONG_ROWS = 20000  # longer than any window's readout: the frame period is the shutter's
# The request that lands on 20000 rows at the full window.
_LONG_MS = exposure_s(_LONG_ROWS, IMG_WIDTH + 1) * 1000
_STEP_S = 60.0
# Set from the sweep's result before the follow-on tests run; empty / None skips them.
# The sweep of 2026-10-01: clean at 13, 12 and 11, nothing framed at 10.
_CHOSEN_P1S = (12, 11)
_HBLANK_P1 = None
# R0x05 adds blanking only above HBMIN (450 clocks at full resolution).
_HBLANK_STEPS = (500, 550, 600)


def _timed_frames(camera, count, *, timeout_s):
    """The next ``count`` distinct stored frames: ``(seq, stored_at, image)`` each."""
    frames = []
    last = None
    deadline = time.monotonic() + timeout_s
    while len(frames) < count and time.monotonic() < deadline:
        ok, image, stored_at, _bits, seq = camera.grab_latest()
        if ok and seq is not None and (last is None or seq > last):
            frames.append((seq, stored_at, image))
            last = seq
        else:
            time.sleep(0.005)
    return frames


def _frame_period_s(camera, count, *, timeout_s):
    """Median interval between consecutive stored frames, and every such interval."""
    frames = _timed_frames(camera, count + 1, timeout_s=timeout_s)
    intervals = [
        round((b_at - a_at).total_seconds(), 4)
        for (a_seq, a_at, _a), (b_seq, b_at, _b) in pairwise(frames)
        if b_seq == a_seq + 1
    ]
    return (float(np.median(intervals)) if intervals else None), intervals


def _integrity(image):
    """Mean, std, and the largest jumps between adjacent row means and column means."""
    img = image.astype(np.float32)
    return (
        float(img.mean()),
        float(img.std()),
        float(np.abs(np.diff(img.mean(axis=1))).max()),
        float(np.abs(np.diff(img.mean(axis=0))).max()),
    )


def _set_p1(camera, p1):
    """Rewrite the PLL with P1 = ``p1``, the sequence connect() uses, with the stream stopped."""
    camera.stop_grabbing()
    camera._program_pll(p1)
    camera.start_grabbing()


def _mean_of(frames):
    return float(np.mean([img.mean() for _seq, _at, img in frames])) if frames else float('nan')


@pytest.mark.fx2_hardware
class TestFX2Timing(_FX2BenchCase):
    def tearDown(self):
        if self.camera.is_connected():
            self.camera._fx2.sensor_reg_write(_REG_HBLANK, 0)
            _set_p1(self.camera, _P1_TODAY)
        super().tearDown()

    def _settle(self, frames=3):
        """Let a change reach the stored frames: the sensor pipelines two."""
        return _timed_frames(self.camera, frames, timeout_s=frames * 3.0 + 5.0)

    def _run_step(self, label, baseline=None):
        """One 60 s stream: counters and image statistics. Returns (clean, jumps)."""
        self._settle()
        self.camera.stream_stats.reset()
        time.sleep(_STEP_S)
        s = self.camera.stream_stats.summary()
        stats = [_integrity(img) for _s, _a, img in _timed_frames(self.camera, 5, timeout_s=10.0)]
        mean = float(np.mean([st[0] for st in stats])) if stats else float('nan')
        std = float(np.mean([st[1] for st in stats])) if stats else float('nan')
        row_jump = max((st[2] for st in stats), default=float('nan'))
        col_jump = max((st[3] for st in stats), default=float('nan'))
        clean = bool(stats) and (
            s['partial_frames'] == 0 and s['shifted_frames'] == 0 and s['usb_errors'] == 0
        )
        if clean and baseline is not None:
            clean = row_jump <= 2 * baseline[0] and col_jump <= 2 * baseline[1]
        logger.info(
            '[FX2 bench] timing %s: %.1f s, %d good / %d partial / %d shifted, %d USB errors, '
            '%.2f fps, %.2f MB/s, mean %.1f std %.1f, max row jump %.2f, max col jump %.2f, '
            'shifted sizes %s, partial sizes %s -> %s',
            label,
            s['elapsed_s'],
            s['good_frames'],
            s['partial_frames'],
            s['shifted_frames'],
            s['usb_errors'],
            s['fps_average'],
            s['throughput_MBps'],
            mean,
            std,
            row_jump,
            col_jump,
            sorted(set(s['shifted_sizes'])),
            sorted(set(s['partial_sizes'])),
            'clean' if clean else 'NOT clean',
        )
        return clean, (row_jump, col_jump)

    def _long_period(self, label):
        """The frame period at 20000 rows, then back to 50 ms."""
        self.camera.exposure_t(_LONG_MS)
        self._settle()
        period, intervals = _frame_period_s(self.camera, 3, timeout_s=20.0)
        self.camera.exposure_t(50)
        self._settle()
        logger.info(
            '[FX2 bench] timing %s: period at %d rows %s s, intervals %s',
            label,
            _LONG_ROWS,
            period,
            intervals,
        )

    def test_measure_the_row_time_at_three_widths(self):
        """A1: the frame period at 20000 rows, and the lit mean, at 1900, 1000 and 500 wide."""
        self.camera.set_frame_size(1900, 1900)
        self.camera.gain(0)
        self.camera.exposure_t(_LONG_MS)
        ma = None
        for candidate in (3, 5, 10, 20, 40):
            self.led.led_on(_BLUE, candidate)
            self._settle()
            mean = _mean_of(_timed_frames(self.camera, 2, timeout_s=10.0))
            logger.info(
                '[FX2 bench] timing A1 light: Blue %d mA at %d rows -> mean %.1f',
                candidate,
                _LONG_ROWS,
                mean,
            )
            if mean >= 60:
                ma = candidate
                break
        if ma is None:
            self.skipTest('no LED current lit the sensor at 20000 rows')
        for w in (1900, 1000, 500):
            self.camera.set_frame_size(w, w)
            # The driver keeps the exposure across a window, not the rows;
            # this measures the period at fixed rows, so it writes them.
            self.camera._fx2.sensor_reg_write(REG_EXPOSURE, _LONG_ROWS)
            self.led.led_on(_BLUE, ma)
            self._settle()
            period, intervals = _frame_period_s(self.camera, 5, timeout_s=20.0)
            lit = _mean_of(_timed_frames(self.camera, 2, timeout_s=10.0))
            self.led.led_off(_BLUE)
            self._settle()
            dark = _mean_of(_timed_frames(self.camera, 2, timeout_s=10.0))
            logger.info(
                '[FX2 bench] timing A1 window %d: period %s s, intervals %s, lit %.1f, dark %.1f, '
                'lit - dark %.1f (Blue %d mA, %d rows, 0 dB)',
                w,
                period,
                intervals,
                lit,
                dark,
                lit - dark,
                ma,
                _LONG_ROWS,
            )

    def test_measure_the_pixel_clock_sweep(self):
        """A2: P1 from 13 down to 8 at 1900x1900; stops at the first step that is not clean."""
        self.camera.set_frame_size(1900, 1900)
        ma = self._light_to_mid_grey()
        baseline = None
        for p1 in (13, 12, 11, 10, 9, 8):
            _set_p1(self.camera, p1)
            clean, jumps = self._run_step(f'A2 P1={p1} (Blue {ma} mA, 50 ms, 0 dB)', baseline)
            if baseline is None:
                baseline = jumps
            self._long_period(f'A2 P1={p1}')
            if not clean:
                logger.info('[FX2 bench] timing A2: the sweep stops at P1=%d', p1)
                break

    def test_measure_the_chosen_clock_at_other_windows(self):
        """A3: each candidate step at 1000, 500, 1896 and 1880."""
        if not _CHOSEN_P1S:
            self.skipTest('set _CHOSEN_P1S from the sweep first')
        ma = self._light_to_mid_grey()
        for p1 in _CHOSEN_P1S:
            _set_p1(self.camera, p1)
            for w in (1000, 500, 1896, 1880):
                self.camera.set_frame_size(w, w)
                self._run_step(f'A3 P1={p1} window {w} (Blue {ma} mA, 50 ms, 0 dB)')

    def test_measure_horizontal_blanking_at_the_failing_clock(self):
        """A4: extra horizontal blanking at the step the sweep stopped on."""
        if _HBLANK_P1 is None:
            self.skipTest('set _HBLANK_P1 from the sweep first')
        self.camera.set_frame_size(1900, 1900)
        ma = self._light_to_mid_grey()
        _set_p1(self.camera, _HBLANK_P1)
        for extra in _HBLANK_STEPS:
            self.camera._fx2.sensor_reg_write(_REG_HBLANK, extra)
            self._run_step(f'A4 P1={_HBLANK_P1} R0x05={extra} (Blue {ma} mA, 50 ms, 0 dB)')


# ---------------------------------------------------------------------------
# Phase B's P0 (FX2 plan section 21.5): what the wire carries per row, the
# conforming Column_Size streamed, dark frames at the black levels, and a
# window change. Measurement only; every register goes back in tearDown. Run
# with --driver-log:
#
#     pytest tests/test_fx2_hardware.py --run-fx2-hardware --driver-log -k P0
#
# The dark-frame test needs a light-tight cover on the light path. Set
# FX2_BENCH_DUMP_DIR to keep the raw wire frames the test-pattern test reads.
# ---------------------------------------------------------------------------

_REG_OUTPUT_CONTROL = 0x07
_OUTPUT_CONTROL_DEFAULT = 0x1F82  # RR R0x07 power-on default; the driver never writes it
_SYNCHRONIZE_CHANGES = 0x0001
_REG_TEST_PATTERN_CONTROL = 0xA0
_REG_TEST_PATTERN_GREEN = 0xA1
_REG_TEST_PATTERN_RED = 0xA2
_REG_TEST_PATTERN_BLUE = 0xA3
_READ_MODE2_TODAY = 0x4040  # Mirror_Column + Row_BLC, as _init_sensor writes it
_READ_MODE2_NO_ROW_BLC = 0x4000  # DS p37: BLC off while a test pattern runs
_ROW_BLACK_TODAY = 0x0000
_REG_BLC = 0x62
_BLC_TODAY = 0x0000  # its default: connect's soft reset leaves it there
# Color field values (12-bit), one distinct bit pattern per channel, so the
# 8-bit image shows which DOUT bits reach the bus and which 2x2 phase is which.
_FIELD_GREEN = 0x0A50
_FIELD_RED = 0x05A0
_FIELD_BLUE = 0x0C30
# RR R0xA0 bits 6:3 = mode, bit 0 = enable.
_PATTERNS = (('color field', 0), ('horizontal gradient', 1), ('walking 1s', 5))
_DARK_FLOOR = 255 * 0.03  # ImagingAPI._DARK_FLOOR_FRACTION on an 8-bit frame


class _WireSpy:
    """Wraps the stream's ``take_frames()``: the length of every frame it ended, and the first few.

    Each is the bytes between two of the device's frame ends, the delimiter
    left out, whether or not the grab loop stores it.
    """

    def __init__(self, stream):
        self._stream = stream
        self._take = stream.take_frames
        self._lock = threading.Lock()
        self.sizes = Counter()
        self.frames = []
        self._keep = 0
        stream.take_frames = self._spy

    def _spy(self):
        ended = self._take()
        with self._lock:
            for frame, _damaged in ended:
                self.sizes[len(frame)] += 1
                if len(self.frames) < self._keep:
                    self.frames.append(bytes(frame))
        return ended

    def reset(self, keep=0):
        with self._lock:
            self.sizes = Counter()
            self.frames = []
            self._keep = keep

    def wait_frames(self, count, timeout_s=15.0):
        self.reset(keep=count)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.frames) >= count:
                    return list(self.frames)
            time.sleep(0.02)
        with self._lock:
            return list(self.frames)

    def remove(self):
        del self._stream.take_frames


def _phase_values(image):
    """The most common value in each 2x2 phase, top-left first."""
    return [
        Counter(image[r::2, c::2].ravel().tolist()).most_common(2)
        for r, c in ((0, 0), (0, 1), (1, 0), (1, 1))
    ]


@pytest.mark.fx2_hardware
class TestFX2PhaseBP0(_FX2BenchCase):
    def setUp(self):
        super().setUp()
        self.spy = _WireSpy(self.camera._fx2.stream)
        self.camera.exposure_t(50)
        self.camera.gain(0)

    def tearDown(self):
        self.spy.remove()
        if self.camera.is_connected():
            write = self.camera._fx2.sensor_reg_write
            for reg, value in (
                (_REG_TEST_PATTERN_CONTROL, 0),
                (REG_READ_MODE2, _READ_MODE2_TODAY),
                (REG_ROW_BLACK, _ROW_BLACK_TODAY),
                (_REG_BLC, _BLC_TODAY),
                (_REG_OUTPUT_CONTROL, _OUTPUT_CONTROL_DEFAULT),
            ):
                write(reg, value)
            self.camera.set_frame_size(1900, 1900)
        super().tearDown()

    def _wire_frame_report(self, label, w, h, frame):
        layout = frame_layout(w, h)
        raw = np.frombuffer(frame, dtype=np.uint8)
        logger.info(
            '[FX2 bench] P0 %s window %d: wire frame %d bytes, layout expects %d (residue mod 1024 %d)',
            label,
            w,
            raw.size,
            layout.frame_bytes,
            raw.size % 1024,
        )
        if raw.size != layout.frame_bytes:
            return
        skip = raw[: layout.skip]
        rows = raw[layout.skip : layout.needed].reshape(h, layout.stride)
        padding = raw[layout.needed :]
        image = rows[:, layout.column : layout.column + w]
        logger.info(
            '[FX2 bench] P0 %s window %d: skip row values %s; last byte of each row %s; padding row values %s',
            label,
            w,
            Counter(skip.tolist()).most_common(4),
            Counter(rows[:, layout.stride - 1].tolist()).most_common(4),
            Counter(padding.tolist()).most_common(4),
        )
        logger.info(
            '[FX2 bench] P0 %s window %d: 2x2 phase values %s; row 0 first 24 %s; row 0 last 8 %s; '
            'row 1 first 8 %s; skip row first 16 %s; padding first 16 %s',
            label,
            w,
            _phase_values(image),
            image[0, :24].tolist(),
            image[0, -8:].tolist(),
            image[1, :8].tolist(),
            skip[:16].tolist(),
            padding[:16].tolist(),
        )

    def test_measure_the_wire_layout_with_test_patterns(self):
        """P0 (a): which sensor column each wire byte is, the row's last byte, the DOUT bits."""
        write = self.camera._fx2.sensor_reg_write
        write(REG_READ_MODE2, _READ_MODE2_NO_ROW_BLC)
        write(_REG_TEST_PATTERN_GREEN, _FIELD_GREEN)
        write(_REG_TEST_PATTERN_RED, _FIELD_RED)
        write(_REG_TEST_PATTERN_BLUE, _FIELD_BLUE)
        dump = os.environ.get('FX2_BENCH_DUMP_DIR')
        for w in (1900, 1896, 1000):
            self.camera.set_frame_size(w, w)
            for name, mode in _PATTERNS:
                write(_REG_TEST_PATTERN_CONTROL, (mode << 3) | 1)
                _new_frames(self.camera, 3, timeout_s=10.0)
                frames = self.spy.wait_frames(2)
                for i, frame in enumerate(frames):
                    self._wire_frame_report(f'pattern {name} #{i}', w, w, frame)
                    if dump:
                        np.save(
                            os.path.join(dump, f'p0a_{name.replace(" ", "_")}_{w}_{i}.npy'),
                            np.frombuffer(frame, dtype=np.uint8),
                        )
        write(_REG_TEST_PATTERN_CONTROL, 0)

    def test_measure_conforming_column_sizes(self):
        """P0 (b): Column_Size w + 1 against RR's 4n - 1 (w - 1, w + 3), 60 s each."""
        ma = self._light_to_mid_grey()
        for w in (1900, 1896, 1880, 1000, 500):
            for label, written in (
                ('w + 1', w + 1),
                ('4n-1 below', w - 1),
                ('4n-1 above', w + 3),
            ):
                self.camera.set_frame_size(w, w)
                if written != column_size_for(w):
                    self.camera._fx2.sensor_reg_write(REG_COL_SIZE, written)
                time.sleep(2.0)
                self.spy.reset()
                self.camera.stream_stats.reset()
                time.sleep(_STEP_S)
                sizes = self.spy.sizes.most_common()
                total = sum(n for _size, n in sizes)
                common = sizes[0][0] if sizes else 0
                glued = sum(n for size, n in sizes if size in (2 * common, 2 * common + 4))
                s = self.camera.stream_stats.summary()
                logger.info(
                    '[FX2 bench] P0 Column_Size window %d %s (R0x04=%d, Blue %d mA, 50 ms): %d wire frames, '
                    'commonest %d bytes (residue mod 1024 %d), glued two-frame chunks %d, sizes %s; '
                    'parser %d good / %d partial / %d shifted',
                    w,
                    label,
                    written,
                    ma,
                    total,
                    common,
                    common % 1024,
                    glued,
                    sizes[:6],
                    s['good_frames'],
                    s['partial_frames'],
                    s['shifted_frames'],
                )

    def test_measure_dark_frames_at_the_black_levels(self):
        """P0 (c): needs a light-tight cover. Exact zeros, mean and phases at each black level."""
        self.led.leds_off()
        self.camera.set_frame_size(1900, 1900)
        write = self.camera._fx2.sensor_reg_write
        for db in (0, 24):
            self.camera.gain(db)
            for row_black, blc in ((0, 0x6000), (0xA8, 0x6000), (0, 0), (0xA8, 0)):
                write(REG_ROW_BLACK, row_black)
                write(_REG_BLC, blc)
                _new_frames(self.camera, 4, timeout_s=10.0)
                images = [img for _s, _a, img in _timed_frames(self.camera, 3, timeout_s=10.0)]
                stack = np.stack(images).astype(np.float32)
                logger.info(
                    '[FX2 bench] P0 dark %d dB R0x49=0x%02X R0x62=0x%04X: exact zeros %.4f, mean %.2f, '
                    'std %.2f, max %d, above the dark floor %.6f, phase means %s',
                    db,
                    row_black,
                    blc,
                    float(np.mean(stack == 0)),
                    float(stack.mean()),
                    float(stack.std()),
                    int(stack.max()),
                    float(np.mean(stack > _DARK_FLOOR)),
                    [
                        round(float(stack[:, r::2, c::2].mean()), 2)
                        for r, c in ((0, 0), (0, 1), (1, 0), (1, 1))
                    ],
                )

    def test_measure_a_window_change(self):
        """P0 (e): what a window change stores, with and without Synchronize_Changes."""
        ma = self._light_to_mid_grey()
        write = self.camera._fx2.sensor_reg_write
        for synchronized in (False, True):
            for w_from, w_to in ((1900, 1000), (1000, 1900), (1900, 1000), (1000, 1900)):
                self.camera.set_frame_size(w_from, w_from)
                _new_frames(self.camera, 3, timeout_s=10.0)
                self.camera.stream_stats.reset()
                started = datetime.now()
                if synchronized:
                    write(_REG_OUTPUT_CONTROL, _OUTPUT_CONTROL_DEFAULT | _SYNCHRONIZE_CHANGES)
                self.camera.set_frame_size(w_to, w_to)
                if synchronized:
                    write(_REG_OUTPUT_CONTROL, _OUTPUT_CONTROL_DEFAULT)
                frames = _timed_frames(self.camera, 5, timeout_s=15.0)
                s = self.camera.stream_stats.summary()
                logger.info(
                    '[FX2 bench] P0 window change %d -> %d, synchronized %s (Blue %d mA, 50 ms): '
                    'first frame after %.2f s; frames %s; %d partial / %d shifted, sizes %s / %s',
                    w_from,
                    w_to,
                    synchronized,
                    ma,
                    (frames[0][1] - started).total_seconds() if frames else -1.0,
                    [
                        (seq, img.shape[1], *(round(v, 1) for v in _integrity(img)))
                        for seq, _at, img in frames
                    ],
                    s['partial_frames'],
                    s['shifted_frames'],
                    sorted(set(s['partial_sizes'])),
                    sorted(set(s['shifted_sizes'])),
                )


# ---------------------------------------------------------------------------
# Phase B's bench (FX2 plan section 21.7): the driver after P1-P3 on the
# LS620. Measurement only, recorded and not asserted. Run with --driver-log:
#
#     pytest tests/test_fx2_hardware.py --run-fx2-hardware --driver-log -k PhaseBBench
#
# Rows 1, 3, 4 and 6 need light on the sensor; row 5 needs a light-tight cover.
# ---------------------------------------------------------------------------

_BENCH_MS = 50.0
_BENCH_WIDTHS = (1900, 1000, 500)


def _model_period_s(w, ms):
    """The data sheet's frame time for a ``w`` x ``w`` window at a ``ms`` request."""
    return frame_time_s(
        column_size_for(w), w + 1, shutter_width_for(ms / 1000.0, column_size_for(w))
    )


@pytest.mark.fx2_hardware
class TestFX2PhaseBBench(_FX2BenchCase):
    def _stream_line(self, label, s):
        logger.info(
            '[FX2 bench] 21.7 %s: %.1f s, %d good / %d partial / %d shifted, %d USB errors, '
            '%.2f fps, %.2f MB/s, shifted sizes %s, partial sizes %s',
            label,
            s['elapsed_s'],
            s['good_frames'],
            s['partial_frames'],
            s['shifted_frames'],
            s['usb_errors'],
            s['fps_average'],
            s['throughput_MBps'],
            sorted(set(s['shifted_sizes'])),
            sorted(set(s['partial_sizes'])),
        )

    def test_bench_row1_one_exposure_at_three_widths(self):
        """Row 1: one request, the same black-subtracted lit mean and the request recorded."""
        ma = self._light_to_mid_grey()
        self.camera.exposure_t(_BENCH_MS)
        lit = {}
        for w in _BENCH_WIDTHS:
            self.camera.set_frame_size(w, w)
            self.led.led_on(_BLUE, ma)
            lit_mean, peak = _settled_mean(self.camera, skip=4, count=5)
            self.led.led_off(_BLUE)
            dark_mean, _ = _settled_mean(self.camera, skip=4, count=5)
            lit[w] = lit_mean - dark_mean
            logger.info(
                '[FX2 bench] 21.7 row 1, %d wide (Blue %d mA, %.0f ms asked): exposure in effect '
                '%.4f ms, lit %.2f (max %d), dark %.2f, lit - dark %.2f, ratio to 1900 %.3f',
                w,
                ma,
                _BENCH_MS,
                self.camera.get_exposure_t(),
                lit_mean,
                peak,
                dark_mean,
                lit[w],
                lit[w] / lit[1900],
            )

    def test_bench_row2_the_frame_period_matches_the_model(self):
        """Row 2: after this connect, the stream is clean and its period is the model's."""
        self.camera.exposure_t(_BENCH_MS)
        for w in _BENCH_WIDTHS:
            self.camera.set_frame_size(w, w)
            _new_frames(self.camera, 3, timeout_s=10.0)
            self.camera.stream_stats.reset()
            period, intervals = _frame_period_s(self.camera, 10, timeout_s=30.0)
            s = self.camera.stream_stats.summary()
            model = _model_period_s(w, _BENCH_MS)
            logger.info(
                '[FX2 bench] 21.7 row 2, %d wide at %.0f ms: period %s s, model %.5f s, '
                'off by %s; intervals %s',
                w,
                _BENCH_MS,
                period,
                model,
                f'{(period / model - 1) * 100:+.2f}%' if period else 'n/a',
                intervals,
            )
            self._stream_line(f'row 2, {w} wide', s)

    def test_bench_row3_sixty_seconds_at_five_windows(self):
        """Row 3: 60 s per window; partial and shifted frames, 1896's and 1880's glue."""
        self.led.led_on(_BLUE, 200)
        self.camera.exposure_t(_BENCH_MS)
        for w in (1900, 1000, 500, 1896, 1880):
            self.camera.set_frame_size(w, w)
            time.sleep(2.0)
            self.camera.stream_stats.reset()
            time.sleep(60.0)
            self._stream_line(f'row 3, {w} wide (Blue 200 mA)', self.camera.stream_stats.summary())

    def test_bench_row4_the_exposure_ladder_at_500_wide(self):
        """Row 4: the stream and the lit mean through exposures that stretch the frame."""
        self.camera.set_frame_size(500, 500)
        self.camera.gain(0)
        self.led.led_on(_BLUE, 5)
        for ms in (50, 178, 400, 800, 2000, 3900, 178, 50):
            applied_us = self.camera.exposure_t(ms)
            _new_frames(self.camera, 3, timeout_s=3 * ms / 1000 + 10.0)
            self.camera.stream_stats.reset()
            frames = _new_frames(self.camera, 4, timeout_s=4 * ms / 1000 + 10.0)
            s = self.camera.stream_stats.summary()
            means = [round(m, 1) for _s, m, _x in frames]
            logger.info(
                '[FX2 bench] 21.7 row 4, 500 wide (Blue 5 mA, 0 dB): asked %d ms, in effect '
                '%.3f ms, model period %.4f s -> means %s, max %s, mean per ms %.4f',
                ms,
                applied_us / 1000,
                _model_period_s(500, ms),
                means,
                [x for _s, _m, x in frames],
                float(np.mean(means)) / (applied_us / 1000) if means else float('nan'),
            )
            self._stream_line(f'row 4, {ms} ms', s)

    def test_bench_row5_a_light_tight_dark_frame(self):
        """Row 5: needs a light-tight cover. The dark frame in the driver's own state."""
        self.led.leds_off()
        self.camera.set_frame_size(1900, 1900)
        self.camera.exposure_t(_BENCH_MS)
        for db in (0, 24):
            self.camera.gain(db)
            _new_frames(self.camera, 4, timeout_s=10.0)
            images = [img for _s, _a, img in _timed_frames(self.camera, 5, timeout_s=10.0)]
            stack = np.stack(images).astype(np.float32)
            per_pixel = stack.mean(axis=0)
            hot = np.argwhere(per_pixel > _DARK_FLOOR)
            logger.info(
                '[FX2 bench] 21.7 row 5, dark, %d dB, %.0f ms: exact zeros %.4f, mean %.3f, '
                'std %.3f, max %d, pixels whose 5-frame mean is above the dark floor %d, '
                'the first 20 (row, col, mean) %s, phase means %s',
                db,
                _BENCH_MS,
                float(np.mean(stack == 0)),
                float(stack.mean()),
                float(stack.std()),
                int(stack.max()),
                len(hot),
                [(int(r), int(c), round(float(per_pixel[r, c]), 1)) for r, c in hot[:20]],
                [
                    round(float(stack[:, r::2, c::2].mean()), 3)
                    for r, c in ((0, 0), (0, 1), (1, 0), (1, 1))
                ],
            )


@pytest.mark.fx2_hardware
class TestFX2HostFramingBench(_FX2BenchCase):
    """Frames end at the device's own end-of-frame packet; the delimiter is only counted."""

    def _stream_line(self, label, s):
        logger.info(
            '[FX2 bench] H1 %s: %.1f s, %d good / %d partial / %d shifted, %d USB errors, '
            'delimiters %d missing / %d wrong, %.2f fps, %.2f MB/s, shifted sizes %s, '
            'partial sizes %s',
            label,
            s['elapsed_s'],
            s['good_frames'],
            s['partial_frames'],
            s['shifted_frames'],
            s['usb_errors'],
            s['delimiters_missing'],
            s['delimiters_wrong'],
            s['fps_average'],
            s['throughput_MBps'],
            sorted(set(s['shifted_sizes'])),
            sorted(set(s['partial_sizes'])),
        )

    def test_h1_row1_sixty_seconds_at_eight_windows(self):
        """Row 1: no frame lost at any window; the delimiter faults counted, not glued."""
        self.led.led_on(_BLUE, 200)
        self.camera.exposure_t(_BENCH_MS)
        for w in (1900, 1896, 1880, 1860, 1852, 1844, 1000, 500):
            self.camera.set_frame_size(w, w)
            time.sleep(2.0)
            self.camera.stream_stats.reset()
            time.sleep(_STEP_S)
            s = self.camera.stream_stats.summary()
            self._stream_line(f'row 1, {w} wide (Blue 200 mA, 50 ms)', s)
            logger.info(
                '[FX2 bench] H1 row 1, %d wide: %.3f fps stored against the model %.3f (%+.2f%%)',
                w,
                s['good_frames'] / s['elapsed_s'],
                1 / _model_period_s(w, _BENCH_MS),
                (s['good_frames'] / s['elapsed_s'] * _model_period_s(w, _BENCH_MS) - 1) * 100,
            )

    def test_h1_row2_five_window_changes(self):
        """Row 2: the old window's frames in flight discarded, every stored frame the new shape."""
        self.led.led_on(_BLUE, 200)
        self.camera.exposure_t(_BENCH_MS)
        for w in (1896, 1880, 1000, 500, 1900):
            _ok, _img, _at, _bits, before = self.camera.grab_latest()
            self.camera.stream_stats.reset()
            self.camera.set_frame_size(w, w)
            # Only frames stored after the change: the one stored before it is the old shape.
            frames = [
                f
                for f in _timed_frames(self.camera, 200, timeout_s=3.0)
                if before is None or f[0] > before
            ]
            s = self.camera.stream_stats.summary()
            shapes = sorted({img.shape for _seq, _at, img in frames})
            logger.info(
                '[FX2 bench] H1 row 2, change to %d wide: %d frames stored in 3 s, shapes %s, '
                '%d partial / %d shifted discarded (sizes %s / %s)',
                w,
                len(frames),
                shapes,
                s['partial_frames'],
                s['shifted_frames'],
                sorted(set(s['partial_sizes'])),
                sorted(set(s['shifted_sizes'])),
            )


def _wire_rows(frame, stride, h):
    """The ``h`` stored rows of one wire frame whose rows are ``stride`` bytes, whole rows."""
    raw = np.frombuffer(frame, dtype=np.uint8)
    skip = stride + 1
    return raw[skip : skip + h * stride].reshape(h, stride)


def _hot_pixel_shift(a, b, rows=2, columns=8):
    """The (row, column) offset at which the most hot pixels of ``a`` are hot in ``b``.

    A hot pixel belongs to one sensor pixel, so two readouts of the same
    sensor columns share them at no column offset.
    """
    h, w = a.shape
    counts = {}
    for dr in range(-rows, rows + 1):
        for dc in range(-columns, columns + 1):
            a_part = a[max(0, -dr) : h - max(0, dr), max(0, -dc) : w - max(0, dc)]
            b_part = b[max(0, dr) : h - max(0, -dr), max(0, dc) : w - max(0, -dc)]
            counts[(dr, dc)] = int(np.count_nonzero(a_part & b_part))
    return max(counts, key=counts.get), counts


@pytest.mark.fx2_hardware
class TestFX2ColumnSizeBench(_FX2BenchCase):
    """Column_Size w + 3 keeps the sensor columns w + 1 gave, by storing from a row's third pixel.

    Needs the optical path capped light-tight: the sensor's hot pixels are
    the marks, each fixed to one sensor column, so no specimen is needed.
    """

    def setUp(self):
        super().setUp()
        self.spy = _WireSpy(self.camera._fx2.stream)

    def tearDown(self):
        self.spy.remove()
        self.camera.set_frame_size(1900, 1900)
        super().tearDown()

    def _mean_wire_rows(self, label, stride, w, h, count=5):
        frames = [f for f in self.spy.wait_frames(count) if len(f) == stride * (h + 2) + 1]
        logger.info(
            '[FX2 bench] H2 row 4, %s: %d wire frames of %d bytes (Column_Size %d)',
            label,
            len(frames),
            stride * (h + 2) + 1,
            stride,
        )
        self.assertTrue(frames, f'no whole wire frame at Column_Size {stride}')
        return np.mean([_wire_rows(f, stride, h) for f in frames], axis=0)

    def test_h2_row4_the_stored_columns(self):
        """Row 4: the column shift from w + 1 to w + 3, with the offset (0) and without it (+2)."""
        self.led.leds_off()
        self.camera.exposure_t(_BENCH_MS)
        self.camera.gain(24)
        w = h = 1900
        self.camera.set_frame_size(w, h)
        self.camera._fx2.sensor_reg_write(REG_COL_SIZE, w + 1)
        time.sleep(2.0)
        before = self._mean_wire_rows('w + 1', w + 1, w, h)[:, :w]

        self.camera.set_frame_size(w, h)
        time.sleep(2.0)
        layout = frame_layout(w, h)
        rows = self._mean_wire_rows('w + 3', layout.stride, w, h)
        images = (
            ('w + 3 with the offset', rows[:, layout.column : layout.column + w]),
            ('w + 3 without it', rows[:, :w]),
        )
        hot_before = before > _DARK_FLOOR
        logger.info(
            '[FX2 bench] H2 row 4, w + 1: %d hot pixels (5-frame mean above %.2f), mean %.3f',
            int(hot_before.sum()),
            _DARK_FLOOR,
            float(before.mean()),
        )
        for label, image in images:
            hot = image > _DARK_FLOOR
            best, counts = _hot_pixel_shift(hot_before, hot)
            ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            logger.info(
                '[FX2 bench] H2 row 4, w + 1 -> %s: %d hot pixels, mean %.3f; best offset '
                '(rows, columns) %s with %d shared; next %s',
                label,
                int(hot.sum()),
                float(image.mean()),
                best,
                counts[best],
                ranked[1:4],
            )
            # A pixel near the floor is hot in some frames' mean and not
            # others; the share is read again for pixels well above it.
            for floor in (15.0, 30.0):
                strong_before = before > floor
                strong = image > floor
                best, counts = _hot_pixel_shift(strong_before, strong)
                logger.info(
                    '[FX2 bench] H2 row 4, w + 1 -> %s, above %.0f: %d / %d hot pixels; best '
                    'offset %s with %d shared',
                    label,
                    floor,
                    int(strong_before.sum()),
                    int(strong.sum()),
                    best,
                    counts[best],
                )
