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

import time
import unittest
from itertools import pairwise

import numpy as np
import pytest

# When --run-fx2-hardware is set, conftest skips installing the usb/usb1
# mocks so the real libusb stack loads here. When the flag is NOT set,
# this import succeeds against the conftest mock and the marker below
# skips the tests at collection time.
from drivers.fx2driver import (
    _ROW_TIME_MS,
    _SHUTTER_OVERHEAD_MS,
    REG_PLL_CFG1,
    REG_PLL_CTRL,
    FX2Camera,
    FX2LEDController,
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

_REG_PLL_CFG2 = 0x12  # P1 divider; the driver writes it as a literal
_REG_HBLANK = 0x05
_P1_TODAY = 13
_LONG_ROWS = 20000  # longer than any window's readout: the frame period is the shutter's
_LONG_MS = _LONG_ROWS * _ROW_TIME_MS - _SHUTTER_OVERHEAD_MS  # the request that lands on 20000 rows
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
    fx2 = camera._fx2
    fx2.sensor_reg_write(REG_PLL_CTRL, 0x0051)
    time.sleep(0.01)
    fx2.sensor_reg_write(REG_PLL_CFG1, 0x1B01)
    time.sleep(0.01)
    fx2.sensor_reg_write(_REG_PLL_CFG2, p1)
    time.sleep(0.01)
    fx2.sensor_reg_write(REG_PLL_CTRL, 0x0053)
    time.sleep(0.2)
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
