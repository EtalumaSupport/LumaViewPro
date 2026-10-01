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
from drivers.fx2driver import FX2Camera, FX2LEDController
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


@pytest.mark.fx2_hardware
class TestFX2Characterization(unittest.TestCase):
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
