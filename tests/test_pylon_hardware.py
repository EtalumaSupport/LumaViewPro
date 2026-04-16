# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Pylon hardware tests — opt-in via --run-pylon-hardware.

These tests require:
  1. Real pypylon SDK installed (not the conftest MagicMock)
  2. A connected Basler camera

Skipped by default. Run with:
    pytest tests/test_pylon_hardware.py --run-pylon-hardware

The `pylon_hardware` marker is gated by conftest.pytest_collection_modifyitems —
test bodies do NOT need their own skip dance.

Mirrors the shape of test_ids_hardware.py so the abstraction is symmetric
across camera vendors. Add coverage as Pylon-specific behaviors come up.
"""
import time
import unittest

import pytest

# When --run-pylon-hardware is set, conftest skips installing the pypylon
# mock so the real SDK loads here. When the flag is NOT set, this import
# succeeds against the conftest mock and the marker below skips the
# tests at collection time.
from drivers.pyloncamera import PylonCamera


@pytest.mark.pylon_hardware
class TestPylon(unittest.TestCase):
    def setUp(self):
        self.camera = PylonCamera()

    def tearDown(self):
        self.camera.disconnect()
        time.sleep(0.5)

    def test_connect_disconnect(self):
        self.assertTrue(self.camera.disconnect())
        self.assertTrue(self.camera.connect())

    def test_grab(self):
        self.assertTrue(self.camera.is_grabbing())
        self.camera.stop_grabbing()
        self.assertFalse(self.camera.is_grabbing())
        self.camera.start_grabbing()
        self.assertTrue(self.camera.is_grabbing())

    def test_pixel_format(self):
        formats = self.camera.get_supported_pixel_formats()
        self.assertTrue(len(formats) > 0)
        self.camera.set_pixel_format(formats[0])
        self.assertEqual(self.camera.get_pixel_format(), formats[0])

    def test_exposure_t(self):
        # Use a short exposure that's well within range for any model
        self.camera.exposure_t(15)
        self.assertAlmostEqual(self.camera.get_exposure_t(), 15.0, delta=0.5)

    def test_binning_size(self):
        # Most Basler models support 1x and 2x; reject 0 and very large.
        self.assertTrue(self.camera.set_binning_size(1))
        self.assertEqual(self.camera.get_binning_size(), 1)
        self.assertFalse(self.camera.set_binning_size(0))

    def test_grab_frame(self):
        time.sleep(0.5)  # Allow grabbing to settle
        result, timestamp = self.camera.grab()
        self.assertTrue(result)
        self.assertIsNotNone(timestamp)
        self.assertIsNotNone(self.camera.array)

    def test_gain(self):
        self.camera.gain(0)  # 0 dB is always in range
        self.assertAlmostEqual(self.camera.get_gain(), 0.0, delta=0.5)


if __name__ == '__main__':
    unittest.main()
