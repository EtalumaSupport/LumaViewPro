# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""IDS Peak hardware tests — opt-in via --run-ids-hardware.

These tests require:
  1. Real ids_peak SDK installed (not the conftest MagicMock)
  2. A connected IDS camera

Skipped by default. Run with:
    pytest tests/test_ids_hardware.py --run-ids-hardware

The `ids_hardware` marker is gated by conftest.pytest_collection_modifyitems —
test bodies do NOT need their own skip dance.
"""
import time
import unittest

import pytest

# When --run-ids-hardware is set, conftest skips installing the ids_peak
# mock so the real SDK can load here. When the flag is NOT set, this
# import succeeds against the conftest mock and the marker below skips
# the tests at collection time.
from drivers.idscamera import IDSCamera


@pytest.mark.ids_hardware
class TestIDS(unittest.TestCase):
    def setUp(self):
        self.camera = IDSCamera()

    def tearDown(self):
        self.camera.disconnect()
        time.sleep(1)

    def test_connect_disconnect(self):
        self.assertTrue(self.camera.disconnect())
        self.assertTrue(self.camera.connect())

    def test_grab(self):
        self.assertTrue(self.camera.is_grabbing())
        self.camera.stop_grabbing()
        self.assertFalse(self.camera.is_grabbing())

    def test_frame_size(self):
        # Valid
        self.camera.set_frame_size(1920, 1528)
        self.assertDictEqual(self.camera.get_frame_size(), {'width': 1920, 'height': 1528})
        # Out of bounds
        self.camera.set_frame_size(1919, 1529)
        self.assertDictEqual(self.camera.get_frame_size(), {'width': 1872, 'height': 1528})
        # Incorrect increment
        self.camera.set_frame_size(1480, 906)
        self.assertDictEqual(self.camera.get_frame_size(), {'width': 1440, 'height': 904})

    def test_pixel_format(self):
        formats = self.camera.get_supported_pixel_formats()
        self.camera.set_pixel_format(formats[0])
        self.assertTrue(self.camera.get_pixel_format() in formats)

    def test_exposure_t(self):
        self.camera.exposure_t(15)
        self.assertAlmostEqual(self.camera.get_exposure_t(), 15.0, delta=0.01)

    def test_binning_size(self):
        self.assertTrue(self.camera.set_binning_size(1))
        self.assertEqual(self.camera.get_binning_size(), 1)
        self.assertTrue(self.camera.set_binning_size(2))
        self.assertEqual(self.camera.get_binning_size(), 2)
        self.assertFalse(self.camera.set_binning_size(3))
        self.assertEqual(self.camera.get_binning_size(), 2)
        self.assertFalse(self.camera.set_binning_size(0))
        self.assertEqual(self.camera.get_binning_size(), 2)

    def test_grab_frame(self):
        time.sleep(1)  # Allow time for the camera to start grabbing
        result, timestamp = self.camera.grab()
        self.assertTrue(result)
        self.assertTrue(len(self.camera.array) == 1528)
        self.assertIsNotNone(timestamp)

    def test_gain(self):
        self.camera.gain(10)
        self.assertAlmostEqual(self.camera.get_gain(), 10.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()
