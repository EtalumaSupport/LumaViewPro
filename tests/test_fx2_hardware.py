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

import pytest

# When --run-fx2-hardware is set, conftest skips installing the usb/usb1
# mocks so the real libusb stack loads here. When the flag is NOT set,
# this import succeeds against the conftest mock and the marker below
# skips the tests at collection time.
from drivers.fx2driver import FX2Camera


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
        result, _timestamp = self.camera.grab()
        self.assertTrue(result)
        self.assertIsNotNone(self.camera.array)
        self.assertEqual(self.camera.array.shape[0], 600)
        self.assertEqual(self.camera.array.shape[1], 800)
