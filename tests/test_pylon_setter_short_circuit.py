# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Pylon setter short-circuit regression tests.

Camera-config-thrash fix (path b): each setter reads the current value
before writing and skips the SDK SetValue + grab-loop bounce when the
value already matches. Bench (SN12062 / SN12073 camera.log) showed
update_camera_config() bouncing the grab loop 3 of 4 connect passes and
a redundant Width/Height re-apply at +21s; these tests pin the
short-circuit behavior so future setter edits don't regress.

Pure-mock tests -- run on every pytest invocation, no hardware flag.
"""

import threading
import unittest
from unittest.mock import MagicMock

from drivers.pyloncamera import PylonCamera


def _mock_camera():
    """Build a PylonCamera with self.active stubbed; bypass __init__.

    Camera.__init__ creates _state_lock + _active that the active
    property reads. We set the private fields directly to avoid the
    connect() chain that __init__ would trigger.
    """
    cam = PylonCamera.__new__(PylonCamera)
    cam._state_lock = threading.Lock()
    cam._active = MagicMock()
    # update_camera_config is the contextmanager that bounces the grab
    # loop. Short-circuit paths must NOT enter it; replace with a mock
    # that records calls so tests can assert it stayed dormant.
    cam.update_camera_config = MagicMock()
    return cam


class TestPixelFormatShortCircuit(unittest.TestCase):
    def test_short_circuits_when_already_set(self):
        cam = _mock_camera()
        cam.active.PixelFormat.GetValue.return_value = 'Mono8'
        cam.get_supported_pixel_formats = MagicMock(return_value=('Mono8', 'Mono12'))

        result = cam.set_pixel_format('Mono8')

        self.assertTrue(result)
        cam.active.PixelFormat.SetValue.assert_not_called()
        cam.update_camera_config.assert_not_called()

    def test_writes_when_changing_format(self):
        cam = _mock_camera()
        cam.active.PixelFormat.GetValue.return_value = 'Mono8'
        cam.get_supported_pixel_formats = MagicMock(return_value=('Mono8', 'Mono12'))

        result = cam.set_pixel_format('Mono12')

        self.assertTrue(result)
        cam.active.PixelFormat.SetValue.assert_called_once_with('Mono12')
        cam.update_camera_config.assert_called_once()


class TestGainShortCircuit(unittest.TestCase):
    def test_short_circuits_when_already_set(self):
        cam = _mock_camera()
        cam.active.Gain.GetValue.return_value = 3.6

        cam.gain(3.6)

        cam.active.Gain.SetValue.assert_not_called()

    def test_short_circuits_within_tolerance(self):
        cam = _mock_camera()
        cam.active.Gain.GetValue.return_value = 3.6005  # within 1e-3 dB

        cam.gain(3.6)

        cam.active.Gain.SetValue.assert_not_called()

    def test_writes_when_outside_tolerance(self):
        cam = _mock_camera()
        cam.active.Gain.GetValue.return_value = 3.6

        cam.gain(4.0)

        cam.active.Gain.SetValue.assert_called_once_with(4.0)

    def test_preserves_gain_selector_contract(self):
        """Even on short-circuit, GainSelector='All' assertion still runs."""
        cam = _mock_camera()
        cam.active.Gain.GetValue.return_value = 3.6

        cam.gain(3.6)

        cam.active.GainSelector.SetValue.assert_called_with('All')


class TestExposureShortCircuit(unittest.TestCase):
    def _cam(self):
        cam = _mock_camera()
        cam.active.ExposureTime.Min = 10.0
        # max_exposure is a derived property reading profile.exposure_max_us
        cam.profile = MagicMock()
        cam.profile.exposure_max_us = 1_000_000.0  # 1000 ms in us
        return cam

    def test_short_circuits_when_already_set(self):
        cam = self._cam()
        cam.active.ExposureTime.GetValue.return_value = 15000.0  # 15 ms in us

        cam.exposure_t(15)

        cam.active.ExposureTime.SetValue.assert_not_called()

    def test_writes_when_changing_exposure(self):
        cam = self._cam()
        cam.active.ExposureTime.GetValue.return_value = 15000.0

        cam.exposure_t(20)  # 20000 us, not equal to current 15000

        cam.active.ExposureTime.SetValue.assert_called_once_with(20000.0)


class TestFrameSizeShortCircuit(unittest.TestCase):
    def test_short_circuits_when_geometry_matches(self):
        cam = _mock_camera()
        cam.active.Width.Max = 3536
        cam.active.Height.Max = 3536
        cam.active.Width.GetValue.return_value = 1900
        cam.active.Height.GetValue.return_value = 1900

        cam.set_frame_size(1900, 1900)

        cam.active.Width.SetValue.assert_not_called()
        cam.active.Height.SetValue.assert_not_called()
        cam.active.BslCenterX.Execute.assert_not_called()
        cam.active.BslCenterY.Execute.assert_not_called()
        cam.update_camera_config.assert_not_called()

    def test_writes_when_geometry_changes(self):
        cam = _mock_camera()
        cam.active.Width.Max = 3536
        cam.active.Height.Max = 3536
        cam.active.Width.GetValue.return_value = 1900
        cam.active.Height.GetValue.return_value = 1900

        cam.set_frame_size(512, 1900)

        cam.active.Width.SetValue.assert_called_once_with(512)
        cam.active.Height.SetValue.assert_called_once_with(1900)
        cam.update_camera_config.assert_called_once()

    def test_short_circuits_against_clamped_dims(self):
        """1901 rounds down to 1900 via the /4*4 clamp; matches current."""
        cam = _mock_camera()
        cam.active.Width.Max = 3536
        cam.active.Height.Max = 3536
        cam.active.Width.GetValue.return_value = 1900
        cam.active.Height.GetValue.return_value = 1900

        cam.set_frame_size(1901, 1901)

        cam.active.Width.SetValue.assert_not_called()
        cam.update_camera_config.assert_not_called()


class TestAutoTargetBrightnessShortCircuit(unittest.TestCase):
    def test_short_circuits_when_already_set(self):
        cam = _mock_camera()
        cam.active.AutoTargetBrightness.GetValue.return_value = 0.5

        cam.update_auto_gain_target_brightness(0.5)

        cam.active.AutoTargetBrightness.SetValue.assert_not_called()

    def test_short_circuits_within_tolerance(self):
        cam = _mock_camera()
        cam.active.AutoTargetBrightness.GetValue.return_value = 0.5005  # within 1e-3

        cam.update_auto_gain_target_brightness(0.5)

        cam.active.AutoTargetBrightness.SetValue.assert_not_called()

    def test_writes_when_outside_tolerance(self):
        cam = _mock_camera()
        cam.active.AutoTargetBrightness.GetValue.return_value = 0.5

        cam.update_auto_gain_target_brightness(0.7)

        cam.active.AutoTargetBrightness.SetValue.assert_called_once_with(0.7)


class TestAutoGainMinMaxShortCircuit(unittest.TestCase):
    def test_short_circuits_when_both_at_target(self):
        cam = _mock_camera()
        cam.active.AutoGainLowerLimit.GetValue.return_value = 0.0
        cam.active.AutoGainUpperLimit.GetValue.return_value = 24.0

        cam.update_auto_gain_min_max(0.0, 24.0)

        cam.active.AutoGainLowerLimit.SetValue.assert_not_called()
        cam.active.AutoGainUpperLimit.SetValue.assert_not_called()

    def test_writes_when_min_changes(self):
        cam = _mock_camera()
        cam.active.AutoGainLowerLimit.GetValue.return_value = 0.0
        cam.active.AutoGainUpperLimit.GetValue.return_value = 24.0

        cam.update_auto_gain_min_max(1.0, 24.0)

        cam.active.AutoGainLowerLimit.SetValue.assert_called_once_with(1.0)
        cam.active.AutoGainUpperLimit.SetValue.assert_called_once_with(24.0)

    def test_writes_when_max_changes(self):
        cam = _mock_camera()
        cam.active.AutoGainLowerLimit.GetValue.return_value = 0.0
        cam.active.AutoGainUpperLimit.GetValue.return_value = 24.0

        cam.update_auto_gain_min_max(0.0, 30.0)

        cam.active.AutoGainLowerLimit.SetValue.assert_called_once_with(0.0)
        cam.active.AutoGainUpperLimit.SetValue.assert_called_once_with(30.0)


if __name__ == '__main__':
    unittest.main()
