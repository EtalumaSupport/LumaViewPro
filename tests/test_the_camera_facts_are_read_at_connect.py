# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scope's capabilities hold what the camera is, read once at connect.

`scope.capabilities` is the one door for the camera's static facts: its
model, its serial number, its firmware version, its timestamp clock and
the largest frame the scope delivers. An unknown fact is None, never a
plausible value: an unrecognised camera is not named after the
placeholder profile it was given, and an unknown frame maximum is not a
size.
"""

from __future__ import annotations

from drivers.simulated_camera import SimulatedCamera
from modules.scope_capabilities import FrameSize
from tests.scope_fakes import capabilities_of


def _camera_reporting(width, height):
    camera = SimulatedCamera()
    # The limits are read from a connected camera.
    camera.active = True
    camera.get_max_frame_size = lambda: {'width': width, 'height': height}
    return camera


def test_a_camera_reporting_more_than_its_sensor_is_bounded_by_the_sensor():
    # The LS850T's daA3840 reports 3860 x 2178; its sensor is 3840 x 2160.
    camera = _camera_reporting(3860, 2178)
    assert camera.profile.native_resolution == {'width': 3840, 'height': 2160}
    assert capabilities_of(camera).camera_limits.frame_max == FrameSize(3840, 2160)


def test_a_camera_reporting_less_than_its_sensor_is_bounded_by_what_it_reports():
    camera = _camera_reporting(1920, 1080)
    assert capabilities_of(camera).camera_limits.frame_max == FrameSize(1920, 1080)


def test_a_camera_with_no_documented_sensor_is_bounded_by_what_it_reports():
    camera = _camera_reporting(3860, 2178)
    camera.profile.native_resolution = {}
    assert capabilities_of(camera).camera_limits.frame_max == FrameSize(3860, 2178)


def test_a_camera_whose_maximum_is_unknown_has_no_maximum():
    camera = _camera_reporting(0, 0)
    camera.get_max_frame_size = lambda: {}
    camera.profile.native_resolution = {}
    assert capabilities_of(camera).camera_limits.frame_max is None


def test_the_camera_identity_is_what_the_camera_reported():
    camera = SimulatedCamera()
    caps = capabilities_of(camera)
    assert caps.camera_model == SimulatedCamera.MODEL_NAME
    assert caps.camera_serial_number == SimulatedCamera.SERIAL_NUMBER
    assert caps.camera_firmware_version == SimulatedCamera.FIRMWARE_VERSION
    assert caps.camera_timestamp_tick_hz == SimulatedCamera.TIMESTAMP_TICK_HZ


def test_a_camera_that_did_not_say_what_it_is_is_unnamed():
    camera = SimulatedCamera()
    camera.model_name = None
    camera._device_serial = None
    camera._device_firmware = None
    camera.timestamp_tick_frequency_hz = None
    caps = capabilities_of(camera)
    assert caps.camera_model is None
    assert caps.camera_serial_number is None
    assert caps.camera_firmware_version is None
    assert caps.camera_timestamp_tick_hz is None


def test_no_camera_has_no_identity():
    caps = capabilities_of(None)
    assert caps.camera_model is None
    assert caps.camera_serial_number is None
    assert caps.camera_firmware_version is None
    assert caps.camera_timestamp_tick_hz is None
