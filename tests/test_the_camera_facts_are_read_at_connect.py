# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scope's capabilities hold what the camera is, read once at connect.

`scope.capabilities` is the one door for the camera's static facts: its
model, its serial number, its timestamp clock and the largest frame the
scope delivers. An unknown fact is None, never a plausible value: an
unrecognised camera is not named after the placeholder profile it was
given, and an unknown frame maximum is not a size.
"""

from __future__ import annotations

from drivers.null_ledboard import NullLEDBoard
from drivers.null_motorboard import NullMotionBoard
from drivers.simulated_camera import SimulatedCamera
from modules.layer_record import UNRESOLVED
from modules.scope_capabilities import ScopeCapabilities


def _capabilities(camera):
    return ScopeCapabilities.from_drivers(
        motion=NullMotionBoard(),
        led=NullLEDBoard(),
        camera=camera,
        layer_identity=UNRESOLVED,
        scope_models={},
    )


def _camera_reporting(width, height):
    camera = SimulatedCamera()
    camera.get_max_frame_size = lambda: {'width': width, 'height': height}
    return camera


def test_a_camera_reporting_more_than_its_sensor_is_bounded_by_the_sensor():
    # The LS850T's daA3840 reports 3860 x 2178; its sensor is 3840 x 2160.
    camera = _camera_reporting(3860, 2178)
    assert camera.profile.native_resolution == {'width': 3840, 'height': 2160}
    assert _capabilities(camera).camera_max_frame_size == (3840, 2160)


def test_a_camera_reporting_less_than_its_sensor_is_bounded_by_what_it_reports():
    camera = _camera_reporting(1920, 1080)
    assert _capabilities(camera).camera_max_frame_size == (1920, 1080)


def test_a_camera_with_no_documented_sensor_is_bounded_by_what_it_reports():
    camera = _camera_reporting(3860, 2178)
    camera.profile.native_resolution = {}
    assert _capabilities(camera).camera_max_frame_size == (3860, 2178)


def test_a_camera_whose_maximum_is_unknown_has_no_maximum():
    camera = _camera_reporting(0, 0)
    camera.get_max_frame_size = lambda: {}
    camera.profile.native_resolution = {}
    assert _capabilities(camera).camera_max_frame_size is None


def test_the_camera_identity_is_what_the_camera_reported():
    camera = SimulatedCamera()
    camera.timestamp_tick_frequency_hz = 1_000_000_000
    caps = _capabilities(camera)
    assert caps.camera_model == SimulatedCamera.MODEL_NAME
    assert caps.camera_serial_number == SimulatedCamera.SERIAL_NUMBER
    assert caps.camera_timestamp_tick_hz == 1_000_000_000


def test_a_camera_that_did_not_say_what_it_is_is_unnamed():
    camera = SimulatedCamera()
    camera.model_name = None
    camera._device_serial = None
    caps = _capabilities(camera)
    assert caps.camera_model is None
    assert caps.camera_serial_number is None
    assert caps.camera_timestamp_tick_hz is None


def test_no_camera_has_no_identity():
    caps = _capabilities(None)
    assert caps.camera_model is None
    assert caps.camera_serial_number is None
    assert caps.camera_timestamp_tick_hz is None
