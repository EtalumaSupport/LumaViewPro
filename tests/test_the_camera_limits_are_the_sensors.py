# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The camera's limits are read once at connect, and the largest frame is the sensor's.

A camera reports its largest frame at the binning it holds. ``capabilities``
is built once at connect, so a camera still at 2x then -- a Basler whose
user set failed to load keeps the binning it last held -- recorded its binned
maximum under the unbinned name, and every 1x frame above half the sensor
was refused for the session. The maximum is multiplied back by the binning
the camera reports, and a failed binning read leaves the report out rather
than shrinking the frame.
"""

from __future__ import annotations

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.scope_capabilities import FrameSize
from tests.scope_fakes import give_camera_capabilities


class _Scope:
    """The one attribute ``give_camera_capabilities`` sets."""


def _capabilities(camera):
    return give_camera_capabilities(_Scope(), camera).capabilities


@pytest.fixture
def camera():
    cam = SimulatedCamera()
    cam.active = True
    cam.open_and_start()
    return cam


def test_a_camera_binned_at_connect_records_the_sensors_frame(camera):
    at_1x = _capabilities(camera)
    camera.set_binning_size(2)

    at_2x = _capabilities(camera)

    assert at_2x.camera_limits.frame_max == at_1x.camera_limits.frame_max
    assert at_2x.camera_limits.frame_max == at_1x.camera_limits.frame_max
    assert at_1x.camera_limits.frame_max == FrameSize(*SimulatedCamera._SENSOR.values())


def test_a_failed_binning_read_does_not_shrink_the_frame(camera, monkeypatch):
    at_1x = _capabilities(camera)
    camera.set_binning_size(2)
    # The drivers' failed-read answer (Camera.get_binning_size: -1).
    monkeypatch.setattr(camera, 'get_binning_size', lambda: -1)

    limits = _capabilities(camera).camera_limits

    assert limits.frame_max == at_1x.camera_limits.frame_max


def test_no_camera_connected_has_no_limits(camera):
    camera.active = None

    assert _capabilities(camera).camera_limits is None


def test_a_maximum_the_camera_reads_as_zero_is_no_maximum(camera, monkeypatch):
    # Zero is the drivers' answer for an inactive camera, never a real cap.
    monkeypatch.setattr(camera, 'get_max_exposure', lambda: 0.0)

    assert _capabilities(camera).camera_limits.exposure_max_ms is None


def test_a_whole_number_maximum_is_a_float(camera, monkeypatch):
    monkeypatch.setattr(camera, 'get_max_exposure', lambda: 500)

    maximum = _capabilities(camera).camera_limits.exposure_max_ms

    assert maximum == 500.0 and isinstance(maximum, float)
