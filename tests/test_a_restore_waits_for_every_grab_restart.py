# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera restore that restarts the grab is not declared failed while it runs.

``restore_camera_state`` waited three value writes' worth for a body that may
restart the grab three times. On a Pylon body whose grab restart took 11 s,
a restore of frame size and pixel format took 22 s, its caller got
``TimeoutError`` at 15 s, and the restore ran on behind it.

Here the simulated camera's grab stop is made slower than a value write's
limit: the restore of two geometry differences must still return, with the
camera as the snapshot holds it.
"""

from __future__ import annotations

import time

from modules.lumascope_api.imaging import ImagingAPI
from tests.test_a_command_for_absent_motion_hardware_is_refused import (
    make_session,  # noqa: F401 -- the fixture this file's tests take
)

_VALUE_LIMIT_S = 0.1
_STOP_S = 0.3


def test_a_restore_of_two_geometry_writes_outlasting_a_value_write_returns(
    make_session, monkeypatch
):
    session = make_session('LS850')
    imaging = session.scope.imaging
    snapshot = imaging.save_camera_state('found')
    imaging.set_pixel_format('Mono12' if snapshot['pixel_format'] != 'Mono12' else 'Mono8')
    imaging.set_frame_size(
        snapshot['frame_size']['width'] - 200, snapshot['frame_size']['height'] - 200
    )

    driver = session.scope._camera_driver
    real_stop = driver.stop_grabbing
    stops = []

    def slow_stop():
        stops.append(time.monotonic())
        time.sleep(_STOP_S)
        real_stop()

    monkeypatch.setattr(driver, 'stop_grabbing', slow_stop)
    monkeypatch.setattr(ImagingAPI, '_CAMERA_WRITE_TIMEOUT_S', _VALUE_LIMIT_S)

    imaging.restore_camera_state(snapshot)

    assert len(stops) == 2, f'the restore restarted the grab {len(stops)} times, not twice'
    now = imaging.save_camera_state('restored')
    assert now['pixel_format'] == snapshot['pixel_format']
    assert now['frame_size'] == snapshot['frame_size']
