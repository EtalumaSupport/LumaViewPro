# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``scope.camera.longest_exposure_ms`` is the longest exposure the camera may be using now.

Every wait for a frame is sized from it: a grab's timeout, a capture's
deadline and its liveness wait, a manual recording's feed-death bound.
Exposure and gain stay open during a recording, and continuous auto-gain
picks its own exposure, so the bound is not the held exposure alone:

- with no auto-gain armed it is the held exposure;
- right after a write made the exposure shorter it is the exposure the
  write replaced, until a frame begun before the write has ended: that
  frame is still exposing under it;
- while auto-gain is armed it is the arm's exposure ceiling, or the sensor
  maximum when the arm was given none;
- with the exposure unknown it is the sensor maximum, the longest a frame
  can take;
- with no camera it is None.

Run on the simulated LS850, whose sensor maximum is 1000 ms.
"""

from __future__ import annotations

import time

import pytest

from modules.exceptions import CameraSettingRejected
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

_AUTO_GAIN = {'target_brightness': 0.5, 'min_gain_db': 0.0, 'max_gain_db': 10.0}


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield s
    finally:
        s.shutdown()


def _sensor_maximum(session):
    return session.scope.capabilities.camera_limits.exposure_max_ms


def test_it_is_the_held_exposure_with_no_auto_gain_armed(session):
    session.camera.set_exposure_ms(40.0)
    assert session.camera.longest_exposure_ms == 40.0


def test_it_is_the_replaced_exposure_until_a_frame_begun_under_it_has_ended(session):
    camera = session.camera
    camera.set_exposure_ms(1000.0)
    camera.set_exposure_ms(300.0)
    assert camera.longest_exposure_ms == 1000.0
    # A frame begun just before the write ends at most the replaced
    # exposure after it; a sleep never returns early, so this is past it.
    time.sleep(1.0)
    assert camera.longest_exposure_ms == 300.0


def test_it_is_the_arms_ceiling_while_auto_gain_is_armed(session):
    camera = session.camera
    camera.set_exposure_ms(40.0)
    camera.set_auto_gain(True, {**_AUTO_GAIN, 'max_exposure_ms': 150.0})
    assert camera.longest_exposure_ms == 150.0
    camera.set_auto_gain(False, dict(_AUTO_GAIN))
    assert camera.longest_exposure_ms == camera.exposure_ms


def test_it_is_the_sensor_maximum_while_auto_gain_is_armed_with_no_ceiling(session):
    camera = session.camera
    camera.set_exposure_ms(40.0)
    camera.set_auto_gain(True, dict(_AUTO_GAIN))
    assert camera.longest_exposure_ms == _sensor_maximum(session) == 1000.0


def test_it_is_the_sensor_maximum_with_the_exposure_unknown(session):
    camera = session.camera
    camera.set_exposure_ms(40.0)
    # Unanswered twice: the write and the API's one resend.
    session.scope.simulation.refuse_camera_write('exposure', 'no_answer', count=2)
    with pytest.raises(CameraSettingRejected):
        camera.set_exposure_ms(20.0)
    assert camera.exposure_ms is None
    assert camera.longest_exposure_ms == _sensor_maximum(session) == 1000.0


def test_it_is_none_with_no_camera(diagnostic_scope):
    assert diagnostic_scope.camera.longest_exposure_ms is None
