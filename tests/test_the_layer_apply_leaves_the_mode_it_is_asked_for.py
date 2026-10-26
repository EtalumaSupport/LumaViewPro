# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``CameraAPI.apply`` leaves the camera in the auto-gain mode it is asked for.

Its ``auto_gain=False`` meant two things: with settings, turn auto-gain off;
with none, leave the mode alone, which the run's manual step relied on to
skip a mode write per step. A caller asking for off with no settings got a
running loop. The mode is now three-way: True arms, False turns it off,
None (the default) leaves it and writes none.
"""

import pytest

from modules.lumascope_api.camera import CameraAutoMode
from modules.exceptions import ArgumentRefusedError

AG_SETTINGS = {'target_brightness': 0.3, 'min_gain_db': 0.0, 'max_gain_db': 20.0}


@pytest.fixture
def armed(sim_scope):
    """The simulated scope's camera with continuous auto-gain running."""
    sim_scope.camera.set_auto_gain(True, AG_SETTINGS)
    assert sim_scope.camera.auto_gain == CameraAutoMode.CONTINUOUS
    return sim_scope


@pytest.fixture
def mode_writes(sim_scope, monkeypatch):
    """Every auto-gain mode the camera body is asked for, passed through to it."""
    driver = sim_scope._camera_driver
    asked = []
    write = driver.auto_gain

    def counted(state, *args, **kwargs):
        asked.append(state)
        return write(state, *args, **kwargs)

    monkeypatch.setattr(driver, 'auto_gain', counted)
    return asked


def test_off_with_no_settings_turns_a_running_loop_off(armed):
    armed.camera.apply(5.0, 20.0, auto_gain=False, channel='BF')

    assert armed.camera.auto_gain == CameraAutoMode.OFF


def test_no_mode_given_leaves_a_running_loop_and_writes_no_mode(armed, mode_writes):
    """The run's manual step: its gain and exposure, and no mode write per step."""
    armed.camera.apply(5.0, 20.0, channel='BF')

    assert mode_writes == []
    assert armed.camera.auto_gain == CameraAutoMode.CONTINUOUS


def test_arming_with_no_settings_is_refused_before_anything_is_written(sim_scope, mode_writes):
    with pytest.raises(ArgumentRefusedError) as refused:
        sim_scope.camera.apply(5.0, 20.0, auto_gain=True, channel='BF')

    assert refused.value.reason == 'missing_key'
    assert mode_writes == []
