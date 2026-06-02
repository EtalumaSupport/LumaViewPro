"""Regression for #539 residual: motion status polls must not emit a full
traceback per poll when the motor is disconnected.

On a USB yank mid-move the motion-monitor thread keeps polling
get_target_status / get_home_status. The driver correctly raises a typed
HardwareError (Rule 29); the API correctly catches it for a state query and
returns the not-at-target sentinel (Rule 8). But it logged logger.exception
(full ERROR traceback) for that expected, handled disconnect -- the stack
traces Eric saw. Now the API short-circuits on `not motor_connected`
(mirroring the home() guard) so the poll never provokes the exception, and a
typed HardwareError at the unplug instant logs a warning WITHOUT a traceback.
A genuinely unexpected error still logs with a traceback.
"""

import types
from unittest.mock import MagicMock

import pytest

from drivers.exceptions import HardwareError
from modules.lumascope_api import motion as motion_mod
from modules.lumascope_api.motion import MotionAPI


def _fake(motor_connected, *, driver_exc=None, has_turret=False):
    driver = MagicMock()
    driver.has_turret.return_value = has_turret
    if driver_exc is not None:
        driver.target_status.side_effect = driver_exc
        driver.home_status.side_effect = driver_exc
    else:
        driver.target_status.return_value = True
        driver.home_status.return_value = True
    return types.SimpleNamespace(
        _scope=types.SimpleNamespace(motor_connected=motor_connected),
        _driver=driver,
    )


@pytest.fixture
def log(monkeypatch):
    fake_log = MagicMock()
    monkeypatch.setattr(motion_mod, 'logger', fake_log)
    return fake_log


@pytest.mark.parametrize('method', ['get_target_status', 'get_home_status'])
def test_disconnected_returns_sentinel_without_touching_driver(method, log):
    fake = _fake(motor_connected=False)
    assert getattr(MotionAPI, method)(fake, 'Z') is False
    fake._driver.target_status.assert_not_called()
    fake._driver.home_status.assert_not_called()
    log.exception.assert_not_called()
    log.warning.assert_not_called()


@pytest.mark.parametrize('method', ['get_target_status', 'get_home_status'])
def test_hardware_error_warns_without_traceback(method, log):
    fake = _fake(motor_connected=True, driver_exc=HardwareError('no response from motor board'))
    assert getattr(MotionAPI, method)(fake, 'Z') is False
    log.warning.assert_called_once()
    log.exception.assert_not_called()  # no full traceback for an expected disconnect


@pytest.mark.parametrize('method', ['get_target_status', 'get_home_status'])
def test_unexpected_error_still_logs_traceback(method, log):
    fake = _fake(motor_connected=True, driver_exc=ValueError('genuinely unexpected'))
    assert getattr(MotionAPI, method)(fake, 'Z') is False
    log.exception.assert_called_once()  # real bug -> keep the traceback
