"""Regression for #539 residual: motion status polls must not emit a full
traceback per poll when the motor is disconnected.

On a USB yank mid-move the motion-monitor thread keeps polling
get_target_status. It logged logger.exception (full ERROR traceback) for
that expected disconnect on every poll -- the stack traces Eric saw.

get_target_status no longer answers a failure as "not at target": with no
controller it is refused before the driver is asked, and a failed read
raises the driver's HardwareError to its caller without logging it. The
monitor, its one polling caller, warns once per move (its own test:
test_a_motion_status_read_never_answers_a_failure.py).
"""

import logging

import pytest

from drivers.exceptions import HardwareError
from modules.exceptions import HardwareCommandRefusedError
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session


@pytest.fixture
def scope(tmp_path):
    session = _session(tmp_path, 'LS850')
    yield session.scope
    session.shutdown()


def test_disconnected_is_refused_without_touching_driver(scope, monkeypatch):
    sent = []
    monkeypatch.setattr(scope._motion_driver, 'is_connected', lambda: False)
    monkeypatch.setattr(scope._motion_driver, 'target_status', lambda ax: sent.append(ax))

    with pytest.raises(HardwareCommandRefusedError) as exc:
        scope.motion.get_target_status('Z')

    assert exc.value.reason == 'not_connected'
    assert sent == []


def test_hardware_error_reaches_the_caller_without_a_traceback(scope, caplog):
    scope._motion_driver._fail_on.add('STATUS_RZ')

    with caplog.at_level(logging.DEBUG), pytest.raises(HardwareError):
        scope.motion.get_target_status('Z')

    assert not [r for r in caplog.records if r.exc_info], 'no full traceback for a failed read'


def test_unexpected_error_reaches_the_caller(scope, monkeypatch):
    def broken(axis):
        raise ValueError('genuinely unexpected')

    monkeypatch.setattr(scope._motion_driver, 'target_status', broken)

    with pytest.raises(ValueError, match='genuinely unexpected'):
        scope.motion.get_target_status('Z')
