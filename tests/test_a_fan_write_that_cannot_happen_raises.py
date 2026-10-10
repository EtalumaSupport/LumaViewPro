# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A fan write either happens or raises, and says why.

The motor controller's own capability check, ``supports_fan()``, is the one
support question: a controller without fan control is refused before
anything is sent, in the one fan-control refusal, whatever the board. A
manual scope is told it has no motors, as every motion command tells it.
"""

from __future__ import annotations

import pytest

from modules.exceptions import HardwareCommandRefusedError, MissingPart
from modules.lumascope_api.diagnostics import DiagnosticsAPI
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session, _spy_wire
from tests.test_tsr_cluster_fix import _make_motor_with_responses


@pytest.fixture
def make_session(tmp_path):
    sessions = []

    def _make(model):
        session = _session(tmp_path, model, homed=False)
        sessions.append(session)
        return session

    yield _make
    for session in sessions:
        session.shutdown()


def _refused(call) -> HardwareCommandRefusedError:
    with pytest.raises(HardwareCommandRefusedError) as caught:
        call()
    return caught.value


def test_a_controller_without_fan_control_is_refused_and_sent_nothing(make_session, monkeypatch):
    # The LS720's TMCM-6110 has no fan control.
    scope = make_session('LS720').scope
    sent = _spy_wire(monkeypatch, scope)

    refusal = _refused(lambda: scope.diagnostics.set_motor_fan_duty(50))

    assert refusal.reason == 'axis_absent'
    assert refusal.missing is MissingPart.FAN_CONTROL
    assert str(refusal) == MissingPart.FAN_CONTROL.sentence
    assert not [c for c in sent if str(c).startswith('FAN')]


def test_firmware_without_the_fan_commands_is_the_same_refusal():
    # An EL-0940 whose firmware lacks FANSPEED answers the probe with ERROR.
    board = _make_motor_with_responses({'FANSPEED': "ERROR: command 'FANSPEED' not found:"})
    for name in ('supports_fan', '_command_supported', '_record_support'):
        setattr(board, name, _bound(board, name))
    sent = []
    answer = board.exchange_command

    def exchange(command, *args, **kwargs):
        sent.append(command)
        return answer(command, *args, **kwargs)

    board.exchange_command = exchange

    refusal = _refused(lambda: DiagnosticsAPI._set_motor_fan_duty_impl(board, 50))

    assert refusal.missing is MissingPart.FAN_CONTROL
    assert sent == ['FANSPEED'], 'only the read-only probe reaches the board'


def test_a_manual_scope_is_told_it_has_no_motors(make_session):
    scope = make_session('LS620').scope

    refusal = _refused(lambda: scope.diagnostics.set_motor_fan_duty(50))

    assert refusal.missing is MissingPart.MOTORS
    assert str(refusal) == 'This microscope has no motors.'


def test_a_controller_with_fan_control_takes_the_write(make_session, monkeypatch):
    scope = make_session('LS850').scope
    written = []
    monkeypatch.setattr(scope._motion_driver, 'set_fan_duty', written.append)

    assert scope.diagnostics.set_motor_fan_duty(50) is None
    assert written == [50]


def _bound(board, name):
    from drivers.motorboard import MotorBoard

    return getattr(MotorBoard, name).__get__(board, type(board))
