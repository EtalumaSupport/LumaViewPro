# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A motor STOP that fails is raised, and each caller says it once.

``stop_motion`` used to post "Motor stop failed ... during shutdown" and
swallow the error, whoever had called it, so a script that stopped the
stage heard nothing and a run stopping a timed-out move said "during
shutdown". Now it raises ``MotorStopFailedError`` chained from the
driver's error, with the stop generation already moved, so no waited move
reads itself as arrived; ``disconnect`` reports it once and still tears
the scope down.
"""

from __future__ import annotations

import pytest

import modules.lumascope_api._lumascope as lumascope_module
import modules.lumascope_api.motion as motion_module
from drivers.exceptions import HardwareError
from modules.exceptions import MotorStopFailedError
from modules.lumascope_api.motion import AxisState
from modules.notification_center import NotificationCenter, Severity
from tests.scope_fakes import build_scope


@pytest.fixture
def centre(monkeypatch):
    c = NotificationCenter(dedup_window_s=10.0)
    c.shown = []
    c.add_listener(c.shown.append, min_severity=Severity.INFO)
    monkeypatch.setattr(motion_module, 'notifications', c)
    monkeypatch.setattr(lumascope_module, 'notifications', c)
    return c


@pytest.fixture
def scope():
    return build_scope(simulate=True)


def _stop_fails(scope, monkeypatch):
    cause = HardwareError('no response from motor board')

    def _dead():
        raise cause

    monkeypatch.setattr(scope._motion_driver, 'motor_stop', _dead)
    return cause


def test_a_failed_stop_raises_chained_and_moves_the_generation(scope, centre, monkeypatch):
    cause = _stop_fails(scope, monkeypatch)
    generation = scope.motion._stop_generation

    with pytest.raises(MotorStopFailedError) as raised:
        scope.motion.stop_motion()

    assert raised.value.__cause__ is cause
    assert raised.value.title == 'Motor Stop Failed'
    assert 'power-cycle the microscope' in str(raised.value)
    assert scope.motion._stop_generation == generation + 1
    assert centre.shown == []


def test_disconnect_reports_a_failed_stop_once_and_finishes(scope, centre, monkeypatch):
    _stop_fails(scope, monkeypatch)

    scope.disconnect()

    assert [(n.severity, n.title, n.message) for n in centre.shown] == [
        (Severity.ERROR, 'Motor Stop Failed', str(MotorStopFailedError()))
    ]
    assert scope.motor_connected is False
    assert all(
        scope.motion.get_axis_state(ax) == AxisState.UNKNOWN for ax in scope.motion._axis_state
    )
