# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A startup home that fails is reported once, and the turret is left alone.

Bring-up waits on the home. When it raises -- the homing fault, or no
motor controller -- bring-up is where that outcome's flight ends, so it
reports it once and skips the turret move, which would be an absolute move
against the reference the home failed to establish. Any other refusal is
not bring-up's to answer and reaches its caller.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from modules.exceptions import HardwareCommandRefusedError, HomingFailedError, MissingPart
from modules.notification_center import Severity
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield s
    finally:
        s.shutdown()


@pytest.fixture
def shown(monkeypatch):
    # A centre of its own: the shared one's dedup window remembers what
    # earlier tests posted, and would swallow this one's popup.
    import modules.notification_center as nc

    centre = nc.NotificationCenter()
    seen = []
    centre.add_listener(seen.append, min_severity=Severity.INFO)
    monkeypatch.setattr(nc, 'notifications', centre)
    return seen


def _raises(exc):
    def _home(axis):
        raise exc

    return _home


def test_a_homing_fault_is_shown_once_and_the_turret_is_not_moved(session, shown):
    turret = []
    fault = HomingFailedError('ALL', 'failed', ('X', 'Y', 'Z', 'T'))
    session.start_application_session(home_fn=_raises(fault), turret_fn=turret.append)

    assert [(n.severity, n.title, n.message) for n in shown] == [
        (Severity.ERROR, 'Homing Failed', 'Homing failed. Position is unknown.')
    ]
    assert turret == []


def test_no_motor_controller_is_shown_once_as_not_connected(session, shown):
    turret = []
    session.start_application_session(
        home_fn=_raises(
            HardwareCommandRefusedError(
                'not_connected', 'home', missing=MissingPart.MOTOR_CONTROLLER
            )
        ),
        turret_fn=turret.append,
    )

    assert [(n.severity, n.title) for n in shown] == [(Severity.WARNING, 'Not Connected')]
    assert turret == []


def test_the_skipped_turret_move_is_not_a_second_error(session, shown, monkeypatch):
    # The reporter logs the failure at its own level; the skip it causes is
    # what bring-up did next, not another failure.
    import modules.scope_session as scope_session_module

    log = MagicMock()
    monkeypatch.setattr(scope_session_module, 'logger', log)
    session.start_application_session(
        home_fn=_raises(HomingFailedError('ALL', 'failed', ('Z',))),
        turret_fn=lambda position: None,
    )

    assert log.error.call_args_list == []
    assert call('startup turret positioning skipped: the stage reference is unknown') in (
        log.info.call_args_list
    )


def test_a_busy_refusal_reaches_the_caller(session, shown):
    refusal = HardwareCommandRefusedError('exclusive_activity_running', 'home', 'protocol')

    with pytest.raises(HardwareCommandRefusedError) as raised:
        session.start_application_session(home_fn=_raises(refusal), turret_fn=lambda position: None)

    assert raised.value is refusal
