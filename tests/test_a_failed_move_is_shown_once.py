# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A move that fails is one object, shown once, titled for what happened.

A driver failure used to post "Move Failed" and then re-raise the driver's
error, which its caller showed again as "Operation failed". A stall or a
lost board during a waited move posted from the motion monitor and then
raised a second, different object from the waiter, shown again. Now the
move raises one ``MoveNotCompletedError``: the monitor records the object
it gave the axis up with before it wakes the waiter, reports it, and the
waiter raises that same object, so the reporter shows it once. A report the
reporter suppressed (the dedup window) leaves it to the person's own
request to show. A jog nobody waits on is shown by the monitor.

Each test drives a real ``Lumascope(simulate=True)`` inside a Session and
reports what the waiter raised exactly as the GUI boundary does.
"""

from __future__ import annotations

import threading

import pytest

import modules.lumascope_api.motion as motion_module
from drivers.exceptions import HardwareError
from modules.exceptions import MoveNotCompletedError
from modules.lumascope_api.motion import AxisState
from modules.notification_center import NotificationCenter, Severity
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

MONITOR_GIVES_UP_S = 0.3
WAITER_BOUND_S = 5.0


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


@pytest.fixture
def centre(monkeypatch):
    c = NotificationCenter(dedup_window_s=10.0)
    c.shown = []
    c.add_listener(c.shown.append, min_severity=Severity.INFO)
    monkeypatch.setattr(motion_module, 'notifications', c)
    return c


def _z_target(motion):
    return motion.get_current_position('Z') + 50.0


def _the_caller_reports(centre, call):
    """What the GUI boundary does with a waited call's outcome."""
    try:
        call()
    except MoveNotCompletedError as e:
        centre.report_outcome(e, solicited=True, category='UI:MOVE_Z')
        return e
    return None


def _z_never_arrives(motion, monkeypatch):
    real_status = motion.get_target_status
    monkeypatch.setattr(
        motion, 'get_target_status', lambda ax: False if ax == 'Z' else real_status(ax)
    )


def _monitor_gives_up_first(motion, monkeypatch):
    # The monitor's stall clock and the waiter's bound are the same number;
    # the waiter's is widened so the monitor's verdict is the one tested.
    monkeypatch.setattr(motion, '_MOTION_SETTLE_TIMEOUT_S', MONITOR_GIVES_UP_S)
    real_wait = motion.wait_until_finished_moving
    monkeypatch.setattr(
        motion, 'wait_until_finished_moving', lambda timeout_s: real_wait(timeout_s=WAITER_BOUND_S)
    )


def test_a_driver_failure_is_one_popup_chained_to_the_drivers_error(session, centre, monkeypatch):
    motion = session.scope.motion
    cause = HardwareError('no response from motor board')

    def _dead(*args, **kwargs):
        raise cause

    monkeypatch.setattr(motion._driver, 'move_abs_pos', _dead)

    raised = _the_caller_reports(
        centre, lambda: motion.move_absolute('Z', _z_target(motion), wait_until_complete=True)
    )

    assert raised.reason == 'driver_failed'
    assert raised.__cause__ is cause
    assert [(n.severity, n.title) for n in centre.shown] == [
        (Severity.ERROR, 'Move Did Not Complete')
    ]
    assert motion.get_axis_state('Z') == AxisState.UNKNOWN


def test_a_stall_during_a_waited_move_is_one_popup_the_monitors(session, centre, monkeypatch):
    motion = session.scope.motion
    _z_never_arrives(motion, monkeypatch)
    _monitor_gives_up_first(motion, monkeypatch)

    raised = _the_caller_reports(
        centre, lambda: motion.move_absolute('Z', _z_target(motion), wait_until_complete=True)
    )

    assert raised.reason == 'stalled'
    assert [n.title for n in centre.shown] == ['Motor Axis Stalled']
    assert centre.shown[0].message == str(raised)


def test_a_second_stall_inside_the_dedup_window_is_still_shown_once(session, centre, monkeypatch):
    motion = session.scope.motion
    _z_never_arrives(motion, monkeypatch)
    _monitor_gives_up_first(motion, monkeypatch)
    _the_caller_reports(
        centre, lambda: motion.move_absolute('Z', _z_target(motion), wait_until_complete=True)
    )
    motion.home('Z')

    second = _the_caller_reports(
        centre, lambda: motion.move_absolute('Z', _z_target(motion), wait_until_complete=True)
    )

    assert second.reason == 'stalled'
    assert [n.title for n in centre.shown] == ['Motor Axis Stalled', 'Motor Axis Stalled']


def test_a_board_lost_during_a_waited_move_is_one_popup_the_monitors(session, centre, monkeypatch):
    motion = session.scope.motion
    driver = motion._driver
    real_connected = driver.is_connected
    monkeypatch.setattr(
        driver,
        'is_connected',
        lambda: False if threading.current_thread().name == 'motion-monitor' else real_connected(),
    )
    monkeypatch.setattr(motion, '_DISCONNECT_FAULT_S', 0.1)

    raised = _the_caller_reports(
        centre, lambda: motion.move_absolute('Z', _z_target(motion), wait_until_complete=True)
    )

    assert raised.reason == 'board_lost'
    assert [n.title for n in centre.shown] == ['Motor Board Disconnected']


def test_a_stalled_jog_nobody_waits_on_is_shown_by_the_monitor(session, centre, monkeypatch):
    motion = session.scope.motion
    _z_never_arrives(motion, monkeypatch)
    monkeypatch.setattr(motion, '_MOTION_SETTLE_TIMEOUT_S', MONITOR_GIVES_UP_S)

    motion.move_relative('Z', 20.0)
    assert motion.wait_until_finished_moving(timeout_s=WAITER_BOUND_S)

    assert motion.get_axis_state('Z') == AxisState.UNKNOWN
    assert [n.title for n in centre.shown] == ['Motor Axis Stalled']


def test_a_later_wait_does_not_raise_an_earlier_stall(session, centre, monkeypatch):
    motion = session.scope.motion
    _z_never_arrives(motion, monkeypatch)
    _monitor_gives_up_first(motion, monkeypatch)
    _the_caller_reports(
        centre, lambda: motion.move_absolute('Z', _z_target(motion), wait_until_complete=True)
    )
    motion.home('Z')

    # The next move's wait ends with Z set UNKNOWN by something other than
    # the monitor: its outcome is that, not the stall before it.
    def _something_else_faults_z(timeout_s):
        motion._set_axis_state('Z', AxisState.UNKNOWN)
        return True

    monkeypatch.setattr(motion, 'wait_until_finished_moving', _something_else_faults_z)
    with pytest.raises(MoveNotCompletedError) as raised:
        motion.move_absolute('Z', _z_target(motion), wait_until_complete=True)

    assert raised.value.reason == 'faulted'
