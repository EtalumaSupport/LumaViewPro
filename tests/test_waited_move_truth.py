# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A waited move tells the truth about arriving.

A move called with ``wait_until_complete`` returns only when its axis
confirmably reached the target. When the wait ends any other way -- the
motion monitor faulted the axis UNKNOWN (a stall, a lost board) or the
wait's bound ran out before the axis arrived -- the move raises
``MoveNotCompletedError`` and the axis stays UNKNOWN. The earlier body
wrote IDLE after the wait whatever it returned, so a stalled turret move
read as a turret in its commanded slot, and every consumer of the axis
state treated a position nobody reached as reached.

Another axis's timeout is that axis's business: a move whose own axis
arrived returns, and the axis that did not arrive keeps the state its own
owner gives it.
"""

import threading

import pytest

from modules.exceptions import MoveNotCompletedError
from modules.lumascope_api.motion import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create_headless(settings=complete_settings(live_folder=str(tmp_path)))
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def _z_target(motion):
    return motion.get_current_position('Z') + 50.0


def test_a_waited_move_that_arrives_returns_idle(session):
    motion = session.scope.motion
    motion.move_absolute('Z', _z_target(motion), wait_until_complete=True)
    assert motion.get_axis_state('Z') == AxisState.IDLE


def test_a_stalled_waited_move_raises_and_the_axis_stays_unknown(session, monkeypatch):
    motion = session.scope.motion
    real_status = motion.get_target_status
    monkeypatch.setattr(
        motion, 'get_target_status', lambda ax: False if ax == 'Z' else real_status(ax)
    )
    monkeypatch.setattr(motion, '_MOTION_SETTLE_TIMEOUT_S', 0.3)

    with pytest.raises(MoveNotCompletedError) as exc:
        motion.move_absolute('Z', _z_target(motion), wait_until_complete=True)

    assert exc.value.axis == 'Z'
    assert motion.get_axis_state('Z') == AxisState.UNKNOWN


def test_a_board_lost_during_the_wait_raises_faulted(session, monkeypatch):
    motion = session.scope.motion
    driver = motion._driver
    real_connected = driver.is_connected
    monkeypatch.setattr(
        driver,
        'is_connected',
        lambda: False if threading.current_thread().name == 'motion-monitor' else real_connected(),
    )
    monkeypatch.setattr(motion, '_DISCONNECT_FAULT_S', 0.1)

    with pytest.raises(MoveNotCompletedError) as exc:
        motion.move_absolute('Z', _z_target(motion), wait_until_complete=True)

    assert exc.value.axis == 'Z'
    assert exc.value.reason == 'faulted'
    assert motion.get_axis_state('Z') == AxisState.UNKNOWN


def test_a_waited_move_whose_axis_never_arrives_raises_timed_out(session, monkeypatch):
    motion = session.scope.motion
    # Nothing confirms arrival once the monitor is stopped, so the wait's
    # bound is what ends it.
    motion._motion_monitor_stop.set()
    motion._motion_wake.set()
    motion._motion_monitor_thread.join(timeout=2.0)
    monkeypatch.setattr(motion, '_MOTION_SETTLE_TIMEOUT_S', 0.3)

    with pytest.raises(MoveNotCompletedError) as exc:
        motion.move_relative('Z', 20.0, wait_until_complete=True)

    assert exc.value.axis == 'Z'
    assert exc.value.reason == 'timed_out'
    assert motion.get_axis_state('Z') == AxisState.UNKNOWN


def test_a_later_move_on_the_same_axis_is_not_faulted_by_this_one(session, monkeypatch):
    """The wait saw every axis stop, then another move on Z began before
    this move looked: that move's cleared event is not this move's timeout."""
    motion = session.scope.motion

    def _stopped_then_a_later_move_starts(timeout_s):
        motion._set_axis_state('Z', AxisState.MOVING)
        return True

    monkeypatch.setattr(motion, 'wait_until_finished_moving', _stopped_then_a_later_move_starts)
    try:
        motion._await_arrival('Z')
        assert motion.get_axis_state('Z') == AxisState.MOVING
    finally:
        motion._set_axis_state('Z', AxisState.IDLE)


def test_the_executor_shows_the_failure_in_its_own_words(monkeypatch):
    """Off an executor worker the user reads the error's message, not the
    generic 'did not complete' line that names no axis and no remedy."""
    import modules.sequential_io_executor as sio
    from modules.notification_center import NotificationCenter, Severity
    from modules.sequential_io_executor import IOTask, SequentialIOExecutor

    def _move():
        raise MoveNotCompletedError('T', 'faulted')

    centre = NotificationCenter(dedup_window_s=10.0)
    seen = []
    centre.add_listener(seen.append, min_severity=Severity.ERROR)
    monkeypatch.setattr(sio, 'notifications', centre)

    executor = SequentialIOExecutor(name='TEST')
    task = IOTask(_move)
    task.set_name(executor.executor_name)
    executor.queue.put(task)
    executor.queue.get()
    result, exception = task.run()
    executor._on_task_done(task, result, exception)

    assert [n.message for n in seen] == [str(MoveNotCompletedError('T', 'faulted'))]


def test_another_axis_timing_out_does_not_fail_an_arrived_move(session, monkeypatch):
    motion = session.scope.motion
    monkeypatch.setattr(motion, '_MOTION_SETTLE_TIMEOUT_S', 0.5)
    # HOMING clears X's arrival event and the monitor never sets it, so the
    # wait times out on X while Z arrives.
    motion._set_axis_state('X', AxisState.HOMING)
    try:
        motion.move_absolute('Z', _z_target(motion), wait_until_complete=True)
        assert motion.get_axis_state('Z') == AxisState.IDLE
        assert motion.get_axis_state('X') == AxisState.HOMING
    finally:
        motion._set_axis_state('X', AxisState.IDLE)
