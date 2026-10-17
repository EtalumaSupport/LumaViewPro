# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A move's verdict is fixed when the move ends.

A move's handle may be waited on long after the move ended. What happened
to the axis since -- a STOP, another move -- is not this move's outcome:
a move that arrived reads arrived, however late its caller waits.
"""

import threading

import pytest

from modules.exceptions import (
    MoveFaultError,
    MoveNotCompletedError,
    MoveStoppedError,
    ScopeDisconnectError,
)
from modules.lumascope_api.motion import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


# How long a wait may take on a slowed host, far past any move here.
_WAIT_S = 10.0


def _reason(handle):
    try:
        handle.result()
    except MoveNotCompletedError as e:
        return e.reason
    return 'arrived'


def _ending(handle):
    """The move's ending as its operation reports it: the error's type and reason, or 'arrived'."""
    try:
        handle.result(timeout=_WAIT_S)
    except MoveNotCompletedError as e:
        return type(e), e.reason
    return 'arrived'


def _held_move(session, axis='X', target=60000.0):
    """A move held part of the way, in flight when the caller acts."""
    motion = session.scope.motion
    motion._driver.set_timing_mode('realistic')
    held = session.scope.simulation.hold_axis(axis)
    handle = motion.start_move_absolute(axis, target)
    assert held.wait(_WAIT_S), 'the move never got part of the way'
    return handle


def test_a_move_in_flight_when_the_scope_is_closed_ends_stopped(session):
    """The close's STOP halted it, and it read 'faulted': the close's
    UNKNOWN settled it with the words of a stall or a lost board."""
    handle = _held_move(session)

    session.scope.disconnect()

    assert _ending(handle) == (MoveStoppedError, 'stopped')


def test_a_move_that_arrives_while_the_close_has_shut_the_lane_ends_arrived(session, monkeypatch):
    """The close shuts the lane first and it read 'status_unread': the
    monitor's arrival read was refused as a client's command, so a close
    slower than the stall window gave the axis up. The window is shortened
    so the close need not be."""
    motion = session.scope.motion
    motion._driver.set_timing_mode('realistic')
    monkeypatch.setattr(motion, '_MOVE_STALL_WINDOW_S', 0.3)
    handle = motion.start_move_absolute('X', motion.get_current_position('X') + 200.0)
    # A stand-in by design: the close's first step reached on its own, as no
    # public member holds a close part way through.
    session.scope._io_executor.shutdown(wait=False)

    ending = _ending(handle)
    session.scope.disconnect()

    assert ending == 'arrived'


def test_a_move_the_closes_failed_stop_was_meant_to_end_ends_stopped(session):
    """The STOP is reported once as itself, and the move it was meant to end
    is recorded as stopped; it read 'faulted'."""
    handle = _held_move(session)
    session.scope.simulation.drop_replies('motor', count=None, command='STOP')

    with pytest.raises(ScopeDisconnectError) as raised:
        session.scope.disconnect()

    assert raised.value.parts == ('motor stop',)
    assert _ending(handle) == (MoveStoppedError, 'stopped')


def test_a_move_a_home_cannot_read_the_position_of_ends_position_unread(session):
    """A Z home reads every axis's position; a failed read of a moving X
    ended X's move 'faulted', the words of a stall or a lost board."""
    handle = _held_move(session)
    session.scope.simulation.drop_replies('motor', count=None, command='TARGET_RX')

    session.scope.motion.home('Z')

    assert _ending(handle) == (MoveFaultError, 'position_unread')


def test_a_move_that_arrived_reads_arrived_after_a_later_move_starts(session):
    """It read 'superseded': the drive count had moved on by the wait."""
    motion = session.scope.motion
    first = motion.start_move_absolute('X', 1000.0)
    motion.wait_until_finished_moving()
    second = motion.start_move_absolute('X', 2000.0)

    assert _reason(first) == 'arrived'
    assert _reason(second) == 'arrived'


def test_a_move_that_arrived_reads_arrived_after_a_later_stop(session):
    """It read 'stopped': the stop generation had moved on by the wait."""
    motion = session.scope.motion
    handle = motion.start_move_absolute('X', 60000.0)
    motion.wait_until_finished_moving()
    motion.stop_motion()

    assert _reason(handle) == 'arrived'


def test_a_move_that_ended_is_answered_while_a_later_one_travels(session):
    """Its wait sat on the axis, so it waited out the later move too."""
    motion = session.scope.motion
    driver = motion._driver
    first = motion.start_move_absolute('X', 1000.0)
    motion.wait_until_finished_moving()
    driver.set_timing_mode('realistic')
    held = session.scope.simulation.hold_axis('X')
    motion.start_move_absolute('X', 60000.0)
    assert held.wait(_WAIT_S), 'the later move never got part of the way'

    reasons = []
    waiter = threading.Thread(target=lambda: reasons.append(_reason(first)), daemon=True)
    waiter.start()
    waiter.join(_WAIT_S)
    alive = waiter.is_alive()
    session.scope.simulation.clear()

    assert not alive, 'the ended move waited on the later one'
    assert reasons == ['arrived']


def test_a_home_after_a_stall_leaves_no_stale_fault(session, monkeypatch):
    """A guard: a wait on an axis whose home failed raises the home's
    'faulted', not the stall of the move before the home, whose object was
    already shown."""
    motion = session.scope.motion
    real_status = motion._read_target_status
    monkeypatch.setattr(
        motion, '_read_target_status', lambda ax: False if ax == 'Z' else real_status(ax)
    )
    monkeypatch.setattr(motion, '_MOVE_STALL_WINDOW_S', 0.5)
    with pytest.raises(MoveNotCompletedError) as stalled:
        motion.move_absolute('Z', motion.get_current_position('Z') + 100.0)
    assert stalled.value.reason == 'stalled'

    motion._set_axis_state('Z', AxisState.HOMING)

    def the_home_fails(axis, timeout_s):
        motion._set_axis_state('Z', AxisState.UNKNOWN)
        return True

    monkeypatch.setattr(motion, '_wait_for_axis_to_stop', the_home_fails)
    with pytest.raises(MoveNotCompletedError) as raised:
        motion.wait_until_finished_moving()

    assert raised.value.reason == 'faulted'
