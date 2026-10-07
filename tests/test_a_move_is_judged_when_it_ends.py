# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A move's verdict is fixed when the move ends.

A move's handle may be waited on long after the move ended. What happened
to the axis since -- a STOP, another move -- is not this move's outcome:
a move that arrived reads arrived, however late its caller waits.
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
        handle.wait()
    except MoveNotCompletedError as e:
        return e.reason
    return 'arrived'


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
    hold = driver.hold_travel('X')
    motion.start_move_absolute('X', 60000.0)
    assert hold.reached.wait(_WAIT_S), 'the later move never got part of the way'

    reasons = []
    waiter = threading.Thread(target=lambda: reasons.append(_reason(first)), daemon=True)
    waiter.start()
    waiter.join(_WAIT_S)
    alive = waiter.is_alive()
    hold.release()

    assert not alive, 'the ended move waited on the later one'
    assert reasons == ['arrived']


def test_a_home_after_a_stall_leaves_no_stale_fault(session, monkeypatch):
    """A guard: a wait on an axis whose home failed raises the home's
    'faulted', not the stall of the move before the home, whose object was
    already shown."""
    motion = session.scope.motion
    real_status = motion.get_target_status
    real_wait = motion._wait_for_move
    monkeypatch.setattr(
        motion, 'get_target_status', lambda ax: False if ax == 'Z' else real_status(ax)
    )
    monkeypatch.setattr(motion, '_MOTION_SETTLE_TIMEOUT_S', 0.5)
    monkeypatch.setattr(motion, '_wait_for_move', lambda move, timeout_s: real_wait(move, _WAIT_S))
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
