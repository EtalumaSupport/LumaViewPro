# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A move is given up when it stops progressing, never because it took long.

While an axis moves, the motion monitor reads its actual position in
microsteps. An axis whose reading has not changed for the stall window
while the board says it has not arrived is stalled: the stage is halted,
the move ends ``MoveFaultError(axis, 'stalled')``, the axis is UNKNOWN,
and the fault is reported once. A move of any length that keeps moving is
never given up. An axis whose position cannot be read from the move's start
is given up the same way, as unread. A monitor that ends leaves no axis
MOVING with nobody to judge it, and refuses every later drive.

Before, the bound was 120 s from the move's start whatever the axis did,
and the stall halted nothing: the motor drove on against whatever held it.
"""

import threading
import time

import pytest

from drivers.exceptions import HardwareError
from modules.exceptions import (
    HardwareCommandRefusedError,
    HomingFaultError,
    MoveFaultError,
)
from modules.lumascope_api import AxisState
from modules.notification_center import Severity
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

# Past the window, with room for a slowed host; far short of the 120 s the
# monitor used to wait.
_GIVE_UP_WAIT_S = 6.0


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850T',
            objective_id='10x Oly',
            turret_objectives={1: '10x Oly', 2: '4x Oly', 3: None, 4: None},
        ),
        simulate=True,
    )
    try:
        home_sim_scope(s.scope)
        s.scope.motion._driver.set_timing_mode('realistic')
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def _faults(centre_posts):
    return [(n.title, n.message) for n in centre_posts if n.severity == Severity.ERROR]


def _ended(operation, timeout_s=_GIVE_UP_WAIT_S):
    try:
        operation.result(timeout=timeout_s)
    except MoveFaultError as e:
        return e.reason
    return 'arrived'


def _stop_count(motion):
    sent = []
    real = motion._driver.motor_stop

    def counted():
        sent.append(time.monotonic())
        return real()

    motion._driver.motor_stop = counted
    return sent


def test_a_held_axis_is_given_up_stalled_and_halted(session, centre_posts):
    motion = session.scope.motion
    halts = _stop_count(motion)
    held = session.scope.simulation.hold_axis('X')
    operation = motion.start_move_absolute('X', 60000.0)
    assert held.wait(_GIVE_UP_WAIT_S), 'the move never reached its hold'
    held_at = time.monotonic()

    assert _ended(operation) == 'stalled'
    given_up_after = time.monotonic() - held_at
    assert motion._MOVE_STALL_WINDOW_S - 0.1 <= given_up_after <= motion._MOVE_STALL_WINDOW_S + 1.0
    assert len(halts) == 1, 'the stall is halted once'
    assert motion.get_axis_state('X') == AxisState.UNKNOWN
    faults = _faults(centre_posts)
    assert len(faults) == 1 and faults[0][0] == 'Motor Axis Stalled', faults


def test_a_move_that_keeps_moving_is_never_given_up(session, monkeypatch, centre_posts):
    """A full-travel Z move takes seconds, and a turret change holds its slot
    number for most of its turn: neither is a stall, at a window far shorter
    than either."""
    motion = session.scope.motion
    monkeypatch.setattr(motion, '_MOVE_STALL_WINDOW_S', 0.5)
    start = time.monotonic()
    motion.move_absolute('Z', motion.get_axis_limits('Z')['max'])
    motion.move_absolute('Z', 0.0)
    assert time.monotonic() - start > 2 * 0.5, 'the moves were too short to show anything'
    motion.move_turret(2)

    assert motion.get_turret_slot() == 2
    assert _faults(centre_posts) == []


def test_an_arriving_move_is_never_given_up(session, monkeypatch, centre_posts):
    """Arrival wins over the stall clock, at its most aggressive."""
    motion = session.scope.motion
    monkeypatch.setattr(motion, '_MOVE_STALL_WINDOW_S', 0.001)
    motion._driver.set_timing_mode('instant')

    assert _ended(motion.start_move_absolute('Z', 1000.0)) == 'arrived'
    assert motion.get_axis_state('Z') == AxisState.IDLE
    assert _faults(centre_posts) == []


def test_an_axis_unread_from_its_first_poll_is_given_up_unread(session, centre_posts):
    """The window starts when the move is armed, not at the first good read."""
    motion = session.scope.motion
    operation = motion.start_move_absolute('X', 60000.0)
    for command in ('STATUS_RX', 'ACTUAL_RX'):
        session.scope.simulation.drop_replies('motor', count=None, command=command)
    try:
        reason = _ended(operation)
    finally:
        session.scope.simulation.clear()

    assert reason in ('status_unread', 'position_unread'), reason
    assert motion.get_axis_state('X') == AxisState.UNKNOWN


def test_a_turret_home_whose_z_park_stalls_ends_in_its_fault(session, centre_posts):
    """The stall's halt is not a person's Stop: the home ends in a homing
    fault, and the monitor goes on judging moves. Before, a halt sent through
    ``stop_motion`` would be refused while a home drives, and the refusal
    would end the monitor's thread."""
    motion = session.scope.motion
    motion.move_absolute('Z', 3000.0)
    session.scope.simulation.hold_axis('Z')

    with pytest.raises(HomingFaultError):
        motion.start_home('T').result(timeout=_GIVE_UP_WAIT_S + 4.0)
    session.scope.simulation.clear()

    assert motion._motion_monitor_thread.is_alive()
    home_sim_scope(session.scope)
    motion._driver.set_timing_mode('instant')
    assert _ended(motion.start_move_absolute('X', 1000.0)) == 'arrived'


def test_a_monitor_that_ends_gives_every_moving_axis_up_and_refuses_later_drives(
    session, centre_posts
):
    motion = session.scope.motion
    driver = motion._driver
    real = driver.current_pos_steps
    broke = threading.Event()

    def defect(axis):
        if axis == 'X':
            broke.set()
            raise RuntimeError('a defect in the read')
        return real(axis)

    # a stand-in by design: a defect in the monitor's own read, which no
    # simulated board fault produces (a board fault is a HardwareError).
    driver.current_pos_steps = defect
    operation = motion.start_move_absolute('X', 60000.0)

    assert _ended(operation) == 'monitor_ended'
    assert broke.is_set()
    assert motion.get_axis_state('X') == AxisState.UNKNOWN
    driver.current_pos_steps = real
    with pytest.raises(HardwareCommandRefusedError) as refused:
        motion.start_move_absolute('Y', 1000.0)
    assert refused.value.reason == 'monitor_ended'
    assert not isinstance(refused.value, HardwareError)
