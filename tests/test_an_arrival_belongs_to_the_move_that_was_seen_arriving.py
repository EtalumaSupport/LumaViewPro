# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An arrival belongs to the move that was seen arriving.

The motion monitor's verdicts -- IDLE on the board's reached bit, UNKNOWN
on a stall -- were written to the axis whatever move owned it by then. A
move that landed between the monitor's status read and its IDLE write
was stamped arrived by the previous move's bit; a Z move retargeted
downward was stamped arrived at its overshoot point; the stall clock ran
per axis, so a retarget inherited the earlier move's time; and the kept
position was read before the exchange that saw the arrival, so
``get_current_position`` after a waited move was short of where the axis
stopped by what it moved between the two reads.

Each test here is one of those races, forced deterministically on the
realistic-timing simulator with a hook on the exchange or an injected
delay, never load. The monitor now notes the move it is judging before
it asks the board, and writes its verdict only if that move still owns
the axis.
"""

import threading
import time

import pytest

import modules.lumascope_api.motion as motion_module
from drivers.exceptions import HardwareError
from modules.exceptions import MoveNotCompletedError
from modules.lumascope_api.motion import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

# A Z microstep on the simulated LS850 is 0.025 um; X and Y 0.078 um.
_MICROSTEP_UM = 0.1
_MONITOR = 'motion-monitor'


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        s.scope.motion._driver.set_timing_mode('realistic')
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def _outcome(handle):
    try:
        handle.wait()
        return 'arrived'
    except MoveNotCompletedError as e:
        return e.reason


def _on_monitor():
    return threading.current_thread().name == _MONITOR


def _start_in_thread(fn):
    t = threading.Thread(target=fn)
    t.start()
    return t


def test_the_position_after_a_waited_move_is_where_the_axis_stopped(session, monkeypatch):
    """The monitor's status read is held until the simulated move ends, so
    the position it read before the status is as stale as it can be. The
    position reported after the wait must still be the board's."""
    motion = session.scope.motion
    driver = motion._driver
    real_status = driver.target_status

    def held_until_the_move_ends(axis):
        if _on_monitor():
            end = driver._move_end_time.get(axis, 0.0)
            while time.monotonic() < end + 0.005:
                time.sleep(0.001)
        return real_status(axis)

    monkeypatch.setattr(driver, 'target_status', held_until_the_move_ends)

    motion.move_absolute('Z', 3000.0)
    assert abs(motion.get_current_position('Z') - driver.current_pos('Z')) <= _MICROSTEP_UM
    motion.move_relative('Z', -500.0)
    assert abs(driver.current_pos('Z') - 2500.0) <= _MICROSTEP_UM
    assert abs(motion.get_current_position('Z') - driver.current_pos('Z')) <= _MICROSTEP_UM


def test_a_move_landing_inside_the_arrival_read_is_not_stamped_by_it(session, monkeypatch):
    """Move B starts inside the monitor's status read that saw move A
    arrive -- after the read, before the write. A is superseded; B's wait
    returns only with the board at B's target."""
    motion = session.scope.motion
    driver = motion._driver
    motion.move_absolute('X', 10000.0)
    real_status = motion.get_target_status
    b = {}
    fired = threading.Event()

    def start_b_inside_the_read(axis):
        reached = real_status(axis)
        if reached and axis == 'X' and _on_monitor() and not fired.is_set():
            fired.set()
            _start_in_thread(lambda: b.update(h=motion.start_move_absolute('X', 13000.0))).join()
        return reached

    monkeypatch.setattr(motion, 'get_target_status', start_b_inside_the_read)
    a = motion.start_move_absolute('X', 10200.0)
    a_out = _outcome(a)
    assert fired.wait(5.0)
    b_out = _outcome(b['h'])

    assert b_out == 'arrived'
    assert driver.target_status('X')
    assert abs(driver.current_pos('X') - 13000.0) <= _MICROSTEP_UM
    assert a_out == 'superseded'


def test_the_previous_targets_bit_read_during_the_send_is_no_arrival(session, monkeypatch):
    """The monitor's status read is forced into the window between move
    B's driver call and its MOVING write, while A's target still reads
    reached. Neither move is stamped arrived by that read."""
    motion = session.scope.motion
    driver = motion._driver
    motion.move_absolute('X', 10000.0)
    real_drive = driver.move_abs_pos
    real_status = motion.get_target_status
    b_in_driver = threading.Event()
    status_seen = threading.Event()
    b_started = threading.Event()
    b = {}

    def drive_held_until_the_monitor_read(axis, pos, **kw):
        if axis == 'X' and pos == 13000.0:
            b_in_driver.set()
            status_seen.wait(5.0)
        return real_drive(axis, pos, **kw)

    def status_held_until_b_is_marked_moving(axis):
        reached = real_status(axis)
        if (
            reached
            and axis == 'X'
            and _on_monitor()
            and b_in_driver.is_set()
            and not status_seen.is_set()
        ):
            status_seen.set()
            b_started.wait(5.0)
        return reached

    monkeypatch.setattr(driver, 'move_abs_pos', drive_held_until_the_monitor_read)
    monkeypatch.setattr(motion, 'get_target_status', status_held_until_b_is_marked_moving)

    a = motion.start_move_absolute('X', 10200.0)
    end = driver._move_end_time['X']
    while time.monotonic() < end - 0.03:
        time.sleep(0.0005)

    def start_b():
        b['h'] = motion.start_move_absolute('X', 13000.0)
        b_started.set()

    _start_in_thread(start_b)
    a_out = _outcome(a)
    assert b_started.wait(5.0)
    b_out = _outcome(b['h'])

    assert b_out == 'arrived'
    assert abs(driver.current_pos('X') - 13000.0) <= _MICROSTEP_UM
    assert a_out == 'superseded'


def test_the_overshoot_legs_arrival_is_nobodys(session, monkeypatch):
    """Move 1 (Z up) is in flight when move 2 (Z down, with overshoot)
    starts. The lane's leg loop, on seeing the leg arrive, holds the final
    target write until the monitor has read that arrival, so the monitor
    reads reached at the overshoot point with move 1 still owning the
    axis. Move 1 is not arrived there; move 2 arrives at its own target."""
    motion = session.scope.motion
    driver = motion._driver
    real_status = driver.target_status
    monitor_read_the_leg = threading.Event()

    def the_lane_waits_for_the_monitors_read(axis):
        reached = real_status(axis)
        if axis == 'Z' and reached and motion._overshoot:
            if _on_monitor():
                monitor_read_the_leg.set()
            else:
                # The lane's leg loop saw the leg arrive: it holds the final
                # target write until the monitor has read that arrival.
                monitor_read_the_leg.wait(5.0)
        return reached

    monkeypatch.setattr(driver, 'target_status', the_lane_waits_for_the_monitors_read)
    motion.move_absolute('Z', 4000.0)
    m1 = motion.start_move_absolute('Z', 6000.0)
    time.sleep(0.1)
    m2 = {}
    t = _start_in_thread(
        lambda: m2.update(h=motion.start_move_absolute('Z', 3000.0, overshoot_enabled=True))
    )
    m1_out = _outcome(m1)
    t.join()
    m2_out = _outcome(m2['h'])

    assert m2_out == 'arrived'
    assert abs(driver.current_pos('Z') - 3000.0) <= _MICROSTEP_UM
    assert m1_out == 'superseded'


def test_a_retargeted_move_is_timed_by_its_own_clock(session, monkeypatch):
    """The board never reports X reached. Move 1 runs 0.5 s of the 1 s
    stall bound before move 2 takes the axis; the monitor faults move 2
    'stalled' only after its own second, not after the earlier move's
    half."""
    motion = session.scope.motion
    real_status = motion.get_target_status
    monkeypatch.setattr(
        motion, 'get_target_status', lambda ax: False if ax == 'X' else real_status(ax)
    )
    monkeypatch.setattr(motion, '_MOTION_SETTLE_TIMEOUT_S', 1.0)
    # The waiter's own bound is kept out of it: the monitor's clock judges.
    real_wait = motion._wait_for_axis_to_stop
    monkeypatch.setattr(
        motion, '_wait_for_axis_to_stop', lambda axis, timeout_s: real_wait(axis, 5.0)
    )

    m1 = motion.start_move_absolute('X', 10000.0)
    time.sleep(0.5)
    t0 = time.monotonic()
    m2 = motion.start_move_absolute('X', 10100.0)
    m2_out = _outcome(m2)
    faulted_after = time.monotonic() - t0

    assert m2_out == 'stalled'
    assert faulted_after >= 0.9, faulted_after
    assert _outcome(m1) == 'superseded'


def test_a_timed_out_wait_does_not_fault_the_move_that_started_in_its_window(session, monkeypatch):
    """Move 1's wait runs out while the stage travels. Move 2 starts in the
    window between the waiter's decision and its UNKNOWN write. Move 2 is
    not faulted; move 1 reads superseded."""
    motion = session.scope.motion
    driver = motion._driver
    motion.move_absolute('X', 10000.0)
    real_wait = motion._wait_for_axis_to_stop
    real_set = motion._set_axis_state
    waiter = threading.current_thread()
    m2 = {}

    monkeypatch.setattr(
        motion, '_wait_for_axis_to_stop', lambda axis, timeout_s: real_wait(axis, 0.3)
    )

    def start_m2_inside_the_waiters_write(axis, state, **kw):
        if state == AxisState.UNKNOWN and threading.current_thread() is waiter and 'h' not in m2:
            _start_in_thread(lambda: m2.update(h=motion.start_move_absolute('X', 10300.0))).join()
        return real_set(axis, state, **kw)

    monkeypatch.setattr(motion, '_set_axis_state', start_m2_inside_the_waiters_write)
    m1 = motion.start_move_absolute('X', 40000.0)
    m1_out = _outcome(m1)
    monkeypatch.setattr(motion, '_set_axis_state', real_set)
    monkeypatch.setattr(motion, '_wait_for_axis_to_stop', real_wait)

    assert 'h' in m2
    assert _outcome(m2['h']) == 'arrived'
    assert abs(driver.current_pos('X') - 10300.0) <= _MICROSTEP_UM
    assert m1_out == 'superseded'


class _SlowClearEvent(threading.Event):
    """An arrival event whose clear is delayed: the preemption stand-in."""

    delay_s = 0.0

    def clear(self):
        if self.delay_s:
            time.sleep(self.delay_s)
        return super().clear()


def test_a_verdict_cannot_land_between_the_moving_write_and_its_event(session, monkeypatch):
    """Y moves so the monitor is awake. X is sent to where it stands, so
    the board reports reached on the first read; X's event clear is
    delayed 80 ms past the MOVING write. The verdict for this move must
    not be undone by its own event clear: the wait returns at once."""
    motion = session.scope.motion
    motion.move_absolute('X', 10000.0)
    ev = _SlowClearEvent()
    ev.set()
    motion._arrival_events['X'] = ev
    monkeypatch.setattr(motion, '_MOTION_SETTLE_TIMEOUT_S', 3.0)
    motion.start_move_absolute('Y', motion.get_current_position('Y') + 3000.0)

    ev.delay_s = 0.08
    h = motion.start_move_absolute('X', 10000.0)
    ev.delay_s = 0.0
    t0 = time.monotonic()
    out = _outcome(h)
    waited = time.monotonic() - t0

    assert out == 'arrived'
    assert waited < 1.0, waited


def test_a_driver_raise_leaves_the_axis_unknown_never_moving(session, monkeypatch):
    """The helper's contract: a raise from the driver call fails the drive,
    so the axis is UNKNOWN and the caller holds 'driver_failed'; the axis
    is never left MOVING with no move owning it."""
    motion = session.scope.motion
    driver = motion._driver

    def no_reply(axis, pos, **kw):
        raise HardwareError('move(Z): no response to the target write')

    monkeypatch.setattr(driver, 'move_abs_pos', no_reply)
    with pytest.raises(MoveNotCompletedError) as exc:
        motion.move_absolute('Z', 3000.0)
    assert exc.value.reason == 'driver_failed'
    assert motion.get_axis_state('Z') == AxisState.UNKNOWN


def test_a_leg_that_never_arrives_fails_the_move_within_the_lanes_bound(session, monkeypatch):
    """The simulated board dies the instant the overshoot leg's target is
    written, so the leg never arrives and, on the simulator, its status
    read never raises. The move fails 'driver_failed' at the leg's bound,
    inside the lane's, with the overshoot flag cleared; with the board
    back, a home and a move run."""
    motion = session.scope.motion
    driver = motion._driver
    monkeypatch.setattr(motion_module, 'OVERSHOOT_LEG_TIMEOUT_S', 0.5)
    monkeypatch.setattr(motion, '_MOTION_WAIT_BASE_S', 3.0)
    motion.move_absolute('Z', 6000.0)
    real_move = driver.move

    def the_board_dies_after_the_legs_write(axis, steps):
        real_move(axis, steps)
        if motion._overshoot:
            driver._fail_after = driver._cmd_count

    monkeypatch.setattr(driver, 'move', the_board_dies_after_the_legs_write)
    t0 = time.monotonic()
    with pytest.raises(MoveNotCompletedError) as exc:
        motion.move_absolute('Z', 3000.0, overshoot_enabled=True)
    failed_after = time.monotonic() - t0

    assert exc.value.reason == 'driver_failed'
    assert failed_after < 2.0, failed_after
    assert motion._overshoot is False
    assert motion.get_axis_state('Z') == AxisState.UNKNOWN

    monkeypatch.setattr(driver, 'move', real_move)
    driver._fail_after = None
    driver.connect()
    home_sim_scope(session.scope)
    motion.move_absolute('Z', 3000.0, overshoot_enabled=True)
    assert abs(driver.current_pos('Z') - 3000.0) <= _MICROSTEP_UM
