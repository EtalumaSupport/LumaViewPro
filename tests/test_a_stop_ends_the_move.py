# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A STOP ends the move.

Once ``stop_motion`` has taken the board's STOP, the stage stops where it
is and no move that was under way reaches its target. Each test here
stops a move on the realistic-timing simulator, whose stage travels for
as long as the real one would, so a stop that did not stop it is seen.
"""

import threading
import time

import pytest

from modules.exceptions import MotorStopFailedError, MoveNotCompletedError
from modules.lumascope_api.motion import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


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


def _reason(handle):
    try:
        handle.wait()
    except MoveNotCompletedError as e:
        return e.reason
    return 'arrived'


def test_a_stop_mid_move_leaves_the_stage_short_and_at_rest(session):
    """The simulated stage stops where the STOP found it. Before, the
    simulator sent no STOP at all, and its STOP command left the move
    running on its old timeline."""
    motion = session.scope.motion
    driver = motion._driver
    handle = motion.start_move_absolute('X', 60000.0)
    time.sleep(0.3)
    motion.stop_motion()

    assert _reason(handle) == 'stopped'
    assert motion.get_axis_state('X') == AxisState.IDLE
    stopped_at = driver.current_pos('X')
    assert 0.0 < stopped_at < 60000.0
    time.sleep(0.3)
    assert driver.current_pos('X') == stopped_at
    assert motion.get_current_position('X') == pytest.approx(stopped_at, abs=0.1)


def test_a_stop_the_board_does_not_answer_fails_and_ends_the_move(session):
    """A STOP with no reply was taken for firmware without STOP: no raise,
    no generation moved, and a move it may have stopped read arrived."""
    motion = session.scope.motion
    driver = motion._driver
    handle = motion.start_move_absolute('X', 60000.0)
    driver._fail_on.add('STOP')

    with pytest.raises(MotorStopFailedError):
        motion.stop_motion()
    driver._fail_on.discard('STOP')

    assert _reason(handle) == 'stopped'


def _wire(driver):
    """Record every command sent to the simulated board, in order."""
    sent = []
    real = driver.exchange_command

    def recorded(command, *args, **kwargs):
        sent.append(command)
        return real(command, *args, **kwargs)

    driver.exchange_command = recorded
    return sent


def test_a_stop_during_the_backlash_leg_ends_the_move_there(session):
    """Row 6: the leg read reached at the STOP's target = actual, and the
    final target went out after the stop."""
    motion = session.scope.motion
    driver = motion._driver
    driver.set_timing_mode('fast')
    motion.move_absolute('Z', 3000.0)
    driver.set_timing_mode('realistic')
    sent = _wire(driver)
    final = f'TARGET_WZ{driver.z_um2ustep(1000.0)}'
    leg = f'TARGET_WZ{driver.z_um2ustep(1000.0 - driver.backlash_um())}'

    def stop_during_the_leg():
        # The leg runs inside the move's body, before start_move_absolute
        # returns, so the stop comes from another thread, once the leg's
        # target is on the wire.
        deadline = time.monotonic() + 5.0
        while leg not in sent and time.monotonic() < deadline:
            time.sleep(0.001)
        time.sleep(0.3)
        motion.stop_motion()

    stopper = threading.Thread(target=stop_during_the_leg)
    stopper.start()
    handle = motion.start_move_absolute('Z', 1000.0, overshoot_enabled=True)
    stopper.join()

    assert _reason(handle) == 'stopped'
    after_stop = sent[sent.index('STOP') :]
    assert not [c for c in after_stop if c.startswith('TARGET_WZ')], after_stop
    assert final not in sent
    assert motion.get_axis_state('Z') == AxisState.IDLE
    stopped_at = driver.current_pos('Z')
    assert 1000.0 < stopped_at < 3000.0
    assert motion.get_current_position('Z') == pytest.approx(stopped_at, abs=0.1)


def test_a_stop_before_the_write_withholds_it(session, monkeypatch):
    """The move read the generation, then the STOP landed before its target
    went out: the write went out anyway and the stage travelled."""
    motion = session.scope.motion
    driver = motion._driver
    sent = _wire(driver)
    real_send = motion._send_drive

    def a_stop_lands_first(axis, send):
        motion.stop_motion()
        return real_send(axis, send)

    monkeypatch.setattr(motion, '_send_drive', a_stop_lands_first)
    handle = motion.start_move_absolute('X', 20000.0)

    assert _reason(handle) == 'stopped'
    assert not [c for c in sent if c.startswith('TARGET_WX')], sent
    assert driver.current_pos('X') == pytest.approx(0.0, abs=0.1)
    assert motion.get_axis_state('X') == AxisState.IDLE


def test_a_move_a_stop_ended_reads_stopped_not_superseded(session, monkeypatch):
    """A move in flight when the STOP lands, and a retarget the stop
    withheld: both read 'stopped'. Before, the withheld move's drive count
    made the first read 'superseded' -- 'going where that move sent it' --
    when nothing was sent."""
    motion = session.scope.motion
    driver = motion._driver
    first = motion.start_move_absolute('X', 60000.0)
    time.sleep(0.3)
    real_send = motion._send_drive

    def a_stop_lands_first(axis, send):
        motion.stop_motion()
        return real_send(axis, send)

    monkeypatch.setattr(motion, '_send_drive', a_stop_lands_first)
    second = motion.start_move_absolute('X', 10000.0)

    assert _reason(second) == 'stopped'
    assert _reason(first) == 'stopped'
    assert motion.get_axis_state('X') == AxisState.IDLE
    assert motion.get_current_position('X') == pytest.approx(driver.current_pos('X'), abs=0.1)


@pytest.fixture
def turret_session(tmp_path):
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
        s.scope.motion.move_absolute('Z', 3000.0)
        s.scope.motion._driver.set_timing_mode('realistic')
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def _turret_change_reason(motion):
    try:
        motion._move_turret_impl(2)
    except MoveNotCompletedError as e:
        return e.reason
    return 'arrived'


def test_a_stop_during_the_turret_leg_leaves_z_parked_and_no_slot(turret_session):
    """A STOP while T turned: the move read 'stopped', and then the Z
    restore drove the objective back up, a move nobody asked for after the
    person stopped the scope."""
    motion = turret_session.scope.motion
    driver = motion._driver
    sent = _wire(driver)

    def stop_during_the_turn():
        deadline = time.monotonic() + 5.0
        while not any(c.startswith('TARGET_WT') for c in sent) and time.monotonic() < deadline:
            time.sleep(0.001)
        time.sleep(0.1)
        motion.stop_motion()

    stopper = threading.Thread(target=stop_during_the_turn)
    stopper.start()
    reason = _turret_change_reason(motion)
    stopper.join()

    assert reason == 'stopped'
    after_stop = sent[sent.index('STOP') :]
    assert not [c for c in after_stop if c.startswith('TARGET_WZ')], after_stop
    assert driver.current_pos('Z') == pytest.approx(0.0, abs=0.1)
    assert motion.get_turret_slot() is None


def test_a_stop_after_the_turret_arrived_leaves_z_parked_and_no_slot(turret_session, monkeypatch):
    """A STOP after T arrived and before the restore: the turn ended
    unremarked, the restore read a generation already past the STOP, and
    the change reported success with Z driven back up."""
    motion = turret_session.scope.motion
    driver = motion._driver
    sent = _wire(driver)
    real_move = motion._move_absolute_impl

    def the_turn_arrives_then_a_stop(axis, *args, **kwargs):
        handle = real_move(axis, *args, **kwargs)
        if axis == 'T':
            real_wait = handle.wait

            def wait_then_stop(*wait_args, **wait_kwargs):
                real_wait(*wait_args, **wait_kwargs)
                motion.stop_motion()

            handle.wait = wait_then_stop
        return handle

    monkeypatch.setattr(motion, '_move_absolute_impl', the_turn_arrives_then_a_stop)
    reason = _turret_change_reason(motion)

    assert reason == 'stopped'
    after_stop = sent[sent.index('STOP') :]
    assert not [c for c in after_stop if c.startswith('TARGET_WZ')], after_stop
    assert driver.current_pos('Z') == pytest.approx(0.0, abs=0.1)
    assert motion.get_turret_slot() is None
