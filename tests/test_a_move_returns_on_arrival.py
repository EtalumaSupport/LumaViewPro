# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A move returns once it has arrived, and says why when it did not.

``move_absolute`` / ``move_relative`` returned as soon as the board took
the command, so a stall, a lost board or a stop never reached a script,
REST or a headless run; only the motion monitor's unsolicited popup told
anyone. They now wait for the axis and raise ``MoveNotCompletedError``.
A caller that works while the axis travels starts the move instead
(``start_move_absolute`` / ``start_move_relative``) and gets the same
verdict from the started move's ``wait()``.

The wait runs in the caller's thread, not on the IO lane, so the lane
takes other work while the stage travels. With the lane no longer
ordering the wait, another move on the same axis can start first; the
earlier move is then superseded, never reported as arrived where the
later move sent the stage.

``wait_until_finished_moving`` waits for the axes moving when it is
called and raises when one ended UNKNOWN or the wait ran out, instead of
returning True for an axis a failed home left unknown, or False on a
timeout its callers discarded.
"""

import inspect
import threading
import time

import pytest

from modules.exceptions import MoveNotCompletedError
from modules.lumascope_api.motion import AxisState, MotionAPI
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import IOTask
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

_TRAVEL_WAIT_S = 30.0


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def _z_target(motion):
    return motion.get_current_position('Z') + 50.0


def _lose_the_board_to_the_monitor(motion, monkeypatch):
    """The monitor sees the board gone and gives a moving axis up as 'board_lost'."""
    driver = motion._driver
    real_connected = driver.is_connected
    monkeypatch.setattr(
        driver,
        'is_connected',
        lambda: False if threading.current_thread().name == 'motion-monitor' else real_connected(),
    )
    monkeypatch.setattr(motion, '_DISCONNECT_FAULT_S', 0.1)


def _far_x(motion) -> float:
    """An X target across most of the travel: seconds of motion in realistic timing."""
    limits = motion.get_axis_limits('X')
    here = motion.get_current_position('X')
    far = limits['max'] * 0.9 if here < (limits['min'] + limits['max']) / 2 else limits['min']
    return far


def _until(predicate, timeout_s=_TRAVEL_WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, 'condition never held'
        time.sleep(0.005)


class TestAMoveWaits:
    def test_a_plain_move_raises_the_fault_the_monitor_gave_its_axis_up_with(
        self, session, monkeypatch
    ):
        motion = session.scope.motion
        _lose_the_board_to_the_monitor(motion, monkeypatch)

        with pytest.raises(MoveNotCompletedError) as exc:
            motion.move_absolute('Z', _z_target(motion))

        assert exc.value.reason == 'board_lost'
        assert motion.get_axis_state('Z') == AxisState.UNKNOWN

    def test_a_started_moves_wait_gives_the_same_verdict(self, session, monkeypatch):
        motion = session.scope.motion
        _lose_the_board_to_the_monitor(motion, monkeypatch)
        move = motion.start_move_absolute('Z', _z_target(motion))

        with pytest.raises(MoveNotCompletedError) as exc:
            move.wait()

        assert exc.value.axis == 'Z'
        assert exc.value.reason == 'board_lost'

    def test_a_started_move_that_arrives_waits_until_idle(self, session):
        motion = session.scope.motion
        target = _z_target(motion)

        motion.start_move_absolute('Z', target).wait()

        assert motion.get_axis_state('Z') == AxisState.IDLE
        assert motion.get_current_position('Z') == pytest.approx(target, abs=1.0)

    def test_no_public_motion_member_takes_a_wait_flag(self):
        flagged = [
            name
            for name, member in inspect.getmembers(MotionAPI, inspect.isfunction)
            if not name.startswith('_')
            and 'wait_until_complete' in inspect.signature(member).parameters
        ]
        assert flagged == []


class TestTheWaitIsOffTheLane:
    def test_the_lane_takes_other_work_while_a_waited_move_travels(self, session):
        motion = session.scope.motion
        motion._driver.set_timing_mode('realistic')
        target = _far_x(motion)
        errors = []

        def _waited_move():
            try:
                motion.move_absolute('X', target)
            except Exception as e:  # reported by the assertion below
                errors.append(e)

        mover = threading.Thread(target=_waited_move)
        mover.start()
        _until(lambda: motion.get_axis_state('X') == AxisState.MOVING)

        session.io_executor.call(IOTask(action=lambda: None), 'lane_probe', timeout_s=5.0)
        still_travelling = motion.get_axis_state('X') == AxisState.MOVING

        mover.join(timeout=_TRAVEL_WAIT_S)
        assert errors == []
        assert still_travelling, 'the lane task waited for the move to arrive'

    def test_a_move_another_caller_supersedes_is_not_reported_arrived(self, session):
        motion = session.scope.motion
        first_target = _z_target(motion)
        first = motion.start_move_absolute('Z', first_target)
        second = motion.start_move_absolute('Z', first_target + 100.0)

        with pytest.raises(MoveNotCompletedError) as exc:
            first.wait()
        second.wait()

        assert exc.value.reason == 'superseded'
        assert motion.get_current_position('Z') == pytest.approx(first_target + 100.0, abs=1.0)
        assert motion.get_axis_state('Z') == AxisState.IDLE


class TestWaitingForMotionTheCallerDidNotStart:
    def test_an_axis_a_failed_home_leaves_unknown_raises(self, session):
        motion = session.scope.motion
        motion._set_axis_state('Z', AxisState.HOMING)
        threading.Timer(0.1, motion._set_axis_state, ('Z', AxisState.UNKNOWN)).start()

        with pytest.raises(MoveNotCompletedError) as exc:
            motion.wait_until_finished_moving(timeout_s=5.0)

        assert exc.value.axis == 'Z'
        assert exc.value.reason == 'faulted'

    def test_a_wait_that_runs_out_raises_and_leaves_the_axis_to_its_move(self, session):
        motion = session.scope.motion
        motion._set_axis_state('Z', AxisState.HOMING)
        try:
            with pytest.raises(MoveNotCompletedError) as exc:
                motion.wait_until_finished_moving(timeout_s=0.2)
            assert exc.value.reason == 'still_moving'
            assert motion.get_axis_state('Z') == AxisState.HOMING
        finally:
            motion._set_axis_state('Z', AxisState.IDLE)

    def test_axes_never_homed_do_not_fail_a_wait_on_the_one_moving(self, session):
        motion = session.scope.motion
        motion._set_axis_state('X', AxisState.UNKNOWN)
        motion._set_axis_state('Y', AxisState.UNKNOWN)
        motion.start_move_absolute('Z', _z_target(motion))

        motion.wait_until_finished_moving(timeout_s=_TRAVEL_WAIT_S)

        assert motion.get_axis_state('Z') == AxisState.IDLE

    def test_a_stop_is_not_a_failure_here(self, session):
        motion = session.scope.motion
        motion._driver.set_timing_mode('realistic')
        move = motion.start_move_absolute('X', _far_x(motion))
        motion.stop_motion()

        motion.wait_until_finished_moving(timeout_s=_TRAVEL_WAIT_S)

        with pytest.raises(MoveNotCompletedError) as exc:
            move.wait()
        assert exc.value.reason == 'stopped'


class TestGoingToAStepWaitsOffTheLane:
    def _protocol_far_from_home(self, session):
        from tests.test_going_to_a_step_is_the_sessions_move import _PLATE, _protocol, _step

        objective = session.scope.runtime_state.get_current_objective_id()
        return _protocol(_PLATE, _step(20.0, 20.0, 5000.0, objective))

    def test_a_move_that_faults_on_the_way_raises_to_the_caller(self, session, monkeypatch):
        motion = session.scope.motion
        protocol = self._protocol_far_from_home(session)
        _lose_the_board_to_the_monitor(motion, monkeypatch)

        with pytest.raises(MoveNotCompletedError) as exc:
            session.go_to_step(protocol, 0)

        assert exc.value.reason == 'board_lost'

    def test_an_led_toggled_while_the_stage_travels_is_not_held_by_it(self, session):
        motion = session.scope.motion
        motion._driver.set_timing_mode('realistic')
        protocol = self._protocol_far_from_home(session)
        errors = []

        def _go():
            try:
                session.go_to_step(protocol, 0)
            except Exception as e:  # reported by the assertion below
                errors.append(e)

        going = threading.Thread(target=_go)
        going.start()
        _until(lambda: motion.get_axis_state('X') == AxisState.MOVING)

        session.scope.illumination.led_on('BF', 10.0)
        still_travelling = motion.get_axis_state('X') == AxisState.MOVING

        going.join(timeout=_TRAVEL_WAIT_S)
        assert errors == []
        assert still_travelling, 'the LED waited for the step to arrive'
