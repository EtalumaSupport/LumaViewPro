# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A stage interlock's refusal reaches the caller as the API's refusal, the axes truthful.

A stage that guards motion with its own inputs (the LS720's TMCM-6110 reads
its lid before every X or Y move and at every poll of a home, and its power
when a home starts) raises ``MotionInterlockError`` from the driver. The API
had no refusal for it: the driver call sites failed the drive, so a move the
lid refused before anything moved marked its axis UNKNOWN and said the board
did not take the command. Now a refusal before anything moved is
``HardwareCommandRefusedError`` naming the interlock, every axis keeping the
state it had; a lid opened while a home moved fails the home naming the lid;
a Stop that ends a home is named as one.

Each test drives a real ``Lumascope(simulate=True)`` through the public
members, with the simulated board standing in for one that has interlocks.
"""

from __future__ import annotations

import threading
import time

import pytest

from drivers.exceptions import MotionInterlockError
from modules.exceptions import (
    HARDWARE_STATE_REASONS,
    INTERLOCK_REASONS,
    HardwareCommandRefusedError,
    HomingFailedError,
    MoveNotCompletedError,
)
from modules.lumascope_api import AxisState
from modules.sequential_io_executor import IOTask
from tests.scope_fakes import build_scope, home_sim_scope

LID_WORDS = "The microscope's lid is open. Close it to move or home the stage."
POWER_WORDS = "The stage has no power. Check the stage's power supply, then home."


@pytest.fixture
def scope():
    scope = home_sim_scope(build_scope(simulate=True, sim_model='LS850', register_atexit=False))
    yield scope
    scope.disconnect()


def _refuse(reason, *, moved=False, stopped=True):
    def refused(*args, **kwargs):
        raise MotionInterlockError(reason, moved=moved, stopped=stopped)

    return refused


class TestTheRefusalsWords:
    def test_each_interlock_reason_has_its_title_and_words(self):
        lid = HardwareCommandRefusedError('lid_open', 'move_absolute')
        power = HardwareCommandRefusedError('stage_unpowered', 'home')

        assert (lid.title, str(lid)) == ('Lid Open', LID_WORDS)
        assert (power.title, str(power)) == ('No Stage Power', POWER_WORDS)

    def test_the_hardware_state_reasons_are_no_board_and_the_interlocks(self):
        assert {'lid_open', 'stage_unpowered'} == INTERLOCK_REASONS
        assert {'not_connected', 'lid_open', 'stage_unpowered'} == HARDWARE_STATE_REASONS

    def test_a_home_the_lid_ended_and_one_a_stop_ended_say_so(self):
        lid = HomingFailedError('ALL', 'lid_open', ('X', 'Y', 'Z'))
        stopped = HomingFailedError('Z', 'stopped', ('Z',))

        assert str(lid) == "Homing stopped: the microscope's lid was opened. Position is unknown."
        assert str(stopped) == 'Z axis homing was stopped. Position is unknown.'


class TestAMoveTheLidRefuses:
    @pytest.mark.parametrize('member', ['move_absolute', 'move_relative'])
    def test_is_refused_naming_the_lid_and_the_axis_stays_known(self, scope, monkeypatch, member):
        monkeypatch.setattr(scope._motion_driver, 'move_abs_pos', _refuse('lid_open'))

        with pytest.raises(HardwareCommandRefusedError) as raised:
            getattr(scope.motion, member)('X', 1000.0)

        assert (raised.value.reason, raised.value.member) == ('lid_open', member)
        assert isinstance(raised.value.__cause__, MotionInterlockError)
        assert scope.motion.get_axis_state('X') == AxisState.IDLE
        assert scope.motion.position_is_known('X')

    def test_a_z_move_in_flight_that_the_refusal_stopped_ends_idle_where_it_stopped(
        self, scope, monkeypatch
    ):
        board = scope._motion_driver
        board.set_timing_mode('realistic')
        start_z = scope.motion.get_current_position('Z')
        scope.motion.start_move_absolute('Z', start_z + 2000.0)
        assert scope.motion.get_axis_state('Z') == AxisState.MOVING

        def lid_open_stops_every_motor(*args, **kwargs):
            # What the 6110's driver does on an open lid: the one stop on
            # every axis, then the refusal.
            board._update_actual('Z')
            board.exchange_command('STOP')
            board._move_end_time['Z'] = 0.0
            raise MotionInterlockError('lid_open', moved=False, stopped=True)

        monkeypatch.setattr(board, 'move_abs_pos', lid_open_stops_every_motor)
        with pytest.raises(HardwareCommandRefusedError):
            scope.motion.move_absolute('X', 1000.0)

        scope.motion.wait_until_finished_moving(timeout_s=10)
        assert scope.motion.get_axis_state('Z') == AxisState.IDLE
        stopped_at = scope.motion.get_current_position('Z')
        assert start_z <= stopped_at < start_z + 2000.0

    def test_a_move_in_flight_on_the_refused_axis_ends_idle_where_it_stopped(
        self, scope, monkeypatch
    ):
        # The move body disarms its axis before the drive; a refusal sent no
        # drive, so the axis must be armed again at the move in flight, or
        # the monitor never ends that move and it stays MOVING.
        board = scope._motion_driver
        board.set_timing_mode('realistic')
        start_z = scope.motion.get_current_position('Z')
        scope.motion.start_move_absolute('Z', start_z + 2000.0)
        assert scope.motion.get_axis_state('Z') == AxisState.MOVING

        def lid_open_stops_every_motor(*args, **kwargs):
            board._update_actual('Z')
            board.exchange_command('STOP')
            board._move_end_time['Z'] = 0.0
            raise MotionInterlockError('lid_open', moved=False, stopped=True)

        monkeypatch.setattr(board, 'move_abs_pos', lid_open_stops_every_motor)
        with pytest.raises(HardwareCommandRefusedError):
            scope.motion.move_absolute('Z', start_z + 1000.0)

        scope.motion.wait_until_finished_moving(timeout_s=5)
        assert scope.motion.get_axis_state('Z') == AxisState.IDLE
        stopped_at = scope.motion.get_current_position('Z')
        assert start_z <= stopped_at < start_z + 2000.0

    def test_a_refusal_after_the_command_moved_fails_the_drive(self, scope, monkeypatch):
        monkeypatch.setattr(scope._motion_driver, 'move_abs_pos', _refuse('lid_open', moved=True))

        with pytest.raises(MoveNotCompletedError) as raised:
            scope.motion.move_absolute('X', 1000.0)

        assert raised.value.reason == 'driver_failed'
        assert scope.motion.get_axis_state('X') == AxisState.UNKNOWN


class TestAHomeTheInterlockRefuses:
    @pytest.mark.parametrize('reason', sorted(INTERLOCK_REASONS))
    @pytest.mark.parametrize('route, call', [('ALL', 'home'), ('Z', 'zhome')])
    def test_before_anything_moved_is_refused_and_every_axis_keeps_its_state(
        self, scope, monkeypatch, reason, route, call
    ):
        before = {axis: scope.motion.get_axis_state(axis) for axis in scope.capabilities.axes}
        monkeypatch.setattr(scope._motion_driver, call, _refuse(reason, stopped=False))

        with pytest.raises(HardwareCommandRefusedError) as raised:
            scope.motion.home(route)

        assert (raised.value.reason, raised.value.member) == (reason, 'home')
        assert {axis: scope.motion.get_axis_state(axis) for axis in before} == before

    def test_a_turret_keeps_its_slot(self, monkeypatch):
        scope = home_sim_scope(
            build_scope(simulate=True, sim_model='LS850T', register_atexit=False)
        )
        try:
            assert scope.motion.get_turret_slot() == 1
            monkeypatch.setattr(scope._motion_driver, 'home', _refuse('lid_open', stopped=False))

            with pytest.raises(HardwareCommandRefusedError):
                scope.motion.home('ALL')

            assert scope.motion.get_turret_slot() == 1
        finally:
            scope.disconnect()

    def test_an_unhomed_axis_refused_stays_unknown_not_homed(self, monkeypatch):
        scope = build_scope(simulate=True, sim_model='LS850', register_atexit=False)
        try:
            monkeypatch.setattr(scope._motion_driver, 'home', _refuse('lid_open', stopped=False))

            with pytest.raises(HardwareCommandRefusedError):
                scope.motion.home('ALL')

            assert scope.motion.get_axis_state('X') == AxisState.UNKNOWN
        finally:
            scope.disconnect()

    @pytest.mark.parametrize('route, call', [('ALL', 'home'), ('Z', 'zhome')])
    def test_a_lid_opened_while_it_moved_fails_it_naming_the_lid(
        self, scope, monkeypatch, route, call
    ):
        monkeypatch.setattr(scope._motion_driver, call, _refuse('lid_open', moved=True))

        with pytest.raises(HomingFailedError) as raised:
            scope.motion.home(route)

        assert raised.value.reason == 'lid_open'
        assert 'lid was opened' in str(raised.value)
        for axis in raised.value.axes:
            assert scope.motion.get_axis_state(axis) == AxisState.UNKNOWN

    @pytest.mark.parametrize('route, call', [('ALL', 'home'), ('Z', 'zhome')])
    def test_a_stop_that_ended_it_is_named_as_a_stop(self, scope, monkeypatch, route, call):
        def stopped_mid_home():
            scope.motion.stop_motion()
            raise RuntimeError('the home was aborted')

        monkeypatch.setattr(scope._motion_driver, call, stopped_mid_home)

        with pytest.raises(HomingFailedError) as raised:
            scope.motion.home(route)

        assert raised.value.reason == 'stopped'
        for axis in raised.value.axes:
            assert scope.motion.get_axis_state(axis) == AxisState.UNKNOWN


class TestTheInterlocksRead:
    def test_a_board_without_interlocks_answers_none(self, scope):
        assert scope.motion.interlocks() == frozenset()

    def test_is_answered_while_the_io_lane_is_busy(self, scope, monkeypatch):
        monkeypatch.setattr(scope._motion_driver, 'interlocks', lambda: frozenset({'lid_open'}))
        running, release = threading.Event(), threading.Event()

        def busy():
            running.set()
            release.wait(10)

        done = scope._io_executor.put(IOTask(action=busy), return_future=True)
        assert running.wait(5), 'the lane never started the task holding it'
        try:
            started = time.monotonic()
            assert scope.motion.interlocks() == {'lid_open'}
            assert time.monotonic() - started < 1.0
        finally:
            release.set()
            done.result(timeout=5)
