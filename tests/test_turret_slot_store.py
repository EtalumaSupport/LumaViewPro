# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The turret slot in the light path has one store, and it tells the truth.

The turret has no encoder. The only thing that can say which slot is in the
light path is the last turret command that returned without error: the
turret move, or a home. ``MotionAPI.get_turret_slot`` answers from that
record, and from nothing else -- never the controller's step counter, which
reports steps issued and reads a whole slot halfway between two.

The record is unknown from the moment a turret command starts until it
succeeds, after any failure, after a stop lands on it, and whenever the
turret's position is lost. Nothing else may move the turret: the public
generic movers refuse T, and the diagnostics channel refuses every motor
verb that moves, stops or repositions a motor, so the support report homes
through the motion API.
"""

import threading
import time

import pytest

from modules.exceptions import MoveNotCompletedError
from modules.lumascope_api.motion import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


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
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def _wait_for_state(motion, axis, state, timeout=5.0):
    deadline = time.monotonic() + timeout
    while motion.get_axis_state(axis) != state:
        if time.monotonic() > deadline:
            raise AssertionError(f'{axis} never reached {state}')
        time.sleep(0.005)


def _move_turret_in_thread(motion, slot):
    outcome = {}

    def _run():
        try:
            motion.move_turret(slot)
            outcome['returned'] = True
        except Exception as e:
            outcome['raised'] = e

    t = threading.Thread(target=_run)
    t.start()
    return t, outcome


class TestTheCommandedSlot:
    def test_a_home_leaves_slot_1(self, session):
        assert session.scope.motion.get_turret_slot() == 1

    def test_a_turret_move_records_its_slot_and_the_lookup_reads_it(self, session):
        motion = session.scope.motion
        motion.move_turret(2)
        assert motion.get_turret_slot() == 2
        assert motion.get_turret_position_for_objective_id('4x Oly') == 2
        assert motion.is_current_turret_position_objective_set() is True

    def test_a_stalled_turret_move_leaves_no_slot(self, session, monkeypatch):
        motion = session.scope.motion
        real_status = motion.get_target_status
        monkeypatch.setattr(
            motion, 'get_target_status', lambda ax: False if ax == 'T' else real_status(ax)
        )
        monkeypatch.setattr(motion, '_MOTION_SETTLE_TIMEOUT_S', 0.3)

        with pytest.raises(MoveNotCompletedError):
            motion.move_turret(2)

        assert motion.get_turret_slot() is None
        # An unknown slot has no objective to name; asking must not crash.
        assert motion.is_current_turret_position_objective_set() is False

    def test_losing_the_turret_position_loses_the_slot(self, session):
        motion = session.scope.motion
        motion._set_axis_state('T', AxisState.UNKNOWN)
        assert motion.get_turret_slot() is None


class TestAStopIsNotAnArrival:
    def _halt_on_stop(self, motion, monkeypatch, *, firmware_has_stop=True):
        """The sim's STOP is a no-op; stand in for the firmware's, which
        sets target = actual so the next status poll reports "reached"."""
        stopped = threading.Event()
        real_status = motion.get_target_status
        monkeypatch.setattr(
            motion,
            'get_target_status',
            lambda ax: stopped.is_set() if ax == 'T' else real_status(ax),
        )

        def _motor_stop():
            stopped.set()
            return firmware_has_stop

        monkeypatch.setattr(motion._driver, 'motor_stop', _motor_stop)

    def test_a_stop_during_a_turret_move_raises_and_leaves_no_slot(self, session, monkeypatch):
        motion = session.scope.motion
        self._halt_on_stop(motion, monkeypatch)

        t, outcome = _move_turret_in_thread(motion, 2)
        _wait_for_state(motion, 'T', AxisState.MOVING)
        motion.stop_motion()
        t.join(timeout=10)

        assert isinstance(outcome.get('raised'), MoveNotCompletedError), outcome
        assert outcome['raised'].reason == 'stopped'
        assert motion.get_turret_slot() is None

    def test_firmware_without_stop_does_not_turn_an_arrival_into_a_stop(self, session, monkeypatch):
        """Field firmware has no STOP: nothing was stopped, so the move that
        arrives is reported as arrived."""
        motion = session.scope.motion
        self._halt_on_stop(motion, monkeypatch, firmware_has_stop=False)

        t, outcome = _move_turret_in_thread(motion, 2)
        _wait_for_state(motion, 'T', AxisState.MOVING)
        motion.stop_motion()
        t.join(timeout=10)

        assert outcome == {'returned': True}
        assert motion.get_turret_slot() == 2


class TestNothingElseMovesTheTurret:
    @pytest.mark.parametrize(
        'call',
        [
            lambda m: m.move_absolute('T', 2),
            lambda m: m.move_absolute_async('T', 2),
            lambda m: m.move_relative('T', 1),
            lambda m: m.move_relative_async('T', 1),
        ],
        ids=['move_absolute', 'move_absolute_async', 'move_relative', 'move_relative_async'],
    )
    def test_the_generic_doors_refuse_the_turret(self, session, call):
        motion = session.scope.motion
        with pytest.raises(ValueError, match='move_turret'):
            call(motion)
        assert motion.get_turret_slot() == 1
        assert motion.get_axis_state('T') == AxisState.IDLE

    @pytest.mark.parametrize(
        'command',
        [
            'HOME',
            'thome',
            ' ZHOME ',
            'CENTER',
            'STOP',
            'TARGET_WT80000',
            'ACTUAL_WZ0',
            'xTARGET_WX5',
            'SPIZ0xA100000000',
            'SPIZ',
        ],
    )
    def test_the_diagnostics_channel_refuses_motor_verbs(self, session, command):
        diag = session.scope.diagnostics
        with pytest.raises(ValueError, match='motion API'):
            diag.send_diagnostic_command('motor', command)
        with pytest.raises(ValueError, match='motion API'):
            diag.send_diagnostic_command_multiline('motor', command)

    @pytest.mark.parametrize('command', ['INFO', 'ACTUAL_RT', 'TARGET_RZ', 'SPIZ0x3500000000'])
    def test_the_diagnostics_channel_still_carries_reads(self, session, command):
        assert isinstance(session.scope.diagnostics.send_diagnostic_command('motor', command), str)

    def test_the_led_board_is_not_the_motor_board(self, session):
        assert isinstance(session.scope.diagnostics.send_diagnostic_command('led', 'HOME'), str)

    def test_the_support_report_homes_through_the_motion_api(self, session):
        from modules.tech_support_report import TechSupportReport

        motion = session.scope.motion
        motion.move_turret(2)

        result = TechSupportReport(scope=session.scope).diag.run_homing_test()

        assert result['passed'] is True, result
        # The turret home left the turret in slot 1, and the record says so.
        assert motion.get_turret_slot() == 1


class TestTheRecordIsLostWhenTheTurretIs:
    def test_a_stop_during_a_turret_home_leaves_no_slot(self, session, monkeypatch):
        """The firmware's homing loop answers STOP; a home it cut short did
        not reach slot 1."""
        motion = session.scope.motion
        real_thome = motion._driver.thome

        def _thome_interrupted():
            result = real_thome()
            motion.stop_motion()
            return result

        monkeypatch.setattr(motion._driver, 'thome', _thome_interrupted)

        motion._home_turret_impl()

        assert motion.get_turret_slot() is None

    def test_a_stop_during_the_full_home_leaves_no_slot(self, session, monkeypatch):
        motion = session.scope.motion
        motion.move_turret(2)
        real_home = motion._driver.home

        def _home_interrupted():
            result = real_home()
            motion.stop_motion()
            return result

        monkeypatch.setattr(motion._driver, 'home', _home_interrupted)

        motion._home_impl()

        assert motion.get_turret_slot() is None

    def test_disconnect_loses_the_slot(self, session):
        motion = session.scope.motion
        assert motion.get_turret_slot() == 1
        motion._disconnect()
        assert motion.get_turret_slot() is None


def test_a_run_names_the_slot_not_the_step_counter(tmp_path, monkeypatch):
    """The engineering filename token is the slot in the light path. The
    step counter, parked between two slots, is not read."""
    from tests.test_composite_run_e2e import (
        headless_settings,
        open_composite_session,
        single_run_dir,
    )

    settings = headless_settings(tmp_path)
    with open_composite_session(settings, engineering_mode=True) as (session, runner):
        motion = session.scope.motion
        slot = motion.get_turret_slot()
        assert slot is not None, 'the session left the turret in no known slot'
        real_position = motion.get_current_position

        def _counter_between_slots(axis=None):
            if axis == 'T':
                return slot + 1.5
            return real_position(axis)

        monkeypatch.setattr(motion, 'get_current_position', _counter_between_slots)
        runner.run_composite(sequence_name='eng', parent_dir=str(tmp_path))

    frames = sorted(p.name for p in single_run_dir(tmp_path).glob('*.tiff'))
    assert frames, 'the run wrote no frames'
    for name in frames:
        assert f'Turret{slot}' in name, name
        assert f'Turret{int(slot + 1.5)}' not in name, name
