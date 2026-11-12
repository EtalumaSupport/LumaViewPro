# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""get_target_position answers the commanded target, or the polled position when none was reached; get_target_positions answers it for every axis.

The target used to live only in the move's ramp profile, cleared the moment
the axis went IDLE, so an arrived axis answered the polled position, a
microstep off the number commanded (100.00234 for a Z of 100.0 on the
simulator). The target is now the move's own and outlives the move; it
gives way to the poll only where no target was reached. An axis whose
position is not known has no target and answers None.
"""

import pytest
from fastapi.testclient import TestClient

from _lumascope.drivers.exceptions import HardwareError
from _lumascope.modules.exceptions import MoveNotCompletedError, PositionOutOfRangeError
from _lumascope.modules.scope_parts import Board
from _lumascope.modules.axes import AxisState
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.motion_moves import axis_move


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        s.scope.motion._driver.set_timing_mode('fast')
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


@pytest.fixture
def turret_session(tmp_path):
    s = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            microscope='LS850T',
            objective_confirmed=True,
            turret_objectives={'1': '4x Oly', '2': '10x Oly', '3': None, '4': None},
        ),
        simulate=True,
    )
    try:
        home_sim_scope(s.scope)
        s.scope.motion._driver.set_timing_mode('fast')
        yield s
    finally:
        s.shutdown()


def test_an_arrived_move_answers_its_commanded_target(session):
    motion = session.scope.motion
    motion.move_z(100.0)
    assert motion.get_target_positions()['Z'] == 100.0
    # Proves the answer above is not the poll.
    assert motion.get_current_position('Z') != 100.0


def test_every_axis_answers_its_commanded_target(session):
    motion = session.scope.motion
    for axis, target in (('X', 100.0), ('Y', 200.0), ('Z', 300.0)):
        axis_move(motion, axis, target)
    assert motion.get_target_positions() == {'X': 100.0, 'Y': 200.0, 'Z': 300.0}


def test_a_stopped_move_answers_the_poll(session):
    motion = session.scope.motion
    driver = motion._driver
    driver.set_timing_mode('realistic')
    held = session.scope.simulation.hold_axis('X')
    handle = motion.start_move_xy(x_um=60000.0)
    held.result(10.0)
    motion.stop_motion()
    with pytest.raises(MoveNotCompletedError):
        handle.result()
    assert motion.get_target_positions()['X'] == motion.get_current_position('X')
    assert motion.get_target_positions()['X'] != 60000.0


def test_a_home_answers_the_poll(session):
    motion = session.scope.motion
    motion.move_z(100.0)
    motion.home('Z')
    assert motion.get_target_positions()['Z'] == motion.get_current_position('Z')


def test_a_refused_move_keeps_the_previous_target(session):
    motion = session.scope.motion
    motion.move_z(100.0)
    with pytest.raises(PositionOutOfRangeError):
        motion.move_z(10_000_000.0)
    assert motion.get_target_positions()['Z'] == 100.0


def test_an_axis_whose_position_is_not_known_has_no_target(tmp_path):
    """Unhomed, no axis has been read: each answers None, in Python and over
    REST, as ``axis_positions`` does, never the cache's 0.0 seed."""
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        assert s.scope.motion.get_target_positions() == {'X': None, 'Y': None, 'Z': None}
        with TestClient(build_app(s)) as client:
            answer = client.get('/api/v1/scope/motion/get_target_positions')
        assert answer.json() == {'X': None, 'Y': None, 'Z': None}
    finally:
        s.shutdown()


def test_a_homing_axis_has_no_target(session):
    motion = session.scope.motion
    motion.move_z(100.0)
    # a stand-in by design: the simulator cannot hold a home part-way
    # (hold_axis holds a move), so the store is put where a home puts it.
    motion._set_axis_state('Z', AxisState.HOMING)
    assert motion.get_target_positions()['Z'] is None


def test_a_lost_motor_board_leaves_no_axis_a_target(session):
    """The board's loss is found by an exchange, with nothing written into
    the axis store: every axis answers None, as ``axis_positions`` does."""
    motion = session.scope.motion
    motion.move_z(100.0)
    session.scope.simulation.unplug(Board.MOTOR)
    with pytest.raises(HardwareError):
        motion.get_actual_position('Z')

    assert motion.get_target_positions() == {'X': None, 'Y': None, 'Z': None}


def test_an_objective_change_returns_z_to_the_commanded_target(turret_session):
    motion = turret_session.scope.motion
    motion.move_z(100.0)
    motion.move_turret(2)
    assert motion.get_target_positions()['Z'] == 100.0
