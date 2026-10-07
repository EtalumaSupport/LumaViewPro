# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""get_target_position answers the commanded target, or the polled position when none was reached.

The target used to live only in the move's ramp profile, cleared the moment
the axis went IDLE, so an arrived axis answered the polled position, a
microstep off the number commanded (100.00234 for a Z of 100.0 on the
simulator). The target is now the move's own and outlives the move; it
gives way to the poll only where no target was reached.
"""

import pytest

from modules.exceptions import MoveNotCompletedError, PositionOutOfRangeError
from modules.lumascope_api.motion import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


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
    motion.move_absolute('Z', 100.0)
    assert motion.get_target_position('Z') == 100.0
    # Proves the answer above is not the poll.
    assert motion.get_current_position('Z') != 100.0


def test_a_stopped_move_answers_the_poll(session):
    motion = session.scope.motion
    driver = motion._driver
    driver.set_timing_mode('realistic')
    hold = driver.hold_travel('X')
    handle = motion.start_move_absolute('X', 60000.0)
    assert hold.reached.wait(10.0)
    motion.stop_motion()
    with pytest.raises(MoveNotCompletedError):
        handle.wait()
    assert motion.get_target_position('X') == motion.get_current_position('X')
    assert motion.get_target_position('X') != 60000.0


def test_a_home_answers_the_poll(session):
    motion = session.scope.motion
    motion.move_absolute('Z', 100.0)
    motion.home('Z')
    assert motion.get_target_position('Z') == motion.get_current_position('Z')


def test_a_refused_move_keeps_the_previous_target(session):
    motion = session.scope.motion
    motion.move_absolute('Z', 100.0)
    with pytest.raises(PositionOutOfRangeError):
        motion.move_absolute('Z', 10_000_000.0)
    assert motion.get_target_position('Z') == 100.0


def test_an_unknown_axis_answers_the_poll(session):
    motion = session.scope.motion
    motion.move_absolute('Z', 100.0)
    motion._set_axis_state('Z', AxisState.UNKNOWN)
    assert motion.get_target_position('Z') == motion.get_current_position('Z')


def test_an_objective_change_returns_z_to_the_commanded_target(turret_session):
    motion = turret_session.scope.motion
    motion.move_absolute('Z', 100.0)
    motion.move_turret(2)
    assert motion.get_target_position('Z') == 100.0
