# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A move to the target an axis already holds sends that axis nothing.

An axis that is IDLE where its last move arrived, at exactly the target
asked for, is not driven again: its board is sent no target, frame validity
is not invalidated for it, and the move's ``Operation`` is done when it is
handed back. A protocol step at the place the previous step left the stage
re-sent every axis's target before, invalidating the frame each time, for
nothing the open-loop board could tell. Anything that ends the arrival -- a
STOP that ended a move, a home, a move elsewhere -- leaves the axis driven
again.

A Z move to a place decides its backlash leg in microsteps: a target a few
nm off the one Z holds rounds to the same microstep and takes no leg. It
reads Z's position only when the leg can fire, and from the arrival when Z
is IDLE where its last move arrived; the board is asked only when Z's place
is not known that way. Each read is a board exchange before the move drives,
about 27 ms on hardware, on every Z move of a protocol run.

Observed on the simulated LS850's board (it has a backlash) through the
commands it is sent and ``FrameValidity.invalidation_counts``.
"""

from __future__ import annotations

import threading

import pytest

from _lumascope.modules.exceptions import MoveStoppedError
from _lumascope.modules.frame_validity import FrameValiditySource
from tests.scope_fakes import build_scope, home_sim_scope

_X_UM = 20_000.0
_Y_UM = 15_000.0
_Z_UM = 1_000.0
_ELSEWHERE_X_UM = 30_000.0


@pytest.fixture
def scope():
    scope = home_sim_scope(build_scope(simulate=True, sim_model='LS850', register_atexit=False))
    yield scope
    scope.disconnect()


def _wire(driver) -> list[str]:
    """Record every command sent to the simulated board, in order."""
    sent: list[str] = []
    real = driver.exchange_command

    def recorded(command, *args, **kwargs):
        sent.append(command)
        return real(command, *args, **kwargs)

    driver.exchange_command = recorded
    return sent


def _wire_but_the_monitor(driver) -> list[str]:
    """Record the commands sent to the simulated board, in order, but the motion monitor's.

    The monitor polls positions on its own thread while an axis moves;
    only the move's own reads decide its backlash leg.
    """
    sent: list[str] = []
    real = driver.exchange_command

    def recorded(command, *args, **kwargs):
        if threading.current_thread().name != 'motion-monitor':
            sent.append(command)
        return real(command, *args, **kwargs)

    driver.exchange_command = recorded
    return sent


def _targets(sent: list[str], axis: str) -> list[str]:
    return [c for c in sent if c.startswith(f'TARGET_W{axis}')]


def test_a_move_to_the_place_every_axis_holds_sends_nothing_and_leaves_frames_valid(scope):
    motion = scope.motion
    motion.move_xy(x_um=_X_UM, y_um=_Y_UM)
    motion.move_z(_Z_UM)
    sent = _wire(scope._motion_driver)
    before = scope.frame_validity.invalidation_counts

    xy = motion.start_move_xy(x_um=_X_UM, y_um=_Y_UM)
    z = motion.start_move_z(_Z_UM)

    assert xy.done() and z.done(), 'a move to the held target was handed back still running'
    assert xy.result() is None and z.result() is None
    assert [c for c in sent if c.startswith('TARGET_W')] == []
    assert scope.frame_validity.invalidation_counts == before


def test_an_axis_whose_move_a_stop_ended_is_driven_to_that_target_again(scope):
    motion = scope.motion
    motion.move_xy(x_um=_X_UM)
    held = scope.simulation.hold_axis('X')
    stopped = motion.start_move_xy(x_um=_ELSEWHERE_X_UM)
    held.result(timeout=10.0)
    motion.stop_motion()
    with pytest.raises(MoveStoppedError):
        stopped.result(timeout=10.0)
    sent = _wire(scope._motion_driver)

    motion.move_xy(x_um=_ELSEWHERE_X_UM)

    assert _targets(sent, 'X') != []
    assert scope._motion_driver.current_pos('X') == pytest.approx(_ELSEWHERE_X_UM, abs=0.1)


def test_an_axis_a_home_took_is_driven_to_its_old_target_again(scope):
    motion = scope.motion
    motion.move_z(_Z_UM)
    before = scope.frame_validity.invalidation_counts[FrameValiditySource.Z_MOVE]
    home_sim_scope(scope)
    homed = scope.frame_validity.invalidation_counts[FrameValiditySource.Z_MOVE]
    sent = _wire(scope._motion_driver)

    motion.move_z(_Z_UM)

    assert _targets(sent, 'Z') != []
    assert homed > before
    assert scope.frame_validity.invalidation_counts[FrameValiditySource.Z_MOVE] > homed


def test_an_axis_that_moved_elsewhere_is_driven_back(scope):
    motion = scope.motion
    motion.move_xy(x_um=_X_UM)
    motion.move_xy(x_um=_ELSEWHERE_X_UM)
    sent = _wire(scope._motion_driver)

    motion.move_xy(x_um=_X_UM)

    assert _targets(sent, 'X') != []


def test_a_z_move_to_the_microstep_z_holds_takes_no_backlash_leg(scope):
    motion = scope.motion
    driver = scope._motion_driver
    assert driver.backlash_um() > 0, 'the board has no backlash, so no move has a leg'
    motion.move_z(_Z_UM)
    held_steps = driver.current_pos_steps('Z')
    # A float just below the position Z reads back, inside its microstep:
    # the read-back is above it, so a leg decided in um fires.
    nearby = driver.current_pos('Z') - 0.001
    assert driver.z_um2ustep(nearby) == held_steps
    sent = _wire(driver)

    motion.move_z(nearby)

    assert _targets(sent, 'Z') == [f'TARGET_WZ{held_steps}']


def test_a_z_move_down_from_an_arrival_takes_its_leg_without_reading_z(scope):
    motion = scope.motion
    driver = scope._motion_driver
    backlash = driver.backlash_um()
    assert backlash > 0, 'the board has no backlash, so no move has a leg'
    motion.move_z(3 * _Z_UM, overshoot_enabled=True)
    sent = _wire_but_the_monitor(driver)

    motion.move_z(_Z_UM, overshoot_enabled=True)

    assert 'ACTUAL_RZ' not in sent
    assert _targets(sent, 'Z') == [
        f'TARGET_WZ{driver.z_um2ustep(_Z_UM - backlash)}',
        f'TARGET_WZ{driver.z_um2ustep(_Z_UM)}',
    ]


def test_a_z_move_whose_leg_cannot_fire_reads_no_z_position(scope):
    """A target within the backlash of the bottom takes no leg wherever Z is,
    so nothing is read to decide it, even with Z's place not known from an
    arrival (the home took it)."""
    motion = scope.motion
    driver = scope._motion_driver
    near_the_bottom = driver.backlash_um() + 50
    sent = _wire_but_the_monitor(driver)

    motion.move_z(near_the_bottom, overshoot_enabled=True)

    assert 'ACTUAL_RZ' not in sent
    assert _targets(sent, 'Z') == [f'TARGET_WZ{driver.z_um2ustep(near_the_bottom)}']
