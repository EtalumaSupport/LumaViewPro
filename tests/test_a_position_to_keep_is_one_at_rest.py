# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A position to keep is read at rest, from an axis that knows where it is.

``MotionAPI.positions_to_keep`` is the one read a caller keeps a position
through: a sweep's centre, a reference written into a result, the Z a
measurement returns to. While an axis moves its number is the motion
monitor's last poll, a place the axis was passing through; while it homes or
has lost its reference, it is the last number the axis reported. Each is
refused, in one snapshot with the number, so a home that starts between a
separate state question and a read cannot pass the number Z left behind
(``focus_z_um`` answered a stale 3000.0 with Z homing that way). An axis the
scope has no motor for is refused, never answered 0.0. Over REST each
refusal is a 409.

The members that keep a position for the user -- ``focus_z_um``, a
bookmark, the session's plate read -- read through it, and are refused as
it is. A homing Z is caught inside a real home, at the read the home makes
while Z is still HOMING; a moving one is held part of the way by the
simulator.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modules.exceptions import (
    AxisMovingError,
    AxisStateUnknownError,
    HardwareCommandRefusedError,
)
from modules.lumascope_api.motion import AxisState
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

# How long a move may take to reach its hold: far past a slowed host, short
# enough that a hold that never engages fails promptly.
_HOLD_WAIT_S = 10.0


def _session(tmp_path, **create):
    settings = complete_settings(live_folder=str(tmp_path))
    s = ScopeSession.create(settings, simulate=True, **create)
    home_sim_scope(s.scope)
    return s


@pytest.fixture
def session(tmp_path):
    s = _session(tmp_path)
    try:
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def _held_z(session):
    """Start a Z move and return it once it has halted part of the way."""
    motion = session.scope.motion
    reached = session.scope.simulation.hold_axis('Z')
    move = motion.start_move_z(motion.get_current_position('Z') + 1000.0)
    assert reached.wait(_HOLD_WAIT_S), 'the move never got part of the way'
    assert motion.get_axis_state('Z') == AxisState.MOVING
    return move


def _while_z_homes(session, ask):
    """Run ``ask`` inside a home of Z, at the read the home makes while Z is HOMING.

    Returns what ``ask`` returned or the exception it raised.
    """
    motion = session.scope.motion
    driver = session.scope._motion_driver
    real_target_pos = driver.target_pos
    answered = []

    def read_with_z_homing(axis):
        if axis == 'Z' and not answered:
            assert motion.get_axis_state('Z') == AxisState.HOMING
            try:
                answered.append(ask())
            except Exception as e:  # the answer under test
                answered.append(e)
        return real_target_pos(axis)

    driver.target_pos = read_with_z_homing
    try:
        motion.home('Z')
    finally:
        driver.target_pos = real_target_pos
    assert answered, 'the home never read Z'
    return answered[0]


def _lose_z(session):
    # A stand-in by design: the fast motor board runs no firmware to fail a
    # home (``fail_home`` refuses it), so Z's lost reference is written as a
    # failed home leaves it.
    session.scope.motion._set_axis_state('Z', AxisState.UNKNOWN)


class TestTheRead:
    def test_at_rest_it_answers_what_axis_positions_answers(self, session):
        motion = session.scope.motion
        motion.move_xy(x_um=20000.0, y_um=30000.0)
        motion.move_z(3000.0)
        kept = motion.positions_to_keep(('X', 'Y', 'Z'), then='keep it')
        snapshot = motion.axis_positions()
        assert kept == {axis: snapshot[axis].position_um for axis in ('X', 'Y', 'Z')}

    def test_a_moving_axis_is_refused(self, session):
        move = _held_z(session)
        with pytest.raises(AxisMovingError) as refused:
            session.scope.motion.positions_to_keep(('Z',), then='keep it')
        assert refused.value.reason == 'axis_moving'
        assert refused.value.axes == ['Z']
        session.scope.simulation.clear()
        move.result(timeout=_HOLD_WAIT_S)

    def test_an_axis_not_asked_may_move(self, session):
        motion = session.scope.motion
        at_rest = motion.axis_positions()['X'].position_um
        move = _held_z(session)
        assert motion.positions_to_keep(('X',), then='keep it') == {'X': at_rest}
        session.scope.simulation.clear()
        move.result(timeout=_HOLD_WAIT_S)

    def test_a_homing_axis_is_refused_never_answered_its_old_number(self, session):
        session.scope.motion.move_z(3000.0)
        answer = _while_z_homes(
            session, lambda: session.scope.motion.positions_to_keep(('Z',), then='keep it')
        )
        assert isinstance(answer, AxisStateUnknownError), answer
        assert answer.reason == 'position_unknown'

    def test_an_axis_that_lost_its_reference_is_refused(self, session):
        _lose_z(session)
        with pytest.raises(AxisStateUnknownError) as refused:
            session.scope.motion.positions_to_keep(('Z',), then='start the sweep')
        assert refused.value.reason == 'position_unknown'
        assert 'start the sweep' in str(refused.value)

    def test_an_axis_the_scope_has_no_motor_for_is_refused_not_answered_zero(self, tmp_path):
        z_only = _session(tmp_path, sim_model='LS820')
        try:
            motion = z_only.scope.motion
            assert 'X' not in z_only.scope.capabilities.axes
            for read in (
                lambda: motion.settled_positions(('X', 'Y')),
                lambda: motion.positions_to_keep(('X', 'Y'), then='keep it'),
            ):
                with pytest.raises(HardwareCommandRefusedError) as refused:
                    read()
                assert refused.value.reason == 'axis_absent'
        finally:
            z_only.shutdown()
            z_only.scope.disconnect()


class TestTheKeepersReadThroughIt:
    """Each member that keeps a position for the user is refused as the read is."""

    def test_nothing_kept_reads_a_homing_z(self, session):
        session.scope.motion.move_z(3000.0)
        keepers = {
            'focus_z_um': lambda: session.scope.protocols.focus_z_um(then='save the focus'),
            'get_current_plate_position': session.get_current_plate_position,
            'save_bookmark': lambda: session.save_bookmark(('Z',)),
        }
        for name, keep in keepers.items():
            answer = _while_z_homes(session, keep)
            assert isinstance(answer, AxisStateUnknownError), (name, answer)
            session.scope.motion.move_z(3000.0)


class TestOverRest:
    @pytest.fixture
    def client(self, session):
        with TestClient(build_app(session)) as client:
            yield client

    def _read(self, client, *axes):
        return client.get(
            '/api/v1/scope/motion/positions_to_keep',
            params=[('axes', axis) for axis in axes] + [('then', 'keep it')],
        )

    def test_at_rest_it_answers(self, session, client):
        session.scope.motion.move_z(3000.0)
        answer = self._read(client, 'Z')
        assert answer.status_code == 200, answer.text
        assert answer.json() == {'Z': session.scope.motion.axis_positions()['Z'].position_um}

    def test_a_moving_axis_is_409(self, session, client):
        move = _held_z(session)
        answer = self._read(client, 'Z')
        assert answer.status_code == 409, answer.text
        assert answer.json()['reason'] == 'axis_moving'
        session.scope.simulation.clear()
        move.result(timeout=_HOLD_WAIT_S)

    def test_an_unknown_axis_is_409(self, session, client):
        _lose_z(session)
        answer = self._read(client, 'Z')
        assert answer.status_code == 409, answer.text
        assert answer.json()['reason'] == 'position_unknown'
