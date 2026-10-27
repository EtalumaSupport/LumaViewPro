# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A position is kept only from an axis at rest.

While an axis moves, the number it answers is the motion monitor's last
poll, behind the board by up to a poll. Every save read that number: on the
simulator, Save Focus and the plate read answered 6955.8 with the board at
7000, and a step was recorded at 5990.3 with the scope at 7000, because
the state was asked apart from the read, after it. Each member that keeps
the live position now takes it from one snapshot of state and number, and
refuses ``axis_moving`` while an axis it records moves, with nothing
written. The move is held part of the way by the simulator, so the axis is
provably travelling when the save is asked.
"""

from __future__ import annotations

import pytest

from modules.exceptions import AxisMovingError
from modules.lumascope_api.motion import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

# How long a move may take to reach its hold: far past a slowed host, short
# enough that a hold that never engages fails promptly.
_HOLD_WAIT_S = 10.0


@pytest.fixture
def session(tmp_path):
    settings = complete_settings(live_folder=str(tmp_path))
    settings['BF']['acquire'] = 'image'
    settings['BF']['focus'] = 5000.0
    s = ScopeSession.create(settings, simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def _held_move(session, axis):
    """Start a move on ``axis`` and return it once it has halted part of the way."""
    motion = session.scope.motion
    reached = session.scope.simulation.hold_axis(axis)
    if axis == 'Z':
        move = motion.start_move_z(motion.get_current_position('Z') + 1000.0)
    else:
        move = motion.start_move_xy(x_um=motion.get_current_position('X') + 10000.0)
    assert reached.wait(_HOLD_WAIT_S), 'the move never got part of the way'
    assert motion.get_axis_state(axis) == AxisState.MOVING
    return move


def _arrive(session, move):
    session.scope.simulation.clear()
    move.result(timeout=_HOLD_WAIT_S)


def test_every_member_that_keeps_the_live_position_refuses_while_z_moves(session):
    protocol = session.create_empty_protocol()
    session.add_step(protocol)
    runner = session.create_protocol_runner()
    before = {
        'bookmark': session.bookmark,
        'focus': session.saved_focus_z_um('BF'),
        'steps': protocol.steps().to_dict('records'),
    }
    keepers = {
        'get_current_plate_position': session.get_current_plate_position,
        'save_bookmark': lambda: session.save_bookmark(('Z',)),
        'save_all_bookmarks': session.save_all_bookmarks,
        'save_focus': lambda: session.save_focus(protocol, 'BF'),
        'apply_focus_to_channel_steps': lambda: session.apply_focus_to_channel_steps(
            protocol, 'BF'
        ),
        'focus_z_um': lambda: session.scope.protocols.focus_z_um(then='save the focus'),
        'add_step': lambda: session.add_step(protocol),
        'update_step': lambda: session.update_step(protocol, 0, channel='BF'),
        'new_protocol': session.new_protocol,
        'start_autofocus': lambda: runner.start_autofocus('BF'),
        'start_composite': runner.start_composite,
        'start_zstack': lambda: runner.start_zstack('BF'),
    }

    move = _held_move(session, 'Z')
    offenders = {}
    for name, keep in keepers.items():
        try:
            keep()
        except AxisMovingError as refused:
            if refused.reason != 'axis_moving' or refused.axes != ['Z']:
                offenders[name] = (refused.reason, refused.axes)
        except Exception as other:  # named in the failure, not swallowed
            offenders[name] = repr(other)
        else:
            offenders[name] = 'kept a position while Z moved'
    _arrive(session, move)

    assert offenders == {}
    assert session.bookmark == before['bookmark']
    assert session.saved_focus_z_um('BF') == before['focus']
    assert protocol.steps().to_dict('records') == before['steps']
    assert not runner.sequenced_capture_runner.run_in_progress()


def test_a_z_bookmark_is_saved_while_x_moves_and_an_x_bookmark_is_not(session):
    move = _held_move(session, 'X')
    try:
        saved = session.save_bookmark(('Z',))
        with pytest.raises(AxisMovingError) as refused:
            session.save_bookmark(('X',))
    finally:
        _arrive(session, move)

    assert saved.z_um == pytest.approx(
        session.scope.motion.axis_positions()['Z'].position_um, abs=0.01
    )
    assert refused.value.reason == 'axis_moving'
    assert refused.value.axes == ['X']


def test_once_the_move_has_ended_each_save_keeps_where_the_axis_stopped(session):
    _arrive(session, _held_move(session, 'Z'))
    at_rest = session.scope.motion.axis_positions()['Z']
    assert at_rest.state == AxisState.IDLE

    assert session.save_bookmark(('Z',)).z_um == pytest.approx(at_rest.position_um, abs=0.01)
    assert session.get_current_plate_position().z_um == pytest.approx(at_rest.position_um, abs=0.01)
    assert session.scope.protocols.focus_z_um(then='save the focus') == at_rest.position_um
