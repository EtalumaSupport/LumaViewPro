"""A request that starts several moves and then fails stops the moves it started.

The clause pinned: ``MotionAPI.start_move_xy``, ``ScopeSession.start_go_to_step``
and ``ScopeSession.go_to_step`` raise with the stage stopped, never while a
move they started still travels with nothing to stop or wait on. Where no
STOP can be sent, or another activity holds the scope, the request's own
failure goes out and nothing is sent; a STOP the board does not take is
what the caller is told.

Each fault is the simulator's own: a dropped reply, a held axis, a refused
camera or LED write.
"""

from __future__ import annotations

import pytest

from _lumascope.drivers.exceptions import BoardPortFailedError
from _lumascope.modules.exceptions import (
    CameraSettingRejected,
    HardwareCommandRefusedError,
    MotorStopFailedError,
    MoveFaultError,
)
from _lumascope.modules.lumascope_api.illumination import LedState
from _lumascope.modules.lumascope_api.motion import HomeTarget
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_going_to_a_step_is_the_sessions_move import _PLATE, _protocol, _step

_X_UM, _Y_UM = 90000.0, 60000.0
_WAIT_S = 30.0


@pytest.fixture
def open_session(tmp_path):
    made = []

    def make(**settings) -> ScopeSession:
        session = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path), **settings),
            simulate=True,
            warn_pre_release=False,
        )
        made.append(session)
        if 'microscope' in settings:
            # the LS720's simulated board is the wire model, which homes as a real one does
            session.scope.motion.home()
        else:
            home_sim_scope(session.scope)
        return session

    yield make
    for session in made:
        session.scope.simulation.clear()
        session.shutdown()


def _drop_y_target(scope, y_um: float) -> None:
    """Y's target write goes unanswered once, so Y's drive fails after X's."""
    steps = scope._motion_driver.position_to_steps('Y', y_um)
    if steps < 0:
        steps += 0x100000000
    scope.simulation.drop_replies('motor', count=1, command=f'TARGET_WY{steps}')


def _a_step_protocol(session, **step):
    objective = session.scope.runtime_state.get_current_objective_id()
    row = _step(10.0, 10.0, 5000.0, objective)
    row.update(step)
    return _protocol(_PLATE, row)


def test_y_refused_after_x_stops_x(open_session):
    session = open_session()
    motion = session.scope.motion
    before = motion.get_current_positions()['X']
    session.scope.simulation.hold_axis('X', 0.5)
    _drop_y_target(session.scope, _Y_UM)

    with pytest.raises(MoveFaultError) as raised:
        motion.start_move_xy(_X_UM, _Y_UM)

    assert (raised.value.reason, raised.value.axis) == ('driver_failed', 'Y')
    assert not motion.is_any_axis_moving()
    assert before < motion.get_current_positions()['X'] < _X_UM


@pytest.mark.parametrize('member', ['start_go_to_step', 'go_to_step'])
def test_a_camera_refusal_after_the_moves_stops_them(open_session, member):
    session = open_session()
    motion = session.scope.motion
    protocol = _a_step_protocol(session, Gain=5.0)
    target_x = session.scope.protocols.step_targets(protocol, 0).x_um
    session.scope.simulation.hold_axis('X', 0.5)
    session.scope.simulation.refuse_camera_write('gain', 'raise', count=5)

    with pytest.raises(CameraSettingRejected):
        getattr(session, member)(protocol, 0)

    assert not motion.is_any_axis_moving()
    assert motion.get_current_positions()['X'] < target_x


def test_a_preview_refusal_after_the_moves_stops_them_and_the_step_is_not_gone_to(open_session):
    """G2 on an LS720, whose preview lights its LED; the re-click is a first visit, so it lights again."""
    session = open_session(microscope='LS720', protocol_led_on=True)
    motion = session.scope.motion
    protocol = _a_step_protocol(session)
    channel = protocol.step_row(step_index=0)['channel']
    # No hold: the wire model travels in real time, so the stage is still on its way
    session.scope.simulation.refuse_led_write(count=1)

    with pytest.raises(BoardPortFailedError):
        session.start_go_to_step(protocol, 0)

    assert not motion.is_any_axis_moving()
    session.scope.simulation.clear()
    session.start_go_to_step(protocol, 0).result(timeout=_WAIT_S)
    lit = [r.name for r in session.scope.illumination.led_states() if r.state == LedState.LIT]
    assert lit == [channel]


def test_a_failure_while_another_activity_holds_the_scope_sends_no_stop(open_session):
    """D4: a home takes the scope between go-to-step's moves and its camera call.

    The STOP is dropped, so a STOP sent would be told as ``MotorStopFailedError``.
    """
    session = open_session()
    motion = session.scope.motion
    protocol = _a_step_protocol(session)
    real_apply = session.apply_channel_camera
    homing = []

    def a_home_takes_the_scope_first(channel):
        # a stand-in by design: the race window between the lane task and the
        # camera call cannot be opened on cue by any simulator member
        homing.append(motion.start_home(HomeTarget.Z))
        return real_apply(channel)

    session.apply_channel_camera = a_home_takes_the_scope_first
    session.scope.simulation.drop_replies('motor', count=1, command='STOP')

    with pytest.raises(HardwareCommandRefusedError) as raised:
        session.start_go_to_step(protocol, 0)

    assert raised.value.reason == 'exclusive_activity_running'
    homing[0].result(timeout=_WAIT_S)


def test_a_stop_the_board_does_not_take_is_what_the_caller_is_told(open_session):
    """D1: the STOP's own error, with the request's failure kept in its chain."""
    session = open_session()
    session.scope.simulation.hold_axis('X', 0.5)
    _drop_y_target(session.scope, _Y_UM)
    session.scope.simulation.drop_replies('motor', count=1, command='STOP')

    with pytest.raises(MotorStopFailedError) as raised:
        session.scope.motion.start_move_xy(_X_UM, _Y_UM)

    chain, link = [], raised.value
    while link is not None:
        chain.append(link)
        link = link.__cause__ or link.__context__
    assert any(isinstance(e, MoveFaultError) and e.reason == 'driver_failed' for e in chain)


def test_a_board_lost_between_the_axes_is_what_the_caller_is_told(open_session, monkeypatch):
    """D3: no STOP can be sent to a board that is gone, so the request's own failure goes out."""
    session = open_session()
    motion = session.scope.motion
    real_drive = motion._drive

    def unplugged_before_y(member, axis, *args, **kwargs):
        # a stand-in by design: the simulator has no member that unplugs the
        # board at one command
        if axis == 'Y':
            session.scope.simulation.unplug('motor')
        return real_drive(member, axis, *args, **kwargs)

    monkeypatch.setattr(motion, '_drive', unplugged_before_y)

    with pytest.raises(MoveFaultError) as raised:
        motion.start_move_xy(_X_UM, _Y_UM)

    assert raised.value.reason == 'board_lost'
