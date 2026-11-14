# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A person's home puts the turret and the focus back when they ask it to.

``MotionAPI.home``: a home leaves Z at the bottom and the turret on slot 1,
unless it puts back -- as the stored setting ``motion.restore_after_home``
says, or as the call's ``put_back`` says for that one home. Putting back
returns the turret to the slot it was on and Z to its target, each only when
it was known before the home. The startup motion always homes plainly, so
every session starts on slot 1.

Each test runs a simulated LS850T (X, Y, Z and a turret) through the API and
reads the result through ``turret_status``, ``target_positions`` and the
simulated board's own turret target: the recorded slot alone could be right
while the board is not.
"""

import pytest

from _lumascope.modules.exceptions import MoveFaultError, MoveStoppedError
from _lumascope.modules.axes import AxisState
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

_SLOT = 3
_FOCUS_UM = 5000.0
_HOLD_WAIT_S = 5.0
# Past the motion monitor's stall window, with room for a slowed host.
_GIVE_UP_WAIT_S = 8.0


@pytest.fixture
def session(tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        home_sim_scope(session.scope)
        yield session
    finally:
        session.shutdown()


def _on_slot_and_focused(scope):
    """The turret on ``_SLOT`` and Z at ``_FOCUS_UM``; the board's turret target there."""
    scope.motion.move_turret(_SLOT)
    scope.motion.move_z(z_um=_FOCUS_UM)
    return scope._motion_driver.target_pos('T')


def _where(scope):
    return (
        scope.motion.turret_status().slot,
        scope._motion_driver.target_pos('T'),
        scope.motion.target_positions()['Z'],
    )


def test_with_the_setting_on_a_home_puts_the_turret_and_focus_back(session):
    scope = session.scope
    board_slot = _on_slot_and_focused(scope)
    session.update_settings('motion.restore_after_home', True)

    scope.motion.home()

    assert _where(scope) == (_SLOT, board_slot, _FOCUS_UM)


def test_as_shipped_a_home_leaves_slot_1_and_the_focus_at_the_bottom(session):
    scope = session.scope
    _on_slot_and_focused(scope)
    assert session.settings['motion']['restore_after_home'] is False, 'precondition: shipped off'

    scope.motion.home()

    slot, _, z = _where(scope)
    assert slot == 1
    assert z != _FOCUS_UM, 'Z must stay where the home left it'


def test_put_back_decides_one_home_whatever_the_setting(session):
    scope = session.scope
    board_slot = _on_slot_and_focused(scope)

    scope.motion.home(put_back=True)
    assert _where(scope) == (_SLOT, board_slot, _FOCUS_UM), 'True puts back with the setting off'

    session.update_settings('motion.restore_after_home', True)
    scope.motion.home(put_back=False)
    assert _where(scope)[0] == 1, 'False homes plainly with the setting on'
    assert _where(scope)[2] != _FOCUS_UM


def test_a_z_home_puts_the_focus_back(session):
    scope = session.scope
    _on_slot_and_focused(scope)

    scope.motion.home('Z', put_back=True)

    assert scope.motion.target_positions()['Z'] == _FOCUS_UM


def test_a_turret_home_puts_the_turret_back_and_z_with_it(session):
    scope = session.scope
    board_slot = _on_slot_and_focused(scope)

    scope.motion.home('T', put_back=True)

    assert _where(scope) == (_SLOT, board_slot, _FOCUS_UM)


@pytest.mark.slow
def test_a_home_puts_back_no_axis_it_did_not_know(session):
    """Z lost before the home stays where the home leaves it; the known turret goes back."""
    scope = session.scope
    board_slot = _on_slot_and_focused(scope)
    scope.motion._driver.set_timing_mode('realistic')
    held = scope.simulation.hold_axis('Z')
    # Upward, so the move takes no overshoot leg that start_move_z would wait on.
    lost = scope.motion.start_move_z(z_um=_FOCUS_UM * 2)
    held.result(_HOLD_WAIT_S)
    with pytest.raises(MoveFaultError):
        lost.result(timeout=_GIVE_UP_WAIT_S)
    scope.simulation.clear()
    assert scope.motion.get_axis_state('Z') == AxisState.UNKNOWN, 'precondition: Z lost'
    scope.motion._driver.set_timing_mode('instant')

    scope.motion.home(put_back=True)

    slot, board, z = _where(scope)
    assert (slot, board) == (_SLOT, board_slot)
    assert z not in (_FOCUS_UM, _FOCUS_UM * 2), 'a Z not known before the home is not put back'


@pytest.mark.slow
def test_a_stop_during_the_put_back_leaves_every_axis_known(session):
    """The put-back runs after the home, so its stop is a move's, not a lost home."""
    scope = session.scope
    scope.motion.move_z(z_um=_FOCUS_UM)
    scope.motion._driver.set_timing_mode('realistic')
    seen = []

    def stop_once_the_put_back_moves(axis, at):
        # Z goes HOMING for the home, then MOVING only for the put-back: the
        # Stop lands while that move is in flight, on the event, not a clock.
        if axis != 'Z':
            return
        seen.append(at.state)
        if at.state == AxisState.MOVING and AxisState.HOMING in seen:
            scope.motion.stop_motion()

    scope.motion.add_position_listener(stop_once_the_put_back_moves)

    with pytest.raises(MoveStoppedError):
        scope.motion.home('Z', put_back=True)
    assert scope.motion.get_axis_state('Z') == AxisState.IDLE
    assert scope.motion.axes_without_position() == {}


def test_the_startup_motion_starts_on_slot_1_whatever_the_setting(session):
    scope = session.scope
    _on_slot_and_focused(scope)
    session.update_settings('motion.restore_after_home', True)

    session.run_application_session()

    assert scope.motion.turret_status().slot == 1
