# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refusal over an unknown axis position says what to do, for every axis at once.

The motion gate's refusal used to read "X position is unknown -- home the
scope before moving it, or pass force=True to move anyway": written for a
script, and shown word for word to a person, because the GUI's background
lane uses a typed error's message as the popup body. It also named one
axis, so a gesture refused on three axes read as a problem with one.

Driven on a real simulated scope that has never been homed, which is
exactly the state a failed home leaves every axis in.
"""

import pytest

from modules.exceptions import AxisStateUnknownError
from modules.lumascope_api import AxisState, Lumascope
from tests.scope_fakes import home_sim_scope


@pytest.fixture
def unhomed_scope():
    scope = Lumascope(simulate=True)
    yield scope
    scope.motion._disconnect()


def test_the_gate_refuses_in_words_a_user_acts_on(unhomed_scope):
    with pytest.raises(AxisStateUnknownError) as excinfo:
        unhomed_scope.motion._move_absolute_impl('X', position=1000)

    message = str(excinfo.value)
    assert message == 'The X position is unknown. Home the scope, then move it.'
    assert 'force' not in message
    assert excinfo.value.axis == 'X'
    assert excinfo.value.axes == {'X': AxisState.UNKNOWN}


def test_one_refusal_names_every_axis():
    error = AxisStateUnknownError(
        {'X': AxisState.UNKNOWN, 'Y': AxisState.UNKNOWN, 'Z': AxisState.UNKNOWN}
    )

    assert str(error) == 'The X, Y and Z positions are unknown. Home the scope, then move it.'
    assert error.axis == 'X'


def test_a_homing_axis_is_waited_for_not_homed_again():
    error = AxisStateUnknownError({'Z': AxisState.HOMING}, then='save the focus')

    assert str(error) == 'Z is still homing. Wait for the home to finish, then save the focus.'


def test_a_lost_axis_beside_a_homing_one_asks_for_a_home():
    error = AxisStateUnknownError({'X': AxisState.UNKNOWN, 'Z': AxisState.HOMING})

    assert str(error) == (
        'Z is still homing; the X position is unknown. Home the scope, then move it.'
    )


# ---------------------------------------------------------------------------
# A gesture that needs several axes is refused once, naming them all, before
# anything is submitted or written.
# ---------------------------------------------------------------------------


@pytest.fixture
def notices(monkeypatch):
    import modules.notification_center as nc

    shown = []
    monkeypatch.setattr(
        nc.notifications,
        'warning',
        lambda category, title, message, **kwargs: shown.append((title, message)),
    )
    return shown


def test_a_gesture_is_refused_once_for_every_unknown_axis(unhomed_scope, notices):
    with pytest.raises(AxisStateUnknownError) as excinfo:
        unhomed_scope.motion.refuse_unknown_positions(
            ('X', 'Y', 'Z'), recording=False, then='go to the step'
        )

    assert excinfo.value.axes == {
        'X': AxisState.UNKNOWN,
        'Y': AxisState.UNKNOWN,
        'Z': AxisState.UNKNOWN,
    }
    assert notices == [
        (
            'Scope Not Homed',
            'The X, Y and Z positions are unknown. Home the scope, then go to the step.',
        )
    ]


def test_an_axis_the_scope_does_not_have_is_not_asked_about(unhomed_scope, notices):
    with pytest.raises(AxisStateUnknownError) as excinfo:
        unhomed_scope.motion.refuse_unknown_positions(
            ('X', 'Q'), recording=False, then='move the stage'
        )

    assert list(excinfo.value.axes) == ['X']


def test_a_homed_scope_is_not_refused(unhomed_scope, notices):
    home_sim_scope(unhomed_scope)

    unhomed_scope.motion.refuse_unknown_positions(
        unhomed_scope.capabilities.axes, recording=True, then='save the bookmark'
    )

    assert notices == []


def test_a_homing_axis_may_be_moved_but_not_recorded(unhomed_scope, notices):
    """Its move queues behind the home and lands; its position is not yet one to save."""
    home_sim_scope(unhomed_scope)
    with unhomed_scope.motion._axis_state_lock:
        unhomed_scope.motion._axis_state['Z'] = AxisState.HOMING

    unhomed_scope.motion.refuse_unknown_positions(('Z',), recording=False, then='move it')
    with pytest.raises(AxisStateUnknownError) as excinfo:
        unhomed_scope.motion.refuse_unknown_positions(('Z',), recording=True, then='save the focus')

    assert str(excinfo.value) == (
        'Z is still homing. Wait for the home to finish, then save the focus.'
    )
    assert len(notices) == 1
