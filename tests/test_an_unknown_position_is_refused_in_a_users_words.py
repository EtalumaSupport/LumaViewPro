# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refusal over an unknown axis position says what to do, for every axis at once.

The motion gate's refusal used to read "X position is unknown -- home the
scope before moving it, or pass force=True to move anyway": written for a
script, and shown word for word to a person, because the GUI's background
lane uses a typed error's message as the popup body. It also named one
axis, so a gesture refused on three axes read as a problem with one. A move
is refused while any axis of the scope is lost, so the refusal names every
lost axis, whichever was asked to move.

Driven on a real simulated scope that has never been homed, which is
exactly the state a failed home leaves every axis in.
"""

import pytest
from fastapi.testclient import TestClient

from _lumascope.modules.exceptions import AxisStateUnknownError
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.settings_fixtures import complete_settings
from _lumascope.modules.axes import AxisState
from _lumascope.modules.notification_center import Severity
from tests.scope_fakes import build_scope, home_sim_scope
from tests.test_a_command_for_absent_motion_hardware_is_refused import _session
from _lumascope.modules.channel_record import release_color


@pytest.fixture
def unhomed_scope():
    scope = build_scope(simulate=True)
    yield scope
    scope.motion._disconnect()


def test_the_gate_refuses_in_words_a_user_acts_on(unhomed_scope):
    with pytest.raises(AxisStateUnknownError) as excinfo:
        unhomed_scope.motion._move_absolute_impl('X', position=1000)

    message = str(excinfo.value)
    assert message == 'The X, Y, Z and T positions are unknown. Home the scope, then move it.'
    assert 'force' not in message
    assert excinfo.value.axes_without_position == dict.fromkeys(
        unhomed_scope.capabilities.axes, AxisState.UNKNOWN
    )


def test_one_refusal_names_every_axis():
    error = AxisStateUnknownError(
        {'X': AxisState.UNKNOWN, 'Y': AxisState.UNKNOWN, 'Z': AxisState.UNKNOWN}
    )

    assert str(error) == 'The X, Y and Z positions are unknown. Home the scope, then move it.'
    assert list(error.axes_without_position) == ['X', 'Y', 'Z']


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


def _warnings(centre_posts):
    return [(n.title, n.message) for n in centre_posts if n.severity == Severity.WARNING]


def test_a_gesture_is_refused_naming_every_unknown_axis(unhomed_scope, centre_posts):
    """Raised naming every lost axis of the scope, the turret the gesture did
    not name included, and posting nothing: the gesture's boundary reports."""
    with pytest.raises(AxisStateUnknownError) as excinfo:
        unhomed_scope.motion.refuse_unknown_positions(
            ('X', 'Y', 'Z'), recording=False, then='go to the step'
        )

    assert excinfo.value.axes_without_position == {
        'X': AxisState.UNKNOWN,
        'Y': AxisState.UNKNOWN,
        'Z': AxisState.UNKNOWN,
        'T': AxisState.UNKNOWN,
    }
    assert _warnings(centre_posts) == []


def test_a_person_is_shown_the_refusal_once_in_words_they_act_on(unhomed_scope, centre_posts):
    """Through the GUI's boundary, as a person's gesture meets it."""
    from ui.ui_helpers import run_reported

    run_reported(
        lambda: unhomed_scope.motion.refuse_unknown_positions(
            ('X', 'Y', 'Z'), recording=False, then='go to the step'
        ),
        None,
        'GO_TO_STEP',
    )

    assert _warnings(centre_posts) == [
        (
            'Scope Not Homed',
            'The X, Y, Z and T positions are unknown. Home the scope, then go to the step.',
        )
    ]


def test_an_axis_the_scope_does_not_have_is_not_asked_about(tmp_path):
    session = _session(tmp_path, 'LS850', homed=False)
    try:
        with pytest.raises(AxisStateUnknownError) as excinfo:
            session.scope.motion.refuse_unknown_positions(
                ('X', 'T'), recording=False, then='move the stage'
            )
    finally:
        session.shutdown()

    assert list(excinfo.value.axes_without_position) == ['X', 'Y', 'Z']


def test_a_homed_scope_is_not_refused(unhomed_scope, centre_posts):
    home_sim_scope(unhomed_scope)

    unhomed_scope.motion.refuse_unknown_positions(
        unhomed_scope.capabilities.axes, recording=True, then='save the bookmark'
    )

    assert _warnings(centre_posts) == []


def test_a_homing_axis_may_be_moved_but_not_recorded(unhomed_scope, centre_posts):
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
    assert _warnings(centre_posts) == []


# ---------------------------------------------------------------------------
# A capture whose position is not known is saved without one, never with the
# last number the axis reported.
# ---------------------------------------------------------------------------


def test_the_well_label_of_an_unknown_position_is_none_not_a_stale_well(unhomed_scope):
    # '' is a plate with no wells; None is a position nobody knows. Both are
    # omitted from the file name, and they stay distinct for the caller.
    assert unhomed_scope.runtime_state.get_well_label() is None


def test_a_hyperstack_states_no_position_it_does_not_have(tmp_path):
    from _lumascope.modules.image_utils import build_hyperstack_output_metadata

    nan = float('nan')
    metadata = build_hyperstack_output_metadata(
        # Frames with no record: the case is the positions alone.
        [None, None],
        channel_names=['BF'],
        channel_colors=[release_color(name) for name in ['BF']],
        plane_positions={
            'PositionX': [nan, nan],
            'PositionY': [nan, nan],
            'PositionZ': [3.0, 3.0],
            'DeltaT': None,
        },
        significant_bits=8,
        pixel_size_um=None,
    )

    assert 'PositionX' not in metadata['Plane']
    assert 'PositionY' not in metadata['Plane']
    assert metadata['Plane']['PositionZ'] == [3.0, 3.0]


def test_a_refusal_names_at_least_one_axis():
    """An unknown position with no axis named is no refusal: it is refused where it is built."""
    with pytest.raises(TypeError):
        AxisStateUnknownError({})


def test_a_wire_client_is_told_which_axes_have_no_position(tmp_path):
    """The problem body names each axis without a position, with its state, as data, for a
    refused move and a step that cannot be saved alike: a client branches on the axes, not
    on the sentence."""
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        session.set_channel_settings('BF', acquire='image')
        with TestClient(build_app(session)) as client:
            moved = client.post('/api/v1/scope/motion/start_move_xy', json={'x_um': 1000.0}).json()
            protocol = client.post('/api/v1/create_empty_protocol').json()['handle']
            added = client.post('/api/v1/add_step', json={'protocol': protocol}).json()
    finally:
        session.shutdown()

    unknown = {'X': 'unknown', 'Y': 'unknown', 'Z': 'unknown'}
    assert (moved['reason'], moved.get('axes_without_position')) == ('position_unknown', unknown)
    assert (added['reason'], added.get('axes_without_position')) == (
        'step_position_unknown',
        unknown,
    )
