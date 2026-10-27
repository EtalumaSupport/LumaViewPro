# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Adding a step is an API capability, and a click that adds nothing is refused there.

The GUI's Add Step handler used to own the whole decision: it read the
layer configs, ran its own turret pre-check with its own popup, ordered
the channels, and called the protocol module directly once per layer --
and when no layer was set to acquire it returned bare, so the click did
nothing and said nothing. The protocols API now owns the add: it refuses
through its one funnel, a typed exception the caller's boundary reports,
and returns the names it inserted, and the Session composes
the inputs from its own settings so a script can add a step the way the
GUI does.
"""

from __future__ import annotations

import logging

import pytest

import modules.config_helpers as config_helpers
from modules.exceptions import ObjectiveUnknownError, ProtocolRunRefusedError
from modules.lumascope_api import AxisState
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.scope_fakes import TEST_TURRET_OBJECTIVES, configure_turret_like_bringup, home_sim_scope
from tests.test_issue_524_extra_z_step_on_objective_change import _empty_protocol_for_add
from tests.test_protocol_execution import scope  # noqa: F401 -- pytest fixture
from tests.test_run_refusal_contract import _capture_notifications
from modules.coord_transformations import PlateLocation
from tests.stock_channels import LED0, LED1

PLATE_POSITION = PlateLocation(x_mm=0.0, y_mm=0.0, z_um=5000.0)
# The active objective a step records: slot 1's, where the home leaves the turret.
OBJECTIVE = TEST_TURRET_OBJECTIVES[1]


def _layer_configs(**acquire_by_layer):
    """The production layer configs with the named layers set to acquire."""
    settings = complete_settings()
    for layer in config_helpers.get_layer_configs(settings):
        settings[layer]['acquire'] = acquire_by_layer.get(layer)
    return config_helpers.get_layer_configs(settings)


def _add(scope, protocol, layer_configs, **kwargs):
    return scope.protocols.add_step(
        protocol,
        layer_configs=layer_configs,
        stim_configs={},
        plate_position=PLATE_POSITION,
        **kwargs,
    )


def clear_the_slot_in_the_light_path(scope):
    """The turret in slot 1 with no objective assigned to it, as bring-up leaves it."""
    configure_turret_like_bringup(scope, {**TEST_TURRET_OBJECTIVES, 1: None})


@pytest.fixture
def turret_in_a_known_slot(scope):
    """A turreted scope knows its slot only after a turret command: home it,
    as bring-up does, so the add is judged on what these tests are about."""
    scope.motion._home_turret_impl()
    return scope


@pytest.mark.usefixtures('turret_in_a_known_slot')
class TestTheApiRefuses:
    def test_no_acquiring_layer_is_refused_and_adds_nothing(self, scope, centre_posts):
        protocol = _empty_protocol_for_add()
        captured = _capture_notifications(centre_posts)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _add(scope, protocol, _layer_configs(), before_step_index=0)

        assert excinfo.value.reason == 'no_acquiring_layer'
        assert list(captured) == []
        assert protocol.num_steps() == 0

    def test_a_slot_with_no_objective_is_refused_and_adds_nothing(self, scope, centre_posts):
        # A step records the objective it was taken with; a slot that holds
        # none has nothing true to record. The class and reason every other
        # member that needs the objective refuses it with.
        protocol = _empty_protocol_for_add()
        clear_the_slot_in_the_light_path(scope)
        captured = _capture_notifications(centre_posts)

        with pytest.raises(ObjectiveUnknownError) as excinfo:
            _add(scope, protocol, _layer_configs(BF='image'), before_step_index=0)

        assert excinfo.value.reason == 'slot_unassigned'
        assert list(captured) == []
        assert protocol.num_steps() == 0

    def test_an_unknown_axis_position_is_refused_before_anything_else(self, scope, centre_posts):
        # An axis that lost its reference keeps answering the last number it
        # reported, so the position handed in looks real and is not. Asked
        # ahead of the objective, whose advice is wrong when the slot itself
        # is what is unknown.
        protocol = _empty_protocol_for_add()
        clear_the_slot_in_the_light_path(scope)
        with scope.motion._axis_state_lock:
            scope.motion._axis_state['Y'] = AxisState.UNKNOWN
            scope.motion._axis_state['Z'] = AxisState.HOMING
        captured = _capture_notifications(centre_posts)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _add(scope, protocol, _layer_configs(BF='image'), before_step_index=0)

        assert (excinfo.value.reason, excinfo.value.title) == (
            'step_position_unknown',
            'Step Not Saved',
        )
        assert excinfo.value.message == (
            'Cannot add the step. Z is still homing; the Y position is unknown. '
            'Home the scope, then add the step.'
        )
        assert list(captured) == []
        assert protocol.num_steps() == 0


@pytest.mark.usefixtures('turret_in_a_known_slot')
class TestTheApiAdds:
    def test_one_step_per_acquiring_layer_in_the_channel_order_given(self, scope):
        protocol = _empty_protocol_for_add()

        names = _add(
            scope,
            protocol,
            _layer_configs(BF='image', **{LED0: 'image'}),
            channel_order=[LED0, 'BF'],
            before_step_index=0,
        )

        assert len(names) == 2
        assert protocol.steps()['Color'].tolist() == [LED0, 'BF']
        assert protocol.steps()['Name'].tolist() == names
        assert protocol.steps()['Objective'].tolist() == [OBJECTIVE, OBJECTIVE]

    def test_layers_absent_from_the_order_follow_it(self, scope):
        protocol = _empty_protocol_for_add()

        _add(
            scope,
            protocol,
            _layer_configs(BF='image', **{LED0: 'image', LED1: 'image'}),
            channel_order=[LED1],
            before_step_index=0,
        )

        colors = protocol.steps()['Color'].tolist()
        assert colors[0] == LED1
        assert set(colors) == {'BF', LED0, LED1}


@pytest.fixture
def session():
    # Homed, as the application homes at startup: a step is a saved
    # position, and an un-homed scope has none to save.
    built = ScopeSession.create(complete_settings(), simulate=True)
    home_sim_scope(built.scope)
    yield built
    try:
        built.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


class TestAScriptAddsAStepThroughTheSession:
    def test_the_session_composes_the_add_from_its_own_settings(self, session):
        for layer in config_helpers.get_layer_configs(session.settings):
            session.settings[layer]['acquire'] = None
        session.settings['BF']['acquire'] = 'image'
        protocol = session.scope.protocols.create_protocol(
            empty_config=session.get_sequenced_capture_config()
        )

        names = session.add_step(protocol, before_step_index=0)

        assert len(names) == 1
        step = protocol.step_row(step_index=0)
        assert step['Color'] == 'BF'
        assert step['Objective'] == session.scope.runtime_state.resolve_current_objective()[0]

    def test_every_channel_takes_the_current_z_and_keeps_its_saved_focus(self, session):
        # The viewed channel's focus is the better guess for every channel,
        # and each saved focus stays where New starts the next protocol.
        saved = {'BF': 7000.0, LED0: 8000.0, LED1: 9000.0}
        for layer in config_helpers.get_layer_configs(session.settings):
            session.settings[layer]['acquire'] = None
        for layer, focus in saved.items():
            session.settings[layer]['acquire'] = 'image'
            session.save_channel_focus(layer, focus)
        session.scope.motion.move_z(5000.0)
        protocol = session.scope.protocols.create_protocol(
            empty_config=session.get_sequenced_capture_config()
        )

        session.add_step(protocol, before_step_index=0)

        steps = protocol.steps()
        assert sorted(steps['Color']) == sorted(saved)
        assert steps['Z'].tolist() == pytest.approx([5000.0] * len(saved))
        assert {layer: session.saved_focus_z_um(layer) for layer in saved} == saved

    def test_a_script_adds_the_layers_it_names_where_it_names_without_changing_the_settings(
        self, session
    ):
        for layer in config_helpers.get_layer_configs(session.settings):
            session.settings[layer]['acquire'] = None
        before = config_helpers.get_layer_configs(session.settings)
        protocol = session.create_empty_protocol()
        at = PlateLocation(x_mm=20.0, y_mm=15.0, z_um=6000.0)

        names = session.add_step(protocol, channels=(LED0, 'BF'), at=at)

        steps = protocol.steps()
        assert len(names) == 2
        assert steps['Color'].tolist() == [LED0, 'BF']
        assert steps['Acquire'].tolist() == ['image', 'image']
        assert steps[['X', 'Y', 'Z']].values.tolist() == [[20.0, 15.0, 6000.0]] * 2
        assert config_helpers.get_layer_configs(session.settings) == before

    def test_the_session_forwards_the_refusal(self, session, centre_posts):
        for layer in config_helpers.get_layer_configs(session.settings):
            session.settings[layer]['acquire'] = None
        protocol = session.scope.protocols.create_protocol(
            empty_config=session.get_sequenced_capture_config()
        )
        captured = _capture_notifications(centre_posts)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            session.add_step(protocol, before_step_index=0)

        assert excinfo.value.reason == 'no_acquiring_layer'
        assert list(captured) == []
        assert protocol.num_steps() == 0


def _only_acquiring(session, *layers):
    for layer in config_helpers.get_layer_configs(session.settings):
        session.settings[layer]['acquire'] = 'image' if layer in layers else None


def _stack_settings(session, *, range_um, step_size_um):
    session.settings['zstack'] = {
        'range': range_um,
        'step_size': step_size_um,
        'position': 'center',
    }


class TestAnAddCanBeAZStack:
    """``add_step(use_zstacking=True)`` stacks the steps it adds, and only those.

    The GUI's Enable Z-Stack box sits beside Add; an Add that ignored it
    ran flat images with the box ticked.
    """

    def test_the_added_steps_are_stacks_and_the_earlier_step_is_left_alone(self, session):
        _only_acquiring(session, 'BF')
        session.scope.motion.move_z(5000.0)
        protocol = session.scope.protocols.create_protocol(
            empty_config=session.get_sequenced_capture_config()
        )
        session.add_step(protocol)
        earlier = protocol.step_row(step_index=0).to_dict()

        _only_acquiring(session, 'BF', LED0)
        _stack_settings(session, range_um=10.0, step_size_um=5.0)
        names = session.add_step(protocol, use_zstacking=True)

        steps = protocol.steps()
        assert steps.iloc[0].to_dict() == earlier
        added = steps.iloc[1:]
        assert added['Name'].tolist() == names
        for color in ('BF', LED0):
            stack = added[added['Color'] == color].sort_values('Z-Slice')
            assert stack['Z-Slice'].tolist() == [0, 1, 2]
            assert stack['Z'].tolist() == pytest.approx([4995.0, 5000.0, 5005.0])
        # One Add at one place is one group, every channel in it: a run focuses it once.
        assert added['Z-Stack Group ID'].nunique() == 1
        assert added['Z'].tolist() == sorted(added['Z'].tolist()), 'not in the order a run visits'

    def test_a_stack_with_no_extent_adds_nothing(self, session):
        _only_acquiring(session, 'BF')
        _stack_settings(session, range_um=0.0, step_size_um=5.0)
        protocol = session.scope.protocols.create_protocol(
            empty_config=session.get_sequenced_capture_config()
        )

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            session.add_step(protocol, use_zstacking=True)

        assert excinfo.value.reason == 'zstack_not_configured'
        assert protocol.num_steps() == 0

    def test_a_stack_past_the_z_travel_adds_nothing(self, session):
        _only_acquiring(session, 'BF', LED0)
        _stack_settings(session, range_um=10.0, step_size_um=5.0)
        z_max = session.scope.motion.get_axis_limits('Z').max_um
        session.scope.motion.move_z(z_max - 1.0)
        protocol = session.scope.protocols.create_protocol(
            empty_config=session.get_sequenced_capture_config()
        )

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            session.add_step(protocol, use_zstacking=True)

        assert excinfo.value.reason == 'zslices_outside_travel'
        assert 'the new BF step' in str(excinfo.value)
        assert protocol.num_steps() == 0


def test_a_slot_with_no_objective_refuses_a_step_as_it_refuses_a_new_protocol(tmp_path):
    """One state, one class and one reason at every member that needs the objective.

    A step added with the turret's slot unassigned was refused
    ``ProtocolRunRefusedError`` ``turret_objective_unset`` while a new
    protocol in the same state was ``ObjectiveUnknownError``
    ``slot_unassigned``: two URNs over the wire for one fact.
    """
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'), simulate=True
    )
    try:
        home_sim_scope(session.scope)
        session.clear_current_turret_objective()
        protocol = session.create_empty_protocol()

        with pytest.raises(ObjectiveUnknownError) as added:
            session.add_step(protocol, channels=('BF',))
        with pytest.raises(ObjectiveUnknownError) as created:
            session.new_protocol()

        assert added.value.reason == created.value.reason == 'slot_unassigned'
        assert protocol.num_steps() == 0
    finally:
        session.shutdown()
