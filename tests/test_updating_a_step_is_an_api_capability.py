# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Updating a step is an API capability, refused for the same reasons as adding one.

The GUI's Update Step handler used to own the whole decision: it read the
live position and the objective, kept the step's channel when only a stim
config was being edited, ran its own turret pre-check with its own popup,
and wrote the step through the protocol module directly. Nothing asked
whether the position it read was one the scope still knew, so after a
failed home Update Step saved the last real position -- a plausible
number the scope no longer vouches for -- and no script or REST caller
could update a step at all. The protocols API now owns the update, with
Add Step's refusals, and the Session composes its inputs.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

import modules.config_helpers as config_helpers
from modules.exceptions import ProtocolRunRefusedError
from modules.lumascope_api import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_adding_a_step_is_an_api_capability import (  # noqa: F401 -- pytest fixture
    turret_in_a_known_slot,
)
from tests.test_issue_524_extra_z_step_on_objective_change import _empty_protocol_for_add
from tests.test_protocol_execution import scope  # noqa: F401 -- pytest fixture
from tests.test_run_refusal_contract import _capture_notifications

ADDED_AT = {'x': 10.0, 'y': 20.0, 'z': 5000.0}
UPDATED_TO = {'x': 30.0, 'y': 40.0, 'z': 6000.0}
OBJECTIVE = '4x Oly'


def _layer_configs(**stim_enabled_by_layer):
    """The production layer configs, BF and Blue set to acquire; the named
    layers' stim config enabled."""
    settings = complete_settings()
    for layer in config_helpers.get_layer_configs(settings):
        settings[layer]['acquire'] = None
    settings['BF']['acquire'] = 'image'
    settings['Blue']['acquire'] = 'image'
    for layer, enabled in stim_enabled_by_layer.items():
        settings[layer]['stim_config'] = {**settings[layer]['stim_config'], 'enabled': enabled}
    return config_helpers.get_layer_configs(settings)


def _protocol_with_one_bf_step(scope):
    protocol = _empty_protocol_for_add()
    scope.protocols.add_step(
        protocol,
        layer_configs={'BF': _layer_configs()['BF']},
        stim_configs={},
        plate_position=ADDED_AT,
        objective_id=OBJECTIVE,
        before_step=0,
    )
    return protocol


def _update(
    scope,
    protocol,
    *,
    layer='Blue',
    objective_id=OBJECTIVE,
    label=None,
    layer_configs=None,
    stim_configs=None,
):
    return scope.protocols.update_step(
        protocol,
        0,
        layer=layer,
        layer_configs=layer_configs if layer_configs is not None else _layer_configs(),
        stim_configs=stim_configs if stim_configs is not None else {},
        plate_position=UPDATED_TO,
        objective_id=objective_id,
        label=label,
    )


def _step(protocol):
    return protocol.step(idx=0).to_dict()


@pytest.mark.usefixtures('turret_in_a_known_slot')
class TestTheApiRefuses:
    def test_an_unknown_axis_position_is_refused_and_the_step_is_unchanged(
        self, scope, monkeypatch
    ):
        # The defect: after a failed home the position read keeps answering
        # the last real position, and Update Step saved it.
        protocol = _protocol_with_one_bf_step(scope)
        before = _step(protocol)
        with scope.motion._axis_state_lock:
            scope.motion._axis_state['X'] = AxisState.UNKNOWN
        monkeypatch.setattr(scope.motion, 'is_current_turret_position_objective_set', lambda: False)
        captured = _capture_notifications(monkeypatch)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _update(scope, protocol)

        assert excinfo.value.reason == 'step_position_unknown'
        assert excinfo.value.message == (
            'Cannot update the step. The X position is unknown. '
            'Home the scope, then update the step.'
        )
        assert len(captured) == 1
        assert _step(protocol) == before

    def test_an_unset_turret_objective_is_refused_and_the_step_is_unchanged(
        self, scope, monkeypatch
    ):
        protocol = _protocol_with_one_bf_step(scope)
        before = _step(protocol)
        monkeypatch.setattr(
            scope, 'capabilities', dataclasses.replace(scope.capabilities, has_turret=True)
        )
        monkeypatch.setattr(scope.motion, 'is_current_turret_position_objective_set', lambda: False)
        captured = _capture_notifications(monkeypatch)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _update(scope, protocol)

        assert excinfo.value.reason == 'turret_objective_unset'
        assert len(captured) == 1
        assert _step(protocol) == before

    def test_an_unknown_objective_is_refused_and_the_step_is_unchanged(self, scope, monkeypatch):
        protocol = _protocol_with_one_bf_step(scope)
        before = _step(protocol)
        captured = _capture_notifications(monkeypatch)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _update(scope, protocol, objective_id=None)

        assert excinfo.value.reason == 'objective_unknown'
        assert len(captured) == 1
        assert _step(protocol) == before


@pytest.mark.usefixtures('turret_in_a_known_slot')
class TestTheApiUpdates:
    def test_the_step_takes_the_position_channel_and_objective_given(self, scope):
        protocol = _protocol_with_one_bf_step(scope)

        name = _update(scope, protocol, objective_id='10x Oly')

        step = _step(protocol)
        assert (step['X'], step['Y'], step['Z']) == (30.0, 40.0, 6000.0)
        assert step['Color'] == 'Blue'
        assert step['Objective'] == '10x Oly'
        assert name == step['Name']

    def test_a_label_renames_and_none_keeps_the_label(self, scope):
        protocol = _protocol_with_one_bf_step(scope)
        original_label = _step(protocol)['Label']

        _update(scope, protocol, label=None)
        assert _step(protocol)['Label'] == original_label

        _update(scope, protocol, label='mine')
        assert _step(protocol)['Label'] == 'mine'
        assert not _step(protocol)['Auto_Named']

    def test_a_layer_with_its_stim_enabled_keeps_the_steps_own_channel(self, scope):
        # Editing a stim config from another layer's drawer is a stim edit,
        # not a channel change: the step keeps acquiring what it acquired.
        protocol = _protocol_with_one_bf_step(scope)

        _update(scope, protocol, layer='Blue', layer_configs=_layer_configs(Blue=True))

        assert _step(protocol)['Color'] == 'BF'

    def test_an_enabled_stim_channel_with_no_frequency_is_saved_disabled(self, scope):
        # The same rule an add applies: a stim channel that cannot pulse is
        # not saved as one that will.
        protocol = _protocol_with_one_bf_step(scope)
        stim = {'enabled': True, 'frequency': 0, 'illumination_ma': 50.0, 'exposure': 10.0}

        _update(scope, protocol, stim_configs={'Blue': stim})

        assert _step(protocol)['Stim_Config']['Blue']['enabled'] is False


@pytest.fixture
def session():
    built = ScopeSession.create(complete_settings(), simulate=True)
    home_sim_scope(built.scope)
    yield built
    try:
        built.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


def _session_protocol(session):
    for layer in config_helpers.get_layer_configs(session.settings):
        session.settings[layer]['acquire'] = None
    session.settings['BF']['acquire'] = 'image'
    protocol = session.scope.protocols.create_protocol(
        empty_config=session.get_sequenced_capture_config()
    )
    session.add_step(protocol, before_step=0)
    return protocol


class TestAScriptUpdatesAStepThroughTheSession:
    def test_the_session_composes_the_update_from_its_own_settings_and_position(self, session):
        protocol = _session_protocol(session)
        session.scope._motion_driver.set_timing_mode('instant')
        session.scope.motion.move_absolute('X', 20000.0, wait_until_complete=True)

        session.update_step(protocol, 0, layer='BF')

        step = _step(protocol)
        position = session.get_current_plate_position()
        assert (step['X'], step['Y'], step['Z']) == (position['x'], position['y'], position['z'])
        assert step['Objective'] == session.scope.runtime_state.resolve_current_objective()[0]

    def test_the_session_passes_the_rename_through(self, session):
        protocol = _session_protocol(session)

        session.update_step(protocol, 0, layer='BF', label='mine')

        assert _step(protocol)['Label'] == 'mine'

    def test_a_lost_position_is_refused_through_the_session(self, session, monkeypatch):
        # The GUI case: the scope was homed, then lost its reference. The
        # position read still answers the last real position.
        protocol = _session_protocol(session)
        before = _step(protocol)
        motion = session.scope.motion
        with motion._axis_state_lock:
            for axis in ('X', 'Y', 'Z'):
                motion._axis_state[axis] = AxisState.UNKNOWN
        captured = _capture_notifications(monkeypatch)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            session.update_step(protocol, 0, layer='BF')

        assert excinfo.value.reason == 'step_position_unknown'
        assert len(captured) == 1
        assert _step(protocol) == before
