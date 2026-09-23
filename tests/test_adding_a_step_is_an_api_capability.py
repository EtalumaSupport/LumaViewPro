# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Adding a step is an API capability, and a click that adds nothing is refused there.

The GUI's Add Step handler used to own the whole decision: it read the
layer configs, ran its own turret pre-check with its own popup, ordered
the channels, and called the protocol module directly once per layer --
and when no layer was set to acquire it returned bare, so the click did
nothing and said nothing. The protocols API now owns the add: it refuses
through its one funnel (one log line, one notification, one typed
exception) and returns the names it inserted, and the Session composes
the inputs from its own settings so a script can add a step the way the
GUI does.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

import modules.config_helpers as config_helpers
from modules.exceptions import ProtocolRunRefusedError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.test_issue_524_extra_z_step_on_objective_change import _empty_protocol_for_add
from tests.test_protocol_execution import scope  # noqa: F401 -- pytest fixture
from tests.test_run_refusal_contract import _capture_notifications

PLATE_POSITION = {'x': 0.0, 'y': 0.0, 'z': 5000.0}
OBJECTIVE = '4x Oly'


def _layer_configs(**acquire_by_layer):
    """The production layer configs with the named layers set to acquire."""
    settings = complete_settings()
    for layer in config_helpers.get_layer_configs(settings):
        settings[layer]['acquire'] = acquire_by_layer.get(layer)
    return config_helpers.get_layer_configs(settings)


def _add(scope, protocol, layer_configs, *, objective_id=OBJECTIVE, **kwargs):
    return scope.protocols.add_step(
        protocol,
        layer_configs=layer_configs,
        stim_configs={},
        plate_position=PLATE_POSITION,
        objective_id=objective_id,
        **kwargs,
    )


@pytest.fixture
def turret_in_a_known_slot(scope):
    """A turreted scope knows its slot only after a turret command: home it,
    as bring-up does, so the add is judged on what these tests are about."""
    assert scope.motion._home_turret_impl()
    return scope


@pytest.mark.usefixtures('turret_in_a_known_slot')
class TestTheApiRefuses:
    def test_no_acquiring_layer_is_refused_once_and_adds_nothing(self, scope, monkeypatch):
        protocol = _empty_protocol_for_add()
        captured = _capture_notifications(monkeypatch)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _add(scope, protocol, _layer_configs(), before_step=0)

        assert excinfo.value.reason == 'no_acquiring_layer'
        assert len(captured) == 1
        assert protocol.num_steps() == 0

    def test_an_unset_turret_objective_is_refused_before_anything_is_read(self, scope, monkeypatch):
        protocol = _empty_protocol_for_add()
        monkeypatch.setattr(
            scope, 'capabilities', dataclasses.replace(scope.capabilities, has_turret=True)
        )
        monkeypatch.setattr(scope.motion, 'is_current_turret_position_objective_set', lambda: False)
        captured = _capture_notifications(monkeypatch)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _add(scope, protocol, _layer_configs(BF='image'), before_step=0)

        assert excinfo.value.reason == 'turret_objective_unset'
        assert len(captured) == 1
        assert protocol.num_steps() == 0

    def test_an_unknown_objective_is_refused_and_adds_nothing(self, scope, monkeypatch):
        # A step records the objective it was taken with; with no one able
        # to say which objective is in the light path, there is nothing
        # true to record.
        protocol = _empty_protocol_for_add()
        captured = _capture_notifications(monkeypatch)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _add(scope, protocol, _layer_configs(BF='image'), objective_id=None, before_step=0)

        assert excinfo.value.reason == 'objective_unknown'
        assert len(captured) == 1
        assert protocol.num_steps() == 0


@pytest.mark.usefixtures('turret_in_a_known_slot')
class TestTheApiAdds:
    def test_one_step_per_acquiring_layer_in_the_channel_order_given(self, scope):
        protocol = _empty_protocol_for_add()

        names = _add(
            scope,
            protocol,
            _layer_configs(BF='image', Blue='image'),
            channel_order=['Blue', 'BF'],
            before_step=0,
        )

        assert len(names) == 2
        assert protocol.steps()['Color'].tolist() == ['Blue', 'BF']
        assert protocol.steps()['Name'].tolist() == names
        assert protocol.steps()['Objective'].tolist() == [OBJECTIVE, OBJECTIVE]

    def test_layers_absent_from_the_order_follow_it(self, scope):
        protocol = _empty_protocol_for_add()

        _add(
            scope,
            protocol,
            _layer_configs(BF='image', Blue='image', Green='image'),
            channel_order=['Green'],
            before_step=0,
        )

        colors = protocol.steps()['Color'].tolist()
        assert colors[0] == 'Green'
        assert set(colors) == {'BF', 'Blue', 'Green'}


@pytest.fixture
def session():
    built = ScopeSession.create(complete_settings(), simulate=True)
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

        names = session.add_step(protocol, before_step=0)

        assert len(names) == 1
        step = protocol.step(idx=0)
        assert step['Color'] == 'BF'
        assert step['Objective'] == session.scope.runtime_state.resolve_current_objective()[0]

    def test_the_session_forwards_the_refusal(self, session, monkeypatch):
        for layer in config_helpers.get_layer_configs(session.settings):
            session.settings[layer]['acquire'] = None
        protocol = session.scope.protocols.create_protocol(
            empty_config=session.get_sequenced_capture_config()
        )
        captured = _capture_notifications(monkeypatch)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            session.add_step(protocol, before_step=0)

        assert excinfo.value.reason == 'no_acquiring_layer'
        assert len(captured) == 1
        assert protocol.num_steps() == 0
