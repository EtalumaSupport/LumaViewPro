# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scope with no Z axis builds no step that autofocuses.

Autofocus moves Z, so a run that asks for it on a scope without a Z
motor is refused. On such a scope the GUI shows no autofocus control,
so a layer's saved autofocus switch -- left on from a configuration used
on a scope with Z -- is one the user can neither see nor clear. Were it
still read, every step and every new protocol would carry it and every
run would be refused. The Session is where the configuration is read
for a step or a run, so it reads the switch as off there, reading
copies and writing nothing back.
"""

from __future__ import annotations

import pytest

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.test_issue_524_extra_z_step_on_objective_change import _empty_protocol_for_add

LAYERS_SAVED_WITH_AUTOFOCUS = ('BF', 'Blue')


def _session(model: str) -> ScopeSession:
    settings = complete_settings()
    settings['microscope'] = model
    for layer in LAYERS_SAVED_WITH_AUTOFOCUS:
        settings[layer]['autofocus'] = True
    settings['BF']['acquire'] = 'image'
    return ScopeSession.create(settings, simulate=True)


@pytest.fixture
def no_z_session():
    session = _session('LS620')
    assert not session.scope.capabilities.has_focus
    return session


def test_an_added_step_does_not_autofocus(no_z_session):
    protocol = _empty_protocol_for_add()

    no_z_session.add_step(protocol, before_step=0)

    assert not protocol.steps()['Auto_Focus'].astype(bool).any()


def test_a_new_protocol_config_does_not_autofocus(no_z_session):
    config = no_z_session.get_sequenced_capture_config()

    assert not any(cfg['autofocus'] for cfg in config['layer_configs'].values())


def test_the_capture_snapshot_does_not_autofocus(no_z_session):
    snapshot = no_z_session.capture_settings_snapshot()

    assert not any(snapshot[layer]['autofocus'] for layer in LAYERS_SAVED_WITH_AUTOFOCUS)


def test_the_saved_switch_is_left_as_the_user_set_it(no_z_session):
    no_z_session.add_step(_empty_protocol_for_add(), before_step=0)
    no_z_session.get_sequenced_capture_config()

    assert all(no_z_session.settings[layer]['autofocus'] for layer in LAYERS_SAVED_WITH_AUTOFOCUS)


def test_a_scope_with_z_keeps_the_saved_switch():
    session = _session('LS850')
    assert session.scope.capabilities.has_focus

    configs = session.get_layer_configs()
    snapshot = session.capture_settings_snapshot()

    for layer in LAYERS_SAVED_WITH_AUTOFOCUS:
        assert configs[layer]['autofocus']
        assert snapshot[layer]['autofocus']
