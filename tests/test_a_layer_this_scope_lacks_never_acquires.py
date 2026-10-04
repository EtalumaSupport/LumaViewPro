# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A layer this scope does not have never acquires, and no step on it runs (#508).

The live configuration could hold ``acquire`` on a layer the connected scope
lacks: a setting saved on another model kept Lumi acquiring on an LS850, and
a protocol's Layer Settings row for Lumi set it acquiring again on load. New
and Add then built Lumi steps, and nothing checked a step's layer against the
scope's, so a run started with them.

Now the store cannot hold it: bring-up turns such a layer off and logs it,
a Layer Settings row for it is dropped and logged, and setting it to acquire
is refused (setting it to acquire nothing is always allowed). Steps that
arrive from outside the store -- a loaded file, a protocol built in memory --
are refused at load and at the run's start, naming the layers and the steps.

Update is refused, as Add is, when the layer the step takes is not set to
acquire: it used to save a step with an empty Acquire that failed at Run.
"""

import pytest

from modules import config_helpers, scope_session
from modules.exceptions import ConfigError, ProtocolRunRefusedError
from modules.layer_record import UNRESOLVED
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_protocol_needs_its_objectives_on_the_turret import _prepare
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_multi_step_protocol,
    executor,
    executors,
    scope,
)

ABSENT = 'layer_not_on_scope'


def _session(tmp_path, microscope, **acquiring):
    settings = complete_settings(live_folder=str(tmp_path), microscope=microscope)
    for layer in config_helpers.get_layer_configs(settings):
        settings[layer]['acquire'] = acquiring.get(layer)
    session = ScopeSession.create(settings, simulate=True)
    home_sim_scope(session.scope)
    return session


@pytest.fixture
def warned(monkeypatch):
    lines = []
    monkeypatch.setattr(
        scope_session.logger, 'warning', lambda msg, *a, **k: lines.append(str(msg))
    )
    return lines


@pytest.fixture
def ls850(tmp_path):
    session = _session(tmp_path, 'LS850', BF='image', Lumi='image')
    yield session
    session.shutdown()
    session.scope.disconnect()


@pytest.fixture
def lumi(tmp_path):
    session = _session(tmp_path, 'Lumi', BF='image')
    yield session
    session.shutdown()
    session.scope.disconnect()


def test_bring_up_turns_off_a_layer_this_scope_lacks(ls850):
    # The fixture's settings had Lumi acquiring, as a file from a Lumi scope would.
    assert ls850.settings['Lumi']['acquire'] is None
    assert ls850.settings['BF']['acquire'] == 'image'


def test_bring_up_logs_the_layer_it_turned_off(tmp_path, warned):
    session = _session(tmp_path, 'LS850', BF='image', Lumi='video')
    try:
        assert any('Lumi' in line and 'acquire nothing' in line for line in warned)
    finally:
        session.shutdown()
        session.scope.disconnect()


def test_new_and_add_build_no_step_for_a_layer_this_scope_lacks(ls850):
    protocol = ls850.create_empty_protocol()
    ls850.add_step(protocol)
    assert set(protocol.steps()['Color']) == {'BF'}


def test_a_layer_this_scope_lacks_is_refused_acquire_and_allowed_nothing(ls850):
    with pytest.raises(ConfigError) as refused:
        ls850.set_layer_acquire('Lumi', 'image')
    assert 'Lumi' in str(refused.value)
    assert ls850.settings['Lumi']['acquire'] is None

    ls850.set_layer_acquire('Lumi', None)


def test_a_layer_settings_row_for_a_layer_this_scope_lacks_is_dropped_and_logged(
    ls850, lumi, tmp_path, warned
):
    # Saved on a Lumi scope: BF steps, and Lumi acquiring in the block.
    lumi.set_layer_acquire('Lumi', 'image')
    protocol = lumi.create_empty_protocol()
    lumi.set_layer_acquire('Lumi', None)
    lumi.add_step(protocol)
    lumi.set_layer_acquire('Lumi', 'image')
    path = lumi.save_protocol(protocol, tmp_path / 'saved_on_lumi.tsv')
    assert 'Lumi\timage' in path.read_text()

    loaded = ls850.load_protocol(path)
    ls850.apply_layer_settings(loaded)

    assert ls850.settings['Lumi']['acquire'] is None
    assert ls850.settings['BF']['acquire'] == 'image'
    assert any('Lumi' in line and 'dropped' in line for line in warned)


def test_a_file_with_a_step_on_a_layer_this_scope_lacks_is_refused_at_load(ls850, lumi, tmp_path):
    lumi.set_layer_acquire('Lumi', 'image')
    protocol = lumi.create_empty_protocol()
    lumi.add_step(protocol)
    path = lumi.save_protocol(protocol, tmp_path / 'lumi_steps.tsv')

    with pytest.raises(ProtocolRunRefusedError) as refused:
        ls850.load_protocol(path)

    assert refused.value.reason == ABSENT
    words = str(refused.value)
    assert 'Lumi' in words
    assert protocol.step(idx=1)['Name'] in words


def test_a_scope_runs_the_lumi_steps_its_own_layers_include(lumi, tmp_path):
    lumi.set_layer_acquire('Lumi', 'image')
    protocol = lumi.create_empty_protocol()
    lumi.add_step(protocol)
    path = lumi.save_protocol(protocol, tmp_path / 'lumi_steps.tsv')

    loaded = lumi.load_protocol(path)

    assert set(loaded.steps()['Color']) == {'BF', 'Lumi'}
    lumi.scope.protocols.refuse_absent_layers(loaded)


def test_a_run_with_a_step_on_a_layer_this_scope_lacks_is_refused(executor, scope, tmp_path):
    lacking = {r.key_name for r in scope.layer_identity.layers} ^ {'Lumi'}
    assert 'Lumi' in lacking, 'the suite scope must lack Lumi for this test'
    protocol = _make_multi_step_protocol([{'name': 'Lumi step', 'color': 'Lumi'}])

    with pytest.raises(ProtocolRunRefusedError) as refused:
        _prepare(executor, protocol, tmp_path)

    assert refused.value.reason == ABSENT
    assert 'Lumi step' in str(refused.value)


def test_a_scope_whose_layers_could_not_be_resolved_says_so(ls850, monkeypatch):
    protocol = ls850.create_empty_protocol()
    ls850.add_step(protocol)
    monkeypatch.setattr(ls850.scope, 'layer_identity', UNRESOLVED)

    with pytest.raises(ProtocolRunRefusedError) as refused:
        ls850.scope.protocols.refuse_absent_layers(protocol)
    assert 'could not be resolved' in str(refused.value)
    assert 'has no BF' not in str(refused.value)

    with pytest.raises(ConfigError) as refused_acquire:
        ls850.set_layer_acquire('BF', 'image')
    assert 'could not be resolved' in str(refused_acquire.value)


def test_update_on_a_layer_not_set_to_acquire_is_refused_and_the_step_kept(ls850):
    protocol = ls850.create_empty_protocol()
    ls850.add_step(protocol)
    before = protocol.steps()
    ls850.set_layer_acquire('BF', None)

    with pytest.raises(ProtocolRunRefusedError) as refused:
        ls850.update_step(protocol, 0, layer='BF')

    assert refused.value.reason == 'no_acquiring_layer'
    assert 'BF' in str(refused.value)
    assert protocol.steps().equals(before)


def test_a_stim_edit_is_judged_on_the_steps_own_channel(ls850):
    # Blue stimulates and does not acquire; the step it edits is BF's, and
    # BF acquires, so the stim edit is not refused.
    protocol = ls850.create_empty_protocol()
    ls850.add_step(protocol)
    ls850.set_layer_acquire('Blue', None)
    ls850.settings['Blue']['stim_config']['enabled'] = True

    ls850.update_step(protocol, 0, layer='Blue')

    assert protocol.step(idx=0)['Color'] == 'BF'
