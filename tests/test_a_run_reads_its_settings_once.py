# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run reads the settings once, at its start, and a user's change during it is theirs.

Two defects with one root, a run reading the live store more than once:

- The composite, autofocus and z-stack members took a snapshot to build
  their protocol, then ``_run`` took a second one for the image mode and
  formats, and the plate position read the live labware. An edit landing
  between the reads gave one run two answers: steps built from one set of
  settings, files saved in another's mode.
- A run snapshotted every layer's autofocus flag and wrote it back at
  cleanup. Nothing in a run reads the flags, so the restore did one thing:
  revert a change the user made while the run was going.
"""

from __future__ import annotations

import copy
import threading
from unittest.mock import MagicMock

import pytest

import modules.config_helpers as config_helpers
from modules.exceptions import ProtocolRunRefusedError
from modules.image_mode import IMAGE_MODE_8BIT, IMAGE_MODE_12BIT_SCIENTIFIC
from modules.image_mode import OUTPUT_FORMAT_OME_TIFF, OUTPUT_FORMAT_TIFF
from modules.run_events import RunEvents
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings
from tests.test_composite_run_config import _settings as _run_settings
from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

_POSITION = {'x': 1.0, 'y': 2.0, 'z': 3.0}


def _at_start():
    settings = _run_settings(acquiring=('BF', 'Green'), sequenced_format=OUTPUT_FORMAT_TIFF)
    settings['image_mode'] = IMAGE_MODE_8BIT
    settings['zstack'] = {
        'range': 20.0,
        'step_size': 5.0,
        'position': 'Current Position at Center',
    }
    return settings


def _edited():
    """What the store holds once a write lands after the run's first read."""
    settings = _at_start()
    settings['image_mode'] = IMAGE_MODE_12BIT_SCIENTIFIC
    settings['image_output_format']['sequenced'] = OUTPUT_FORMAT_OME_TIFF
    settings['protocol']['labware'] = '6 well microplate'
    return settings


def _runner_over_a_store_edited_after_the_first_read():
    """A ProtocolRunner whose session answers the first settings read with
    the settings at the start and every later read with an edited store."""
    from modules.protocol_runner import ProtocolRunner

    reads = []

    def snapshot():
        reads.append(None)
        return _at_start() if len(reads) == 1 else _edited()

    session = MagicMock()
    session.settings = _edited()
    session.capture_settings_snapshot.side_effect = snapshot
    session.get_settings_snapshot.side_effect = snapshot
    session.plate_position_on.return_value = dict(_POSITION)
    session.objective_helper.get_objective_info.return_value = {'magnification': 10}
    return ProtocolRunner(session)


_MEMBERS = {
    'composite': lambda runner: runner.start_composite(),
    'autofocus': lambda runner: runner.run_autofocus(layer='BF'),
    'zstack': lambda runner: runner.run_zstack(layer='BF'),
    'single_scan': lambda runner: runner.run_single_scan(protocol=MagicMock()),
    'protocol': lambda runner: runner.run_protocol(protocol=MagicMock()),
}


def _expected_image_config(member):
    if member == 'composite':
        return config_helpers.get_composite_image_capture_config_from_settings(_at_start())
    return config_helpers.get_image_capture_config_from_settings(_at_start())


class TestEveryRunKindTakesItsSettingsFromOneSnapshot:
    @pytest.mark.parametrize('member', sorted(_MEMBERS))
    def test_the_image_config_is_the_snapshot_taken_at_the_start(self, member):
        runner = _runner_over_a_store_edited_after_the_first_read()
        _MEMBERS[member](runner)
        prepared = runner._executor.prepare.call_args.kwargs
        assert prepared['image_capture_config'] == _expected_image_config(member)

    @pytest.mark.parametrize('member', ['composite', 'autofocus', 'zstack'])
    def test_the_position_is_stated_on_the_snapshots_plate(self, member):
        runner = _runner_over_a_store_edited_after_the_first_read()
        _MEMBERS[member](runner)
        runner.session.plate_position_on.assert_called_once_with('96 well microplate')


@pytest.fixture
def session(tmp_path):
    built = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    home_sim_scope(built.scope)
    yield built
    built.shutdown()


def test_a_layers_autofocus_switched_during_a_run_is_still_switched_after_it(session, tmp_path):
    session.settings['BF']['autofocus'] = False
    files_written = threading.Event()

    def users_switch(*_scan):
        with session.settings_lock:
            session.settings['BF']['autofocus'] = True

    run = session.create_protocol_runner().run_single_scan(
        protocol=_build_real_protocol([copy.deepcopy(_make_single_step_protocol().step(idx=0))]),
        sequence_name='autofocus_switch',
        parent_dir=str(tmp_path),
        events=RunEvents(
            scan_started=users_switch,
            files_written=lambda run_dir, files: files_written.set(),
        ),
    )
    assert files_written.wait(COMPLETION_TIMEOUT), 'the run never finished its files'
    assert run.wait(timeout_s=COMPLETION_TIMEOUT) is not None

    assert session.settings['BF']['autofocus'] is True


@pytest.fixture
def unhomed_session(tmp_path):
    built = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    yield built
    built.shutdown()


@pytest.mark.parametrize('member', ['composite', 'autofocus', 'zstack'])
def test_before_a_home_the_run_is_refused_as_not_homed(unhomed_session, member):
    """The position is read on the snapshot's plate without asking about
    the axes; the run's own gate answers, as it does for every run kind."""
    settings = unhomed_session.settings
    for layer in config_helpers.get_layer_configs(settings):
        settings[layer]['acquire'] = 'image' if layer in ('BF', 'Green') else None
    settings['zstack'].update({'range': 20.0, 'step_size': 5.0})
    runner = unhomed_session.create_protocol_runner()
    with pytest.raises(ProtocolRunRefusedError) as refused:
        _MEMBERS[member](runner)
    assert refused.value.reason == 'position_unknown'
