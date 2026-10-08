# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol run and a single scan say who asked for them and in which mode.

The protocol panel's Run and Scan start through these two members, the same
calls a script makes, so the members take what the panel knows and a script
does not: the trigger the panel's strings are named by, and the engineering
mode a plugin may have flipped after the session was built. A caller that
passes neither gets the API's own trigger and the session's mode.
"""

from unittest.mock import MagicMock

import pytest

from tests.test_run_zstack_entry_point import _prepared, _runner


def _run(runner, member, **kwargs):
    protocol = MagicMock()
    getattr(runner, member)(protocol, **kwargs)
    return protocol


@pytest.mark.parametrize(
    'member, api_trigger',
    [('run_protocol', 'api_protocol'), ('run_single_scan', 'api_scan')],
)
class TestTheCallerNamesItsRun:
    def test_a_script_gets_the_apis_trigger_and_the_sessions_mode(self, member, api_trigger):
        runner = _runner()
        runner.session.engineering_mode = True

        _run(runner, member)

        assert _prepared(runner)['run_trigger_source'] == api_trigger
        assert _prepared(runner)['engineering_mode'] is True

    def test_the_panel_gets_its_own_trigger_and_the_sessions_mode(self, member, api_trigger):
        runner = _runner()
        runner.session.engineering_mode = False

        _run(runner, member, run_trigger_source='protocol')

        assert _prepared(runner)['run_trigger_source'] == 'protocol'
        assert _prepared(runner)['engineering_mode'] is False

    def test_the_run_is_of_the_protocol_handed_in(self, member, api_trigger):
        runner = _runner()

        protocol = _run(runner, member)

        assert _prepared(runner)['protocol'] is protocol


class TestTheRunReadsOneSnapshotOfTheSettings:
    """A run reads its settings from one copy taken under the lock.

    The panel's Run reaches the runner on the worker pool while the GUI
    thread may be writing the store; a run reading the live dict there could
    take half of an edit. The run's values come from the session's snapshot.
    """

    def test_the_runs_settings_are_the_snapshots(self, tmp_path):
        import copy

        runner = _runner()
        snapshot = copy.deepcopy(runner.session.settings)
        snapshot['live_folder'] = str(tmp_path)
        snapshot['keep_led_between_steps'] = True
        snapshot['protocol']['autogain']['target_brightness'] = 0.123
        runner.session.get_settings_snapshot.return_value = snapshot

        _run(runner, 'run_protocol')

        prepared = _prepared(runner)
        assert prepared['parent_dir'] == tmp_path.resolve() / 'ProtocolData'
        assert prepared['keep_led_between_steps'] is True
        assert prepared['autogain_settings']['target_brightness'] == 0.123


class TestTheScanLogNamesTheFormatInForce:
    def test_the_scan_line_names_the_pixel_format_the_camera_delivers(self, monkeypatch):
        import modules.protocol_runner as protocol_runner

        lines = []
        monkeypatch.setattr(protocol_runner.logger, 'info', lambda msg, *a, **k: lines.append(msg))
        runner = _runner()
        runner.session.scope.imaging.pixel_format_cached = 'Mono8'

        _run(runner, 'run_single_scan')

        (scan,) = [line for line in lines if '[Protocol] scan' in line]
        assert 'pixel_format=Mono8' in scan
        assert 'capture_depth=' not in scan
