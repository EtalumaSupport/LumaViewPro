# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol run and a single scan run in the session's mode, on the protocol handed in.

The protocol panel's Run and Scan start through these two members, the same
calls a script makes. The engineering mode is the session's, read at the
start, so a plugin that flipped it after the session was built is obeyed.
Who started the run is not the member's to say: ``prepare`` reads it from
the host's stamp on the calling thread (``modules.caller``).
"""

import pytest

from tests.test_run_refusal_contract import _make_single_step_protocol
from tests.test_start_zstack_entry_point import _prepared, _runner


def _run(runner, member, **kwargs):
    protocol = _make_single_step_protocol()
    getattr(runner, member)(protocol, **kwargs)
    return protocol


@pytest.mark.parametrize('member', ['start_protocol', 'start_single_scan'])
class TestTheRunIsTheSessions:
    @pytest.mark.parametrize('mode', [True, False])
    def test_the_run_gets_the_sessions_mode(self, member, mode):
        runner = _runner()
        runner.session.engineering_mode = mode

        _run(runner, member)

        assert _prepared(runner)['engineering_mode'] is mode

    def test_the_run_is_of_the_protocol_handed_in(self, member):
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

        _run(runner, 'start_protocol')

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
        runner.session.scope.camera.pixel_format = 'Mono8'

        _run(runner, 'start_single_scan')

        (scan,) = [line for line in lines if '[Protocol] scan' in line]
        assert 'pixel_format=Mono8' in scan
        assert 'capture_depth=' not in scan
