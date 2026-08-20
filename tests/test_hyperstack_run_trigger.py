# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The hyperstack build triggers from the RUNNER's completion, not the GUI.

The build trigger lived in the GUI tier (five protocol-settings call
sites plus zstack), read its config from the live UI, and therefore
never fired for a headless / L2 run. The runner owns every other
end-of-run action, holds the run's own immutable config snapshot, and
holds the file executor whose idle protocol queue is the
all-files-flushed signal -- so the trigger lives there, and a run
started from any host gets its stacks.
"""

import threading
from unittest.mock import patch

import pytest

from modules.image_mode import OUTPUT_FORMAT_HYPERSTACK, ImageCaptureConfig
from modules.protocol_state_machine import SequencedCaptureRunMode
from tests.protocol_drives import bare_capture_runner


def _hyperstack_runner(tmp_path, run_mode=SequencedCaptureRunMode.FULL_PROTOCOL):
    runner = bare_capture_runner()
    runner._run_mode = run_mode
    runner._image_capture_config = ImageCaptureConfig.from_image_mode(
        '8bit', output_format_sequenced=OUTPUT_FORMAT_HYPERSTACK
    )
    runner._run_dir = tmp_path
    runner.file_io_executor.is_protocol_queue_active.return_value = False
    runner._scope.capabilities.has_turret = False
    return runner


def _join(thread):
    assert isinstance(thread, threading.Thread)
    thread.join(timeout=5.0)
    assert not thread.is_alive()


class TestRunnerHyperstackTrigger:
    def test_completion_builds_hyperstacks_without_any_ui(self, tmp_path):
        runner = _hyperstack_runner(tmp_path)
        with patch('modules.stack_builder.build_hyperstacks_for_run') as build:
            thread = runner._start_hyperstack_build()
            _join(thread)
        build.assert_called_once_with(run_dir=tmp_path, has_turret=False)

    def test_waits_for_the_protocol_file_queue_to_drain(self, tmp_path, monkeypatch):
        import modules.sequenced_capture_runner as scr

        monkeypatch.setattr(scr, '_HYPERSTACK_QUEUE_POLL_S', 0.01)
        runner = _hyperstack_runner(tmp_path)
        # Queue active for the first two polls, then idle.
        runner.file_io_executor.is_protocol_queue_active.side_effect = [True, True, False]
        with patch('modules.stack_builder.build_hyperstacks_for_run') as build:
            thread = runner._start_hyperstack_build()
            _join(thread)
        build.assert_called_once()
        assert runner.file_io_executor.is_protocol_queue_active.call_count == 3

    @pytest.mark.parametrize(
        'mutate',
        [
            lambda r: setattr(r, '_run_mode', SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN),
            lambda r: setattr(
                r, '_image_capture_config', ImageCaptureConfig.from_image_mode('8bit')
            ),
            lambda r: setattr(r, '_run_dir', None),
        ],
        ids=['autofocus_mode', 'non_hyperstack_format', 'no_run_dir'],
    )
    def test_no_build_when_not_applicable(self, tmp_path, mutate):
        runner = _hyperstack_runner(tmp_path)
        mutate(runner)
        with patch('modules.stack_builder.build_hyperstacks_for_run') as build:
            assert runner._start_hyperstack_build() is None
        build.assert_not_called()

    @pytest.mark.parametrize(
        'run_mode',
        [SequencedCaptureRunMode.SINGLE_SCAN, SequencedCaptureRunMode.SINGLE_ZSTACK],
        ids=lambda m: m.value,
    )
    def test_scan_and_zstack_modes_also_build(self, tmp_path, run_mode):
        runner = _hyperstack_runner(tmp_path, run_mode=run_mode)
        with patch('modules.stack_builder.build_hyperstacks_for_run') as build:
            thread = runner._start_hyperstack_build()
            _join(thread)
        build.assert_called_once()


def test_gui_tier_no_longer_owns_the_trigger():
    # The GUI-tier trigger read the LIVE UI config and was unreachable
    # from a headless host; its deletion is load-bearing for the
    # relocation (a surviving copy would be a second, drifting path).
    import modules.config_ui_getters as config_ui_getters

    assert not hasattr(config_ui_getters, 'create_hyperstacks_if_needed')
