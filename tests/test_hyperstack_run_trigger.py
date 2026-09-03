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

import datetime
import threading
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# Run-mode exhaustiveness: every kind states what the engine does with it
# ---------------------------------------------------------------------------

# What each run kind commits the engine to, at the two decision points that
# branch on run mode. The guard below diffs these keys against the enum, so a
# new run kind reds the suite until someone states its behavior here and the
# parametrized tests prove it -- rather than inheriting whichever branch it
# happens to fall through to. That fall-through is invisible at review time:
# both decisions are written as a single named mode against an else.
#
#   blocks_hyperstack_build -- the run mode alone suppresses the per-well
#       stack build, whatever the run's output format says.
#   derives_scans_from_duration -- scan count comes from the protocol's
#       period and duration; otherwise max_scans is used verbatim.
RUN_MODE_ENGINE_BEHAVIOR = {
    SequencedCaptureRunMode.FULL_PROTOCOL: {
        'blocks_hyperstack_build': False,
        'derives_scans_from_duration': True,
    },
    SequencedCaptureRunMode.SINGLE_SCAN: {
        'blocks_hyperstack_build': False,
        'derives_scans_from_duration': False,
    },
    SequencedCaptureRunMode.SINGLE_ZSTACK: {
        'blocks_hyperstack_build': False,
        'derives_scans_from_duration': False,
    },
    SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN: {
        'blocks_hyperstack_build': True,
        'derives_scans_from_duration': False,
    },
    # A composite never stacks, but the run mode is not what stops it: its
    # config assembly resolves a format the merge can read back, which
    # leaves the format gate shut. Pinned from the assembler below.
    SequencedCaptureRunMode.SINGLE_COMPOSITE: {
        'blocks_hyperstack_build': False,
        'derives_scans_from_duration': False,
    },
}


def test_every_run_mode_states_its_engine_behavior():
    declared = set(RUN_MODE_ENGINE_BEHAVIOR)
    actual = set(SequencedCaptureRunMode)
    assert declared == actual, (
        'RUN_MODE_ENGINE_BEHAVIOR drifted from SequencedCaptureRunMode. '
        f'undeclared run modes: {sorted(m.name for m in actual - declared)}; '
        f'stale rows: {sorted(m.name for m in declared - actual)}. '
        'State what the engine does with each kind at both decision points, '
        'so a new run kind cannot silently inherit another kind branch.'
    )


@pytest.mark.parametrize('run_mode', list(RUN_MODE_ENGINE_BEHAVIOR), ids=lambda m: m.value)
def test_run_mode_hyperstack_applicability_matches_its_declared_row(run_mode, tmp_path):
    runner = _hyperstack_runner(tmp_path, run_mode=run_mode)
    with patch('modules.stack_builder.build_hyperstacks_for_run'):
        result = runner._start_hyperstack_build()
        if isinstance(result, threading.Thread):
            _join(result)
    blocked = result is None
    assert blocked == RUN_MODE_ENGINE_BEHAVIOR[run_mode]['blocks_hyperstack_build']


@pytest.mark.parametrize('run_mode', list(RUN_MODE_ENGINE_BEHAVIOR), ids=lambda m: m.value)
def test_run_mode_scan_count_matches_its_declared_row(run_mode):
    from modules.sequenced_capture_runner import SequencedCaptureRunner

    protocol = MagicMock()
    protocol.period.return_value = datetime.timedelta(seconds=60)
    protocol.duration.return_value = datetime.timedelta(seconds=300)

    n_scans = SequencedCaptureRunner._calculate_num_scans(
        protocol=protocol, run_mode=run_mode, max_scans=1
    )
    if RUN_MODE_ENGINE_BEHAVIOR[run_mode]['derives_scans_from_duration']:
        # 300/60 == 5 scans, clamped by max_scans to 1 -- the clamp is what
        # distinguishes it from the verbatim branch, so read the unclamped
        # count too.
        assert (
            SequencedCaptureRunner._calculate_num_scans(
                protocol=protocol, run_mode=run_mode, max_scans=None
            )
            == 5
        )
        assert n_scans == 1
    else:
        assert n_scans == 1
        assert (
            SequencedCaptureRunner._calculate_num_scans(
                protocol=protocol, run_mode=run_mode, max_scans=None
            )
            is None
        ), 'a non-derived run mode must pass max_scans through verbatim'


def test_a_composite_run_never_carries_the_hyperstack_format():
    # The composite's non-applicability is enforced at config assembly, not
    # by a run-mode branch; this is the other half of the row above.
    import modules.config_helpers as config_helpers
    from tests.test_composite_run_config import _settings

    config = config_helpers.get_composite_image_capture_config_from_settings(
        _settings(acquiring=('BF', 'Blue'), sequenced_format=OUTPUT_FORMAT_HYPERSTACK)
    )
    assert config.output_format_sequenced != OUTPUT_FORMAT_HYPERSTACK
