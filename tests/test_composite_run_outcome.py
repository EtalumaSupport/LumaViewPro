# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A composite is a terminal, API-observable outcome.

An L2 caller that starts a composite must learn, through the runner
surface, whether the MERGE succeeded and where the artifact landed -- or
get a typed failure. A run that reports 'completed' while the merged file
is missing, with the merge error reaching only a notification, hides the
failure from every headless caller: exactly the boundary the composite
run kind exists to fix.

The merge itself, and the run-scoped outcome object every terminal path
resolves, are the next build stage. The contract is pinned here first so
that stage has a target it must satisfy, rather than being declared done
against a surface nobody asserted.
"""

from unittest.mock import MagicMock

import pytest

from modules.exceptions import CaptureError
from modules.protocol_state_machine import SequencedCaptureRunMode
from modules.run_outcome import PendingRunOutcome, RunEnding
from tests.test_composite_run_config import _settings


def _runner(acquiring=('BF', 'Blue')):
    """A ProtocolRunner over a mocked session and engine.

    Real config assembly, mocked engine: this pins what the composite
    surface RETURNS, not what the engine does with the plan.
    """
    from modules.protocol_runner import ProtocolRunner

    session = MagicMock()
    session.settings = _settings(acquiring=acquiring)
    session.get_current_plate_position.return_value = {'x': 1.0, 'y': 2.0, 'z': 3.0}
    session.objective_helper.get_objective_info.return_value = {'magnification': 10}
    return ProtocolRunner(session)


class TestCompositeRunsAsItsOwnKind:
    def test_the_run_is_prepared_as_a_composite(self):
        runner = _runner()
        runner.run_composite()
        run_mode = runner._executor.prepare.call_args.kwargs['run_mode']
        assert run_mode is SequencedCaptureRunMode.SINGLE_COMPOSITE

    def test_the_run_captures_exactly_one_scan(self):
        runner = _runner()
        runner.run_composite()
        assert runner._executor.prepare.call_args.kwargs['max_scans'] == 1

    def test_the_trigger_source_names_the_api_caller(self):
        runner = _runner()
        runner.run_composite()
        trigger = runner._executor.prepare.call_args.kwargs['run_trigger_source']
        assert trigger == 'api_composite'

    def test_pre_run_lit_channels_are_restored_at_the_end(self):
        # A composite ends by handing the scope back as it was found; the
        # scan/protocol default of forcing every LED dark is a different
        # policy and would leave an interactive user's channel off.
        runner = _runner()
        runner.run_composite()
        assert runner._executor.prepare.call_args.kwargs['leds_state_at_end'] == (
            'return_to_original'
        )

    def test_frames_are_saved(self):
        # The merge reads its per-channel frames back off disk, so a
        # composite that saved nothing has nothing to merge.
        runner = _runner()
        runner.run_composite()
        assert runner._executor.prepare.call_args.kwargs['enable_image_saving'] is True

    def test_the_scan_and_protocol_paths_still_force_leds_dark(self):
        # The end-state became a parameter for the composite's sake; the
        # two callers that had it hardcoded must be unchanged by that.
        from modules.protocol import Protocol

        runner = _runner()
        protocol = MagicMock(spec=Protocol)
        runner.run_single_scan(
            protocol=protocol,
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        )
        assert runner._executor.prepare.call_args.kwargs['leds_state_at_end'] == 'off'

        runner.run_protocol(
            protocol=protocol,
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        )
        assert runner._executor.prepare.call_args.kwargs['leds_state_at_end'] == 'off'


class TestCompositeOutcomeIsObservable:
    """B17: the merge, not the capture, is what the caller learns about."""

    _COMPLETED = RunEnding('completed', 'completed', 'Protocol Complete', 'The run finished.')

    def _runner_whose_run(
        self, ending, *, merged=False, artifact_path=None, merge_reason='', settle=True
    ):
        runner = _runner()
        settled = PendingRunOutcome()
        if settle:
            token = settled.arm(ending)
            settled.resolve(
                token, merged=merged, artifact_path=artifact_path, merge_reason=merge_reason
            )
        runner._executor.start.return_value = settled
        return runner

    def test_run_composite_returns_the_merged_artifact_path(self):
        runner = self._runner_whose_run(
            self._COMPLETED, merged=True, artifact_path='/runs/1/A1_Composite_1.tiff'
        )

        assert runner.run_composite() == '/runs/1/A1_Composite_1.tiff', (
            'an L2 caller must learn WHERE the merged composite landed; '
            'returning nothing makes a missing artifact indistinguishable '
            'from a successful merge'
        )

    @pytest.mark.parametrize('merge_reason', ['merge_timeout', 'merge_failed', 'no_run_dir'])
    def test_a_completed_run_whose_merge_produced_nothing_names_the_merge(self, merge_reason):
        runner = self._runner_whose_run(self._COMPLETED, merge_reason=merge_reason)

        with pytest.raises(CaptureError) as excinfo:
            runner.run_composite()
        assert excinfo.value.reason == merge_reason, (
            'the failure must CARRY its machine-readable cause, so a caller '
            'can tell an aborted run from a failed merge from a timeout '
            'without pattern-matching the prose'
        )
        assert merge_reason in str(excinfo.value), 'the human-readable message still names it too'

    @pytest.mark.parametrize(
        ('status', 'reason'),
        [('aborted', 'stopped'), ('failed', 'motion_timeout'), ('failed', 'disk_space_critical')],
    )
    def test_a_run_that_never_completed_names_its_own_ending(self, status, reason):
        # There was no merge to blame: the run died first, so merge_reason
        # is empty and the code the caller gets is why the RUN stopped.
        # Collapsing both into one field is what made 'aborted' and
        # 'merge_failed' indistinguishable to anyone deciding what to retry.
        runner = self._runner_whose_run(
            RunEnding(status, reason, 'Protocol Stopped', 'The run did not finish.')
        )

        with pytest.raises(CaptureError) as excinfo:
            runner.run_composite()
        assert excinfo.value.reason == reason, (
            f'a run that ended {status!r} for {reason!r} reported '
            f'{excinfo.value.reason!r} to its caller'
        )

    def test_an_outcome_that_never_settles_raises_rather_than_hanging(self):
        runner = self._runner_whose_run(self._COMPLETED, settle=False)

        with pytest.raises(CaptureError) as excinfo:
            runner.run_composite(merge_timeout_s=0.05)
        assert excinfo.value.reason == 'merge_timeout'
        assert 'wedged' in str(excinfo.value)

    def test_a_refused_run_reaches_the_caller_as_a_refusal(self):
        # "Started, but with no outcome to wait on" is no longer a state
        # that can occur: start() either commits and hands back this run's
        # own outcome, or refuses. What the caller must never do is wait on
        # a run that does not exist, and that is still pinned here -- the
        # refusal has to arrive as a refusal rather than as a merge failure.
        from modules.exceptions import ProtocolRunRefusedError

        runner = _runner()
        runner._executor.start.side_effect = ProtocolRunRefusedError(
            'already_running', 'Run refused', 'Another run is already in progress.'
        )

        with pytest.raises(ProtocolRunRefusedError):
            runner.run_composite()
