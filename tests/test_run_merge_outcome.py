# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every transition of the run-outcome machine.

A caller that starts a run must learn how the RUN ended -- and, for a
composite, whether the MERGE produced an artifact. Those are two
different answers and the machine carries both, because "the run
aborted" and "the run completed but merged nothing" are not the same
thing to anyone deciding what to do next. Three facts make holding them
harder than a single result slot:

  - Run cleanup executes twice on a NORMAL run: the loop's 'completed'
    call, then the safety net's 'failed' call, whose early return sits
    inside the try so the finally runs again with a contradictory
    ending.
  - On the 'completed' pass the merge has not run yet, so that pass can
    only arm the outcome.
  - Teardown (session shutdown, discarding the run's unwritten inputs)
    has to settle an outcome the merge thread already owns, or a caller
    blocks forever on a merge that can no longer finish.

Each test below is one row of the machine's transition table. The
second-cleanup-pass row is the one that matters most: a two-state design
reports 'failed' over the real merge result on every successful run.
"""

import dataclasses
import pathlib
import threading

import pytest

from _lumascope.modules.exceptions import ArgumentRefusedError
from _lumascope.modules.run_outcome import (
    ARMED,
    PENDING,
    RESOLVED,
    CaptureTally,
    FailedCapture,
    FocusFailure,
    MergeReason,
    PendingRunOutcome,
    RunEnding,
    RunOutcome,
    RunStatus,
    StepFrames,
)

_COMPLETED = RunEnding('completed', 'completed', 'The run finished.')
_ABORTED = RunEnding('aborted', 'stopped', 'Stopped by the user.')
_FAILED = RunEnding(
    'failed', 'position_lost', 'The stage did not arrive.', axes_without_position={'Z': 'unknown'}
)
_SHUTDOWN = RunEnding('aborted', 'shutdown', 'The session shut down.')

_ARTIFACT = '/runs/1/Composite/A1_Composite_1.tiff'


def _record_merge(outcome, token, path=_ARTIFACT):
    return outcome.resolve(token, artifact_path=pathlib.Path(path), merge_reason=None)


class TestArming:
    def test_a_fresh_outcome_is_pending(self):
        assert PendingRunOutcome().state == PENDING

    def test_completed_cleanup_arms_and_gets_a_token(self):
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        assert token is not None
        assert outcome.state == ARMED

    def test_arming_twice_yields_no_second_token(self):
        outcome = PendingRunOutcome()
        outcome.arm(_COMPLETED)
        assert outcome.arm(_COMPLETED) is None, (
            'a second arm must be refused; two merge threads both believing '
            'they own the outcome is how one overwrites the other'
        )

    def test_an_already_settled_run_cannot_be_armed(self):
        # The decline-to-start path: cleanup settled the run, so no merge
        # should begin at all.
        outcome = PendingRunOutcome()
        outcome.resolve_if_pending(_ABORTED)
        assert outcome.arm(_COMPLETED) is None


class TestStatus:
    def test_a_settled_run_reads_its_status_as_a_run_status(self):
        # A plain string of the set is the member it names, so a caller
        # comparing with the string and one comparing with the member agree.
        outcome = PendingRunOutcome()
        outcome.resolve_if_pending(RunEnding('aborted', 'stopped', 'Stopped by the user.'))
        settled = outcome.wait(timeout_s=0.1)
        assert type(settled.status) is RunStatus
        assert settled.status == RunStatus.ABORTED == 'aborted'

    def test_an_ending_in_no_status_of_the_set_fails_where_it_is_made(self):
        # Not later, inside a resolver: those run in cleanup's finally and
        # never raise, or the run's claim on the scope would leak.
        with pytest.raises(ValueError, match='RunStatus'):
            RunEnding('finished', 'completed', 'The run finished.')

    def test_an_outcome_built_with_no_status_of_the_set_is_refused(self):
        outcome = PendingRunOutcome()
        outcome.resolve_if_pending(_COMPLETED)
        settled = outcome.wait(timeout_s=0.1)
        with pytest.raises(ArgumentRefusedError) as refused:
            dataclasses.replace(settled, status='finished')
        assert (refused.value.reason, refused.value.argument) == (
            'not_offered',
            'RunOutcome.status',
        )


class TestCleanupResolver:
    def test_a_non_completed_cleanup_settles_the_run(self):
        outcome = PendingRunOutcome()
        assert outcome.resolve_if_pending(_ABORTED) is True
        assert outcome.state == RESOLVED
        settled = outcome.wait(timeout_s=0.1)
        assert (settled.status, settled.reason) == ('aborted', 'stopped')
        assert settled.artifact_path is None

    def test_a_run_with_no_merge_carries_no_merge_reason(self):
        # A scan has no merge to fail, so there is nothing for the merge
        # field to say. A code invented here ('not_a_composite_run') read
        # as a failure to every caller branching on that field.
        outcome = PendingRunOutcome()
        outcome.resolve_if_pending(_COMPLETED)
        settled = outcome.wait(timeout_s=0.1)
        assert settled.status == 'completed'
        assert settled.merge_reason is None, (
            f'a run that never had a merge reported {settled.merge_reason!r} '
            f'as the reason its merge produced nothing'
        )

    def test_a_completed_run_whose_merge_never_started_says_so(self):
        outcome = PendingRunOutcome()
        outcome.resolve_if_pending(_COMPLETED, MergeReason.MERGE_NOT_STARTED)
        settled = outcome.wait(timeout_s=0.1)
        assert settled.status == 'completed', (
            'the merge failing to start does not retroactively fail the run'
        )
        assert settled.merge_reason == 'merge_not_started'

    def test_the_second_cleanup_pass_cannot_touch_an_armed_outcome(self):
        # THE row this machine exists for. On a normal run the loop's
        # 'completed' pass arms, then the safety net's 'failed' pass
        # arrives; if it could resolve, every successful composite would
        # report failed and the real artifact path would be discarded.
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)

        assert outcome.resolve_if_pending(_FAILED) is False
        assert outcome.state == ARMED

        assert _record_merge(outcome, token) is True
        assert outcome.wait(timeout_s=0.1).artifact_path is not None

    def test_the_second_cleanup_pass_cannot_touch_a_resolved_outcome(self):
        outcome = PendingRunOutcome()
        outcome.resolve_if_pending(_ABORTED)
        assert outcome.resolve_if_pending(_FAILED) is False
        assert outcome.wait(timeout_s=0.1).reason == 'stopped', (
            'first-wins: the ending the run actually stopped for is the one that survives'
        )


class TestMergeThreadResolver:
    def test_the_token_holder_records_the_artifact(self):
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        assert _record_merge(outcome, token, '/runs/1/c.tiff') is True
        settled = outcome.wait(timeout_s=0.1)
        assert settled.artifact_path is not None
        assert settled.artifact_path == pathlib.Path('/runs/1/c.tiff')

    def test_the_token_holder_can_record_a_typed_failure(self):
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        outcome.resolve(
            token,
            artifact_path=None,
            merge_reason=MergeReason.MERGE_ERROR,
            merge_failure='write_batch_timeout',
        )
        settled = outcome.wait(timeout_s=0.1)
        # The merge's word and the failure's own word, each in its field.
        assert (settled.merge_reason, settled.merge_failure) == (
            'merge_error',
            'write_batch_timeout',
        )

    def test_the_merge_thread_cannot_restate_how_the_run_ended(self):
        # The merge reports on the merge. The run's status and reason come
        # from the ending recorded at arm() -- a thread that could set them
        # would be free to call a completed run failed because its own
        # step went wrong, which is the collapse this shape exists to stop.
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        outcome.resolve(token, artifact_path=None, merge_reason=MergeReason.MERGE_ERROR)
        settled = outcome.wait(timeout_s=0.1)
        assert (settled.status, settled.reason) == ('completed', 'completed'), (
            f'a failed merge rewrote the run as {settled.status!r}/{settled.reason!r}'
        )
        assert settled.merge_reason == 'merge_error'

    def test_a_wrong_token_is_refused(self):
        outcome = PendingRunOutcome()
        outcome.arm(_COMPLETED)
        assert _record_merge(outcome, 'not-the-token') is False

    def test_a_stale_thread_reporting_after_teardown_is_a_no_op(self):
        # Shutdown settled the run while the merge was still going; the
        # merge finishing afterward must not reopen it.
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        outcome.force_resolve(MergeReason.CLEANUP_ERROR, fallback=_SHUTDOWN)

        assert _record_merge(outcome, token) is False
        assert outcome.wait(timeout_s=0.1).merge_reason == 'cleanup_error'

    def test_resolving_twice_keeps_the_first_answer(self):
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        _record_merge(outcome, token, '/first.tiff')
        assert _record_merge(outcome, token, '/second.tiff') is False
        assert outcome.wait(timeout_s=0.1).artifact_path == pathlib.Path('/first.tiff')


class TestTeardownResolver:
    def test_shutdown_settles_a_pending_run_with_the_callers_ending(self):
        # PENDING means the run never reached cleanup, so it recorded no
        # ending of its own; the teardown is the only thing that knows why
        # it stopped, and its ending is the run's.
        outcome = PendingRunOutcome()
        assert outcome.force_resolve(MergeReason.CLEANUP_ERROR, fallback=_SHUTDOWN) is True
        settled = outcome.wait(timeout_s=0.1)
        assert (settled.status, settled.reason) == ('aborted', 'shutdown')
        assert settled.merge_reason == 'cleanup_error'

    def test_shutdown_keeps_the_ending_an_armed_run_already_reported(self):
        # The executors are about to be torn down without waiting, so the
        # merge cannot finish and a blocked caller must be released. But
        # ARMED means cleanup already told the run-complete subscribers
        # this run completed; answering a waiter with 'aborted' would put
        # the two channels in contradiction about the same run.
        outcome = PendingRunOutcome()
        outcome.arm(_COMPLETED)
        assert outcome.force_resolve(MergeReason.CLEANUP_ERROR, fallback=_SHUTDOWN) is True
        assert outcome.state == RESOLVED
        settled = outcome.wait(timeout_s=0.1)
        assert (settled.status, settled.reason) == ('completed', 'completed'), (
            f'teardown overwrote the ending the run had already reported: '
            f'{settled.status!r}/{settled.reason!r}'
        )
        assert settled.merge_reason == 'cleanup_error', (
            'what settled it is why no artifact followed, not how the run ended'
        )

    def test_teardown_does_not_overwrite_a_finished_merge(self):
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        _record_merge(outcome, token, '/done.tiff')
        assert outcome.force_resolve(MergeReason.CLEANUP_ERROR, fallback=_SHUTDOWN) is False
        assert outcome.wait(timeout_s=0.1).artifact_path == pathlib.Path('/done.tiff')


# Each fact a run records: its setter, a value no "nothing recorded" default
# equals, and the outcome field it reaches.
_RECORDED_FACTS = (
    ('record_autofocus_data', pathlib.Path('/runs/1/af.csv'), 'autofocus_data_path'),
    ('record_autofocus_focus', 1234.5, 'autofocus_z_um'),
    (
        'record_captures',
        CaptureTally(
            asked=4,
            captured=3,
            failed=(FailedCapture(scan_index=0, step_index=2, step_name='C1', message='timeout'),),
            saturated=(StepFrames(step_index=1, step_name='B1', count=1, first_scan_index=0),),
            dark=(),
        ),
        'captures',
    ),
    ('record_cleanup_failures', ('Return to position',), 'cleanup_failures'),
    ('record_video_save_wait', 2.5, 'video_save_wait_s'),
    ('record_focus_written', True, 'focus_written'),
    (
        'record_focus_failures',
        (FocusFailure(step_index=0, step_name='A1', count=1, first_scan_index=0, message='flat'),),
        'focus_failures',
    ),
)

# The outcome's fields that are not a recorded fact: the ending's and the
# merge's.
_NOT_RECORDED = {
    'status',
    'reason',
    'title',
    'message',
    'unconnected_parts',
    'axes_without_position',
    'artifact_path',
    'merge_reason',
    'merge_failure',
}


def _settle_by_cleanup(pending):
    pending.resolve_if_pending(_COMPLETED)


def _settle_by_teardown(pending):
    pending.force_resolve(MergeReason.CLEANUP_ERROR, fallback=_SHUTDOWN)


def _settle_by_merge(pending):
    _record_merge(pending, pending.arm(_COMPLETED))


def test_every_recorded_fact_reaches_the_outcome_whichever_path_settles_it():
    """A run settles down cleanup's path, a teardown's or the merge thread's,
    and every fact it recorded reaches its outcome on each: a fact missed on
    one path would read there as its "nothing recorded" value, with nothing
    raised, for a run that recorded it.

    The table is checked against the outcome's fields first, so a fact added
    to the outcome fails here until it joins the table.
    """
    fields = {f.name for f in dataclasses.fields(RunOutcome)} - _NOT_RECORDED
    tabled = {field for _, _, field in _RECORDED_FACTS}
    assert tabled == fields, (
        f'outcome fields with no row: {sorted(fields - tabled)}; '
        f'rows with no field: {sorted(tabled - fields)}'
    )

    dropped = []
    for settle in (_settle_by_cleanup, _settle_by_teardown, _settle_by_merge):
        pending = PendingRunOutcome()
        for setter, value, _ in _RECORDED_FACTS:
            getattr(pending, setter)(value)
        settle(pending)
        settled = pending.wait(timeout_s=1.0)
        assert settled is not None, f'{settle.__name__} did not settle the run'
        dropped += [
            f'{settle.__name__}: {field}'
            for _, value, field in _RECORDED_FACTS
            if getattr(settled, field) != value
        ]
    assert dropped == [], f'recorded facts the outcome did not carry: {dropped}'


class TestWaiting:
    def test_the_wait_is_bounded(self):
        assert PendingRunOutcome().wait(timeout_s=0.05) is None, (
            'an unsettled outcome must time out rather than block a caller forever'
        )

    def test_none_is_distinct_from_a_resolved_failure(self):
        # A caller has to tell "never reported" from "reported that nothing
        # was merged"; collapsing them hides a wedged run.
        never = PendingRunOutcome()
        reported = PendingRunOutcome()
        reported.resolve_if_pending(_ABORTED)

        assert never.wait(timeout_s=0.05) is None
        settled = reported.wait(timeout_s=0.05)
        assert settled is not None and settled.artifact_path is None
        assert settled.reason == 'stopped'

    def test_a_waiter_wakes_when_the_merge_lands(self):
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        threading.Timer(0.05, lambda: _record_merge(outcome, token, '/late.tiff')).start()

        settled = outcome.wait(timeout_s=5.0)
        assert settled is not None and settled.artifact_path == pathlib.Path('/late.tiff')


def _race_three_resolvers() -> tuple[int, str]:
    """Fire all three resolvers at one outcome simultaneously.

    Returns how many of them reported winning, and the final state.
    """
    outcome = PendingRunOutcome()
    token = outcome.arm(_COMPLETED)
    start = threading.Barrier(3)
    wins = []

    def _try(resolve):
        start.wait()
        if resolve():
            wins.append(resolve)

    threads = [
        threading.Thread(target=_try, args=(lambda: _record_merge(outcome, token),)),
        threading.Thread(
            target=_try,
            args=(lambda: outcome.force_resolve(MergeReason.CLEANUP_ERROR, fallback=_SHUTDOWN),),
        ),
        threading.Thread(target=_try, args=(lambda: outcome.resolve_if_pending(_FAILED),)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    return len(wins), outcome.state


def test_concurrent_resolvers_produce_exactly_one_answer():
    """Cleanup, teardown and the merge thread can race; one must win.

    Repeated because a lost race is a scheduling accident: a single pass
    can pick the safe interleaving and report green on a broken machine.
    """
    for _ in range(50):
        win_count, state = _race_three_resolvers()
        assert win_count == 1, f'expected exactly one resolver to win, got {win_count}'
        assert state == RESOLVED
