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

import threading

from modules.run_outcome import (
    ARMED,
    PENDING,
    RESOLVED,
    PendingRunOutcome,
    RunEnding,
)

_COMPLETED = RunEnding('completed', 'completed', 'Protocol Complete', 'The run finished.')
_ABORTED = RunEnding('aborted', 'stopped', 'Protocol Stopped', 'Stopped by the user.')
_FAILED = RunEnding('failed', 'motion_timeout', 'Protocol Failed', 'The stage did not arrive.')
_SHUTDOWN = RunEnding('aborted', 'shutdown', 'Session Shutdown', 'The session shut down.')

_ARTIFACT = '/runs/1/Composite/A1_Composite_1.tiff'


def _record_merge(outcome, token, path=_ARTIFACT):
    return outcome.resolve(token, merged=True, artifact_path=path, merge_reason='')


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


class TestCleanupResolver:
    def test_a_non_completed_cleanup_settles_the_run(self):
        outcome = PendingRunOutcome()
        assert outcome.resolve_if_pending(_ABORTED) is True
        assert outcome.state == RESOLVED
        settled = outcome.wait(timeout_s=0.1)
        assert (settled.status, settled.reason) == ('aborted', 'stopped')
        assert settled.merged is False and settled.artifact_path is None

    def test_a_run_with_no_merge_carries_an_empty_merge_reason(self):
        # A scan has no merge to fail, so there is nothing for the merge
        # field to say. A code invented here ('not_a_composite_run') read
        # as a failure to every caller branching on that field.
        outcome = PendingRunOutcome()
        outcome.resolve_if_pending(_COMPLETED)
        settled = outcome.wait(timeout_s=0.1)
        assert settled.status == 'completed'
        assert settled.merge_reason == '', (
            f'a run that never had a merge reported {settled.merge_reason!r} '
            f'as the reason its merge produced nothing'
        )

    def test_a_completed_run_whose_merge_never_started_says_so(self):
        outcome = PendingRunOutcome()
        outcome.resolve_if_pending(_COMPLETED, 'merge_not_started')
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
        assert outcome.wait(timeout_s=0.1).merged is True

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
        assert settled.merged is True
        assert settled.artifact_path == '/runs/1/c.tiff'

    def test_the_token_holder_can_record_a_typed_failure(self):
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        outcome.resolve(token, merged=False, artifact_path=None, merge_reason='merge_timeout')
        assert outcome.wait(timeout_s=0.1).merge_reason == 'merge_timeout'

    def test_the_merge_thread_cannot_restate_how_the_run_ended(self):
        # The merge reports on the merge. The run's status and reason come
        # from the ending recorded at arm() -- a thread that could set them
        # would be free to call a completed run failed because its own
        # step went wrong, which is the collapse this shape exists to stop.
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        outcome.resolve(token, merged=False, artifact_path=None, merge_reason='merge_error')
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
        outcome.force_resolve('shutdown', fallback=_SHUTDOWN)

        assert _record_merge(outcome, token) is False
        assert outcome.wait(timeout_s=0.1).merge_reason == 'shutdown'

    def test_resolving_twice_keeps_the_first_answer(self):
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        _record_merge(outcome, token, '/first.tiff')
        assert _record_merge(outcome, token, '/second.tiff') is False
        assert outcome.wait(timeout_s=0.1).artifact_path == '/first.tiff'


class TestTeardownResolver:
    def test_shutdown_settles_a_pending_run_with_the_callers_ending(self):
        # PENDING means the run never reached cleanup, so it recorded no
        # ending of its own; the teardown is the only thing that knows why
        # it stopped, and its ending is the run's.
        outcome = PendingRunOutcome()
        assert outcome.force_resolve('shutdown', fallback=_SHUTDOWN) is True
        settled = outcome.wait(timeout_s=0.1)
        assert (settled.status, settled.reason) == ('aborted', 'shutdown')
        assert settled.merge_reason == 'shutdown'

    def test_shutdown_keeps_the_ending_an_armed_run_already_reported(self):
        # The executors are about to be torn down without waiting, so the
        # merge cannot finish and a blocked caller must be released. But
        # ARMED means cleanup already told the run-complete subscribers
        # this run completed; answering a waiter with 'aborted' would put
        # the two channels in contradiction about the same run.
        outcome = PendingRunOutcome()
        outcome.arm(_COMPLETED)
        assert outcome.force_resolve('shutdown', fallback=_SHUTDOWN) is True
        assert outcome.state == RESOLVED
        settled = outcome.wait(timeout_s=0.1)
        assert (settled.status, settled.reason) == ('completed', 'completed'), (
            f'teardown overwrote the ending the run had already reported: '
            f'{settled.status!r}/{settled.reason!r}'
        )
        assert settled.merge_reason == 'shutdown', (
            'the shutdown is why no artifact followed, not how the run ended'
        )

    def test_teardown_does_not_overwrite_a_finished_merge(self):
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        _record_merge(outcome, token, '/done.tiff')
        assert outcome.force_resolve('shutdown', fallback=_SHUTDOWN) is False
        assert outcome.wait(timeout_s=0.1).artifact_path == '/done.tiff'

    def test_settle_unfinished_never_raises(self):
        # It runs from teardown paths and from cleanup's finally, ahead of
        # the activity-claim release; a raise there would leak the claim
        # and refuse every future run.
        outcome = PendingRunOutcome()
        outcome.settle_unfinished('shutdown', fallback=_SHUTDOWN)
        outcome.settle_unfinished('shutdown', fallback=_SHUTDOWN)
        assert outcome.wait(timeout_s=0.1).merge_reason == 'shutdown'


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
        assert settled is not None and settled.merged is False
        assert settled.reason == 'stopped'

    def test_a_waiter_wakes_when_the_merge_lands(self):
        outcome = PendingRunOutcome()
        token = outcome.arm(_COMPLETED)
        threading.Timer(0.05, lambda: _record_merge(outcome, token, '/late.tiff')).start()

        settled = outcome.wait(timeout_s=5.0)
        assert settled is not None and settled.artifact_path == '/late.tiff'


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
            target=_try, args=(lambda: outcome.force_resolve('shutdown', fallback=_SHUTDOWN),)
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
