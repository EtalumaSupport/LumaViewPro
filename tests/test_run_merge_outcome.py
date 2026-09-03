# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every transition of the run-merge outcome machine.

A caller that starts a composite must learn whether the MERGE produced an
artifact, not merely that the capture ended. Three facts make that harder
than a single result slot:

  - Run cleanup executes twice on a NORMAL run: the loop's 'completed'
    call, then the safety net's 'failed' call, whose early return sits
    inside the try so the finally runs again with a contradictory status.
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
    MergeOutcome,
    RunMergeOutcome,
)


def _merged(path='/runs/1/Composite/A1_Composite_1.tiff'):
    return MergeOutcome(merged=True, artifact_path=path, reason='')


class TestArming:
    def test_a_fresh_outcome_is_pending(self):
        assert RunMergeOutcome().state == PENDING

    def test_completed_cleanup_arms_and_gets_a_token(self):
        outcome = RunMergeOutcome()
        token = outcome.arm()
        assert token is not None
        assert outcome.state == ARMED

    def test_arming_twice_yields_no_second_token(self):
        outcome = RunMergeOutcome()
        outcome.arm()
        assert outcome.arm() is None, (
            'a second arm must be refused; two merge threads both believing '
            'they own the outcome is how one overwrites the other'
        )

    def test_an_already_settled_run_cannot_be_armed(self):
        # The decline-to-start path: cleanup settled the run, so no merge
        # should begin at all.
        outcome = RunMergeOutcome()
        outcome.resolve_if_pending('aborted')
        assert outcome.arm() is None


class TestCleanupResolver:
    def test_a_non_completed_cleanup_settles_the_run(self):
        outcome = RunMergeOutcome()
        assert outcome.resolve_if_pending('aborted') is True
        assert outcome.state == RESOLVED
        assert outcome.wait(timeout_s=0.1) == MergeOutcome(False, None, 'aborted')

    def test_the_second_cleanup_pass_cannot_touch_an_armed_outcome(self):
        # THE row this machine exists for. On a normal run the loop's
        # 'completed' pass arms, then the safety net's 'failed' pass
        # arrives; if it could resolve, every successful composite would
        # report failed and the real artifact path would be discarded.
        outcome = RunMergeOutcome()
        token = outcome.arm()

        assert outcome.resolve_if_pending('failed') is False
        assert outcome.state == ARMED

        assert outcome.resolve(token, _merged()) is True
        assert outcome.wait(timeout_s=0.1).merged is True

    def test_the_second_cleanup_pass_cannot_touch_a_resolved_outcome(self):
        outcome = RunMergeOutcome()
        outcome.resolve_if_pending('aborted')
        assert outcome.resolve_if_pending('failed') is False
        assert outcome.wait(timeout_s=0.1).reason == 'aborted', (
            'first-wins: the status the run actually ended with is the one that survives'
        )


class TestMergeThreadResolver:
    def test_the_token_holder_records_the_artifact(self):
        outcome = RunMergeOutcome()
        token = outcome.arm()
        assert outcome.resolve(token, _merged('/runs/1/c.tiff')) is True
        settled = outcome.wait(timeout_s=0.1)
        assert settled.merged is True
        assert settled.artifact_path == '/runs/1/c.tiff'

    def test_the_token_holder_can_record_a_typed_failure(self):
        outcome = RunMergeOutcome()
        token = outcome.arm()
        outcome.resolve(token, MergeOutcome(False, None, 'merge_timeout'))
        assert outcome.wait(timeout_s=0.1).reason == 'merge_timeout'

    def test_a_wrong_token_is_refused(self):
        outcome = RunMergeOutcome()
        outcome.arm()
        assert outcome.resolve('not-the-token', _merged()) is False

    def test_a_stale_thread_reporting_after_teardown_is_a_no_op(self):
        # Shutdown settled the run while the merge was still going; the
        # merge finishing afterward must not reopen it.
        outcome = RunMergeOutcome()
        token = outcome.arm()
        outcome.force_resolve('shutdown')

        assert outcome.resolve(token, _merged()) is False
        assert outcome.wait(timeout_s=0.1).reason == 'shutdown'

    def test_resolving_twice_keeps_the_first_answer(self):
        outcome = RunMergeOutcome()
        token = outcome.arm()
        outcome.resolve(token, _merged('/first.tiff'))
        assert outcome.resolve(token, _merged('/second.tiff')) is False
        assert outcome.wait(timeout_s=0.1).artifact_path == '/first.tiff'


class TestTeardownResolver:
    def test_shutdown_settles_a_pending_run(self):
        outcome = RunMergeOutcome()
        assert outcome.force_resolve('shutdown') is True
        assert outcome.wait(timeout_s=0.1).reason == 'shutdown'

    def test_shutdown_settles_an_armed_run(self):
        # The executors are about to be torn down without waiting, so the
        # merge cannot finish; a caller blocked on it must be released.
        outcome = RunMergeOutcome()
        outcome.arm()
        assert outcome.force_resolve('shutdown') is True
        assert outcome.state == RESOLVED

    def test_discarding_the_runs_inputs_settles_an_armed_run(self):
        outcome = RunMergeOutcome()
        outcome.arm()
        assert outcome.force_resolve('inputs_discarded') is True
        assert outcome.wait(timeout_s=0.1).reason == 'inputs_discarded', (
            'a discard must report its own cause, never a merge timeout -- '
            'the inputs were thrown away, the merge did not stall'
        )

    def test_teardown_does_not_overwrite_a_finished_merge(self):
        outcome = RunMergeOutcome()
        token = outcome.arm()
        outcome.resolve(token, _merged('/done.tiff'))
        assert outcome.force_resolve('shutdown') is False
        assert outcome.wait(timeout_s=0.1).artifact_path == '/done.tiff'

    def test_settle_unfinished_never_raises(self):
        # It runs from teardown paths and from cleanup's finally, ahead of
        # the activity-claim release; a raise there would leak the claim
        # and refuse every future run.
        outcome = RunMergeOutcome()
        outcome.settle_unfinished('shutdown')
        outcome.settle_unfinished('shutdown')
        assert outcome.wait(timeout_s=0.1).reason == 'shutdown'


class TestWaiting:
    def test_the_wait_is_bounded(self):
        assert RunMergeOutcome().wait(timeout_s=0.05) is None, (
            'an unsettled outcome must time out rather than block a caller forever'
        )

    def test_none_is_distinct_from_a_resolved_failure(self):
        # A caller has to tell "never reported" from "reported that nothing
        # was merged"; collapsing them hides a wedged run.
        never = RunMergeOutcome()
        reported = RunMergeOutcome()
        reported.resolve_if_pending('aborted')

        assert never.wait(timeout_s=0.05) is None
        assert reported.wait(timeout_s=0.05) == MergeOutcome(False, None, 'aborted')

    def test_a_waiter_wakes_when_the_merge_lands(self):
        outcome = RunMergeOutcome()
        token = outcome.arm()
        threading.Timer(0.05, lambda: outcome.resolve(token, _merged('/late.tiff'))).start()

        settled = outcome.wait(timeout_s=5.0)
        assert settled is not None and settled.artifact_path == '/late.tiff'


def _race_three_resolvers() -> tuple[int, str]:
    """Fire all three resolvers at one outcome simultaneously.

    Returns how many of them reported winning, and the final state.
    """
    outcome = RunMergeOutcome()
    token = outcome.arm()
    start = threading.Barrier(3)
    wins = []

    def _try(resolve):
        start.wait()
        if resolve():
            wins.append(resolve)

    threads = [
        threading.Thread(target=_try, args=(lambda: outcome.resolve(token, _merged()),)),
        threading.Thread(target=_try, args=(lambda: outcome.force_resolve('shutdown'),)),
        threading.Thread(target=_try, args=(lambda: outcome.resolve_if_pending('failed'),)),
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
