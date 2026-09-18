# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Which run holds the microscope has one owner: the activity claim.

The claim already knew whether anything held the scope; it did not know
WHICH run, so that question was answered from a runner field no run-end
ever cleared -- between runs the getter named a run that was over, and
every correct reader was correct only because it paired that name with
a liveness term of its own. Kind and run identity are now one snapshot,
taken and released with the run.
"""

from __future__ import annotations

import threading

import pytest

from modules.activity_claim import ActivityClaim
from modules.exceptions import ProtocolRunRefusedError
from tests.protocol_drives import bare_capture_runner, scr_run_kwargs


class TestTheClaimCarriesTheRun:
    def test_a_claim_reports_its_kind_and_the_run_behind_it(self):
        claim = ActivityClaim()

        assert claim.try_claim('protocol', run_trigger_source='zstack')

        holder = claim.holder
        assert holder.kind == 'protocol'
        assert holder.run_trigger_source == 'zstack'
        assert claim.owner == 'protocol', 'the kind stays readable on its own'

    def test_an_activity_that_is_not_a_run_carries_no_trigger(self):
        claim = ActivityClaim()

        assert claim.try_claim('recording')

        assert claim.holder.kind == 'recording'
        assert claim.holder.run_trigger_source is None

    def test_the_holder_is_gone_on_release(self):
        claim = ActivityClaim()
        claim.try_claim('protocol', run_trigger_source='scan')

        claim.release('protocol')

        assert claim.holder is None
        assert claim.owner is None

    def test_a_release_by_a_non_holder_still_says_who_holds_it(self):
        claim = ActivityClaim()
        claim.try_claim('protocol', run_trigger_source='scan')

        with pytest.raises(RuntimeError, match="held by 'protocol'"):
            claim.release('recording')

    def test_one_snapshot_is_published_at_a_time(self):
        """Kind and trigger travel together or not at all: a reader
        cannot observe the kind of one claimant beside the trigger of
        another, however hard the claim is cycled underneath it."""
        claim = ActivityClaim()
        seen = set()
        stop = threading.Event()

        def _reader():
            while not stop.is_set():
                holder = claim.holder
                if holder is not None:
                    seen.add((holder.kind, holder.run_trigger_source))

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()
        try:
            for _ in range(500):
                claim.try_claim('protocol', run_trigger_source='scan')
                claim.release('protocol')
                claim.try_claim('recording')
                claim.release('recording')
        finally:
            stop.set()
            reader.join(timeout=5.0)

        assert seen <= {('protocol', 'scan'), ('recording', None)}, (
            f'a half-written holder was observed: {seen}'
        )


class TestTheRunnerReadsTheClaim:
    """A real SequencedCaptureRunner, not a mock of one.

    The live-run half of this contract needs a run that actually
    commits, which needs a session: it is pinned in
    tests/test_run_state_strand.py beside the harness that has one.
    """

    def test_the_getter_is_empty_before_any_run(self):
        runner = bare_capture_runner()

        assert runner.run_trigger_source() is None

    def test_a_second_runs_refusal_names_the_holder_off_the_claim(self):
        session_claim = ActivityClaim()
        session_claim.try_claim('recording')
        runner = bare_capture_runner(activity_claim=session_claim)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            runner.prepare(**scr_run_kwargs())

        assert excinfo.value.reason == 'exclusive_activity_running'
        assert excinfo.value.holder == 'recording'
        assert excinfo.value.holder_trigger is None, 'a recording has no trigger to name'


class TestTheFlagAndTheClaimEndTogether:
    """The run flag and the claim describe the same run.

    They are cleared deep in two different places -- the flag inside
    run_cleanup, the claim in the caller's finally -- so a cleanup that
    raised in between used to leave a run holding nothing that still
    reported itself in progress. In that state the owner's own Stop is
    refused in the name of a run whose trigger reads as nobody's.
    """

    def test_a_cleanup_that_raises_still_ends_the_run(self, monkeypatch):
        from tests.protocol_drives import autofocus_snapshot, protocol_step, scan_ready_runner

        runner = scan_ready_runner(
            protocol_step(),
            _original_led_states=None,
            _return_to_position=None,
            _protocol_execution_record=None,
            _autofocus_snapshot=autofocus_snapshot(states={}),
            _run_dir=None,
        )
        assert runner._activity_claim.try_claim('protocol', run_trigger_source='test')
        runner._activity_claim_held = True
        assert runner.run_in_progress()

        def _boom(**_kwargs):
            raise RuntimeError('cleanup died before it cleared the run flag')

        monkeypatch.setattr('modules.sequenced_capture_runner.run_cleanup', _boom)
        monkeypatch.setattr(runner, '_start_hyperstack_build', lambda: None)

        from modules.run_outcome import RunEnding

        with pytest.raises(RuntimeError, match='cleanup died'):
            runner._cleanup_inner(RunEnding('aborted', 'stopped', 'Stopped', 'Stopped by test'))

        assert not runner.run_in_progress(), (
            'a run that released the scope still reports itself in progress'
        )
        assert runner._activity_claim.holder is None
        assert runner.run_trigger_source() is None
