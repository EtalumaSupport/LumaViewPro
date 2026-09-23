# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""One store answers "is a run happening", and every exit restores it.

A run's phase was stored twice: a boolean Event maintained beside the
state machine. Cleanup's finally restored the Event, the activity claim
and the LED lease -- but never the state. So a cleanup that raised left
the state non-IDLE with the Event clear, and the next start() passed its
already-running gate (which read the Event alone), took the activity
claim and the LED lease, and then raised ValueError transitioning into
RUNNING -- after both were taken and before the try that would have
unwound them. Claim and lease leaked for the life of the process, and
every later run AND video recording was refused.

The strands below are constructed directly: three passes over the
cleanup tail found the chain reachable but no trigger that enters it,
so a test that waited for one would be pinning a hope.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.protocol_state_machine import PROTOCOL_STATE_TRANSITIONS, ProtocolState
from modules.run_outcome import RunEnding
from tests.protocol_drives import bare_capture_runner, scr_run_kwargs

_ABORTED = RunEnding('aborted', 'stopped', 'Protocol Stopped', 'Stopped by test')


def _started_runner():
    """A runner past prepare()+start(): claim held, lease held, RUNNING.

    The dispatched Future is left UNRESOLVED, which is what start() reads
    as "the run loop is live": a resolved one means the dispatch was
    refused and start() unwinds the run it just committed. Nothing ever
    runs the loop, so the run stays committed for the test to unwind by
    hand.
    """
    runner = bare_capture_runner()
    runner.protocol_thread.run_protocol.return_value = Future()
    runner.protocol_thread.aborted = threading.Event()
    runner.start(runner.prepare(**scr_run_kwargs()))
    return runner


def test_a_cleanup_that_raises_past_the_phase_change_still_ends_idle(monkeypatch):
    """The burn, end to end: strand, then start the next run.

    run_cleanup moves the run into COMPLETING early and has no enclosing
    try/finally of its own, so a raise in its tail is what strands the
    state one transition short of IDLE.
    """
    import modules.sequenced_capture_runner as scr

    runner = _started_runner()

    def raise_after_the_phase_change(**kwargs):
        kwargs['set_state_fn'](ProtocolState.COMPLETING)
        raise RuntimeError('cleanup tail raised')

    monkeypatch.setattr(scr, 'run_cleanup', raise_after_the_phase_change)

    with pytest.raises(RuntimeError, match='cleanup tail raised'):
        runner._cleanup(_ABORTED)

    assert runner._state is ProtocolState.IDLE, (
        'cleanup raised and left the run stranded outside IDLE'
    )
    assert runner._activity_claim.holder is None, 'the activity claim outlived the run'

    # The researcher presses Run again. This is the whole point: at the
    # defect it raised ValueError out of start() and burned the claim.
    runner.start(runner.prepare(**scr_run_kwargs()))
    assert runner.run_in_progress(), 'the next run did not start after a raising cleanup'


def test_a_cleanup_that_raises_before_the_phase_change_still_ends_idle(monkeypatch):
    """The same guarantee from RUNNING, which the transition table once
    could not express: a table with no RUNNING -> IDLE models a run that
    always succeeds, and the restore itself would raise inside the block
    whose whole purpose is to run on every path."""
    runner = _started_runner()

    # Raises in the drain wait at the head of cleanup's try -- before
    # run_cleanup is reached, so the phase never changes.
    writer = MagicMock()
    writer.video_busy = True
    writer.wait_for_video_drains.side_effect = RuntimeError('drain wait raised')
    runner._image_writer = writer

    with pytest.raises(RuntimeError, match='drain wait raised'):
        runner._cleanup(_ABORTED)

    assert runner._state is ProtocolState.IDLE, 'a run that stopped at RUNNING did not end idle'
    assert runner._activity_claim.holder is None, 'the activity claim outlived the run'
    runner.start(runner.prepare(**scr_run_kwargs()))
    assert runner.run_in_progress(), 'the next run did not start after a raising cleanup'


def test_idle_is_reachable_from_every_state():
    """A run that has stopped is idle whatever phase it stopped in."""
    for state in ProtocolState:
        if state is ProtocolState.IDLE:
            continue
        assert ProtocolState.IDLE in PROTOCOL_STATE_TRANSITIONS[state], (
            f'a run stopped in {state.value} cannot be restored to idle'
        )


def test_the_run_predicate_has_exactly_one_store():
    """Stops the fix being undone by reintroducing the shadow flag."""
    runner = bare_capture_runner()

    assert not hasattr(runner, '_run_in_progress_event'), 'a second store for the run phase is back'

    assert not runner.run_in_progress()
    runner._set_state(ProtocolState.RUNNING)
    assert runner.run_in_progress(), 'the predicate does not answer from the state machine'
    runner._set_state(ProtocolState.IDLE)
    assert not runner.run_in_progress()


def test_a_live_error_run_reads_as_in_progress():
    """ERROR is written on three live paths and held through teardown.

    A predicate that answered "not in progress" for it would make
    wait_for_run_idle return True at shutdown with the teardown still
    running, and reset() / force_reset() silently early-return.
    """
    runner = _started_runner()

    runner._set_state(ProtocolState.ERROR)

    assert runner.run_in_progress(), 'a run unwinding through ERROR reads as no run at all'


def test_a_run_that_died_still_ends_idle():
    """ERROR is held through the whole teardown -- cleanup needs it to know
    the run was a fault -- so the restore is what finally ends it."""
    runner = _started_runner()
    runner._set_state(ProtocolState.ERROR)

    runner._cleanup(RunEnding('failed', 'run_loop_crashed', 'Protocol Crashed', 'died'))

    assert runner._state is ProtocolState.IDLE, 'a failed run never became idle again'
    assert runner._activity_claim.holder is None, 'the activity claim outlived the run'


def test_the_cleanup_tail_refuses_the_next_run_by_name():
    """Through the tail the run is still the run, and says so.

    The tail used to answer "no run in progress" while still holding the
    activity claim, so the next click was refused as another EXCLUSIVE
    ACTIVITY -- a recording, in the message the user read -- rather than
    as the protocol run it actually was.
    """
    runner = _started_runner()
    refusals: list[ProtocolRunRefusedError] = []
    release_lease = runner._release_scan_led_lease

    def probe_the_tail():
        # First statement of cleanup's finally: run_cleanup has returned
        # and the claim has not been released yet.
        try:
            runner.prepare(**scr_run_kwargs())
        except ProtocolRunRefusedError as refusal:
            refusals.append(refusal)
        release_lease()

    runner._release_scan_led_lease = probe_the_tail
    runner._cleanup(_ABORTED)

    assert refusals, 'a run starting inside the cleanup tail was not refused at all'
    assert refusals[0].reason == 'already_running', (
        f'the cleanup tail refused by the wrong name: {refusals[0].reason}'
    )
    assert runner._state is ProtocolState.IDLE, 'the tail never ended the run'


def test_a_cleanup_whose_steps_fail_still_ends_the_run():
    """Cleanup is fault-tolerant by design: every step runs, failures are
    collected into one summary. The run still has to end -- the guarantee
    that used to sit at the bottom of run_cleanup and now sits in the
    finally that outlives a raise."""
    runner = _started_runner()
    runner._cancel_all_scheduled_events = MagicMock(side_effect=RuntimeError('cancel boom'))

    runner._cleanup(_ABORTED)

    assert runner._state is ProtocolState.IDLE, (
        'a cleanup with a failing step left the run un-ended'
    )
    assert runner._activity_claim.holder is None, 'the activity claim outlived the run'
