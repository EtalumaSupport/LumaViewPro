# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run-end subscriber is HANDED the finished run's directory.

Reading it back off the runner is reachable-wrong, not theoretically
wrong: while a run's files drain, the protocol leg re-enables the
z-stack, composite and standalone-autofocus starters, so the user can
commit a successor whose own run directory replaces the field before
the pending run_ended or files_written handler runs. The post-processing plugins then
receive the successor's directory described as the finished run's.
"""

from __future__ import annotations

import pathlib

import pytest

from modules.run_events import RunEvents
from tests.test_audit_fixes import _run_cleanup_kwargs


@pytest.fixture
def fired():
    """What the run's events were handed, in order; headless delivery runs inline."""
    return []


def _events(fired, *, files=True):
    return RunEvents(
        run_ended=lambda outcome, run_dir, protocol: fired.append(('run_ended', run_dir)),
        files_written=(
            (lambda run_dir, files: fired.append(('files_written', run_dir))) if files else None
        ),
    )


class _HeldFileLane:
    """A FILE lane that takes a write and holds it until the test runs it."""

    def __init__(self):
        self.held = []

    def put(self, task, return_future=False):
        from modules.sequential_io_executor import ENQUEUED

        self.held.append(task)
        return ENQUEUED

    def run_held(self):
        while self.held:
            task = self.held.pop(0)
            task.action(*task.args, **task.kwargs)


def _end_the_run(kwargs, events, run_dir):
    """The runner's end of the run, after cleanup: run_ended and the files'
    completion are built by value before the release and told after it."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from modules.protocol_cleanup import send_run_ended
    from modules.sequenced_capture_runner import SequencedCaptureRunner

    runner = SimpleNamespace(
        _disable_saving_artifacts=True,
        _protocol_execution_record=None,
        _events=events,
        _protocol=None,
        _run_dir=run_dir,
        _on_run_idle=None,
        _run_mode=None,
        LOGGER_NAME='TEST',
    )
    batch = kwargs['write_batch']
    run = MagicMock()
    # A plain run's outcome has settled by the release.
    run._pending.when_settled = lambda tell: tell(MagicMock(status='completed'))
    files_written = SequencedCaptureRunner._close_run_writes(runner, batch, kwargs['ending'], run)
    # After the release.
    send_run_ended(events, run, protocol=None, run_dir=run_dir)
    batch.when_complete(files_written)


class TestBothCompletionLegsCarryTheDirectory:
    def test_the_drained_leg_hands_the_directory_to_both_events(self, fired, monkeypatch):
        from modules.protocol_cleanup import run_cleanup

        run_dir = pathlib.Path('/tmp/the_finished_run')
        events = _events(fired)
        kwargs = _run_cleanup_kwargs()

        run_cleanup(**kwargs)
        # Nothing left to drain: the batch completes at the close, and both
        # events fire once the run has let go.
        _end_the_run(kwargs, events, run_dir)

        assert fired == [('run_ended', run_dir), ('files_written', run_dir)]

    def test_the_draining_leg_hands_the_directory_to_the_deferred_event(self, fired):
        from modules.protocol_cleanup import run_cleanup
        from modules.protocol_image_writer import RunWriteBatch

        run_dir = pathlib.Path('/tmp/the_finished_run')
        events = _events(fired)
        lane = _HeldFileLane()
        kwargs = _run_cleanup_kwargs(write_batch=RunWriteBatch(lane))
        # One of the run's writes is still on its way to the disk.
        kwargs['write_batch'].submit(lambda: None, {}, what='The image x', pace_until=None)

        run_cleanup(**kwargs)
        _end_the_run(kwargs, events, run_dir)

        # The deferred leg waits rather than fires; the last write landing
        # completes the batch, and files_written must still carry THIS
        # run's directory then -- which is the whole point of passing by
        # value.
        assert fired == [('run_ended', run_dir)]
        lane.run_held()
        assert fired[-1] == ('files_written', run_dir)

    def test_a_run_that_made_no_directory_carries_none(self, fired):
        """A run that failed at start nulls its directory before cleanup;
        the dispatch returns early on the None rather than inventing a
        path."""
        from modules.protocol_cleanup import run_cleanup

        events = _events(fired, files=False)
        kwargs = _run_cleanup_kwargs()

        run_cleanup(**kwargs)
        _end_the_run(kwargs, events, None)

        assert fired == [('run_ended', None)]


class TestTheRunnerHandsOverItsOwnDirectory:
    def test_cleanup_is_given_the_directory_the_run_created(self, fired, monkeypatch):
        """The runner reads its own field once, on the cleanup path, with
        the claim still held -- not later, when run_ended is sent."""
        from modules.protocol_image_writer import RunWriteBatch
        from modules.run_outcome import PendingRunOutcome
        from modules.sequenced_capture_runner import RunHandle
        from tests.protocol_drives import protocol_step, scan_ready_runner

        run_dir = pathlib.Path('/tmp/this_runs_dir')
        runner = scan_ready_runner(
            protocol_step(),
            _run_dir=run_dir,
            _original_led_states=None,
            _return_to_position=None,
            _protocol_execution_record=None,
        )
        # The run start() would have committed, with its writes.
        runner._run_outcome = PendingRunOutcome()
        runner._write_batch = RunWriteBatch(runner.file_io_executor)
        run = runner._run_handle = RunHandle(runner, runner._run_outcome, runner._write_batch)

        runner._events = _events(fired, files=False)

        # The stack build is the next statement after cleanup and belongs
        # to a different contract; this drive stops at the handover. The
        # directory rides the run's run_ended, captured by value in cleanup
        # and sent once the run has let go and its outcome has settled.
        monkeypatch.setattr(runner, '_start_hyperstack_build', lambda: None)
        monkeypatch.setattr('modules.sequenced_capture_runner.run_cleanup', lambda **kw: True)
        from modules.run_outcome import RunEnding

        after_end = []
        runner._cleanup_inner(
            RunEnding('completed', 'completed', 'Done', 'The run finished.'), run, after_end
        )
        # A successor's setup replaces the runner's field before run_ended
        # goes out; run_ended still carries this run's.
        runner._run_dir = pathlib.Path('/tmp/the_successors_dir')
        for tell in after_end:
            tell()

        assert fired == [('run_ended', run_dir)]
