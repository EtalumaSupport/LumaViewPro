# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A completion subscriber is HANDED the finished run's directory.

Reading it back off the runner is reachable-wrong, not theoretically
wrong: while a run's files drain, the protocol leg re-enables the
z-stack, composite and standalone-autofocus starters, so the user can
commit a successor whose own run directory replaces the field before
the pending completion callback runs. The post-processing plugins then
receive the successor's directory described as the finished run's.
"""

from __future__ import annotations

import pathlib

import pytest

from modules.protocol_callbacks import ProtocolCallbacks
from tests.test_audit_fixes import _run_cleanup_kwargs


@pytest.fixture
def fired(monkeypatch):
    """Cleanup's UI scheduling, run inline so the callbacks land here."""
    monkeypatch.setattr('modules.protocol_cleanup._schedule_ui', lambda fn, timeout=0: fn(0))
    return []


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


def _close_the_runs_writes(kwargs, callbacks, run_dir):
    """The runner's close of the run's batch, after cleanup: the directory
    is captured by value at the close, as the runner does it."""
    from types import SimpleNamespace

    from modules.sequenced_capture_runner import SequencedCaptureRunner

    runner = SimpleNamespace(
        _disable_saving_artifacts=True,
        _protocol_execution_record=None,
        _callbacks=callbacks,
        _protocol=None,
        _run_dir=run_dir,
        _on_run_idle=None,
        _run_mode=None,
        LOGGER_NAME='TEST',
    )
    SequencedCaptureRunner._close_run_writes(runner, kwargs['write_batch'], kwargs['run_complete'])


class TestBothCompletionLegsCarryTheDirectory:
    def test_the_drained_leg_hands_the_directory_to_both_callbacks(self, fired, monkeypatch):
        from modules.protocol_cleanup import run_cleanup

        run_dir = pathlib.Path('/tmp/the_finished_run')
        callbacks = ProtocolCallbacks(
            run_complete=lambda **kw: fired.append(('run_complete', kw.get('run_dir'))),
            files_complete=lambda **kw: fired.append(('files_complete', kw.get('run_dir'))),
        )
        kwargs = _run_cleanup_kwargs(callbacks=callbacks, run_dir=run_dir)

        run_cleanup(**kwargs)
        # Nothing left to drain: the batch completes at the close, and both
        # callbacks have fired.
        _close_the_runs_writes(kwargs, callbacks, run_dir)

        assert fired == [('run_complete', run_dir), ('files_complete', run_dir)]

    def test_the_draining_leg_hands_the_directory_to_the_deferred_callback(self, fired):
        from modules.protocol_cleanup import run_cleanup
        from modules.protocol_image_writer import RunWriteBatch

        run_dir = pathlib.Path('/tmp/the_finished_run')
        callbacks = ProtocolCallbacks(
            run_complete=lambda **kw: fired.append(('run_complete', kw.get('run_dir'))),
            files_complete=lambda **kw: fired.append(('files_complete', kw.get('run_dir'))),
        )
        lane = _HeldFileLane()
        kwargs = _run_cleanup_kwargs(
            callbacks=callbacks, run_dir=run_dir, write_batch=RunWriteBatch(lane)
        )
        # One of the run's writes is still on its way to the disk.
        kwargs['write_batch'].submit(lambda: None, {}, what='The image x', pace_until=None)

        run_cleanup(**kwargs)
        _close_the_runs_writes(kwargs, callbacks, run_dir)

        # The deferred leg waits rather than fires; the last write landing
        # completes the batch, and files_complete must still carry THIS
        # run's directory then -- which is the whole point of passing by
        # value.
        assert fired == [('run_complete', run_dir)]
        lane.run_held()
        assert fired[-1] == ('files_complete', run_dir)

    def test_a_run_that_made_no_directory_carries_none(self, fired):
        """A run that failed at start nulls its directory before cleanup;
        the dispatch returns early on the None rather than inventing a
        path."""
        from modules.protocol_cleanup import run_cleanup

        callbacks = ProtocolCallbacks(
            run_complete=lambda **kw: fired.append(('run_complete', kw.get('run_dir'))),
        )
        kwargs = _run_cleanup_kwargs(callbacks=callbacks, run_dir=None)

        run_cleanup(**kwargs)
        _close_the_runs_writes(kwargs, callbacks, None)

        assert fired == [('run_complete', None)]


class TestTheRunnerHandsOverItsOwnDirectory:
    def test_cleanup_is_given_the_directory_the_run_created(self, fired, monkeypatch):
        """The runner reads its own field once, on the cleanup path, with
        the claim still held -- not later, from a subscriber."""
        from modules.protocol_image_writer import RunWriteBatch
        from modules.run_outcome import PendingRunOutcome
        from tests.protocol_drives import autofocus_snapshot, protocol_step, scan_ready_runner

        run_dir = pathlib.Path('/tmp/this_runs_dir')
        runner = scan_ready_runner(
            protocol_step(),
            _run_dir=run_dir,
            _original_led_states=None,
            _return_to_position=None,
            _protocol_execution_record=None,
            _autofocus_snapshot=autofocus_snapshot(states={}),
        )
        # The run start() would have committed, with its writes.
        run = PendingRunOutcome()
        runner._run_outcome = run
        runner._write_batch = RunWriteBatch(runner.file_io_executor)

        runner._callbacks = ProtocolCallbacks(
            run_complete=lambda **kw: fired.append(('run_complete', kw.get('run_dir'))),
        )

        # The stack build is the next statement after cleanup and belongs
        # to a different contract; this drive stops at the handover. The
        # directory rides the run's run_complete notice, which cleanup sends:
        # the stand-in sends the notice it was given, as cleanup does.
        monkeypatch.setattr(runner, '_start_hyperstack_build', lambda: None)
        monkeypatch.setattr(
            'modules.sequenced_capture_runner.run_cleanup',
            lambda **kw: kw['run_complete'].send() or True,
        )
        from modules.run_outcome import RunEnding

        runner._cleanup_inner(RunEnding('completed', 'completed', 'Done', 'The run finished.'), run)

        assert fired == [('run_complete', run_dir)]
