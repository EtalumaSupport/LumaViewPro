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


class TestBothCompletionLegsCarryTheDirectory:
    def test_the_drained_leg_hands_the_directory_to_both_callbacks(self, fired, monkeypatch):
        from modules.protocol_cleanup import run_cleanup

        run_dir = pathlib.Path('/tmp/the_finished_run')
        kwargs = _run_cleanup_kwargs(
            callbacks=ProtocolCallbacks(
                run_complete=lambda **kw: fired.append(('run_complete', kw.get('run_dir'))),
                files_complete=lambda **kw: fired.append(('files_complete', kw.get('run_dir'))),
            ),
            run_dir=run_dir,
        )
        # Nothing left to drain: both callbacks fire on the spot.
        kwargs['file_io_executor'].is_protocol_queue_active.return_value = False

        run_cleanup(**kwargs)

        assert fired == [('run_complete', run_dir), ('files_complete', run_dir)]

    def test_the_draining_leg_hands_the_directory_to_the_deferred_callback(self, fired):
        from modules.protocol_cleanup import run_cleanup

        run_dir = pathlib.Path('/tmp/the_finished_run')
        kwargs = _run_cleanup_kwargs(
            callbacks=ProtocolCallbacks(
                run_complete=lambda **kw: fired.append(('run_complete', kw.get('run_dir'))),
                files_complete=lambda **kw: fired.append(('files_complete', kw.get('run_dir'))),
            ),
            run_dir=run_dir,
        )
        file_io_executor = kwargs['file_io_executor']
        file_io_executor.is_protocol_queue_active.return_value = True

        run_cleanup(**kwargs)

        # The deferred leg registers rather than fires; the executor calls
        # it when the queue empties, and it must still carry THIS run's
        # directory then -- which is the whole point of passing by value.
        assert fired == [('run_complete', run_dir)]
        registered = file_io_executor.set_protocol_complete_callback.call_args.kwargs['callback']
        registered()
        assert fired[-1] == ('files_complete', run_dir)

    def test_a_run_that_made_no_directory_carries_none(self, fired):
        """A run that failed at start nulls its directory before cleanup;
        the dispatch returns early on the None rather than inventing a
        path."""
        from modules.protocol_cleanup import run_cleanup

        kwargs = _run_cleanup_kwargs(
            callbacks=ProtocolCallbacks(
                run_complete=lambda **kw: fired.append(('run_complete', kw.get('run_dir'))),
            ),
            run_dir=None,
        )
        kwargs['file_io_executor'].is_protocol_queue_active.return_value = False

        run_cleanup(**kwargs)

        assert fired == [('run_complete', None)]


class TestTheRunnerHandsOverItsOwnDirectory:
    def test_cleanup_is_given_the_directory_the_run_created(self, monkeypatch):
        """The runner reads its own field once, on the cleanup path, with
        the claim still held -- not later, from a subscriber."""
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

        seen = {}
        # The stack build is the next statement after cleanup and belongs
        # to a different contract; this drive stops at the handover.
        monkeypatch.setattr(runner, '_start_hyperstack_build', lambda: None)
        monkeypatch.setattr(
            'modules.sequenced_capture_runner.run_cleanup',
            lambda **kw: seen.update(kw) or True,
        )
        from modules.run_outcome import RunEnding

        runner._cleanup_inner(RunEnding('completed', 'completed', 'Done', 'The run finished.'))

        assert seen['run_dir'] == run_dir
