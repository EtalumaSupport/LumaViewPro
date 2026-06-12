# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for AF Characterization data save on every exit path.

Bench observation: `<live_folder>/Autofocus Characterization/<timestamp>/`
folders were appearing on disk with no CSV or plot inside. AF
Characterization is a diagnostic tool enabled in engineering mode; the
failure data is the WHOLE POINT of running it.

Root cause: `_save_autofocus_data` was queued only from the success
branch inside `_iterate()` (the `if self._last_pass:` block). Any
non-success exit (user abort, exception, degenerate-curve abort)
skipped the save. Combined with the eager mkdir at
`_allocate_results_dir`, the folder was created up front but stayed
empty whenever AF didn't complete successfully.

Fix: move the save queue to `run()`'s `finally` block, gated on
`_save_results_to_file`. Extend any unpromoted `_af_data_pass` into
`_af_data_full` first so a mid-pass abort still produces data on
disk. Keep the empty-data early-return inside `_save_autofocus_data`
itself.
"""

from __future__ import annotations

import pathlib
import threading

import pandas as pd
import pytest

from modules.autofocus_runner import AutofocusRunner
from modules.exceptions import AutofocusAborted
from tests.af_drives import af_runner_and_scope, drive_af


class TestSaveQueuedOnEveryExitPath:
    """The diagnostic-data save must be queued on EVERY run() exit
    (success, abort, exception), gated on save_results_to_file. A
    success-branch-only save (the original bug) leaves the eager-made
    results dir empty whenever AF does not complete."""

    def _queued_save_tasks(self, runner):
        return [
            call.args[0]
            for call in runner._file_io_executor.protocol_put.call_args_list
            if call.args[0].action == runner._save_autofocus_data
        ]

    def test_success_exit_queues_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, _scope = af_runner_and_scope()
        drive_af(runner, save_results_to_file=True, results_dir=tmp_path)
        assert self._queued_save_tasks(runner), (
            'a successful AF run must queue _save_autofocus_data'
        )

    def test_exception_exit_queues_save(self, tmp_path, monkeypatch):
        """The save fires from the finally path: an AF loop that raises
        must still queue the diagnostic save."""
        runner, scope = af_runner_and_scope()
        scope.imaging.capture_and_wait.side_effect = RuntimeError('camera fault')
        with pytest.raises(RuntimeError, match='camera fault'):
            drive_af(runner, save_results_to_file=True, results_dir=tmp_path)
        assert self._queued_save_tasks(runner), (
            'an AF run that raises must still queue _save_autofocus_data -- '
            'the failure data is the whole point of the diagnostic'
        )

    def test_mid_pass_abort_promotes_and_queues_save(self, tmp_path, monkeypatch):
        """An abort after one sample must promote the unpromoted in-pass
        data and queue the save, so partial scans land on disk."""
        abort_event = threading.Event()

        def score_then_abort(image):
            abort_event.set()
            return 7.0

        monkeypatch.setattr('modules.autofocus_functions.focus_function', score_then_abort)
        runner, _scope = af_runner_and_scope()
        with pytest.raises(AutofocusAborted):
            drive_af(
                runner,
                abort_event=abort_event,
                save_results_to_file=True,
                results_dir=tmp_path,
            )
        assert self._queued_save_tasks(runner), (
            'an aborted AF run must still queue _save_autofocus_data'
        )
        assert len(runner._af_data_full) == 1 and runner._af_data_pass == [], (
            'the mid-pass sample must be promoted to _af_data_full before '
            'the save is queued, or the CSV lands empty'
        )

    def test_no_save_queued_when_flag_off(self, monkeypatch):
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, _scope = af_runner_and_scope()
        drive_af(runner, save_results_to_file=False)
        assert not self._queued_save_tasks(runner), (
            "non-engineering-mode AF runs must not queue the diagnostic save"
        )


class TestSaveAutofocusDataBehavior:
    """Behavioral: _save_autofocus_data preserves its empty-data
    early-return contract (handles abort-before-any-frame), and
    writes a CSV when data is present."""

    def _stub_runner(self, tmp_path: pathlib.Path) -> AutofocusRunner:
        """Construct a runner via __new__ and populate just the slots
        _save_autofocus_data reads. Avoids the heavy __init__ scope /
        executor / objective_loader graph."""
        runner = AutofocusRunner.__new__(AutofocusRunner)
        runner._af_data_full = []
        runner._results_dir = tmp_path
        return runner

    def test_empty_data_early_returns_without_writing(self, tmp_path):
        """When _af_data_full is empty (true no-data abort), the save
        must early-return -- writing an empty CSV would be confusing."""
        runner = self._stub_runner(tmp_path)
        runner._af_data_full = []
        runner._save_autofocus_data()

        files = list(tmp_path.iterdir())
        assert files == [], f'Expected no files written for empty data, found: {files}'

    def test_populated_data_writes_csv(self, tmp_path):
        """Smoke: populated _af_data_full produces a CSV file in the
        results dir. Plot may or may not write depending on matplotlib
        availability; the CSV is the authoritative diagnostic data."""
        runner = self._stub_runner(tmp_path)
        runner._af_data_full = [
            {'position': 1000.0, 'score': 10.5},
            {'position': 1010.0, 'score': 25.3},
            {'position': 1020.0, 'score': 18.7},
        ]
        runner._save_autofocus_data()

        csvs = list(tmp_path.glob('autofocus_data_*.csv'))
        assert len(csvs) == 1, (
            f'Expected exactly 1 autofocus_data_*.csv, found: '
            f'{[p.name for p in tmp_path.iterdir()]}'
        )
        df = pd.read_csv(csvs[0])
        assert list(df.columns) == ['position', 'score']
        assert len(df) == 3
        assert df['position'].tolist() == [1000.0, 1010.0, 1020.0]


class TestAllocateResultsDirStillEagerMkdir:
    """The eager mkdir in _allocate_results_dir is kept after the fix.
    Reason: the per-run timestamped subdir is the durable marker that
    'an AF run started at <ts>'. With the save now in the finally,
    the directory + CSV pair tells the full story (CSV present means
    AF collected at least one frame; CSV absent means AF aborted
    before any frame). Removing the eager mkdir would lose that
    distinction."""

    def test_allocate_results_dir_creates_parent_and_subdir(self, tmp_path):
        runner = AutofocusRunner.__new__(AutofocusRunner)
        parent = tmp_path / 'Autofocus Characterization'
        # Method does mkdir + returns the timestamped subdir.
        result = runner._allocate_results_dir(parent)
        assert parent.exists(), 'Parent dir must be eagerly created'
        assert result.exists(), 'Timestamped subdir must be eagerly created'
        assert result.parent == parent
