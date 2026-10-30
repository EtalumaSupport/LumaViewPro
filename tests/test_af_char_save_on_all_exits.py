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

import modules.autofocus_functions as autofocus_functions
from modules.autofocus_runner import AutofocusRunner
from modules.exceptions import AutofocusAborted
from modules.protocol_image_writer import RunWriteBatch
from modules.sequential_io_executor import SequentialIOExecutor
from tests.af_drives import af_runner_on, raises_the_failure_from, sweep


@pytest.fixture
def write_batch():
    """A run's write batch over a real file lane."""
    lane = SequentialIOExecutor(name='FILE')
    lane.start()
    yield RunWriteBatch(lane)
    lane.shutdown(wait=True)


def _saved_rows(runner, write_batch) -> int | None:
    """Rows of the characterization CSV once the run's writes have landed; None when none was written."""
    write_batch.close()
    assert write_batch.wait_complete(10.0), "the run's writes never landed"
    path = runner.saved_data_path()
    return None if path is None else len(pd.read_csv(path))


def _scored_then(monkeypatch, then):
    """Score the first frame for real, then do ``then`` at the second."""
    score = autofocus_functions.autofocus_score
    scored = []

    def scoring(image):
        scored.append(image)
        if len(scored) == 2:
            then()
        return score(image)

    monkeypatch.setattr('modules.autofocus_functions.autofocus_score', scoring)


class TestSaveLandsOnEveryExitPath:
    """The diagnostic data must be saved on EVERY run() exit (success,
    abort, exception), gated on save_results_to_file. A
    success-branch-only save (the original bug) leaves the eager-made
    results dir empty whenever AF does not complete."""

    def test_success_exit_saves(self, sim_scope, tmp_path, write_batch):
        runner = af_runner_on(sim_scope)
        sweep(
            runner,
            sim_scope,
            save_results_to_file=True,
            results_dir=tmp_path,
            write_batch=write_batch,
        )
        assert _saved_rows(runner, write_batch), 'a successful AF run must save its data'

    def test_exception_exit_saves(self, sim_scope, tmp_path, write_batch, monkeypatch):
        """The save fires from the finally path: an AF loop that raises
        must still save the diagnostic data it collected."""
        runner = af_runner_on(sim_scope)

        def broken():
            raise RuntimeError('focus measure broke')

        _scored_then(monkeypatch, broken)
        with raises_the_failure_from(RuntimeError):
            sweep(
                runner,
                sim_scope,
                save_results_to_file=True,
                results_dir=tmp_path,
                write_batch=write_batch,
            )
        assert _saved_rows(runner, write_batch) == 1, (
            'an AF run that raises must still save the sample it scored -- '
            'the failure data is the whole point of the diagnostic'
        )

    def test_mid_pass_abort_promotes_and_saves(self, sim_scope, tmp_path, write_batch, monkeypatch):
        """An abort after one sample must promote the unpromoted in-pass
        data and save it, so partial scans land on disk."""
        runner = af_runner_on(sim_scope)
        abort_event = threading.Event()
        score = autofocus_functions.autofocus_score

        def score_then_abort(image):
            abort_event.set()
            return score(image)

        monkeypatch.setattr('modules.autofocus_functions.autofocus_score', score_then_abort)
        with pytest.raises(AutofocusAborted):
            sweep(
                runner,
                sim_scope,
                abort_event=abort_event,
                save_results_to_file=True,
                results_dir=tmp_path,
                write_batch=write_batch,
            )
        assert _saved_rows(runner, write_batch) == 1, (
            'the mid-pass sample must be promoted before the save, or the CSV lands empty'
        )

    def test_no_save_when_flag_off(self, sim_scope, tmp_path, write_batch):
        runner = af_runner_on(sim_scope)
        sweep(
            runner,
            sim_scope,
            save_results_to_file=False,
            results_dir=tmp_path,
            write_batch=write_batch,
        )
        assert _saved_rows(runner, write_batch) is None, (
            'non-engineering-mode AF runs must not save the diagnostic data'
        )
        assert list(tmp_path.iterdir()) == [], 'nothing may be written where it was not asked for'


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
