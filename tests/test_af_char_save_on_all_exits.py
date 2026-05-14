# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for issue #650 -- AF Characterization folder empty.

Bench observation 2026-05-14: Eric reported `<live_folder>/Autofocus
Characterization/<timestamp>/` folders appearing on disk with no CSV
or plot inside. AF Characterization is a diagnostic tool enabled in
engineering mode; the failure data is the WHOLE POINT of running it.

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

import ast
import datetime
import pathlib
from unittest.mock import MagicMock

import pandas as pd
import pytest

from modules.autofocus_runner import AutofocusRunner


def _autofocus_runner_source() -> str:
    return (pathlib.Path(__file__).resolve().parent.parent
            / "modules" / "autofocus_runner.py").read_text()


def _function_source(source: str, func_name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            text = ast.get_source_segment(source, node)
            if text is None:
                raise AssertionError(f"could not extract source for {func_name!r}")
            return text
    raise AssertionError(f"function {func_name!r} not found in source")


def _finally_block_of_run(source: str) -> str:
    """Return the source text of the `finally` clause inside `run()`.
    AST-walked so we don't false-match a `finally` in some nested
    helper. The finally-clause body is the list of stmts under the
    Try node whose parent's name is 'run'."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Try) and sub.finalbody:
                    parts = [ast.get_source_segment(source, stmt)
                             for stmt in sub.finalbody]
                    return "\n".join(p for p in parts if p is not None)
    raise AssertionError("run() finally block not found")


class TestSaveQueuedFromFinallyNotIterate:
    """Structural: the save queue lives in run()'s finally, not in
    _iterate()'s success branch. Locks the fix shape so a future
    refactor that moves it back to _iterate fires the regression."""

    def test_iterate_does_not_queue_save(self):
        body = _function_source(_autofocus_runner_source(), "_iterate")
        assert "self._save_autofocus_data" not in body, (
            "_iterate() must not queue _save_autofocus_data -- the save "
            "must fire from run()'s finally block so abort, exception, "
            "and degenerate-curve exits also save the diagnostic data."
        )

    def test_finally_block_queues_save(self):
        finally_text = _finally_block_of_run(_autofocus_runner_source())
        assert "self._save_autofocus_data" in finally_text, (
            "run()'s finally block must queue _save_autofocus_data so "
            "AF Characterization data lands on disk on every exit path."
        )

    def test_finally_block_gates_on_save_results_to_file(self):
        finally_text = _finally_block_of_run(_autofocus_runner_source())
        assert "self._save_results_to_file" in finally_text, (
            "Save queue in finally must be gated on _save_results_to_file "
            "so non-engineering-mode AF runs don't write empty CSVs."
        )

    def test_finally_block_promotes_partial_pass_data(self):
        """A mid-coarse-pass abort leaves data in `_af_data_pass` that
        never got promoted to `_af_data_full`. The finally block must
        promote it before queueing the save, or partial scans land an
        empty CSV even with the new save path."""
        finally_text = _finally_block_of_run(_autofocus_runner_source())
        assert "self._af_data_full.extend(self._af_data_pass)" in finally_text, (
            "run()'s finally block must extend _af_data_pass into "
            "_af_data_full before queueing the save so mid-pass aborts "
            "still produce diagnostic data."
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
        assert files == [], (
            f"Expected no files written for empty data, found: {files}"
        )

    def test_populated_data_writes_csv(self, tmp_path):
        """Smoke: populated _af_data_full produces a CSV file in the
        results dir. Plot may or may not write depending on matplotlib
        availability; the CSV is the authoritative diagnostic data."""
        runner = self._stub_runner(tmp_path)
        runner._af_data_full = [
            {"position": 1000.0, "score": 10.5},
            {"position": 1010.0, "score": 25.3},
            {"position": 1020.0, "score": 18.7},
        ]
        runner._save_autofocus_data()

        csvs = list(tmp_path.glob("autofocus_data_*.csv"))
        assert len(csvs) == 1, (
            f"Expected exactly 1 autofocus_data_*.csv, found: "
            f"{[p.name for p in tmp_path.iterdir()]}"
        )
        df = pd.read_csv(csvs[0])
        assert list(df.columns) == ["position", "score"]
        assert len(df) == 3
        assert df["position"].tolist() == [1000.0, 1010.0, 1020.0]


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
        parent = tmp_path / "Autofocus Characterization"
        # Method does mkdir + returns the timestamped subdir.
        result = runner._allocate_results_dir(parent)
        assert parent.exists(), "Parent dir must be eagerly created"
        assert result.exists(), "Timestamped subdir must be eagerly created"
        assert result.parent == parent
