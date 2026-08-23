# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The api_examples are executable: each runs green in-suite, plus one
standalone subprocess witness.

Each example is the SAME code path both ways -- the examples ARE the
smoke corpus (a hand-rolled per-file import environment was the defect
this replaces). In-suite, the conftest's heavy-dep mocks serve the
imports; standalone, the real installed deps do. The subprocess witness
proves the standalone form still works outside the suite's mock
environment; one witness suffices because the import environment is the
only thing that differs between the two forms.

Per-example wall time is budgeted (each builds a full simulate-mode
scope; see PERFORMANCE_BUDGETS.md `api_example_runtime_s`) -- the
budget is enforced by review of suite timings, not a per-test timer,
so a slow CI box cannot flake the suite on wall clock.
"""

import pathlib
import runpy
import subprocess
import sys

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'api_examples'
EXAMPLES = [
    'basic_capture',
    'multi_channel_capture',
    'z_stack',
    'protocol_execution',
]


@pytest.mark.parametrize('name', EXAMPLES)
def test_example_runs_in_suite(name):
    """Run the example's __main__ path in-process under the suite mocks."""
    runpy.run_path(str(EXAMPLES_DIR / f'{name}.py'), run_name='__main__')


def test_basic_capture_standalone_subprocess():
    """The one standalone witness: the example runs in its own interpreter,
    outside the suite's mock environment, against the real installed deps."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / 'basic_capture.py')],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f'stderr tail: {proc.stderr[-2000:]}'
