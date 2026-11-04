# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's summary names the Python it ran on, under xdist and -q.

LumaViewPro supports 3.12 and 3.13, and a test can pass on one and fail on
the other; two runs of one commit that disagree are told apart by this line.
The owner is `pytest_terminal_summary` in `tests/conftest.py`.
"""

import platform
import subprocess
import sys

from tests.ast_seams import REPO_ROOT

_CHEAP_TEST = 'tests/test_ratchets.py::test_an_observation_states_its_pin_the_way_its_rule_needs'


def test_an_xdist_run_names_its_interpreter_in_its_summary():
    """The push suite runs under `-n auto` with `-q` from `addopts`, which drops
    pytest's own platform header; the controller's summary states the version
    and the executable instead."""
    run = subprocess.run(
        [sys.executable, '-m', 'pytest', '-n', '2', '-p', 'no:cacheprovider', _CHEAP_TEST],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert f'Python {platform.python_version()} at {sys.executable}' in run.stdout, run.stdout
