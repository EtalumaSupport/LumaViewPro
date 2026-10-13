# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The ratchet summary: what a test observed reaches the end of the run, under xdist too.

The owner is `tests/ratchets.py` and the two conftest hooks that carry a
worker's observations to the controller.
"""

import subprocess
import sys

import pytest

from tests import ratchets
from tests.ast_seams import REPO_ROOT

_RATCHET = (
    'tests/guards/test_the_gui_edits_protocols_through_the_api.py'
    '::test_the_gui_makes_no_new_protocol_edit'
)


def test_a_ratchet_observed_on_a_worker_is_in_the_controllers_summary():
    """An xdist run's log carries the table.

    The controller imports no test module, so an import-time registry is
    empty there: before the observations travelled in `workeroutput`, no
    push-suite log carried a ratchet line (2026-10-12). Two workers, one
    ratchet test, the table names it.
    """
    run = subprocess.run(
        [sys.executable, '-m', 'pytest', '-n', '2', '-p', 'no:cacheprovider', _RATCHET],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'ratchets: current count beside its pin' in run.stdout, run.stdout
    assert 'GUI: protocol edits made outside the API' in run.stdout, run.stdout


def test_an_observation_states_its_pin_the_way_its_rule_needs():
    """A pinned rule with no pin, or an announced count with one, is refused
    at the observation rather than printed wrong."""
    with pytest.raises(ValueError, match='equal'):
        ratchets.observe('x', 1, rule='equal')
    with pytest.raises(ValueError, match='announce'):
        ratchets.observe('x', 1, 1, rule='announce')
    with pytest.raises(ValueError, match='unknown ratchet rule'):
        ratchets.observe('x', 1, 1, rule='floor')
