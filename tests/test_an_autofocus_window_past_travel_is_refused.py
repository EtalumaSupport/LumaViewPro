# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus window past Z's travel is refused, and reported as that refusal.

The simulated-scope tests in `test_integration.py` drive the first, extended
and refined windows. These pin the two paths a real sweep reaches too rarely
to drive there: the final approach from below, which can leave the travel
when the last window's floor sits within one AF_min step of the minimum, and
the report, which names the refusal rather than an unexpected error.
"""

from __future__ import annotations

import pytest

import modules.autofocus_runner as autofocus_runner_module
from modules.exceptions import AutofocusFailedError
from tests.af_drives import AF_CENTER_Z, af_runner_and_scope, drive_af


@pytest.fixture
def reported(monkeypatch):
    """Every outcome autofocus reports."""
    outcomes = []
    monkeypatch.setattr(
        autofocus_runner_module.notifications,
        'report_outcome',
        lambda outcome, **kwargs: outcomes.append(outcome),
    )
    return outcomes


def _z_moves(scope):
    return [c.args[1] for c in scope.motion.move_absolute.call_args_list if c.args[0] == 'Z']


def test_a_final_approach_past_the_travel_minimum_is_refused(monkeypatch, reported):
    # The stand-in completes each pass on one sample: the coarse pass picks
    # the centre, the last pass the floor of its window, which is the travel
    # minimum; the approach from one AF_min step below it would leave travel.
    monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 1.0)
    runner, scope = af_runner_and_scope()
    floor = AF_CENTER_Z - 30.0  # the refined window's floor: centre - AF_max
    scope.motion.get_axis_limits.return_value = {'min': floor, 'max': 14000.0}
    bests = iter([AF_CENTER_Z, floor])
    runner._find_best = lambda df: next(bests)

    with pytest.raises(AutofocusFailedError) as refused:
        drive_af(runner)

    assert refused.value.reason == 'out_of_travel'
    assert all(z >= floor for z in _z_moves(scope)), _z_moves(scope)


def test_a_refusal_is_reported_once_as_itself(monkeypatch, reported):
    monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 1.0)
    runner, scope = af_runner_and_scope()
    # The first window (centre +- AF_range) reaches below this minimum.
    scope.motion.get_axis_limits.return_value = {'min': AF_CENTER_Z - 5.0, 'max': 14000.0}

    with pytest.raises(AutofocusFailedError):
        drive_af(runner)

    assert [o.reason for o in reported] == ['out_of_travel']
