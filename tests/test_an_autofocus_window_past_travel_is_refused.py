# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus window past Z's travel is refused before the stage goes there.

A move past the travel used to be refused mid-sweep, which ended the
autofocus as an unexpected error, and the bottom window was clamped to 0.
The runner judges each window against Z's travel before any move into it
and never narrows one to fit: a focus beyond the travel would come back as
the travel's edge, a plausible wrong result. The first, refined and
extended windows are driven on the simulated scope through the autofocus
thread, the way a run drives them. The final approach from below, which a
real sweep reaches too rarely to drive there, is pinned on the drive
helper's stand-in scope (``tests/af_drives.py``) until a start that reaches
it on the simulator is found.
"""

from __future__ import annotations

import time

import pytest

import modules.autofocus_runner as autofocus_runner_module
from modules.autofocus_runner import AutofocusRunner
from modules.autofocus_thread import AutofocusThread
from modules.exceptions import AutofocusFailedError
from tests.af_drives import AF_CENTER_Z, af_runner_and_scope, drive_af
from tests.protocol_drives import held_run_claim, run_identity
from tests.scope_fakes import home_sim_scope


@pytest.fixture
def scope(sim_scope):
    """The simulated scope with Z known, streaming for the sweep's frames."""
    return home_sim_scope(sim_scope)


def _autofocus_from(scope, z, monkeypatch, peak):
    """Run one autofocus from ``z`` whose score peaks at ``peak``.

    The score falls with distance from ``peak``, whatever the frame
    shows, so where the sweep goes is decided by the curve alone.
    """
    monkeypatch.setattr(
        'modules.autofocus_functions.focus_function',
        lambda image: 1e6 - abs(scope.motion.get_current_position('Z') - peak),
    )
    scope.motion.move_absolute('Z', z)
    while scope.motion.is_moving():
        time.sleep(0.01)
    thread = AutofocusThread(afe=AutofocusRunner(scope=scope))
    thread.start()
    try:
        return thread.run_autofocus(
            run=run_identity('autofocus'),
            objective_id='10x Oly',
            led_color='BF',
            led_illumination=50.0,
            camera_gain=1.0,
            camera_exposure=50.0,
            led_lease=scope.illumination.acquire_led_lease('protocol', claim=held_run_claim()),
        ).result(timeout=15.0)
    finally:
        thread.stop(timeout=2.0)


@pytest.mark.parametrize('edge', ['min', 'max'])
def test_a_first_window_past_travel_is_refused_before_the_stage_moves(scope, monkeypatch, edge):
    # Started at a travel limit, the first window reaches past it. The
    # bottom used to be clamped to 0 and the top ran on until a sweep
    # move was refused as an unexpected error.
    limit = scope.motion.get_axis_limits('Z')[edge]
    with pytest.raises(AutofocusFailedError) as refused:
        _autofocus_from(scope, limit, monkeypatch, peak=limit)

    assert refused.value.reason == 'out_of_travel'
    assert scope.motion.get_current_position('Z') == pytest.approx(limit)


@pytest.mark.parametrize('edge', ['min', 'max'])
def test_a_sweep_led_past_travel_is_refused_and_z_goes_back(scope, monkeypatch, edge):
    # The first window fits, but the curve peaks at the travel limit: at
    # the bottom the refined window opens below it, at the top the
    # extension runs past it. Either is refused before the stage moves
    # there, and Z returns to where the autofocus started.
    limit = scope.motion.get_axis_limits('Z')[edge]
    af_range = scope.objective_helper.get_objective_info(objective_id='10x Oly')['AF_range']
    start = limit + (af_range + 5.0) * (1 if edge == 'min' else -1)
    with pytest.raises(AutofocusFailedError) as refused:
        _autofocus_from(scope, start, monkeypatch, peak=limit)

    assert refused.value.reason == 'out_of_travel'
    # The restore is a move; read Z once it has arrived, within a motor
    # step, since the stage lands on its own step grid.
    while scope.motion.is_moving():
        time.sleep(0.01)
    assert scope.motion.get_current_position('Z') == pytest.approx(start, abs=0.01)


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
    """The runner reports the refusal and then raises it: today's funnel,
    which ruling R2 of 2026-10-09 (a funnel raises only; the caller reports)
    flips with its build. Until then this pins what the code does."""
    monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 1.0)
    runner, scope = af_runner_and_scope()
    # The first window (centre +- AF_range) reaches below this minimum.
    scope.motion.get_axis_limits.return_value = {'min': AF_CENTER_Z - 5.0, 'max': 14000.0}

    with pytest.raises(AutofocusFailedError):
        drive_af(runner)

    assert [o.reason for o in reported] == ['out_of_travel']
