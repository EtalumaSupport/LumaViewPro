# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus window past Z's travel is refused before the stage goes there.

A move past the travel used to be refused mid-sweep, which ended the
autofocus as an unexpected error, and the bottom window was clamped to 0.
The runner judges each window against Z's travel before any move into it
and never narrows one to fit: a focus beyond the travel would come back as
the travel's edge, a plausible wrong result. The first, refined and
extended windows are driven on the simulated scope through the autofocus
thread, the way a run drives them; the final approach from below is driven
on the runner itself, from a start and a curve chosen so every window
before it fits.
"""

from __future__ import annotations

import math
import time

import pytest

import modules.autofocus_runner as autofocus_runner_module
from modules.autofocus_runner import AutofocusRunner
from modules.autofocus_thread import AutofocusThread
from modules.exceptions import AutofocusFailedError, CaptureError, RefusalCause
from modules.notification_center import OutcomeKind, outcome_of
from tests.af_drives import af_runner_on, park_z, sweep, z_asked
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
        'modules.autofocus_functions.autofocus_score',
        lambda image: 1e6 - abs(scope.motion.get_current_position('Z') - peak),
    )
    scope.motion.move_z(z)
    while scope.motion.is_any_axis_moving():
        time.sleep(0.01)
    thread = AutofocusThread(afe=AutofocusRunner(scope=scope))
    thread.start()
    try:
        return thread.run_autofocus(
            run=run_identity(),
            objective_id='10x Oly',
            led_color='BF',
            led_illumination=5.0,
            camera_gain=1.0,
            camera_exposure=50.0,
            led_lease=scope.illumination.acquire_led_lease('protocol', claim=held_run_claim()),
        ).result(timeout=15.0)
    finally:
        thread.stop(timeout=2.0)


@pytest.mark.parametrize('edge', ['min_um', 'max_um'])
def test_a_first_window_past_travel_is_refused_before_the_stage_moves(scope, monkeypatch, edge):
    # Started at a travel limit, the first window reaches past it. The
    # bottom used to be clamped to 0 and the top ran on until a sweep
    # move was refused as an unexpected error.
    limit = getattr(scope.motion.get_axis_limits('Z'), edge)
    with pytest.raises(CaptureError) as refused:
        _autofocus_from(scope, limit, monkeypatch, peak=limit)

    # Declined before anything moved, for where Z stands: a refusal, not a
    # failure, and the same autofocus started further from the limit runs.
    outcome = outcome_of(refused.value)
    assert (outcome.kind, outcome.cause, outcome.reason) == (
        OutcomeKind.REFUSAL,
        RefusalCause.STATE,
        'out_of_travel',
    )
    assert scope.motion.get_current_position('Z') == pytest.approx(limit)


@pytest.mark.parametrize('edge', ['min_um', 'max_um'])
def test_a_sweep_led_past_travel_is_refused_and_z_goes_back(scope, monkeypatch, edge):
    # The first window fits, but the curve peaks at the travel limit: at
    # the bottom the refined window opens below it, at the top the
    # extension runs past it. Either is refused before the stage moves
    # there, and Z returns to where the autofocus started.
    limit = getattr(scope.motion.get_axis_limits('Z'), edge)
    af_range = scope.objectives['10x Oly'].autofocus_half_range_um
    start = limit + (af_range + 5.0) * (1 if edge == 'min_um' else -1)
    with pytest.raises(AutofocusFailedError) as refused:
        _autofocus_from(scope, start, monkeypatch, peak=limit)

    # The sweep had moved the stage when it found it could not go on: a
    # failure, where the same refusal before the first move is not.
    assert refused.value.reason == 'travel_exhausted'
    assert outcome_of(refused.value).kind == OutcomeKind.FAULT
    # The restore is a move; read Z once it has arrived, within a motor
    # step, since the stage lands on its own step grid.
    while scope.motion.is_any_axis_moving():
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


def test_a_final_approach_past_the_travel_minimum_is_refused(sim_scope, monkeypatch):
    # A curve peaking 2 um above the floor, swept from 123 um above it:
    # the first window's samples (step 36) start at floor + 48, its nearest
    # to the peak; the refined window (+-36, step 12) starts at floor + 12;
    # the last window (+-12, step 8) finds the peak at the floor, and the
    # approach from one step below it would leave travel.
    runner = af_runner_on(sim_scope)
    floor = sim_scope.motion.get_axis_limits('Z').min_um
    peak = floor + 2.0
    monkeypatch.setattr(
        'modules.autofocus_functions.autofocus_score',
        lambda image: (
            1e6 * math.exp(-(((sim_scope.motion.get_current_position('Z') - peak) / 15.0) ** 2))
        ),
    )
    park_z(sim_scope, floor + 123.0)
    asked = z_asked(sim_scope, monkeypatch)

    with pytest.raises(AutofocusFailedError) as refused:
        sweep(runner, sim_scope)

    assert refused.value.reason == 'travel_exhausted'
    assert all(z >= floor for z in asked), asked


def test_a_refusal_is_reported_once_as_itself(sim_scope, reported):
    """The runner reports the first window's refusal and raises the same object.

    The autofocus worker is the boundary of its own flight: inside a run
    its future is read only by a log line, so no caller above it reports.
    """
    runner = af_runner_on(sim_scope)
    # At the floor, the first window (centre +- AF_range) reaches below it.
    park_z(sim_scope, sim_scope.motion.get_axis_limits('Z').min_um)

    with pytest.raises(CaptureError) as raised:
        sweep(runner, sim_scope)

    assert raised.value.reason == 'out_of_travel'
    assert outcome_of(raised.value).kind == OutcomeKind.REFUSAL
    assert reported == [raised.value]


def test_an_unexpected_failure_is_raised_as_the_failure_it_reported(
    sim_scope, monkeypatch, reported
):
    """One failure is one object: the one reported, raised, its cause chained."""

    def broken(image):
        raise RuntimeError('focus measure broke')

    runner = af_runner_on(sim_scope)
    monkeypatch.setattr('modules.autofocus_functions.autofocus_score', broken)

    with pytest.raises(AutofocusFailedError) as raised:
        sweep(runner, sim_scope)

    assert raised.value.reason == 'unexpected_error'
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert reported == [raised.value]
