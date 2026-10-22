# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A sweep a person stopped leaves Z where the stop left it.

An autofocus that ends without choosing a focus puts Z back where it began,
so a failed sweep is a no-op in position. A stop is not a failure: the
person pressed STOP or the run's Stop, and putting Z back is a move made
after they did (Eric, 2026-10-18: leave Z where it stopped). The run's
cleanup aborts the sweep with ``stay_put`` for those endings; any other
abort, a fault's included, still restores.
"""

from __future__ import annotations

import threading

import pytest

from modules.exceptions import AutofocusStopped
from tests.af_drives import AF_CENTER_Z, af_runner_and_scope, drive_af
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT
from tests.test_a_stop_press_ends_the_live_run import (  # noqa: F401 (the fixture)
    session,
    start_an_autofocus_held_mid_sweep,
)


def _z_moves(scope):
    """Every absolute Z move autofocus issued, started or waited, in order."""
    return [
        call.args[0] for call in scope.motion.mock_calls if call[0] in ('start_move_z', 'move_z')
    ]


def _aborted_sweep(stay_put: bool):
    """Drive a sweep whose abort is already set; return the Z moves it made."""
    runner, scope = af_runner_and_scope()
    abort = threading.Event()
    abort.set()
    stays = threading.Event()
    if stay_put:
        stays.set()
    with pytest.raises(AutofocusStopped):
        drive_af(runner, abort_event=abort, stay_put=stays)
    return _z_moves(scope)


def test_a_stopped_sweep_leaves_z_where_it_stopped():
    moves = _aborted_sweep(stay_put=True)

    assert AF_CENTER_Z not in moves[1:], f'Z was put back after the stop: {moves}'


def test_an_abort_that_is_no_stop_still_puts_z_back():
    moves = _aborted_sweep(stay_put=False)

    assert moves and moves[-1] == AF_CENTER_Z, moves


# The run's cleanup reads its ending to decide: each ending in STOP_REASONS
# leaves Z, any other puts it back. Each is a different ending reaching it.
@pytest.mark.parametrize(
    'end, reason',
    [
        (lambda session, run: run.stop(), 'stopped'),
        (lambda session, run: session.scope.motion.stop_motion(), 'stage_stopped'),
    ],
    ids=['the_runs_stop', 'a_stop_press'],
)
def test_an_autofocus_run_stopped_mid_sweep_leaves_z_where_it_stopped(
    session, end, reason, monkeypatch
):
    motion = session.scope.motion
    # Every Z target asked, so a red says which it was: a move back to 5000
    # (the sweep put Z back) or none (5000.0 read where Z had stopped).
    asked = []
    for member in ('start_move_z', 'move_z'):
        real = getattr(motion, member)

        def spy(position, *args, _real=real, _member=member, **kwargs):
            asked.append((_member, position))
            return _real(position, *args, **kwargs)

        monkeypatch.setattr(motion, member, spy)
    run = start_an_autofocus_held_mid_sweep(session)

    end(session, run)
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', reason)
    z = motion.get_current_position(axis='Z')
    assert z != 5000.0, (
        f'Z was put back after the stop: Z {z} ({motion.get_axis_state("Z")}), '
        f'Z asked {asked}, outcome {outcome.message!r}'
    )


def test_an_autofocus_run_a_fault_ends_mid_sweep_puts_z_back(session):
    run = start_an_autofocus_held_mid_sweep(session)

    session.sequenced_capture_runner.abort_run_fatal('camera_failure', 'The camera failed.')
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('failed', 'camera_failure')
    assert session.scope.motion.get_current_position(axis='Z') == 5000.0
