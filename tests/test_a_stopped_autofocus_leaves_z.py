# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A sweep a person stopped leaves Z where the stop left it.

An autofocus that ends without choosing a focus puts Z back where it began,
so a failed sweep is a no-op in position. A stop is not a failure: the
person pressed STOP or the run's Stop, and putting Z back is a move made
after they did (Eric, 2026-10-18: leave Z where it stopped). The run's
cleanup aborts the sweep with ``stay_put`` for those endings; any other
abort, a fault's included, still restores.

A press of STOP also withholds every Z move the sweep asks after it: the
sweep runs under its run's ask, so a move it asks before the run's cleanup
has aborted it is asked before the press too. Before, the sweep's thread
asked each move as it made it, and one made after the press drove.
"""

from __future__ import annotations

import threading

import pytest

from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT
from tests.test_a_stop_press_ends_the_live_run import (  # noqa: F401 (the fixture)
    session,
    start_an_autofocus_held_mid_sweep,
)


@pytest.fixture
def z_asked(session, monkeypatch):
    """Every absolute Z target the session's scope is asked for, in order, as (member, target).

    A run's put-back is judged by the targets asked, not by Z's position:
    while the held sweep move is ``moving`` the position is the cache the
    motion monitor refreshes, and under load it still reads the pre-move
    5000.0 with no move back asked.
    """
    motion = session.scope.motion
    asked = []
    for member in ('start_move_z', 'move_z'):
        real = getattr(motion, member)

        def spy(position, *args, _real=real, _member=member, **kwargs):
            asked.append((_member, position))
            return _real(position, *args, **kwargs)

        monkeypatch.setattr(motion, member, spy)
    return asked


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
    session, end, reason, z_asked
):
    run = start_an_autofocus_held_mid_sweep(session)
    held_at = len(z_asked)

    end(session, run)
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', reason)
    assert 5000.0 not in [target for _, target in z_asked[held_at:]], (
        f'Z was put back after the stop: Z asked {z_asked}, outcome {outcome.message!r}'
    )


def test_an_autofocus_run_a_fault_ends_mid_sweep_puts_z_back(session, z_asked, centre_posts):
    run = start_an_autofocus_held_mid_sweep(session)
    held_at = len(z_asked)

    session._sequenced_capture_runner.abort_run_fatal('camera_failure', 'The camera failed.')
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('failed', 'camera_failure')
    assert ('move_z', 5000.0) in z_asked[held_at:], f'Z asked {z_asked}'
    # Asked is not arrived: a put-back the run's halt withheld is asked too.
    assert session.scope.motion.get_current_position('Z') == pytest.approx(5000.0, abs=0.5)
    assert 'Z Position Not Restored' not in [n.title for n in centre_posts]


# Far past a slowed host, short enough that a sweep that never asks fails promptly.
_SWEEP_WAIT_S = 30.0


def test_a_sweep_move_asked_after_a_stop_press_drives_nothing(session, monkeypatch):
    """The sweep's next Z move, asked after the press and before the sweep is aborted, is withheld."""
    motion = session.scope.motion
    run = start_an_autofocus_held_mid_sweep(session)
    real_start_move_z = motion.start_move_z
    asked = threading.Event()
    go_on = threading.Event()

    # A seam by design: the sweep's next move waits here, before the real
    # call, until the press has been made and the sweep aborted, so the move
    # is asked in the window every time rather than when scheduling allows.
    def held_until_pressed(position, *args, **kwargs):
        if not asked.is_set():
            asked.set()
            assert go_on.wait(_SWEEP_WAIT_S)
        return real_start_move_z(position, *args, **kwargs)

    monkeypatch.setattr(motion, 'start_move_z', held_until_pressed)
    session.scope.simulation.clear()
    assert asked.wait(_SWEEP_WAIT_S), 'the sweep never asked its next Z move'
    z_at_press = motion.get_current_position('Z')

    motion.stop_motion()
    assert session.autofocus_thread.aborted.wait(_SWEEP_WAIT_S), (
        "the run's end never aborted the sweep"
    )
    go_on.set()
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert motion.get_current_position('Z') == pytest.approx(z_at_press, abs=0.5), (
        'a sweep move asked after the STOP drove'
    )
