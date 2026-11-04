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


def test_an_autofocus_run_a_fault_ends_mid_sweep_puts_z_back(session, z_asked):
    run = start_an_autofocus_held_mid_sweep(session)
    held_at = len(z_asked)

    session.sequenced_capture_runner.abort_run_fatal('camera_failure', 'The camera failed.')
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('failed', 'camera_failure')
    assert ('move_z', 5000.0) in z_asked[held_at:], f'Z asked {z_asked}'
