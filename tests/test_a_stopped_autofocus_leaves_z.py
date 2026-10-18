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


def _z_moves(scope):
    """Every absolute Z move autofocus issued, started or waited, in order."""
    return [
        call.args[1]
        for call in scope.motion.mock_calls
        if call[0] in ('start_move_absolute', 'move_absolute') and call.args and call.args[0] == 'Z'
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
