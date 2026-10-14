# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run that ends while its move travels halts the stage where it is.

A run that ends any way but finishing -- a Stop, or a fault -- halts its
move before it puts the scope back, so the stage stops short of the step it
was heading for. Before, the run's Stop only signalled the run loop, and the
move went on to the step's target with no STOP sent; the one halt a run had
was its step-motion bound's own.

Each test holds the run's first X move part of the way along and ends the
run there: a move the run's ending halted leaves the board driving nowhere
past where the stage stopped; a move nothing halted still has the step's
target, and the stage goes on to it once the hold lets go.
"""

from __future__ import annotations

import pytest

from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings
from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

# How long the run's move may take to reach its hold: far past a slowed
# host, short enough that a hold that never engages fails promptly.
_HOLD_WAIT_S = 10.0


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def _start_held(session, tmp_path):
    """Start a one-step scan whose X move halts part of the way; return the run and X's target."""
    held = session.scope.simulation.hold_axis('X')
    step = {**_make_single_step_protocol().step(idx=0), 'X': 60.0, 'Y': 20.0}
    run = session.create_protocol_runner().run_single_scan(
        protocol=_build_real_protocol([step]), parent_dir=str(tmp_path)
    )
    assert held.wait(_HOLD_WAIT_S), "the run's move never got part of the way"
    return run, session.scope.motion.get_target_position('X')


def _where_the_board_drives_x(session):
    """The X target the motor board holds now: where the stage is driving.

    A STOP sets the board's target to where the stage is; a move nothing
    halted keeps the step's. Read off the board, not the API: once a fault
    gives the axis up as unknown, the API stops reading it, and its cached
    position stays short while the stage drives on.
    """
    return session.scope.motion._driver.target_pos('X')


def test_a_stopped_run_halts_its_move(session, tmp_path):
    run, target = _start_held(session, tmp_path)

    run.stop()
    outcome = run.wait(timeout_s=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('aborted', 'stopped')
    assert _where_the_board_drives_x(session) < target - 1000.0, (
        'the stopped run left its move driving'
    )


def test_a_run_ended_by_its_moves_fault_halts_the_move(session, tmp_path, monkeypatch):
    """The move's own stall bound ends the run; the run's step-motion bound
    is far longer, so it is not the one that ends it."""
    monkeypatch.setattr(session.scope.motion, '_MOTION_SETTLE_TIMEOUT_S', 0.5)
    run, target = _start_held(session, tmp_path)

    outcome = run.wait(timeout_s=COMPLETION_TIMEOUT)

    assert (outcome.status, outcome.reason) == ('failed', 'position_lost')
    assert _where_the_board_drives_x(session) < target - 1000.0, (
        'the faulted run left its move driving'
    )
