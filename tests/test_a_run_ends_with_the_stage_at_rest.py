# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's end is the stage at rest.

A caller told the run is over can start the next at once: the run's
teardown waits for every move it left in flight -- its halt's STOP, a
z-stack's return to where the stack was centred -- to read stopped before
the run ends. Before, a z-stack's run ended while Z still travelled back,
and the next start was refused ``axis_moving``. The halted case is in
``tests/test_a_run_that_ends_mid_move_halts_the_stage.py``, beside the
held move it needs.

The run's halt is the scope's own stop, not a press: a z-stack its own Stop
ends still goes back to its centre. A person's press of STOP withholds every
move asked before it, the return among them, so a z-stack a press ends
leaves Z where it stopped (Eric, 2026-11-08).
"""

from __future__ import annotations

import threading

import pytest

from _lumascope.modules.run_events import RunEvents
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT


@pytest.fixture
def session(tmp_path):
    settings = complete_settings(live_folder=str(tmp_path))
    settings['BF']['acquire'] = 'image'
    settings['zstack'] = {'range': 20.0, 'step_size': 5.0, 'position': 'center'}
    s = ScopeSession.create(settings, simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def test_a_zstack_that_returns_to_its_centre_ends_with_z_at_rest(session):
    motion = session.scope.motion
    motion.move_z(5000.0)

    run = session.protocol_runner().start_zstack('BF')
    outcome = run.result(timeout=COMPLETION_TIMEOUT)

    assert outcome.status == 'completed', outcome
    assert not motion.is_any_axis_moving(), 'the z-stack ended while Z still travelled back'
    assert motion.get_current_position('Z') == pytest.approx(5000.0, abs=1.0)


# Far past a slowed host, short enough that a stack that never reaches its
# second slice fails promptly.
_SLICE_WAIT_S = 30.0


def _a_zstack_ended_mid_stack(session, end):
    """Run a z-stack around Z 5000, end it with ``end()`` from another thread at its second slice; return its outcome.

    The run's thread waits in the second slice's frame callback until the
    ending has been asked, so the stack is ended mid-stack every time.
    """
    second_slice = threading.Event()
    ended = threading.Event()
    slices = []

    def on_frame(*_args):
        slices.append(None)
        if len(slices) == 2:
            second_slice.set()
            ended.wait(_SLICE_WAIT_S)

    run = session.protocol_runner().start_zstack('BF', events=RunEvents(frame_captured=on_frame))
    assert second_slice.wait(_SLICE_WAIT_S), 'the stack never reached its second slice'
    ender = threading.Thread(target=end, args=(run,))
    ender.start()
    ender.join(_SLICE_WAIT_S)
    ended.set()
    return run.result(timeout=COMPLETION_TIMEOUT)


def test_a_zstack_its_own_stop_ends_returns_to_its_centre(session):
    motion = session.scope.motion
    motion.move_z(5000.0)

    outcome = _a_zstack_ended_mid_stack(session, lambda run: run.stop())

    assert (outcome.status, outcome.reason) == ('aborted', 'stopped')
    assert motion.get_current_position('Z') == pytest.approx(5000.0, abs=1.0), (
        "the run's halt withheld the stack's return to its centre"
    )


def test_a_zstack_a_stop_press_ends_leaves_z_where_it_stopped(session):
    motion = session.scope.motion
    motion.move_z(5000.0)

    outcome = _a_zstack_ended_mid_stack(session, lambda run: motion.stop_motion())

    assert (outcome.status, outcome.reason) == ('aborted', 'stage_stopped')
    assert motion.get_current_position('Z') != pytest.approx(5000.0, abs=1.0), (
        'the stack went back to its centre after a press of STOP'
    )
