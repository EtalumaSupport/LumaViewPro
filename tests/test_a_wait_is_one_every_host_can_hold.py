# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A wait a caller passes is from 0 to 30 days, refused where it enters past that.

Python's waits raise ``OverflowError`` past ``threading.TIMEOUT_MAX``, about
292 years on macOS and 49.7 days on Windows. ``wait_until_finished_moving``
handed a caller's ``1e12`` straight to an arrival event, so while an axis
moved the wait raised out of itself and REST answered a 500; the bad-request
guard went red only when an earlier route had left an axis moving. A run's
waiting form started its run first and then overflowed, leaving the caller
no handle to it. Every member that takes a wait now refuses one outside
``WAIT_S`` with ``out_of_range`` before anything starts, the same on every
host.
"""

from __future__ import annotations

import pytest

from modules.exceptions import ArgumentRefusedError
from modules.finite_number import WAIT_S
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

_HOLD_WAIT_S = 30.0


@pytest.fixture
def session(tmp_path):
    from tests.test_a_run_needs_every_axis_position import _settings

    built = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    home_sim_scope(built.scope)
    yield built
    built.shutdown()


@pytest.fixture
def z_held_moving(session):
    """Z mid-move and held there, so a wait on the moving axes has one to wait for."""
    motion = session.scope.motion
    held = session.scope.simulation.hold_axis('Z')
    motion.start_move_z(motion.get_current_position(axis='Z') + 500.0)
    assert held.wait(_HOLD_WAIT_S), 'the Z move never reached the hold'
    yield
    session.scope.simulation.clear()
    motion.stop_motion()


def _refused(call) -> ArgumentRefusedError:
    with pytest.raises(ArgumentRefusedError) as refused:
        call()
    return refused.value


@pytest.mark.parametrize('timeout_s', [1e12, -1.0], ids=['past_every_host', 'negative'])
def test_a_wait_on_a_moving_axis_outside_the_range_is_refused(session, z_held_moving, timeout_s):
    refused = _refused(lambda: session.scope.motion.wait_until_finished_moving(timeout_s=timeout_s))

    assert (refused.reason, refused.argument, refused.limits) == (
        'out_of_range',
        'timeout_s',
        WAIT_S,
    )


def test_a_run_waiting_form_refuses_its_wait_before_starting_the_run(session, tmp_path):
    from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

    step = dict(_make_single_step_protocol().step_row(step_index=0))
    # Held, a run that started would still be live when the wait returned.
    session.scope.simulation.hold_axis('Z')
    try:
        refused = _refused(
            lambda: session.create_protocol_runner().run_single_scan(
                protocol=_build_real_protocol([step]), parent_dir=str(tmp_path), timeout_s=1e12
            )
        )
        started = session.sequenced_capture_runner.run_in_progress()
    finally:
        session.scope.simulation.clear()

    assert (refused.reason, refused.argument) == ('out_of_range', 'timeout_s')
    assert not started, 'the run started before its wait was refused'


def test_a_capture_whose_sum_would_wait_past_the_range_is_refused(session):
    # Each time is in range; two frames at the longest delay are not.
    refused = _refused(
        lambda: session.scope.imaging.capture_still(sum_count=2, sum_delay_s=WAIT_S[1])
    )

    assert (refused.reason, refused.limits) == ('out_of_range', WAIT_S)
