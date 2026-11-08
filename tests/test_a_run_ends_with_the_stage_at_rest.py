# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's end is the stage at rest.

A caller told the run is over can start the next at once: the run's
teardown waits for every move it left in flight -- its halt's STOP, a
z-stack's return to where the stack was centred -- to read stopped before
the run ends. Before, a z-stack's run ended while Z still travelled back,
and the next start was refused ``axis_moving``. The halted case is in
``tests/test_a_run_that_ends_mid_move_halts_the_stage.py``, beside the
held move it needs.
"""

from __future__ import annotations

import pytest

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
