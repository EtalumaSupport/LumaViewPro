# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A STOP ends the move.

Once ``stop_motion`` has taken the board's STOP, the stage stops where it
is and no move that was under way reaches its target. Each test here
stops a move on the realistic-timing simulator, whose stage travels for
as long as the real one would, so a stop that did not stop it is seen.
"""

import time

import pytest

from modules.exceptions import MoveNotCompletedError
from modules.lumascope_api.motion import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        s.scope.motion._driver.set_timing_mode('realistic')
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def _reason(handle):
    try:
        handle.wait()
    except MoveNotCompletedError as e:
        return e.reason
    return 'arrived'


def test_a_stop_mid_move_leaves_the_stage_short_and_at_rest(session):
    """The simulated stage stops where the STOP found it. Before, the
    simulator sent no STOP at all, and its STOP command left the move
    running on its old timeline."""
    motion = session.scope.motion
    driver = motion._driver
    handle = motion.start_move_absolute('X', 60000.0)
    time.sleep(0.3)
    motion.stop_motion()

    assert _reason(handle) == 'stopped'
    assert motion.get_axis_state('X') == AxisState.IDLE
    stopped_at = driver.current_pos('X')
    assert 0.0 < stopped_at < 60000.0
    time.sleep(0.3)
    assert driver.current_pos('X') == stopped_at
    assert motion.get_current_position('X') == pytest.approx(stopped_at, abs=0.1)
