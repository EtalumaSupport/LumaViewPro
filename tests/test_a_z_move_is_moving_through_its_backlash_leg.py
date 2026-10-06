# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Z move is motion from its first target write, its backlash leg included.

A Z move down first drives below its target, then approaches it from below.
The leg is nearly the whole move: from 3000 um to 1000 um it runs to 975 um.
Before, the axis went MOVING only after the leg, so through it
``wait_until_finished_moving`` returned at once, the state and the positions
read idle at the start, and frame validity had no Z move pending -- a
capture made then was judged valid, and a recording stamped the idle start
position into its frames. Each reader is asked here, from another thread,
while the leg runs on the realistic-timing simulator.
"""

import threading

import pytest

from modules.lumascope_api.motion import AxisState
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()
        s.scope.disconnect()


def test_every_reader_sees_the_leg_as_the_move(session):
    motion = session.scope.motion
    driver = motion._driver
    validity = session.scope.imaging.frame_validity
    driver.set_timing_mode('fast')
    motion.move_absolute('Z', 3000.0)
    driver.set_timing_mode('realistic')

    leg_um = 1000.0 - driver.backlash_um()
    leg_written = threading.Event()
    real_write = driver.move_abs_pos

    def recorded_write(axis, pos, *args, **kwargs):
        result = real_write(axis, pos, *args, **kwargs)
        if axis == 'Z' and pos == leg_um:
            leg_written.set()
        return result

    driver.move_abs_pos = recorded_write
    seen = {}

    def observe_the_leg():
        assert leg_written.wait(5.0)
        seen['state'] = motion.get_axis_state('Z')
        seen['position_state'] = motion.axis_positions()['Z'].state
        seen['z_move_unsettled'] = 'z_move' in validity.unsettled_motion_sources()
        seen['stage_in_leg'] = driver.current_pos('Z')
        motion.wait_until_finished_moving(timeout_s=10.0)
        seen['state_after_wait'] = motion.get_axis_state('Z')
        seen['stage_after_wait'] = driver.current_pos('Z')

    observer = threading.Thread(target=observe_the_leg)
    observer.start()
    handle = motion.start_move_absolute('Z', 1000.0, overshoot_enabled=True)
    observer.join(15.0)
    handle.wait()

    assert seen['stage_in_leg'] > 1000.0, 'the observer did not read during the leg'
    assert seen['state'] == AxisState.MOVING
    assert seen['position_state'] == AxisState.MOVING
    assert seen['z_move_unsettled'], 'a frame grabbed during the leg would be judged valid'
    assert seen['state_after_wait'] == AxisState.IDLE
    assert seen['stage_after_wait'] == pytest.approx(1000.0, abs=1.0), (
        'the wait returned before the move arrived'
    )
