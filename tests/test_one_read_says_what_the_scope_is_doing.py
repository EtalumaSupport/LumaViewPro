# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""One read says what the scope is doing.

A client that connects to a running session -- a REST client above all --
has to ask four owners before it can act: what the session is still
doing, where each axis is, which parts came up, and whether the camera is
grabbing. ``session.status`` answers all four from those owners, so
each field here is checked against the owner it reads, idle and in the
middle of a home, and against the camera after its stream stops.
"""

import pytest

from modules import live_work
from modules.lumascope_api import AxisState


@pytest.fixture
def sim_session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings
    from tests.test_a_run_needs_every_axis_position import _settings

    session = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(session.scope)
        yield session
    finally:
        session.shutdown()


def test_an_idle_session_answers_each_field_from_its_owner(sim_session):
    from modules.scope_session import Status

    status = sim_session.status

    assert isinstance(status, Status)
    assert status.live_work == sim_session.live_work
    assert status.axes == sim_session.scope.motion.axis_positions()
    assert all(axis.state == AxisState.IDLE for axis in status.axes.values())
    assert status.parts == sim_session.bring_up_record().parts
    assert status.camera_streaming is True


def test_a_home_in_progress_is_the_work_and_no_axis_has_a_position(sim_session):
    driver = sim_session.scope._motion_driver
    real_target_pos = driver.target_pos
    seen = []

    def read_status_mid_home(axis):
        seen.append(sim_session.status)
        return real_target_pos(axis)

    driver.target_pos = read_status_mid_home
    sim_session.scope.motion.home()

    assert seen, 'the home never read a position'
    status = seen[0]
    assert [item.kind for item in status.live_work.work] == [live_work.HOME]
    assert {axis.state for axis in status.axes.values()} == {AxisState.HOMING}
    assert all(axis.position is None for axis in status.axes.values())


def test_a_stopped_camera_reads_not_streaming(sim_session):
    sim_session.scope.imaging.stop_streaming()

    assert sim_session.status.camera_streaming is False
