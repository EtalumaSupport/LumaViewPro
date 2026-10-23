# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""One read says what the scope is doing.

A client that connects to a running session -- a REST client above all --
has to ask several owners before it can act: what the session is still
doing, which controls the Session locks for a person and why, whether a
run would be refused now for the scope's state, where each axis is, which
parts came up, and whether the camera is grabbing. ``session.status``
answers them all from those owners, so each field here is checked against
the owner it reads: idle, in the middle of a home, under a diagnostic and
during a recording, and against the camera after its stream stops. The
lock is a policy, not a refusal; the run's refusal is the one a start
raises.
"""

import pathlib

import pytest

from modules import live_work, wire_encoding
from modules.exceptions import ProtocolRunRefusedError
from modules.lumascope_api import AxisState

COMPLETION_TIMEOUT = 30


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
    assert status.lock == sim_session.controls_lock
    assert (status.lock.run_named, status.lock.recording) == (None, False)
    assert (status.lock.run_lockout, status.lock.controls_locked) == (False, False)
    assert status.run_refused is None


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
    sim_session.scope.imaging.disable_streaming()

    assert sim_session.status.camera_streaming is False


def test_a_held_scope_locks_the_controls_and_names_the_refusal_a_start_raises(
    sim_session, tmp_path
):
    protocol = sim_session.create_empty_protocol()
    sim_session.set_channel_acquire('BF', 'image')
    sim_session.add_step(protocol)

    with sim_session.diagnostic_claim():
        status = sim_session.status
        with pytest.raises(ProtocolRunRefusedError) as refused:
            sim_session.create_protocol_runner().start_single_scan(
                protocol, parent_dir=str(tmp_path)
            )

    assert status.lock.run_named == 'A diagnostic is in progress.'
    assert (status.lock.run_lockout, status.lock.controls_locked) == (True, True)
    assert status.run_refused.reason == refused.value.reason == 'exclusive_activity_running'


def test_a_live_recording_locks_the_controls_and_not_the_run_surface(sim_session):
    operation = sim_session.manual_recording.start()
    try:
        lock = sim_session.status.lock
    finally:
        operation.stop()
    operation.result(timeout=COMPLETION_TIMEOUT)

    assert (lock.run_named, lock.recording) == (None, True)
    assert (lock.run_lockout, lock.controls_locked) == (False, True)
    assert sim_session.status.lock.recording is False


def test_the_status_crosses_a_wire_with_its_lock_and_its_refusal(sim_session):
    def refuse(v):
        raise AssertionError(f'{v!r} has no place in a status')

    with sim_session.diagnostic_claim():
        sent = wire_encoding.encode(
            sim_session.status, live_folder=pathlib.Path(), handle=refuse, job=refuse
        )

    assert sent['lock'] == {
        'run_named': 'A diagnostic is in progress.',
        'control_named': None,
        'recording': False,
        'run_lockout': True,
        'controls_locked': True,
        'named': 'A diagnostic is in progress.',
    }
    assert sent['run_refused']['kind'] == 'refusal'
    assert sent['run_refused']['reason'] == 'exclusive_activity_running'
