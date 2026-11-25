# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``ScopeSession.something_to_stop`` says whether a press of STOP has anything to stop.

The GUI's STOP is lit from it and a wire client reads it in ``status``: True
while anything holds the scope or a run is live, while an axis moves, and
while an axis's position is unknown on a connected board; False at rest, and
with the motor board lost and nothing running, when a press would only be
refused. Before, STOP was lit at all times on a scope with motors.
"""

from __future__ import annotations

import datetime
import threading

import pytest

from _lumascope.modules.exceptions import HardwareCommandRefusedError
from _lumascope.modules.run_events import RunEvents
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings
from tests.test_a_stop_press_ends_the_live_run import _steps
from tests.test_run_refusal_contract import _build_real_protocol

_WAIT_S = 10.0


@pytest.fixture
def unhomed(tmp_path):
    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        yield s
    finally:
        s.shutdown()


@pytest.fixture
def session(unhomed):
    home_sim_scope(unhomed.scope)
    return unhomed


def test_a_homed_scope_at_rest_has_nothing_to_stop(session):
    assert session.something_to_stop is False
    assert session.status.something_to_stop is False


def test_an_axis_whose_position_is_unknown_lights_stop(unhomed):
    assert unhomed.scope.motion.axes_without_position
    assert unhomed.something_to_stop is True


def test_a_moving_axis_lights_stop(session):
    held = session.scope.simulation.hold_axis('Z')
    move = session.scope.motion.start_move_z_by(200.0)
    held.result(_WAIT_S)
    try:
        assert session.something_to_stop is True
    finally:
        session.scope.motion.stop_motion()
        move.exception(_WAIT_S)


def test_a_manual_recording_lights_stop_and_is_served_in_status(session):
    recording = session.manual_recording.start()
    try:
        assert session.something_to_stop is True
        assert session.status.something_to_stop is True
    finally:
        recording.stop()
        recording.result(timeout=COMPLETION_TIMEOUT)
    assert session.something_to_stop is False


def test_a_protocol_waiting_for_its_next_scan_lights_stop(session, tmp_path):
    """Nothing moves between scans; the run holding the scope is what a press ends."""
    first_scan_ended = threading.Event()
    protocol = _build_real_protocol(_steps(60.0))
    protocol.modify_time_params(
        period=datetime.timedelta(seconds=60), duration=datetime.timedelta(seconds=180)
    )
    run = session.protocol_runner().start_protocol(
        protocol,
        parent_dir=str(tmp_path),
        events=RunEvents(scan_ended=lambda *_args: first_scan_ended.set()),
    )
    assert first_scan_ended.wait(COMPLETION_TIMEOUT), 'the first scan never ended'
    try:
        assert not session.scope.motion.is_any_axis_moving()
        assert session.something_to_stop is True
    finally:
        session.scope.motion.stop_motion()
        run.result(timeout=COMPLETION_TIMEOUT)


def test_a_lost_board_with_nothing_running_has_nothing_to_stop(session):
    session.scope.simulation.unplug('motor')
    # The loss is found by the next exchange, as on a real board.
    with pytest.raises(HardwareCommandRefusedError):
        session.scope.motion.move_z(100.0)
    assert not session.scope.motor_connected

    assert session.something_to_stop is False
