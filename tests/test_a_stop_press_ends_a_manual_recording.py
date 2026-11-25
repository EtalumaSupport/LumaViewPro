# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Pressing STOP ends a manual recording, as it ends the live run.

``stop_motion`` is the one STOP every client presses: the GUI's red button,
REST, the SDK. A manual recording it finds going ends as the person's stop,
``STOPPED``. Before, a press during a movie stopped the stage and left the
recording going. A press is told to its listeners before its refusal, so a
recording still going when the motor board is lost ends on the press too,
and only the STOP that could not be sent is refused.
"""

from __future__ import annotations

import pytest

from _lumascope.modules.exceptions import HardwareCommandRefusedError
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.video_recording import RecordingEndReason
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import _settings

_RESULT_S = 30.0


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def test_a_press_ends_a_manual_recording_as_the_persons_stop(session):
    recording = session.manual_recording.start()

    session.scope.motion.stop_motion()
    recording.result(timeout=_RESULT_S)

    assert session.manual_recording.end_reason == RecordingEndReason.STOPPED


def test_a_press_with_the_motor_board_lost_still_ends_the_recording(session):
    recording = session.manual_recording.start()
    session.scope.simulation.unplug('motor')
    # The loss is found by the next exchange, as on a real board.
    with pytest.raises(HardwareCommandRefusedError):
        session.scope.motion.move_z(100.0)
    assert not session.scope.motor_connected

    with pytest.raises(HardwareCommandRefusedError) as refused:
        session.scope.motion.stop_motion()
    recording.result(timeout=_RESULT_S)

    assert refused.value.reason == 'not_connected'
    assert session.manual_recording.end_reason == RecordingEndReason.STOPPED
