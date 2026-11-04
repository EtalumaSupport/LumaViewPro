# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report's fan test waits the spin times its board states.

The fan test sets a duty, waits for the fan to reach it, and reads the
tachometer. It waited a fixed 2 s and 3 s on every board, so a report on
the simulator's fast tier waited 5 s for a fan the simulator does not spin.
The board's driver states how long its fan takes, the diagnostics API
answers it, and the report waits that.
"""

import time
import types

import pytest

from modules import tech_support_report
from modules.scope_session import ScopeSession
from modules.tech_support_report import FirmwareDiagnostics
from tests.settings_fixtures import complete_settings


@pytest.fixture
def make_session(tmp_path):
    sessions = []

    def _make(model):
        session = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path / model), microscope=model), simulate=True
        )
        sessions.append(session)
        return session

    yield _make
    for session in sessions:
        session.shutdown()


def test_the_fan_test_waits_the_spin_times_the_board_states(make_session, monkeypatch):
    session = make_session('LS850')
    board = session.scope._motion_driver
    waits = []
    clock = types.SimpleNamespace(sleep=waits.append, monotonic=time.monotonic)
    monkeypatch.setattr(tech_support_report, 'time', clock)

    fan = FirmwareDiagnostics(scope=session.scope).verify_fan_tachometer()

    assert fan['supported'] is True
    assert waits == [board.fan_spin_up_s, board.fan_spin_down_s]


def test_the_api_answers_the_boards_spin_times(make_session):
    session = make_session('LS850')
    board = session.scope._motion_driver

    assert session.scope.diagnostics.motor_fan_spin_times() == {
        'spin_up_s': board.fan_spin_up_s,
        'spin_down_s': board.fan_spin_down_s,
    }


def test_a_scope_without_fan_control_has_no_spin_times(make_session):
    session = make_session('LS620')

    assert session.scope.diagnostics.motor_fan_spin_times() is None
