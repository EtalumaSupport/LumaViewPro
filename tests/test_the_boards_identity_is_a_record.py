# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each board's identity is a record, as it reported it at connect, None where unknown.

``get_motor_info`` and ``get_led_info`` answered bare dicts whose unknowns
were an empty string, ``'unknown'`` or ``'FX2-Classic'``: a model that could
be read as a model, a serial number stamped into every saved image. They
answer ``MotorBoardInfo`` and ``LedBoardInfo``, every unknown None, and the
LED answer leaves whether a board is connected to ``led_connected``. Each
board's command set is a member of its own set, ``MotorCommandSet`` and
``LedCommandSet``.
"""

from __future__ import annotations

import pytest

from modules.lumascope_api.diagnostics import (
    LedBoardInfo,
    LedCommandSet,
    MotorBoardInfo,
    MotorCommandSet,
)
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope='LS850T'), simulate=True
    )
    yield s
    s.shutdown()


def test_a_connected_board_reports_its_identity(session):
    motor = session.scope.diagnostics.get_motor_info()
    assert motor == MotorBoardInfo(
        model='LS850T',
        serial_number='SIM-001',
        firmware_version='2.0.1',
        firmware_date=None,
        command_set=MotorCommandSet.TEXT,
    )
    led = session.scope.diagnostics.get_led_info()
    assert led == LedBoardInfo(
        firmware_version='2.0.1', firmware_date=None, command_set=LedCommandSet.V2
    )
    assert (type(motor.command_set), type(led.command_set)) == (MotorCommandSet, LedCommandSet)


def test_the_identity_is_served_from_what_the_board_reported_at_connect(session):
    # It is read for every saved frame's metadata; a FULLINFO here would put
    # a serial round trip on every capture.
    driver = session.scope._motion_driver
    asked = []
    real_fullinfo = driver.fullinfo
    driver.fullinfo = lambda *a, **k: asked.append(1) or real_fullinfo(*a, **k)
    assert session.scope.diagnostics.get_motor_info().serial_number == 'SIM-001'
    assert asked == []


def test_with_no_motor_board_every_field_is_unknown(tmp_path):
    # An LS620 runs the null motor board.
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope='LS620'), simulate=True
    )
    try:
        assert s.scope.diagnostics.get_motor_info() == MotorBoardInfo(None, None, None, None, None)
    finally:
        s.shutdown()


def test_with_no_led_board_every_field_is_unknown(session, monkeypatch):
    monkeypatch.setattr(session.scope._led_driver, 'is_connected', lambda: False)
    assert session.scope.diagnostics.get_led_info() == LedBoardInfo(None, None, None)
