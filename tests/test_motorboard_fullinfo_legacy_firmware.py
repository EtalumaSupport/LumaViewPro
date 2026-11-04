# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: MotorBoard.fullinfo() handles legacy firmware quietly.

Field firmware that predates the FULLINFO command replies with an
UNKNOWN_CMD error (e.g. ``{"err": "UNKNOWN_CMD", ..., "cmd": "FULLINFO"}``)
instead of a ``Model:`` / ``Serial:`` line. fullinfo() then failed to find
'Model:' and logged an ERROR on every connect:

  [ERROR] motorboard.py - [XYZ Class ] Failed to parse FULLINFO response:
  '{"err": "UNKNOWN_CMD", ...}' ('Model:' is not in list)

On a legacy unit that fired per connect and buried genuine failures -- the
same noise class the VOLTAGE / DRVSTAT / FANSPEED diagnostic probes already
suppress. fullinfo() now recognises the UNKNOWN_CMD reply as an expected
capability gap, logs it at INFO, and falls back to model/serial None, unknown.
A genuinely unparseable response (not an unsupported-command reply) still
logs ERROR so real faults stay visible.

The levels are read from caplog: the lines motorboard's own code logged.

A parsed reply and an unparseable one are the field firmware's, on the
firmware-backed simulator. The UNKNOWN_CMD envelope is a reply the
emulation carries no firmware for (the open question K67 in
SIM_AT_THE_WIRE_KNOWLEDGE), so that one case is still a scripted
exchange.
"""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock

import pytest

import drivers.motorboard as motorboard_mod
from tests.log_capture import records_of
from tests.wire_boards import motor_on_the_wire

_UNKNOWN_CMD = '{"err": "UNKNOWN_CMD", "ok": false, "msg": "unknown command", "cmd": "FULLINFO"}'


def _recording(caplog):
    """From here on, the lines motorboard logs at ``level``, read as ``rec(level)``."""
    caplog.clear()
    caplog.set_level(logging.INFO)

    def rec(level):
        return [r.getMessage() for r in records_of(caplog, motorboard_mod) if r.levelno == level]

    return rec


def _make_board(response, caplog):
    """Build a MotorBoard stub with exchange_command patched, its lines read from caplog."""
    from drivers.motorboard import MotorBoard

    board = MotorBoard.__new__(MotorBoard)
    board.exchange_command = MagicMock(return_value=response)
    board._state_lock = threading.Lock()
    board._has_turret = False
    return board, _recording(caplog)


def test_unknown_cmd_returns_fallback_without_error(caplog):
    board, rec = _make_board(_UNKNOWN_CMD, caplog)
    result = board.fullinfo()
    # Assert the identity fields, not the whole record: this test's subject is
    # the LOG LEVEL, and the record carries other fields (axis presence, homed
    # state) whose fallback values are pinned in
    # test_fresh_process_reads_homed_state.py.
    assert (result['model'], result['serial_number']) == (None, None)
    # Expected legacy-firmware capability gap -- must NOT log at ERROR.
    assert rec(logging.ERROR) == [], f'unexpected ERROR logs: {rec(logging.ERROR)}'
    # The condition is still recorded, once, at INFO.
    assert any('not supported on this firmware' in m for m in rec(logging.INFO))


@pytest.mark.firmware_tier
def test_unparseable_response_still_logs_error(caplog):
    # A non-UNKNOWN_CMD garbage response is a genuine fault: keep the ERROR
    # so real parse failures are not silently swallowed by the legacy path.
    board, sim = motor_on_the_wire()
    rec = _recording(caplog)
    sim.garble_next_reply()
    result = board.fullinfo()
    assert (result['model'], result['serial_number']) == (None, None)
    assert any('Failed to parse FULLINFO' in m for m in rec(logging.ERROR))


@pytest.mark.firmware_tier
def test_valid_response_parses_model_and_serial(caplog):
    board, _sim = motor_on_the_wire('LS850T', 'XYZT')
    rec = _recording(caplog)
    result = board.fullinfo()
    assert result['model'] == 'LS850T'
    assert result['serial_number'] == 'SIM-0000'
    # The trailing 'T' marks a turret-equipped scope.
    assert board.has_turret() is True
    assert rec(logging.ERROR) == []
