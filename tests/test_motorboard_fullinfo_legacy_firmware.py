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
capability gap, logs it at INFO, and falls back to model/serial = 'unknown'.
A genuinely unparseable response (not an unsupported-command reply) still
logs ERROR so real faults stay visible.

motorboard logs through lvp_logger (propagate=False, custom file handlers),
which pytest's caplog cannot see; spy on the module logger directly instead.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

_UNKNOWN_CMD = '{"err": "UNKNOWN_CMD", "ok": false, "msg": "unknown command", "cmd": "FULLINFO"}'


class _RecLogger:
    """Records (level, formatted-message) for each log call."""

    def __init__(self):
        self.calls = []

    def _rec(self, level, msg, args):
        self.calls.append((level, msg % args if args else msg))

    def info(self, msg, *args, **kwargs):
        self._rec('info', msg, args)

    def error(self, msg, *args, **kwargs):
        self._rec('error', msg, args)

    def warning(self, msg, *args, **kwargs):
        self._rec('warning', msg, args)

    def debug(self, *args, **kwargs):
        pass

    def messages(self, level):
        return [m for lvl, m in self.calls if lvl == level]


def _make_board(response, monkeypatch):
    """Build a MotorBoard stub with exchange_command + logger patched."""
    from drivers.motorboard import MotorBoard

    board = MotorBoard.__new__(MotorBoard)
    board.exchange_command = MagicMock(return_value=response)
    board._state_lock = threading.Lock()
    board._has_turret = False
    rec = _RecLogger()
    monkeypatch.setattr('drivers.motorboard.logger', rec)
    return board, rec


def test_unknown_cmd_returns_fallback_without_error(monkeypatch):
    board, rec = _make_board(_UNKNOWN_CMD, monkeypatch)
    result = board.fullinfo()
    # Assert the identity fields, not the whole record: this test's subject is
    # the LOG LEVEL, and the record carries other fields (axis presence, homed
    # state) whose fallback values are pinned in
    # test_fresh_process_reads_homed_state.py.
    assert (result['model'], result['serial_number']) == ('unknown', 'unknown')
    # Expected legacy-firmware capability gap -- must NOT log at ERROR.
    assert rec.messages('error') == [], f'unexpected ERROR logs: {rec.messages("error")}'
    # The condition is still recorded, once, at INFO.
    assert any('not supported on this firmware' in m for m in rec.messages('info'))


def test_unparseable_response_still_logs_error(monkeypatch):
    # A non-UNKNOWN_CMD garbage response is a genuine fault: keep the ERROR
    # so real parse failures are not silently swallowed by the legacy path.
    board, rec = _make_board('garbage nonsense with no fields', monkeypatch)
    result = board.fullinfo()
    assert (result['model'], result['serial_number']) == ('unknown', 'unknown')
    assert any('Failed to parse FULLINFO' in m for m in rec.messages('error'))


def test_valid_response_parses_model_and_serial(monkeypatch):
    resp = 'Model: LS720T Serial: SN12345 X present: True'
    board, rec = _make_board(resp, monkeypatch)
    result = board.fullinfo()
    assert result['model'] == 'LS720T'
    assert result['serial_number'] == 'SN12345'
    assert result['_raw'] == resp
    # The trailing 'T' marks a turret-equipped scope.
    assert board._has_turret is True
    assert rec.messages('error') == []
