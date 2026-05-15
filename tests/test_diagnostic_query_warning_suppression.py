# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: MotorBoard._diagnostic_query suppresses
FIRMWARE ERROR warnings on legacy firmware.

VOLTAGE / DRVSTAT_<axis> / FANSPEED / FAN:<duty> are diagnostic
commands added to motor firmware revisions after 2024-09-10. On
the legacy firmware still in the field, each of these returns
``ERROR: command 'X' not found``, and pre-fix the WARNING
"FIRMWARE ERROR: VOLTAGE -> ERROR..." fired from
``serialboard.exchange_command`` BEFORE ``_diagnostic_query``
caught the response and returned None. The TSR was hitting all
six commands during its diagnostic phase, surfacing six WARNINGs
to the user-visible log on every TSR generation against legacy
firmware.

Fix: ``_diagnostic_query`` passes ``expect_unsupported=True`` to
``exchange_command``, so the WARNING is suppressed for these
probes. The helper still returns None to the caller and the caller
still treats None as "INCONCLUSIVE -- firmware does not support."

Same shape as ``motor_stop`` (commit 3f3553c) and tracked under
GitHub issue #654.
"""

from __future__ import annotations

from unittest.mock import MagicMock


class TestDiagnosticQueryCapabilityProbe:
    def _make_board(self, response):
        from drivers.motorboard import MotorBoard
        board = MotorBoard.__new__(MotorBoard)
        board.exchange_command = MagicMock(return_value=response)
        return board

    def test_voltage_probe_passes_expect_unsupported_flag(self):
        board = self._make_board(response='24V=OK 5V=5.18 3V3=3.31 1V2=1.24')
        board.read_voltages()
        assert board.exchange_command.call_count == 1
        args, kwargs = board.exchange_command.call_args
        assert args[0] == 'VOLTAGE'
        assert kwargs.get('expect_unsupported') is True, (
            "read_voltages must route through _diagnostic_query which "
            "passes expect_unsupported=True so the FIRMWARE ERROR "
            "warning is suppressed on legacy firmware."
        )

    def test_drvstat_probe_passes_expect_unsupported_flag(self):
        board = self._make_board(response='ERROR: command \'DRVSTAT_Z\' not found:')
        result = board.read_drv_status('Z')
        assert result is None, (
            "ERROR response must be swallowed and returned as None "
            "(unsupported-firmware indicator), not propagated."
        )
        assert board.exchange_command.call_count == 1
        args, kwargs = board.exchange_command.call_args
        assert args[0] == 'DRVSTAT_Z'
        assert kwargs.get('expect_unsupported') is True

    def test_fanspeed_probe_passes_expect_unsupported_flag(self):
        board = self._make_board(response='ERROR: command \'FANSPEED\' not found:')
        result = board.read_fanspeed()
        assert result is None
        assert board.exchange_command.call_count == 1
        args, kwargs = board.exchange_command.call_args
        assert args[0] == 'FANSPEED'
        assert kwargs.get('expect_unsupported') is True

    def test_diagnostic_query_returns_response_when_supported(self):
        """When firmware supports the command, the response passes
        through unchanged (no None substitution)."""
        from drivers.motorboard import MotorBoard
        board = MotorBoard.__new__(MotorBoard)
        board.exchange_command = MagicMock(return_value='real-response-data')
        result = board._diagnostic_query('VOLTAGE')
        assert result == 'real-response-data'

    def test_diagnostic_query_returns_none_on_error_response(self):
        """When firmware rejects with ERROR prefix, the response is
        swallowed and None is returned (caller treats as inconclusive)."""
        from drivers.motorboard import MotorBoard
        board = MotorBoard.__new__(MotorBoard)
        board.exchange_command = MagicMock(
            return_value="ERROR: command 'VOLTAGE' not found:"
        )
        result = board._diagnostic_query('VOLTAGE')
        assert result is None
