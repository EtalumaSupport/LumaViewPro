# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Motor firmware command-family capability probes.

The driver answers "does the connected firmware implement STOP / the
fan commands / the diagnostic queries" via probe-and-cache predicates
(supports_motor_stop / supports_fan / supports_diagnostics), mirroring
the LED board's firmware-stim probe, so callers gate on the probe
answer instead of firmware version strings or per-call ERROR handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_board(response):
    """MotorBoard stub with exchange_command monkeypatched."""
    from drivers.motorboard import MotorBoard

    board = MotorBoard.__new__(MotorBoard)
    board.exchange_command = MagicMock(return_value=response)
    return board


class TestSupportsPredicates:
    def test_error_response_caches_unsupported(self):
        board = _make_board("ERROR: command 'FANSPEED' not found:")
        assert board.supports_fan() is False
        assert board._supports_fan_cached is False

    def test_clean_response_caches_supported(self):
        board = _make_board('1200')
        assert board.supports_diagnostics() is True
        assert board._supports_diagnostics_cached is True

    def test_no_response_is_inconclusive_and_uncached(self):
        """No reply (board absent / wedged) is not a capability answer:
        return False but do NOT cache, so a healthy later connection
        re-probes."""
        board = _make_board(None)
        assert board.supports_fan() is False
        assert not hasattr(board, '_supports_fan_cached')

    def test_cached_answer_skips_the_wire(self):
        board = _make_board('1200')
        board.supports_fan()
        board.supports_fan()
        assert board.exchange_command.call_count == 1

    def test_probes_use_read_only_commands_with_suppression(self):
        """supports_fan probes FANSPEED (never FAN:<duty>, which would
        change fan state) and supports_diagnostics probes VOLTAGE; both
        opt into FIRMWARE ERROR suppression for the probe."""
        board = _make_board('OK')
        board.supports_fan()
        args, kwargs = board.exchange_command.call_args
        assert args[0] == 'FANSPEED'
        assert kwargs.get('expect_unsupported') is True

        board2 = _make_board('OK')
        board2.supports_diagnostics()
        args, kwargs = board2.exchange_command.call_args
        assert args[0] == 'VOLTAGE'
        assert kwargs.get('expect_unsupported') is True
