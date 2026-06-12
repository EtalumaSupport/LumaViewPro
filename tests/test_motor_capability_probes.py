# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Motor firmware command-family capability probes + ScopeCapabilities
registration.

The driver answers "does the connected firmware implement STOP / the
fan commands / the diagnostic queries" via probe-and-cache predicates
(supports_motor_stop / supports_fan / supports_diagnostics), mirroring
the LED board's firmware-stim probe. ScopeCapabilities.from_drivers
registers the answers at boot as has_motor_stop / has_fan /
has_diagnostics so callers gate on scope.capabilities.* instead of
firmware version strings or per-call ERROR handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from modules.scope_capabilities import ScopeCapabilities


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


class _FakeMotion:
    """Minimal motion driver for from_drivers: present axes plus the
    three capability predicates."""

    def __init__(self, stop=True, fan=True, diagnostics=False):
        self._stop = stop
        self._fan = fan
        self._diagnostics = diagnostics

    def detect_present_axes(self):
        return ('X', 'Y', 'Z')

    def get_microscope_model(self):
        return 'LS850'

    def supports_motor_stop(self):
        return self._stop

    def supports_fan(self):
        return self._fan

    def supports_diagnostics(self):
        return self._diagnostics


class TestCapabilitiesRegistration:
    def test_from_drivers_registers_predicate_answers(self):
        caps = ScopeCapabilities.from_drivers(
            motion=_FakeMotion(stop=True, fan=True, diagnostics=False),
            led=None,
            camera=None,
        )
        assert caps.has_motor_stop is True
        assert caps.has_fan is True
        assert caps.has_diagnostics is False

    def test_missing_predicates_default_false(self):
        """A driver without the predicate methods (older / minimal
        implementations) yields False, never raises."""

        class _BareMotion:
            def detect_present_axes(self):
                return ()

        caps = ScopeCapabilities.from_drivers(motion=_BareMotion(), led=None, camera=None)
        assert caps.has_motor_stop is False
        assert caps.has_fan is False
        assert caps.has_diagnostics is False

    def test_supports_helper_routes_by_token(self):
        caps = ScopeCapabilities.from_drivers(
            motion=_FakeMotion(stop=True, fan=False, diagnostics=True),
            led=None,
            camera=None,
        )
        assert caps.supports('motor_stop') is True
        assert caps.supports('fan') is False
        assert caps.supports('diagnostics') is True
