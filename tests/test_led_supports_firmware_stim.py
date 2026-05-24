# Copyright Etaluma, Inc.
"""Regression tests for LEDBoard.supports_firmware_stim probe + the
ScopeCapabilities `has_firmware_stim` flag plumbing.

Probe shape (per `drivers/ledboard.py::LEDBoard.supports_firmware_stim`):
- Sends `STIM 0 0 1 2 1\\n` (intentionally invalid mA=0)
- v3.0.8+ firmware replies with `STIM: mA must be > 0` (parser recognized)
- Pre-v3.0.8 firmware echoes the command + returns `Command not recognized`

Result is cached on the LEDBoard instance via `_supports_stim_cached` so
subsequent calls return without re-probing the bus.

The capability flag lands on `ScopeCapabilities` as `has_firmware_stim`
(immutable per-Lumascope-instance fact, sourced at Lumascope.__init__
from LEDBoard.supports_firmware_stim()). Caller gate is
`scope.capabilities.supports('firmware_stim')`.
"""

import threading
from unittest.mock import MagicMock


class TestSupportsFirmwareStimProbe:
    """Probe-detection logic on LEDBoard."""

    def _make_led(self, readline_responses):
        """Build a real LEDBoard with the driver attribute mocked.

        readline_responses is a list of bytes (or callables) that mock
        what `self.driver.readline()` returns, one per call. Empty bytes
        keeps the probe-loop polling; non-empty bytes is parsed.
        """
        from drivers.ledboard import LEDBoard

        led = LEDBoard.__new__(LEDBoard)
        led._lock = threading.RLock()
        led._label = '[LED Class ]'
        led.driver = MagicMock()
        led.driver.timeout = 0.1
        responses = list(readline_responses)
        led.driver.readline = MagicMock(
            side_effect=lambda: responses.pop(0) if responses else b''
        )
        led.driver.reset_input_buffer = MagicMock()
        led.driver.write = MagicMock()
        led.driver.in_waiting = 0
        led.driver.read = MagicMock(return_value=b'')
        return led

    def test_v308_plus_firmware_returns_true(self):
        """Modern firmware reply (`STIM: mA must be > 0`) -> probe True."""
        led = self._make_led([b'STIM: mA must be > 0\r\n'])
        assert led.supports_firmware_stim() is True

    def test_pre_v308_firmware_returns_false(self):
        """Legacy firmware reply (`Command not recognized`) -> probe False."""
        led = self._make_led([b'Command not recognized\r\n'])
        assert led.supports_firmware_stim() is False

    def test_stim_diag_response_also_recognized(self):
        """STIM_DIAG: prefix (alternate parser-recognized form) -> True."""
        led = self._make_led([b'STIM_DIAG: probe ok\r\n'])
        assert led.supports_firmware_stim() is True

    def test_command_echo_then_recognized_response(self):
        """Some firmware echoes the command first, then replies. The
        probe loop should ignore the echo + accept the second line."""
        led = self._make_led([
            b'STIM 0 0 1 2 1\r\n',  # command echo -- neither STIM: nor "not recognized"
            b'STIM: mA must be > 0\r\n',  # parser recognition
        ])
        assert led.supports_firmware_stim() is True

    def test_no_response_returns_false(self):
        """Disconnected / silent firmware -> probe deadline expires -> False."""
        led = self._make_led([])  # all readlines return b''
        # Force the readline mock to always return empty bytes (no responses)
        led.driver.readline = MagicMock(return_value=b'')
        assert led.supports_firmware_stim() is False

    def test_result_is_cached_after_first_call(self):
        """Second call should return cached value without re-probing."""
        led = self._make_led([b'STIM: mA must be > 0\r\n'])
        first = led.supports_firmware_stim()
        write_count_after_first = led.driver.write.call_count
        second = led.supports_firmware_stim()
        assert first is True
        assert second is True
        # Cached: no new probe-write happened on the second call
        assert led.driver.write.call_count == write_count_after_first

    def test_driver_none_returns_false_without_probing(self):
        """No serial driver attached -> probe short-circuits to False."""
        from drivers.ledboard import LEDBoard

        led = LEDBoard.__new__(LEDBoard)
        led._lock = threading.RLock()
        led._label = '[LED Class ]'
        led.driver = None
        assert led.supports_firmware_stim() is False


class TestSimulatedLEDBoardParityStub:
    """SimulatedLEDBoard must implement supports_firmware_stim() so
    sim-mode Lumascope instances populate ScopeCapabilities.has_firmware_stim
    without raising AttributeError."""

    def test_sim_default_is_false(self):
        """Default sim board reports no STIM support (matches pre-v3.0.8)."""
        from drivers.simulated_ledboard import SimulatedLEDBoard

        sim = SimulatedLEDBoard()
        assert sim.supports_firmware_stim() is False

    def test_sim_configurable_true(self):
        """Constructor flag injects STIM support for tests that need it."""
        from drivers.simulated_ledboard import SimulatedLEDBoard

        sim = SimulatedLEDBoard(supports_firmware_stim=True)
        assert sim.supports_firmware_stim() is True


class TestNullLEDBoardParityStub:
    """NullLEDBoard must implement supports_firmware_stim() so a no-
    hardware Lumascope construction populates the capability without
    AttributeError. No-hardware = no STIM support."""

    def test_null_returns_false(self):
        from drivers.null_ledboard import NullLEDBoard

        null = NullLEDBoard()
        assert null.supports_firmware_stim() is False


class TestScopeCapabilitiesPlumbing:
    """ScopeCapabilities.from_drivers probes led.supports_firmware_stim
    and exposes the result via .has_firmware_stim + .supports('firmware_stim')."""

    def _build_caps(self, led_stub):
        """Build a ScopeCapabilities snapshot with a stub LED driver."""
        from drivers.null_motorboard import NullMotionBoard
        from modules.scope_capabilities import ScopeCapabilities

        return ScopeCapabilities.from_drivers(
            motion=NullMotionBoard(),
            led=led_stub,
            camera=None,
        )

    def test_has_firmware_stim_true_when_driver_reports_true(self):
        from drivers.simulated_ledboard import SimulatedLEDBoard

        caps = self._build_caps(SimulatedLEDBoard(supports_firmware_stim=True))
        assert caps.has_firmware_stim is True
        assert caps.supports('firmware_stim') is True

    def test_has_firmware_stim_false_when_driver_reports_false(self):
        from drivers.simulated_ledboard import SimulatedLEDBoard

        caps = self._build_caps(SimulatedLEDBoard(supports_firmware_stim=False))
        assert caps.has_firmware_stim is False
        assert caps.supports('firmware_stim') is False

    def test_has_firmware_stim_false_for_null_led_board(self):
        from drivers.null_ledboard import NullLEDBoard

        caps = self._build_caps(NullLEDBoard())
        assert caps.has_firmware_stim is False
        assert caps.supports('firmware_stim') is False

    def test_old_driver_without_method_defaults_to_false(self):
        """Driver without supports_firmware_stim() -- the realistic
        AttributeError pre-rollout scenario -- gets False via _probe's
        AttributeError fallback (per the Rule 8 capability-probe corollary)."""
        from drivers.null_motorboard import NullMotionBoard
        from modules.scope_capabilities import ScopeCapabilities

        old_led = MagicMock(spec=['available_channels', 'available_colors'])
        # spec= restricts attrs: supports_firmware_stim raises AttributeError
        old_led.available_channels = MagicMock(return_value=())
        old_led.available_colors = MagicMock(return_value=())

        caps = ScopeCapabilities.from_drivers(
            motion=NullMotionBoard(),
            led=old_led,
            camera=None,
        )
        assert caps.has_firmware_stim is False
