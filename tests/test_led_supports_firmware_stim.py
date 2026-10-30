# Copyright Etaluma, Inc.
"""Regression tests for LEDBoard.supports_firmware_stim probe + the
ScopeCapabilities `has_firmware_stim` flag plumbing.

Probe shape (per `drivers/ledboard.py::LEDBoard.supports_firmware_stim`):
- Sends `STIM 0 0 1 2 1\\n` (intentionally invalid mA=0)
- v3.0.8+ firmware replies with `STIM: mA must be > 0` (parser recognized)
- Pre-v3.0.8 firmware echoes the command + returns `Command not recognized`

Only an answer the board gave is cached on the LEDBoard instance (via
`_supports_stim_cached`), so subsequent calls return without re-probing the
bus; a probe the board did not answer raises and is asked again.

The capability flag lands on `ScopeCapabilities` as `has_firmware_stim`
(immutable per-Lumascope-instance fact, sourced at Lumascope.__init__
from LEDBoard.supports_firmware_stim()). Caller gate is
`scope.capabilities.has_firmware_stim`.
"""

import logging
import threading
from unittest.mock import MagicMock

import pytest

from drivers.ledboard import LEDBoard
from drivers.sim_wire.backend import SimWireBackend
from tests.wire_boards import led_on_the_wire


def _stim_probes(caplog):
    return [
        r for r in caplog.records if r.name == 'LVP.serial' and 'STIM 0 0 1 2 1' in r.getMessage()
    ]


class TestSupportsFirmwareStimProbe:
    """Probe-detection logic on LEDBoard.

    The field LED firmware's answers run on the firmware-backed simulator;
    a v3.0.8+ reply has no emulated firmware yet (an LED image built from the
    STIM branch is a stage 12 gap), so those cases script the port."""

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
        led.driver.readline = MagicMock(side_effect=lambda: responses.pop(0) if responses else b'')
        led.driver.reset_input_buffer = MagicMock()
        led.driver.write = MagicMock()
        led.driver.in_waiting = 0
        led.driver.read = MagicMock(return_value=b'')
        return led

    def test_v308_plus_firmware_returns_true(self):
        """Modern firmware reply (`STIM: mA must be > 0`) -> probe True."""
        led = self._make_led([b'STIM: mA must be > 0\r\n'])
        assert led.supports_firmware_stim() is True

    @pytest.mark.firmware_tier
    def test_pre_v308_firmware_returns_false(self):
        """Legacy firmware reply (`Command not recognized`) -> probe False:
        the field LED firmware's own answer."""
        led, _sim = led_on_the_wire()
        assert led.supports_firmware_stim() is False

    def test_stim_diag_response_also_recognized(self):
        """STIM_DIAG: prefix (alternate parser-recognized form) -> True."""
        led = self._make_led([b'STIM_DIAG: probe ok\r\n'])
        assert led.supports_firmware_stim() is True

    def test_command_echo_then_recognized_response(self):
        """Some firmware echoes the command first, then replies. The
        probe loop should ignore the echo + accept the second line."""
        led = self._make_led(
            [
                b'STIM 0 0 1 2 1\r\n',  # command echo -- neither STIM: nor "not recognized"
                b'STIM: mA must be > 0\r\n',  # parser recognition
            ]
        )
        assert led.supports_firmware_stim() is True

    @pytest.mark.firmware_tier
    def test_no_reply_by_the_deadline_raises_and_the_next_call_asks_again(self, caplog):
        """The reply is held past the probe's 2.5 s deadline (a dropped reply
        on this board takes the echo line, and the answer behind it still
        arrives): not an answer, so nothing is cached and the next call asks
        the firmware again, which answers it has no STIM."""
        from drivers.exceptions import BoardNoReplyError

        led, sim = led_on_the_wire()
        sim.delay_next_reply(3.0)
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            with pytest.raises(
                BoardNoReplyError, match=r'^STIM 0 0 1 2 1: no reply from the board$'
            ):
                led.supports_firmware_stim()
            assert led.supports_firmware_stim() is False
        assert len(_stim_probes(caplog)) == 2, 'nothing was cached, so the firmware was asked again'

    def test_lines_that_are_neither_answer_are_unreadable_and_not_cached(self):
        """A board that talks but never says STIM or not-recognized gave no answer."""
        from drivers.exceptions import BoardReplyUnreadableError

        led = self._make_led([b'>>> \r\n', b'Traceback (most recent call last):\r\n'])
        with pytest.raises(BoardReplyUnreadableError, match=r'^STIM 0 0 1 2 1: the board answered'):
            led.supports_firmware_stim()
        assert not hasattr(led, '_supports_stim_cached')

    def test_a_port_that_fails_after_the_probe_is_sent_is_no_reply_and_closes_it(self):
        # a stand-in by design: the simulated wire has no member that fails
        # the port between the probe's write and its read
        import serial

        from drivers.exceptions import BoardNoReplyError

        led = self._make_led([])
        led.driver.readline = MagicMock(side_effect=serial.SerialException('device gone'))
        with pytest.raises(BoardNoReplyError, match=r'after the probe was sent: device gone$'):
            led.supports_firmware_stim()
        assert led.driver is None, 'a failed port is closed, as the wire closes it'

    @pytest.mark.firmware_tier
    def test_result_is_cached_after_first_call(self, caplog):
        """Second call should return cached value without re-probing."""
        led, _sim = led_on_the_wire()
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            first = led.supports_firmware_stim()
            second = led.supports_firmware_stim()
        assert first is second is False
        assert len(_stim_probes(caplog)) == 1, 'the cached answer went to the wire'

    def test_a_closed_port_is_the_port_failure_not_no_stim(self, caplog):
        """No LED board on the bus -> no driver -> the port's failure, never
        asked of the wire and nothing cached."""
        from drivers.exceptions import BoardPortFailedError

        led = LEDBoard(backend=SimWireBackend(None, None))
        assert led.driver is None
        with (
            caplog.at_level(logging.INFO, logger='LVP.serial'),
            pytest.raises(BoardPortFailedError, match=r'^STIM 0 0 1 2 1: the port is closed$'),
        ):
            led.supports_firmware_stim()
        assert _stim_probes(caplog) == []
        assert not hasattr(led, '_supports_stim_cached')


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
    and exposes the result via .has_firmware_stim."""

    def _build_caps(self, led_stub):
        """Build a ScopeCapabilities snapshot with a stub LED driver."""
        from drivers.null_motorboard import NullMotionBoard
        from modules.layer_record import UNRESOLVED
        from modules.scope_capabilities import ScopeCapabilities

        return ScopeCapabilities.from_drivers(
            motion=NullMotionBoard(),
            led=led_stub,
            camera=None,
            channel_identity=UNRESOLVED,
            scope_models={},
            timing_record=None,
        )

    def test_has_firmware_stim_true_when_driver_reports_true(self):
        from drivers.simulated_ledboard import SimulatedLEDBoard

        caps = self._build_caps(SimulatedLEDBoard(supports_firmware_stim=True))
        assert caps.has_firmware_stim is True
        assert caps.has_firmware_stim is True

    def test_has_firmware_stim_false_when_driver_reports_false(self):
        from drivers.simulated_ledboard import SimulatedLEDBoard

        caps = self._build_caps(SimulatedLEDBoard(supports_firmware_stim=False))
        assert caps.has_firmware_stim is False
        assert caps.has_firmware_stim is False

    def test_has_firmware_stim_false_for_null_led_board(self):
        from drivers.null_ledboard import NullLEDBoard

        caps = self._build_caps(NullLEDBoard())
        assert caps.has_firmware_stim is False
        assert caps.has_firmware_stim is False

    def test_old_driver_without_method_defaults_to_false(self):
        """Driver without supports_firmware_stim() -- the realistic
        AttributeError pre-rollout scenario -- gets False via _probe's
        AttributeError fallback (per the Rule 8 capability-probe corollary)."""
        from drivers.null_motorboard import NullMotionBoard
        from modules.layer_record import UNRESOLVED
        from modules.scope_capabilities import ScopeCapabilities

        old_led = MagicMock(spec=['available_channels'])
        # spec= restricts attrs: supports_firmware_stim raises AttributeError
        old_led.available_channels = MagicMock(return_value=())

        caps = ScopeCapabilities.from_drivers(
            motion=NullMotionBoard(),
            led=old_led,
            camera=None,
            channel_identity=UNRESOLVED,
            scope_models={},
            timing_record=None,
        )
        assert caps.has_firmware_stim is False
