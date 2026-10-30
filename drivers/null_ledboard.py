# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Null-object LED board -- no-op implementation of the LEDBoard interface.

Used when no LED hardware is present (e.g., LEDBoard connection failure).
All methods return safe defaults: currents return -1, state queries return
False/disabled, commands are silently dropped.

This eliminates the need for ``if self.led is None`` guards throughout the
codebase (the API handles missing hardware gracefully).

The Lumascope API assigns ``self.led = NullLEDBoard()`` instead of
``self.led = None``, so callers never need to check for None.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from drivers.registry import led_registry

logger = logging.getLogger('LVP.drivers.null_ledboard')


@led_registry.register('null', priority=0)
class NullLEDBoard:
    """No-op LED board that satisfies the full LEDBoard interface.

    Attributes match what ``lumascope_api.py`` and other callers access
    directly (``driver``, ``found``, ``port``, etc.).
    """

    # No board, so no text wire: the diagnostic door sends it nothing.
    takes_text_commands: ClassVar[bool] = False

    # The addresses a real EL-0940 LED board drives (LEDBoard).
    _ADDRESSES: ClassVar[tuple[int, ...]] = tuple(range(8))

    def __init__(self):
        self.driver = True  # truthy sentinel -- satisfies `not self.led.driver`
        self.found = False
        self.port = None
        self.is_v2 = False

        logger.debug('[NULL LED  ] NullLEDBoard initialized (no LED hardware)')

    def available_channels(self) -> tuple[int, ...]:
        """Return the supported LED addresses.

        The same addresses as a real RP2040 LED board so callers using
        the NullLEDBoard fallback see consistent ranges and silently
        no-op rather than raise ValueError on an address 0-7.
        """
        return self._ADDRESSES

    # ------------------------------------------------------------------
    # Core LED methods (no-ops)
    # ------------------------------------------------------------------
    def led_on(self, channel, mA, block=False, timeout_s=5.0) -> None:
        """Null implementation: no-op."""
        pass

    def led_off(self, channel) -> None:
        """Null implementation: no-op."""
        pass

    def led_on_fast(self, channel, mA) -> None:
        """Null implementation: no-op."""
        pass

    def led_off_fast(self, channel) -> None:
        """Null implementation: no-op."""
        pass

    def leds_off(self) -> None:
        """Null implementation: no-op."""
        pass

    def leds_off_fast(self) -> None:
        """Null implementation: no-op."""
        pass

    def leds_enable(self) -> None:
        """Null implementation: no-op."""
        pass

    def leds_disable(self) -> None:
        """Null implementation: no-op."""
        pass

    # State-query methods have been retired; see ledboard.py for
    # rationale.

    def get_status(self):
        """Null implementation: returns sentinel value.

        Returns:
            None: Always.
        """
        return None

    def supports_firmware_stim(self) -> bool:
        """Null implementation: no hardware = no firmware STIM support."""
        return False

    def max_ma(self) -> int:
        """No board, so no legal current above zero. Nothing compares against
        it: the API refuses LED commands on a Null board before its range
        guard runs."""
        return 0

    def commanded_ma(self, mA: float) -> float:
        """No board, so nothing is commanded. Unreached for the same reason
        as ``max_ma``."""
        return 0.0

    def wait_until_on(self, timeout_s=5.0) -> None:
        """Null implementation: no-op."""
        pass

    # ------------------------------------------------------------------
    # ADC / calibration (no-ops)
    # ------------------------------------------------------------------
    def read_led_current(self, channel):
        """Null implementation: returns sentinel value.

        Returns:
            None: Always.
        """
        return None

    # ------------------------------------------------------------------
    # Connection (no-ops)
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Null implementation: no-op."""
        pass

    def disconnect(self) -> None:
        """Null implementation: no-op."""
        pass

    def is_connected(self) -> bool:
        """Null implementation: never connected.

        Returns:
            bool: Always False.
        """
        return False

    # ------------------------------------------------------------------
    # Write-only (no-ops)
    # ------------------------------------------------------------------
    def _write_command_fast(self, command):
        pass

    def _safety_leds_off(self):
        pass

    def _on_disconnect(self):
        pass
