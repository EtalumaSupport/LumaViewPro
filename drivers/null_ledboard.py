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
import threading
from typing import ClassVar

from drivers.registry import led_registry

logger = logging.getLogger('LVP.drivers.null_ledboard')


@led_registry.register('null', priority=0)
class NullLEDBoard:
    """No-op LED board that satisfies the full LEDBoard interface.

    Attributes match what ``lumascope_api.py`` and other callers access
    directly (``driver``, ``found``, ``port``, etc.).
    """

    _COLOR_TO_CH: ClassVar[dict] = {
        'Blue': 0,
        'Green': 1,
        'Red': 2,
        'BF': 3,
        'PC': 4,
        'DF': 5,
    }
    _CH_TO_COLOR: ClassVar[dict] = {v: k for k, v in _COLOR_TO_CH.items()}

    def __init__(self):
        self.driver = True  # truthy sentinel -- satisfies `not self.led.driver`
        self.found = False
        self.port = None
        self.is_v2 = False
        self._state_lock = threading.Lock()
        self.led_ma = dict.fromkeys(self._COLOR_TO_CH, -1)

        logger.debug('[NULL LED  ] NullLEDBoard initialized (no LED hardware)')

    def available_channels(self) -> tuple:
        """Return the supported LED channel numbers.

        Returns the same 6-channel range as a real RP2040 LED board so
        callers using the NullLEDBoard fallback see consistent ranges
        and silently no-op rather than raise ValueError on channel 0-5.

        Returns:
            tuple: Channel numbers (ints) supported by this board.
        """
        return tuple(self._COLOR_TO_CH.values())

    def available_colors(self) -> tuple:
        """Return the supported LED color names.

        Returns:
            tuple: Color name strings supported by this board.
        """
        return tuple(self._COLOR_TO_CH.keys())

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

    def wait_until_on(self, timeout_s=5.0) -> None:
        """Null implementation: no-op."""
        pass

    # ------------------------------------------------------------------
    # Channel mapping
    # ------------------------------------------------------------------
    def color2ch(self, color) -> int:
        """Convert color name to numerical channel.

        Args:
            color: Color name (e.g. 'BF', 'Red', 'Blue').

        Returns:
            int: Channel number (0-5). Defaults to 3 (BF) for unknown names.
        """
        return self._COLOR_TO_CH.get(color, 3)

    def ch2color(self, channel) -> str:
        """Convert numerical channel to color name.

        Args:
            channel: Channel number (0-5).

        Returns:
            str: Color name. Defaults to 'BF' for unknown channels.
        """
        return self._CH_TO_COLOR.get(channel, 'BF')

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

    def exchange_command(self, command, **kwargs):
        """Null implementation: returns sentinel value.

        Returns:
            None: Always.
        """
        return None

    # ------------------------------------------------------------------
    # Write-only (no-ops)
    # ------------------------------------------------------------------
    def _write_command_fast(self, command):
        pass

    def _safety_leds_off(self):
        pass

    def _on_disconnect(self):
        pass
