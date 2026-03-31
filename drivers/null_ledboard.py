# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Null-object LED board -- no-op implementation of the LEDBoard interface.

Used when no LED hardware is present (e.g., LEDBoard connection failure).
All methods return safe defaults: currents return -1, state queries return
False/disabled, commands are silently dropped.

This eliminates the need for ``if self.led is None`` guards throughout the
codebase (Rule 8: API handles missing hardware gracefully).

The Lumascope API assigns ``self.led = NullLEDBoard()`` instead of
``self.led = None``, so callers never need to check for None.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger('LVP.drivers.null_ledboard')


class NullLEDBoard:
    """No-op LED board that satisfies the full LEDBoard interface.

    Attributes match what ``lumascope_api.py`` and other callers access
    directly (``driver``, ``found``, ``port``, etc.).
    """

    def __init__(self):
        self.driver = True  # truthy sentinel -- satisfies `not self.led.driver`
        self.found = False
        self.port = None
        self.is_v2 = False
        self._state_lock = threading.Lock()
        self.led_ma = {color: -1 for color in ('BF', 'PC', 'DF', 'Red', 'Blue', 'Green')}

        logger.debug('[NULL LED  ] NullLEDBoard initialized (no LED hardware)')

    # ------------------------------------------------------------------
    # Core LED methods (no-ops)
    # ------------------------------------------------------------------
    def led_on(self, channel, mA, block=False, timeout=5.0): pass
    def led_off(self, channel): pass
    def led_on_fast(self, channel, mA): pass
    def led_off_fast(self, channel): pass
    def leds_off(self): pass
    def leds_off_fast(self): pass
    def leds_enable(self): pass
    def leds_disable(self): pass

    # ------------------------------------------------------------------
    # State queries (return safe defaults)
    # ------------------------------------------------------------------
    def get_led_ma(self, color): return -1
    def is_led_on(self, color): return False
    def get_led_state(self, color): return {'enabled': False, 'illumination': -1}
    def get_led_states(self):
        return {c: {'enabled': False, 'illumination': -1} for c in self.led_ma}
    def get_status(self): return None
    def wait_until_on(self, timeout=5.0): pass

    # ------------------------------------------------------------------
    # Channel mapping
    # ------------------------------------------------------------------
    def color2ch(self, color):
        return {'Blue': 0, 'Green': 1, 'Red': 2, 'BF': 3, 'PC': 4, 'DF': 5}.get(color, 3)

    def ch2color(self, channel):
        return {0: 'Blue', 1: 'Green', 2: 'Red', 3: 'BF', 4: 'PC', 5: 'DF'}.get(channel, 'BF')

    # ------------------------------------------------------------------
    # ADC / calibration (no-ops)
    # ------------------------------------------------------------------
    def read_led_current(self, channel): return None

    # ------------------------------------------------------------------
    # Connection (no-ops)
    # ------------------------------------------------------------------
    def connect(self): pass
    def disconnect(self): pass
    def is_connected(self): return False
    def exchange_command(self, command, **kwargs): return None

    # ------------------------------------------------------------------
    # Write-only (no-ops)
    # ------------------------------------------------------------------
    def _write_command_fast(self, command): pass
    def _safety_leds_off(self): pass
    def _on_disconnect(self): pass
