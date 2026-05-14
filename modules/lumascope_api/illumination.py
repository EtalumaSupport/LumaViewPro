# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""IlluminationAPI -- sub-API for LED / illuminator control.

Phase 1 of Wave 7 decomposition. Thin delegating facade over the
Lumascope composition root. Bodies still live on Lumascope; later
phases relocate them and migrate callers.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.2 for the canonical
method list. Channel-spec widening (set_channel / clear_channel) lands
in a later Wave 7 phase paired with the STATE-LED-1 collapse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.protocols import LEDBoardProtocol


class IlluminationAPI:
    """Illumination sub-API. Forwards to Lumascope composition root."""

    def __init__(self, scope: 'Lumascope', driver: 'LEDBoardProtocol') -> None:
        self._scope = scope
        self._driver = driver

    # --- Sync control ---
    def led_on(self, *args, **kwargs):
        return self._scope.led_on(*args, **kwargs)

    def led_off(self, *args, **kwargs):
        return self._scope.led_off(*args, **kwargs)

    def leds_off(self, *args, **kwargs):
        return self._scope.leds_off(*args, **kwargs)

    def led_on_fast(self, *args, **kwargs):
        return self._scope.led_on_fast(*args, **kwargs)

    def led_off_fast(self, *args, **kwargs):
        return self._scope.led_off_fast(*args, **kwargs)

    def leds_off_fast(self, *args, **kwargs):
        return self._scope.leds_off_fast(*args, **kwargs)

    # --- Async control ---
    def led_on_async(self, *args, **kwargs):
        return self._scope.led_on_async(*args, **kwargs)

    def led_off_async(self, *args, **kwargs):
        return self._scope.led_off_async(*args, **kwargs)

    def leds_off_async(self, *args, **kwargs):
        return self._scope.leds_off_async(*args, **kwargs)

    def led_on_sync(self, *args, **kwargs):
        return self._scope.led_on_sync(*args, **kwargs)

    def leds_off_sync(self, *args, **kwargs):
        return self._scope.leds_off_sync(*args, **kwargs)

    # --- State ---
    def get_led_ma(self, *args, **kwargs):
        return self._scope.get_led_ma(*args, **kwargs)

    def led_enabled(self, *args, **kwargs):
        return self._scope.led_enabled(*args, **kwargs)

    def led_illumination(self, *args, **kwargs):
        return self._scope.led_illumination(*args, **kwargs)

    @property
    def led_states(self) -> dict:
        return self._scope.led_states

    def get_led_state(self, *args, **kwargs):
        return self._scope.get_led_state(*args, **kwargs)

    def get_led_states(self, *args, **kwargs):
        return self._scope.get_led_states(*args, **kwargs)

    def get_led_status(self, *args, **kwargs):
        return self._scope.get_led_status(*args, **kwargs)

    # --- Save / restore ---
    def save_led_state(self, *args, **kwargs):
        return self._scope.save_led_state(*args, **kwargs)

    def restore_led_state(self, *args, **kwargs):
        return self._scope.restore_led_state(*args, **kwargs)

    def leds_off_owned(self, *args, **kwargs):
        return self._scope.leds_off_owned(*args, **kwargs)

    def leds_enable(self, *args, **kwargs):
        return self._scope.leds_enable(*args, **kwargs)

    def leds_disable(self, *args, **kwargs):
        return self._scope.leds_disable(*args, **kwargs)

    # --- Wait ---
    def wait_until_led_on(self, *args, **kwargs):
        return self._scope.wait_until_led_on(*args, **kwargs)

    # --- Channel mapping ---
    def ch2color(self, *args, **kwargs):
        return self._scope.ch2color(*args, **kwargs)

    def color2ch(self, *args, **kwargs):
        return self._scope.color2ch(*args, **kwargs)

    # --- Listeners ---
    def add_led_listener(self, *args, **kwargs):
        return self._scope.add_led_listener(*args, **kwargs)

    def remove_led_listener(self, *args, **kwargs):
        return self._scope.remove_led_listener(*args, **kwargs)
