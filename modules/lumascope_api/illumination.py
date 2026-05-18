# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""IlluminationAPI -- sub-API for LED / illuminator control.

Wave 7 Phase 3c onward. Stateless methods own their bodies here;
stateful bodies and the LED state slots relocate in Phase 3d.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.2 for the canonical
method list. Channel-spec widening (set_channel / clear_channel) lands
in a later Wave 7 phase paired with the STATE-LED-1 collapse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lvp_logger import logger
from modules.sequential_io_executor import IOTask

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.protocols import LEDBoardProtocol


class IlluminationAPI:
    """Illumination sub-API. Stateless bodies live here; stateful
    bodies and LED state slots relocate in Phase 3d."""

    def __init__(self, scope: 'Lumascope', driver: 'LEDBoardProtocol') -> None:
        self._scope = scope
        self._driver = driver

    # --- Sync control (stateful -- bodies in Phase 3d) ---
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

    # --- Async control (stateless -- bodies here, Phase 3c) ---
    def leds_off_async(self, *, callback=None) -> None:
        """Submit ``leds_off`` to the io_executor.

        No-op if LED disconnected.

        Args:
            callback: Optional completion callback.
        """
        if not self._scope.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        ex = self._scope._require_executor(self._scope._io_executor, 'leds_off_async')
        ex.put(IOTask(action=self.leds_off, callback=callback))
        logger.info('[SCOPE API ] leds_off_async()')

    def led_on_async(self, channel, illumination, *, callback=None,
                     cb_kwargs=None, owner: str = '') -> None:
        """Submit ``led_on(channel, illumination)`` to the io_executor.

        Args:
            channel: Channel number or color name.
            illumination: Illumination current in mA.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
            owner: Optional ownership tag for the LED state.
        """
        if not self._scope.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        kwargs = {'owner': owner} if owner else {}
        ex = self._scope._require_executor(self._scope._io_executor, 'led_on_async')
        ex.put(IOTask(
            action=self.led_on,
            args=(channel, illumination),
            kwargs=kwargs,
            callback=callback,
            cb_kwargs=cb_kwargs,
        ))

    def led_off_async(self, channel, *, callback=None, cb_kwargs=None,
                      owner: str = '') -> None:
        """Submit ``led_off(channel)`` to the io_executor.

        Args:
            channel: Channel number or color name.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
            owner: Optional ownership tag; only matching owner can turn
                off the channel.
        """
        if not self._scope.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        kwargs = {'channel': channel}
        if owner:
            kwargs['owner'] = owner
        ex = self._scope._require_executor(self._scope._io_executor, 'led_off_async')
        ex.put(IOTask(
            action=self.led_off,
            kwargs=kwargs,
            callback=callback,
            cb_kwargs=cb_kwargs,
        ))

    def led_on_sync(self, channel, illumination, *, timeout=5,
                    owner: str = '') -> None:
        """Run ``led_on`` through the io_executor and block until done.

        Args:
            channel: Channel number or color name.
            illumination: Illumination current in mA.
            timeout: Max seconds to wait for completion.
            owner: Optional ownership tag for the LED state.
        """
        if not self._scope.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        kwargs = {'owner': owner} if owner else {}
        ex = self._scope._require_executor(self._scope._io_executor, 'led_on_sync')
        task = IOTask(action=self.led_on, args=(channel, illumination),
                      kwargs=kwargs)
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout)

    def leds_off_sync(self, *, timeout=5) -> None:
        """Run ``leds_off`` through the io_executor and block until done.

        Args:
            timeout: Max seconds to wait for completion.
        """
        if not self._scope.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        ex = self._scope._require_executor(self._scope._io_executor, 'leds_off_sync')
        task = IOTask(action=self.leds_off)
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout)

    # --- State (stateful -- bodies in Phase 3d) ---
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

    def get_led_status(self) -> 'int | None':
        """Get the LED board status register.

        Returns:
            Driver-defined status object (typically int bitfield), or
            None if no LED board is connected.
        """
        if not self._driver:
            return None
        return self._driver.get_status()

    # --- Save / restore (stateful -- bodies in Phase 3d) ---
    def save_led_state(self, *args, **kwargs):
        return self._scope.save_led_state(*args, **kwargs)

    def restore_led_state(self, *args, **kwargs):
        return self._scope.restore_led_state(*args, **kwargs)

    def leds_off_owned(self, *args, **kwargs):
        return self._scope.leds_off_owned(*args, **kwargs)

    # --- Enable / disable (stateless -- bodies here, Phase 3c) ---
    def leds_enable(self) -> None:
        """Enable all LED channels (allows them to be turned on)."""
        if not self._driver:
            return
        self._driver.leds_enable()

    def leds_disable(self) -> None:
        """Disable all LED channels (prevents them from turning on)."""
        if not self._driver:
            return
        self._driver.leds_disable()

    # --- Wait (stateless -- body here, Phase 3c) ---
    def wait_until_led_on(self) -> None:
        """Block until the LED board confirms an LED is on."""
        if not self._driver:
            return
        self._driver.wait_until_on()

    # --- Channel mapping (stateless -- bodies here, Phase 3c) ---
    def ch2color(self, channel: int) -> 'str | None':
        """Convert a channel number to its color name string.

        Args:
            channel: Channel number (0=Blue, 1=Green, 2=Red, 3=BF, 4=PC, 5=DF).

        Returns:
            Color name (e.g. "Blue", "BF"), or None if LED board unavailable.
        """
        if not self._driver:
            return None
        return self._driver.ch2color(channel)

    def color2ch(self, color: str) -> 'int | None':
        """Convert a color name string to its channel number.

        Args:
            color: Color name ("Blue", "Green", "Red", "BF", "PC", "DF").

        Returns:
            Channel number (0-5), or None if LED board unavailable.
        """
        if not self._driver:
            return None
        return self._driver.color2ch(color)

    # --- Listeners (stateful -- bodies in Phase 3d) ---
    def add_led_listener(self, *args, **kwargs):
        return self._scope.add_led_listener(*args, **kwargs)

    def remove_led_listener(self, *args, **kwargs):
        return self._scope.remove_led_listener(*args, **kwargs)
