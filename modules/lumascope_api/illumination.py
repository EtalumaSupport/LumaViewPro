# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""IlluminationAPI -- sub-API for LED / illuminator control.

Wave 7 Phase 3d -- bodies and state slots fully relocated from
Lumascope. IlluminationAPI owns _led_state (Rule 2 SoT), _led_owners,
_led_listeners, and the three locks that serialize their access plus
LED-driver I/O.

See docs/PLUGIN_API_DESIGN_2026-05-09.md sec 2.2. Channel-spec
widening (set_channel / clear_channel) is deferred -- a future
phase, not Wave 7.
"""

from __future__ import annotations

import logging as _logging
import os
import threading
from typing import TYPE_CHECKING

from lib import profile_trace
from lvp_logger import logger
from modules.sequential_io_executor import IOTask

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.protocols import LEDBoardProtocol

_api_log = _logging.getLogger('LVP.api')


class IlluminationAPI:
    """Illumination sub-API. Owns LED state, ownership tracking, and
    listener registry. Stateful bodies live here post-Phase 3d.
    """

    def __init__(self, scope: 'Lumascope', driver: 'LEDBoardProtocol') -> None:
        self._scope = scope
        # driver argument kept for API compatibility but unused; `_driver`
        # is a @property that re-resolves `self._scope._led_driver` so
        # disconnect / reconnect / test hot-swap propagate without
        # rebinding IlluminationAPI. Same pattern as MotionAPI._driver
        # (Wave 7 Phase 2c precedent).
        del driver  # noqa -- intentionally unused, kept for backward call sites

        # LED change listeners -- push-based UI update mechanism. Each
        # listener is called with (color, enabled, mA, owner) whenever
        # any LED channel changes state. Fires from the thread that
        # caused the change, so listeners MUST schedule UI work via
        # Clock.schedule_once.
        self._led_listeners_lock = threading.Lock()
        self._led_listeners: list = []

        # LED state -- API-level source of truth (Rule 2). The API was
        # always supposed to own LED state, but the implementation only
        # got as far as ownership + observers + save/restore. State
        # queries (get_led_ma, led_enabled, etc.) still delegated to
        # the driver -- which worked for LEDBoard (has an internal
        # led_ma dict) but broke for FX2LEDController (thin translator,
        # returns sentinels). This dict is the primary store, analogous
        # to _pos_cache for motor position. Updated inside led_on /
        # led_off / leds_off; read by all state-query methods. See
        # docs/AUDIT_LED_STATE_FX2.md.
        # Each entry: color -> {'enabled': True, 'illumination_ma': float, 'owner': str}
        self._led_state: dict[str, dict] = {}

        # LED ownership tracking -- prevents subsystems from turning
        # off LEDs they did not turn on. Each led_on with an owner
        # records who claimed the channel. led_off with a non-matching
        # owner is a no-op. leds_off() without owner is the "nuclear"
        # option (shutdown only).
        self._led_owner_lock = threading.Lock()
        self._led_owners: dict[str, str] = {}  # color -> owner tag

        # LED driver I/O serialization. Split from the old global
        # _hw_lock to allow LED stim pulses during camera grabs and
        # motor moves. Threading audit sec 10.2 -- wrapped with
        # TimedLock for contention tracing.
        self._led_lock = profile_trace.TimedLock(
            threading.RLock(), name="illumination._led_lock"
        )

    @property
    def _driver(self) -> 'LEDBoardProtocol':
        """Resolve the LED driver via the composition root each access.

        Lumascope's `_led_driver` slot is reassigned on disconnect /
        reconnect and during tests that hot-swap drivers. Re-resolving
        here keeps IlluminationAPI in sync without rebinding.
        """
        return self._scope._led_driver

    # --- Sync control ---
    def led_on(self, channel, mA, block: bool = False, owner: str = '') -> None:
        """Turn on an LED channel at the specified current.

        Args:
            channel: Channel number (0-5) or color name string.
            mA: Illumination current in milliamps.
            block: If True, wait for confirmation from the LED board.
            owner: Optional ownership tag (e.g. 'autofocus', 'protocol').
                If set, only ``led_off`` / ``leds_off_owned`` with the same
                owner can turn this channel off.  Empty string (default) means
                no ownership tracking.

        Raises:
            ValueError: If channel or mA is out of range.
        """
        if not self._driver:
            return

        if isinstance(channel, str):
            channel = self.color2ch(color=channel)

        valid_channels = self._driver.available_channels()
        if channel not in valid_channels:
            raise ValueError(f"LED channel must be one of {valid_channels}, got {channel}")
        if not isinstance(mA, (int, float)) or mA < 0 or mA > self._scope.LED_MAX_MA:
            raise ValueError(f"LED current must be 0-{self._scope.LED_MAX_MA} mA, got {mA}")

        # Skip redundant command if channel is already on at the same current
        color_name = self.ch2color(channel)
        if color_name:
            current_ma = self.get_led_ma(color_name)
            # Rule 12 workaround: _led_state cache-equality trace for the
            # slider > ~150 mA silent-fail bench investigation. Gated by
            # LVP_FX2_DEBUG_WIRE env var to match drivers/fx2driver.py.
            # Remove together with fx2driver._FX2_DEBUG_WIRE block after
            # the 2026-04-21 bench session.
            if os.environ.get("LVP_FX2_DEBUG_WIRE") == "1":
                cached_entry = self._led_state.get(color_name)
                is_enabled = self.led_enabled(color_name)
                try:
                    delta = (None if current_ma is None
                             else abs(float(mA) - float(current_ma)))
                except Exception:
                    delta = 'ERR'
                _api_log.info(
                    '[FX2 LED diag] led_on cache-check color=%s '
                    'new_mA=%r (type=%s) cached_mA=%r (type=%s) '
                    'delta=%r enabled=%s cache_entry=%r',
                    color_name, mA, type(mA).__name__,
                    current_ma, type(current_ma).__name__,
                    delta, is_enabled, cached_entry,
                )
            if current_ma is not None and abs(float(mA) - float(current_ma)) < 0.01:
                if self.led_enabled(color_name):
                    return

        with self._led_lock:
            self._driver.led_on(channel, mA, block=block)
        self._scope.imaging.frame_validity.invalidate('led')
        _api_log.info(f'led_on ch={channel} mA={mA} owner={owner!r}')

        # Update API-level state cache + ownership (Rule 2). Unconditional
        # -- empty owner ('') is recorded too, fixing AUDIT_LED_STATE_FX2.md
        # Bug 3 where UI clicks were never tracked because of an `if owner:`
        # gate that excluded empty strings.
        color_name = self.ch2color(channel)
        if color_name:
            with self._led_owner_lock:
                self._led_state[color_name] = {
                    'enabled': True,
                    'illumination_ma': float(mA),
                    'owner': owner,
                }
                self._led_owners[color_name] = owner
            self._fire_led_listeners(color_name, True, float(mA), owner)

    def led_off(self, channel, owner: str = '') -> None:
        """Turn off an LED channel.

        Args:
            channel: Channel number (0-5) or color name string.
            owner: If set, only turn off if this owner currently owns
                the channel.  A non-matching owner is a no-op (logged).
                Empty string (default) turns off unconditionally.

        Raises:
            ValueError: If channel is out of range.
        """
        if not self._driver:
            return

        if isinstance(channel, str):
            channel = self.color2ch(color=channel)

        valid_channels = self._driver.available_channels()
        if channel not in valid_channels:
            raise ValueError(f"LED channel must be one of {valid_channels}, got {channel}")

        # Skip if channel is already off. Now reads from the API-level
        # _led_state cache, which is correct for both LEDBoard and FX2.
        # Pre-fix this delegated to the driver's get_led_state, which for
        # FX2 always returned False -- making led_off a complete no-op
        # (AUDIT_LED_STATE_FX2.md Bug 2).
        color_name = self.ch2color(channel)
        if color_name and not self.led_enabled(color_name):
            return

        # Check ownership -- if caller specifies an owner, only allow if it matches
        if owner and color_name:
            with self._led_owner_lock:
                entry = self._led_state.get(color_name, {})
                current_owner = entry.get('owner', '')
                if current_owner and current_owner != owner:
                    _api_log.debug(f'led_off blocked: ch={channel} owner={owner!r} '
                                   f'but owned by {current_owner!r}')
                    return

        with self._led_lock:
            self._driver.led_off(channel)
        self._scope.imaging.frame_validity.invalidate('led')
        _api_log.info(f'led_off ch={channel} owner={owner!r}')

        # Clear from API-level state cache + ownership
        if color_name:
            with self._led_owner_lock:
                self._led_state.pop(color_name, None)
                self._led_owners.pop(color_name, None)
            self._fire_led_listeners(color_name, False, 0.0, owner)

    def leds_off(self) -> None:
        """Turn off all LEDs (nuclear -- ignores ownership, clears all owners)."""
        if not self._driver:
            return
        with self._led_lock:
            self._driver.leds_off()
        with self._led_owner_lock:
            self._led_owners.clear()
            self._led_state.clear()
        self._scope.imaging.frame_validity.invalidate('led')
        _api_log.info('leds_off')
        for color in self._driver.available_colors():
            self._fire_led_listeners(color, False, 0.0, '')

    def led_on_fast(self, channel, mA) -> None:
        """Turn on an LED with write-only (no read-back) for time-critical pulses.

        Args:
            channel: Channel number (0-5) or color name string.
            mA: Illumination current in milliamps.

        Raises:
            ValueError: If channel or mA is out of range.
        """
        if not self._driver:
            return
        if isinstance(channel, str):
            channel = self.color2ch(color=channel)
        valid_channels = self._driver.available_channels()
        if channel not in valid_channels:
            raise ValueError(f"LED channel must be one of {valid_channels}, got {channel}")
        if not isinstance(mA, (int, float)) or mA < 0 or mA > self._scope.LED_MAX_MA:
            raise ValueError(f"LED current must be 0-{self._scope.LED_MAX_MA} mA, got {mA}")
        with self._led_lock:
            self._driver.led_on_fast(channel, mA)
        self._scope.imaging.frame_validity.invalidate('led')
        color_name = self.ch2color(channel)
        if color_name:
            self._fire_led_listeners(color_name, True, float(mA), '')

    def led_off_fast(self, channel) -> None:
        """Turn off an LED with write-only (no read-back) for time-critical pulses.

        Args:
            channel: Channel number (0-5) or color name string.

        Raises:
            ValueError: If channel is out of range.
        """
        if not self._driver:
            return
        if isinstance(channel, str):
            channel = self.color2ch(color=channel)
        valid_channels = self._driver.available_channels()
        if channel not in valid_channels:
            raise ValueError(f"LED channel must be one of {valid_channels}, got {channel}")
        with self._led_lock:
            self._driver.led_off_fast(channel)
        self._scope.imaging.frame_validity.invalidate('led')
        color_name = self.ch2color(channel)
        if color_name:
            self._fire_led_listeners(color_name, False, 0.0, '')

    def leds_off_fast(self) -> None:
        """Turn off all LEDs with write-only (no read-back) for time-critical pulses."""
        if not self._driver:
            return
        with self._led_lock:
            self._driver.leds_off_fast()
        self._scope.imaging.frame_validity.invalidate('led')
        with self._led_owner_lock:
            self._led_state.clear()
        for color in self._driver.available_colors():
            self._fire_led_listeners(color, False, 0.0, '')

    # --- Async control ---
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

    def led_on_async(self, channel, mA, *, callback=None,
                     cb_kwargs=None, owner: str = '') -> None:
        """Submit ``led_on(channel, mA)`` to the io_executor.

        Args:
            channel: Channel number or color name.
            mA: LED current in milliamps.
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
            args=(channel, mA),
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

    def led_on_sync(self, channel, mA, *, timeout_s=5,
                    owner: str = '') -> None:
        """Run ``led_on`` through the io_executor and block until done.

        Args:
            channel: Channel number or color name.
            mA: LED current in milliamps.
            timeout_s: Max seconds to wait for completion.
            owner: Optional ownership tag for the LED state.
        """
        if not self._scope.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        kwargs = {'owner': owner} if owner else {}
        ex = self._scope._require_executor(self._scope._io_executor, 'led_on_sync')
        task = IOTask(action=self.led_on, args=(channel, mA),
                      kwargs=kwargs)
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout_s)

    def leds_off_sync(self, *, timeout_s=5) -> None:
        """Run ``leds_off`` through the io_executor and block until done.

        Args:
            timeout_s: Max seconds to wait for completion.
        """
        if not self._scope.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return
        ex = self._scope._require_executor(self._scope._io_executor, 'leds_off_sync')
        task = IOTask(action=self.leds_off)
        fut = ex.put(task, return_future=True)
        if fut:
            fut.result(timeout=timeout_s)

    # --- State ---
    def get_led_ma(self, color: str) -> float:
        """Get the current illumination level for an LED channel.

        Reads from the API-level _led_state cache (Rule 2). Does NOT
        delegate to the driver -- see AUDIT_LED_STATE_FX2.md Bug 4.

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            Illumination in milliamps, or -1 if channel is off or
            LED board unavailable.
        """
        if not self._driver:
            return -1
        with self._led_owner_lock:
            entry = self._led_state.get(color)
            return entry['illumination_ma'] if entry else -1.0

    def led_enabled(self, color: str) -> bool:
        """Whether a specific LED channel is currently on.

        Reads from the API-level _led_state cache (Rule 2). Pre-fix,
        this delegated to the driver's get_led_state, which for
        FX2LEDController always returned False -- making led_off a
        complete no-op (AUDIT_LED_STATE_FX2.md Bug 2).

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            True if the channel is currently on.
        """
        if not self._driver:
            return False
        with self._led_owner_lock:
            return self._led_state.get(color) is not None

    def led_illumination(self, color: str) -> float:
        """Current mA for an LED channel, or -1 if off.

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            Illumination in milliamps, or -1 if off / unavailable.
        """
        return self.get_led_ma(color)

    @property
    def led_states(self) -> dict:
        """Snapshot of all LED states {color: {enabled, illumination}}.

        Returns:
            Mapping of color -> {'enabled': bool, 'illumination_ma': float}.
            Empty if no LED board is connected.
        """
        if not self._driver:
            return {}
        with self._led_owner_lock:
            return {
                color: {'enabled': True, 'illumination_ma': entry['illumination_ma']}
                for color, entry in self._led_state.items()
            }

    def get_led_state(self, color: str) -> dict:
        """Get the on/off state and illumination for an LED channel.

        Reads from the API-level _led_state cache (Rule 2).

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            {'enabled': bool, 'illumination_ma': float}.
        """
        if not self._driver:
            return {'enabled': False, 'illumination_ma': -1}
        with self._led_owner_lock:
            entry = self._led_state.get(color)
            if entry is None:
                return {'enabled': False, 'illumination_ma': -1}
            return {'enabled': True, 'illumination_ma': entry['illumination_ma']}

    def get_led_states(self) -> dict:
        """Get state and illumination for all LED channels.

        Returns states for ALL channels the driver supports (not just
        currently-on channels).

        Returns:
            Mapping of color -> {'enabled': bool, 'illumination_ma': float}
            for every channel the driver supports. Empty if no LED
            board is connected.
        """
        if not self._driver:
            return {}
        all_colors = self._driver.available_colors()
        with self._led_owner_lock:
            return {
                color: (
                    {'enabled': True, 'illumination_ma': self._led_state[color]['illumination_ma']}
                    if color in self._led_state
                    else {'enabled': False, 'illumination_ma': -1}
                )
                for color in all_colors
            }

    def get_led_status(self) -> 'int | None':
        """Get the LED board status register.

        Returns:
            Driver-defined status object (typically int bitfield), or
            None if no LED board is connected.
        """
        if not self._driver:
            return None
        return self._driver.get_status()

    # --- Save / restore ---
    def save_led_state(self, tag: str) -> dict:
        """Snapshot the current LED state for later restoration.

        Args:
            tag: Descriptive name for the snapshot (for logging).

        Returns:
            Snapshot suitable for passing to ``restore_led_state``.
        """
        states = self.get_led_states()
        with self._led_owner_lock:
            owners = dict(self._led_owners)
        snapshot = {'tag': tag, 'states': states, 'owners': owners}
        _api_log.info(f'save_led_state tag={tag}: '
                      f'{[c for c, s in states.items() if s.get("enabled")]}')
        return snapshot

    def restore_led_state(self, snapshot: dict, owner: str = '') -> None:
        """Restore LEDs to a previously saved state.

        Turns off channels owned by *owner* (or all if owner is empty),
        then re-enables channels that were on in the snapshot.

        Args:
            snapshot: Return value from ``save_led_state``.
            owner: If set, only turn off channels currently owned by
                this owner before restoring.
        """
        if not snapshot:
            return
        tag = snapshot.get('tag', '?')
        saved_states = snapshot.get('states', {})
        _api_log.info(f'restore_led_state tag={tag}')

        # Turn off what the owner turned on
        if owner:
            self.leds_off_owned(owner)
        else:
            self.leds_off()

        # Restore channels that were on in the snapshot
        for color, state in saved_states.items():
            if state.get('enabled', False):
                mA = state.get('illumination_ma', 0)
                if mA and mA > 0:
                    ch = self.color2ch(color)
                    if ch is not None:
                        saved_owner = snapshot.get('owners', {}).get(color, '')
                        self.led_on(channel=ch, mA=mA, owner=saved_owner)

    def leds_off_owned(self, owner: str) -> None:
        """Turn off only the LED channels owned by *owner*.

        Channels owned by other subsystems are left alone.

        Args:
            owner: The owner tag whose channels should be turned off.
        """
        if not self._driver or not owner:
            return
        with self._led_owner_lock:
            channels_to_off = [color for color, own in self._led_owners.items()
                               if own == owner]
            for color in channels_to_off:
                self._led_owners.pop(color, None)
                self._led_state.pop(color, None)
        for color in channels_to_off:
            ch = self.color2ch(color)
            if ch is not None:
                with self._led_lock:
                    self._driver.led_off(ch)
                self._scope.imaging.frame_validity.invalidate('led')
                _api_log.info(f'led_off ch={ch} (owned release by {owner})')
                self._fire_led_listeners(color, False, 0.0, owner=owner)

    # --- Enable / disable ---
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

    # --- Wait ---
    def wait_until_led_on(self, timeout_s: float = 5.0) -> bool:
        """Block until the LED board confirms an LED is on.

        Mirrors motion.wait_until_finished_moving in shape.

        Args:
            timeout_s: Maximum seconds to wait (default 5s).

        Returns:
            bool: True if confirmed on, False on timeout / no driver /
            firmware lacks STATUS (current state until v3.1 firmware).
        """
        if not self._driver:
            return False
        return self._driver.wait_until_on(timeout_s)

    # --- Channel mapping ---
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

    # --- Listeners ---
    def add_led_listener(self, listener) -> None:
        """Register a callback for LED state changes.

        The listener is called with ``(color, enabled, mA, owner)`` whenever
        any LED channel changes state.  It fires from the thread that caused
        the change, so listeners **must** schedule UI work via
        ``Clock.schedule_once``.

        Args:
            listener: ``callable(color: str, enabled: bool, mA: float, owner: str)``
        """
        with self._led_listeners_lock:
            self._led_listeners.append(listener)

    def remove_led_listener(self, listener) -> None:
        """Unregister an LED listener.

        Args:
            listener: A callable previously passed to ``add_led_listener``.
                Silently ignores listeners that are not currently registered.
        """
        with self._led_listeners_lock:
            try:
                self._led_listeners.remove(listener)
            except ValueError:
                pass

    def _fire_led_listeners(self, color: str, enabled: bool, mA: float,
                            owner: str = '') -> None:
        """Notify all LED listeners of a state change on *color*."""
        with self._led_listeners_lock:
            listeners = list(self._led_listeners)
        for fn in listeners:
            try:
                fn(color, enabled, mA, owner)
            except Exception as ex:
                _api_log.debug(f'led listener error: {ex}')
