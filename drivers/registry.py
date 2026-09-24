# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Driver registry -- retires the hardcoded if/elif chains in Lumascope.__init__.

Pre-B2, `Lumascope.__init__` hardcoded driver selection:

    if simulate:
        self.motion = SimulatedMotorBoard(model=sim_model)
    else:
        self.motion = MotorBoard()
    # ... same pattern for LED and camera ...

Adding a new hardware variant (e.g., the FX2 driver for Lumaview Classic,
or a TMC5240-based RP2350 motor board) meant editing this constructor
and coupling it to every driver class. New hardware should mean a new
driver with zero changes above this layer.

B2 replaces the chains with three registries (motor / LED / camera).
Each driver self-registers via decorator:

    # drivers/motorboard.py
    from drivers.registry import motor_registry

    @motor_registry.register('rp2040', priority=100)
    class MotorBoard(SerialBoard): ...

    # drivers/null_motorboard.py
    @motor_registry.register('null', priority=0)
    class NullMotionBoard: ...

    # drivers/simulated_motorboard.py
    @motor_registry.register('sim', priority=100, is_simulator=True)
    class SimulatedMotorBoard: ...

Lumascope asks the registry for a driver:

    self.motion = motor_registry.create(motor_kind, simulate=simulate, **kw)

where `motor_kind` is 'auto' (default) or a specific name like 'rp2040'.

**Auto mode:** tries registered drivers in descending priority order,
returns the first one whose constructor succeeds. If all real drivers
fail, falls back to the registered 'null' driver (or raises if no null
is registered). In simulate mode, only `is_simulator=True` drivers are
considered.

**Composite hardware -- FX2 / Lumaview Classic pattern:** The FX2 chip
exposes both a camera and an LED controller over a single shared USB
connection. `FX2Camera` (camera registry) and `FX2LEDController` (LED
registry) register as two independent drivers -- conditionally, gated on
the pyusb / libusb prerequisites being importable -- that internally share
a module-level `_FX2Connection` singleton. From the registry's point of
view they're separate entries; the shared connection is an implementation
detail inside the fx2driver module. Note: this pattern applies to
camera+LED only -- LS720 motion uses a standalone USB-to-serial motor
controller, not the FX2, and registers as its own independent motor driver.

See `tests/test_driver_registry.py::TestRegistryAccommodatesCompositeHardware`
for the composite-hardware registry pattern and `tests/test_fx2_driver.py`
for the FX2 driver's own coverage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

logger = logging.getLogger('LVP.drivers.registry')


class DriverNotLiveError(RuntimeError):
    """A driver selected by name constructed, but its hardware is not live."""


def _disconnect_quietly(instance: Any) -> None:
    """Release a rejected driver's port and threads; a failure here is
    secondary to the rejection already decided."""
    try:
        if hasattr(instance, 'disconnect'):
            instance.disconnect()
    except Exception as e:
        logger.debug(
            f'[registry] {type(instance).__name__} disconnect() after rejection raised '
            f'({type(e).__name__}: {e})'
        )


@dataclass
class _RegistryEntry:
    cls: type[Any]
    priority: int
    is_simulator: bool


class DriverRegistry:
    """Name-keyed driver registry with auto-detect support.

    One instance per driver kind (motor / LED / camera). Populated at
    import time via the `@register` decorator; queried at runtime via
    `create()`. The registry is not thread-safe for registration -- all
    `@register` calls happen once at module import, before any thread
    has a chance to call `create()`.
    """

    def __init__(self, kind: str):
        self._kind = kind  # 'motor' | 'led' | 'camera' -- for error messages
        self._entries: dict[str, _RegistryEntry] = {}

    def register(self, name: str, *, priority: int = 50, is_simulator: bool = False) -> Callable:
        """Decorator. Registers a driver class under `name`.

        Args:
            name: Short registry key (e.g. 'rp2040', 'pylon', 'fx2', 'null').
                Must be unique per registry.
            priority: Auto-detect ordering. Higher = tried first. Real
                drivers for current production hardware typically use
                100. Fallback / legacy drivers use lower values (50, 80).
                Null drivers MUST be 0 so they sort last.
            is_simulator: True if this driver is for simulate=True mode.
                Simulators and real drivers live in separate priority
                orders -- `create(simulate=True)` only considers simulators
                and `create(simulate=False)` only considers real drivers.
        """

        def decorator(cls):
            if name in self._entries:
                existing = self._entries[name].cls.__name__
                raise ValueError(
                    f'{self._kind} driver name {name!r} already registered '
                    f'to {existing}; cannot also register {cls.__name__}'
                )
            self._entries[name] = _RegistryEntry(
                cls=cls,
                priority=priority,
                is_simulator=is_simulator,
            )
            logger.debug(
                f'[registry] {self._kind}: registered {name!r} -> '
                f'{cls.__name__} (priority={priority}, sim={is_simulator})'
            )
            return cls

        return decorator

    @staticmethod
    def _liveness_failure(instance: Any) -> tuple[str, str] | None:
        """Why a constructed driver is not a live one, or None when it is.

        Three ways a constructor returns without a working board, each a
        (verdict, reason) pair:

        - ``found``: the driver reports ``found=False`` instead of raising
          (the serial-board pattern: no port matched its VID/PID).
        - ``connected``: the port was discovered but ``open()`` failed
          (another program holds it), and the driver swallowed that and
          left itself half-built; handed on, it crashes later on its
          first command rather than here.
        - ``responsive``: the port opened and stayed open, but the board
          answered zero bytes to the whole connect-time reset sequence and
          will reject every command until a power cycle; selected anyway,
          the operator gets one failure per command for the session.

        A check the driver does not implement passes.
        """
        if getattr(instance, 'found', True) is False:
            return 'found', 'found=False'
        if hasattr(instance, 'is_connected'):
            try:
                connected = instance.is_connected()
            except Exception as e:
                connected = False
                logger.debug(
                    f'[registry] {type(instance).__name__} is_connected() raised '
                    f'({type(e).__name__}: {e}); treating as not-connected'
                )
            if not connected:
                return 'connected', 'constructed but is_connected()=False (port held / open failed)'
        if hasattr(instance, 'is_responsive'):
            try:
                responsive = instance.is_responsive()
            except Exception as e:
                responsive = False
                logger.debug(
                    f'[registry] {type(instance).__name__} is_responsive() raised '
                    f'({type(e).__name__}: {e}); treating as not-responsive'
                )
            if not responsive:
                return (
                    'responsive',
                    'connected but not responding (zero bytes to the connect sequence)',
                )
        return None

    def get(self, name: str) -> type[Any]:
        """Return the class registered under `name`, or raise ValueError."""
        entry = self._entries.get(name)
        if entry is None:
            available = sorted(self._entries.keys())
            raise ValueError(
                f'No {self._kind} driver registered as {name!r}. Available: {available}'
            )
        return entry.cls

    def create(self, name: str = 'auto', *, simulate: bool = False, **kwargs) -> Any:
        """Construct a driver instance.

        Args:
            name: Registry key or 'auto'. With 'auto', tries drivers in
                descending priority order, returning the first whose
                constructor succeeds. Falls back to the registered 'null'
                driver if all real drivers fail.
            simulate: If True, only considers drivers with
                `is_simulator=True`. If False, only considers real
                drivers (`is_simulator=False`).
            **kwargs: Forwarded to the driver constructor. Callers can
                pass driver-specific args like `model='LS850T'` for
                SimulatedMotorBoard or `z_position_func=...` for
                SimulatedCamera.

        Returns:
            A driver instance. Never returns None -- always raises or
            returns a concrete driver (possibly the null fallback).

        Raises:
            ValueError: If `name` is not 'auto' and isn't registered,
                or if 'auto' finds no candidates at all.
        """
        if name != 'auto':
            # Selecting by NAME has no fallback: the caller asked for this
            # driver and nothing else, so a board that constructed but never
            # came up is a failure to raise, not a null driver to hand back.
            # The liveness verdict is the auto path's, so a dead board is
            # refused the same way on both branches instead of reaching the
            # API layer indistinguishable from a live one.
            cls = self.get(name)
            instance = cls(**kwargs)
            failure = self._liveness_failure(instance)
            if failure is not None:
                _, why = failure
                _disconnect_quietly(instance)
                raise DriverNotLiveError(f'{self._kind} driver {cls.__name__} ({name!r}) {why}')
            return instance

        # Auto mode -- pick by priority, filtered by simulate flag.
        candidates = sorted(
            (e for e in self._entries.values() if e.is_simulator == simulate),
            key=lambda e: -e.priority,
        )

        if not candidates:
            raise ValueError(
                f'{self._kind} registry has no '
                f'{"simulator" if simulate else "real"} drivers. '
                f'Auto-create cannot proceed.'
            )

        # Try each candidate in priority order. Skip the null driver
        # until all real drivers have been attempted -- null is the
        # explicit fallback, not a candidate.
        real_candidates = [e for e in candidates if e.priority > 0]
        null_candidates = [e for e in candidates if e.priority == 0]

        last_error: Exception | None = None
        found_false_names: list[str] = []
        not_connected_names: list[str] = []
        not_responsive_names: list[str] = []
        for entry in real_candidates:
            try:
                instance = entry.cls(**kwargs)
                failure = self._liveness_failure(instance)
                if failure is not None:
                    verdict, why = failure
                    {
                        'found': found_false_names,
                        'connected': not_connected_names,
                        'responsive': not_responsive_names,
                    }[verdict].append(entry.cls.__name__)
                    if verdict == 'responsive':
                        logger.warning(
                            f'[registry] {self._kind}: {entry.cls.__name__} {why}; '
                            f'refusing it. The board needs a power cycle.'
                        )
                    else:
                        logger.debug(
                            f'[registry] {self._kind}: {entry.cls.__name__} {why}, '
                            f'trying next candidate'
                        )
                    if verdict != 'found':
                        _disconnect_quietly(instance)
                    continue
                # Every rejection above is logged; without this the
                # SUCCESS path was the only silent one, so which driver
                # actually won could only be inferred from the ABSENCE
                # of a fallback warning -- and the Null* drivers
                # announce themselves at debug, so on a default-level
                # field log it could not be inferred at all.
                logger.info(f'[registry] {self._kind}: using {entry.cls.__name__}')
                return instance
            except Exception as e:
                last_error = e
                logger.debug(
                    f'[registry] {self._kind}: {entry.cls.__name__} failed '
                    f'({type(e).__name__}: {e}), trying next candidate'
                )

        # All real drivers exhausted -- fall back to null if one is registered.
        # Three distinct cases the operator needs to be able to tell apart:
        #   1. last_error set                -- at least one driver raised.
        #      Log the error type + message (no traceback -- see below).
        #   2. found_false_names non-empty   -- at least one driver instantiated
        #      but reported `found=False` (SerialBoard with no port, FX2 with
        #      no device). Name the drivers that were tried so the operator
        #      can see WHICH hardware path was probed.
        #   3. neither                       -- registry actually empty for
        #      this kind. Misconfiguration; the import that wires up the
        #      driver is missing.
        # Pre-fix the case-3 message used to fire for case-2 too, which made
        # "no real drivers attempted (none registered)" look like an empty
        # registry when actually `found=False` had been returned. Bit us
        # 2026-04-15 chasing why FX2 wasn't in the picture.
        for entry in null_candidates:
            if last_error is not None:
                # Name the error type + message but not the traceback: a
                # missing board at startup is an expected fallback, and the
                # stack is always the same driver connect() chain. The type
                # and message are the diagnostic payload.
                logger.warning(
                    f'[registry] {self._kind}: all real drivers failed, '
                    f'falling back to {entry.cls.__name__}. '
                    f'Last error: {type(last_error).__name__}: {last_error}'
                )
            elif not_responsive_names:
                logger.warning(
                    f'[registry] {self._kind}: all real drivers connected but '
                    f'returned zero bytes to the connect sequence -- the board '
                    f'is powered and enumerated but its firmware is not '
                    f'answering, which needs a power cycle. Tried: '
                    f'{", ".join(not_responsive_names)}. '
                    f'Falling back to {entry.cls.__name__}'
                )
            elif not_connected_names:
                logger.warning(
                    f'[registry] {self._kind}: all real drivers constructed '
                    f'but reported is_connected()=False (port held by another '
                    f'process, or open() failed). Tried: '
                    f'{", ".join(not_connected_names)}. '
                    f'Falling back to {entry.cls.__name__}'
                )
            elif found_false_names:
                logger.warning(
                    f'[registry] {self._kind}: all real drivers reported '
                    f'found=False (no hardware detected). Tried: '
                    f'{", ".join(found_false_names)}. '
                    f'Falling back to {entry.cls.__name__}'
                )
            else:
                logger.info(
                    f'[registry] {self._kind}: registry is empty for this '
                    f'kind (no real drivers registered -- check that the '
                    f'driver module is imported somewhere). Falling back '
                    f'to {entry.cls.__name__}'
                )
            return entry.cls()

        # No null driver registered -- raise with the last real-driver error.
        if last_error is not None:
            raise last_error
        raise ValueError(f'{self._kind} registry has no real drivers and no null fallback.')

    def registered_names(self) -> list[str]:
        """For tests and diagnostics."""
        return sorted(self._entries.keys())


# Three global registries -- one per driver kind.
motor_registry = DriverRegistry('motor')
led_registry = DriverRegistry('led')
camera_registry = DriverRegistry('camera')
