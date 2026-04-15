# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Driver registry — retires the hardcoded if/elif chains in Lumascope.__init__.

Pre-B2, `Lumascope.__init__` hardcoded driver selection:

    if simulate:
        self.motion = SimulatedMotorBoard(model=sim_model)
    else:
        self.motion = MotorBoard()
    # ... same pattern for LED and camera ...

Adding a new hardware variant (e.g., the FX2 driver for Lumaview Classic,
or a TMC5240-based RP2350 motor board) meant editing this constructor
and coupling it to every driver class. Rule 10 ("New hardware = new
driver, zero changes above") was violated.

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

**Composite hardware — FX2 / Lumaview Classic pattern:** The FX2 chip
exposes both a camera and an LED controller over a single shared USB
connection. Stage 3 will register `FX2Camera` in the camera registry and
`FX2LEDController` in the LED registry as two independent drivers that
internally share a module-level `_FX2Connection` singleton. From the
registry's point of view they're separate entries; the shared connection
is an implementation detail inside the fx2driver module. Note: this
pattern applies to camera+LED only — LS720 motion uses a standalone
USB-to-serial motor controller, not the FX2, and will register as its
own independent motor driver.

See `tests/test_driver_registry.py::TestRegistryAccommodatesCompositeHardware`
for a test double that proves this pattern works before the real FX2
driver lands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Type

logger = logging.getLogger('LVP.drivers.registry')


@dataclass
class _RegistryEntry:
    cls: Type[Any]
    priority: int
    is_simulator: bool


class DriverRegistry:
    """Name-keyed driver registry with auto-detect support.

    One instance per driver kind (motor / LED / camera). Populated at
    import time via the `@register` decorator; queried at runtime via
    `create()`. The registry is not thread-safe for registration — all
    `@register` calls happen once at module import, before any thread
    has a chance to call `create()`.
    """

    def __init__(self, kind: str):
        self._kind = kind  # 'motor' | 'led' | 'camera' — for error messages
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
                orders — `create(simulate=True)` only considers simulators
                and `create(simulate=False)` only considers real drivers.
        """
        def decorator(cls):
            if name in self._entries:
                existing = self._entries[name].cls.__name__
                raise ValueError(
                    f"{self._kind} driver name {name!r} already registered "
                    f"to {existing}; cannot also register {cls.__name__}"
                )
            self._entries[name] = _RegistryEntry(
                cls=cls, priority=priority, is_simulator=is_simulator,
            )
            logger.debug(
                f'[registry] {self._kind}: registered {name!r} → '
                f'{cls.__name__} (priority={priority}, sim={is_simulator})'
            )
            return cls
        return decorator

    def get(self, name: str) -> Type[Any]:
        """Return the class registered under `name`, or raise ValueError."""
        entry = self._entries.get(name)
        if entry is None:
            available = sorted(self._entries.keys())
            raise ValueError(
                f"No {self._kind} driver registered as {name!r}. "
                f"Available: {available}"
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
            A driver instance. Never returns None — always raises or
            returns a concrete driver (possibly the null fallback).

        Raises:
            ValueError: If `name` is not 'auto' and isn't registered,
                or if 'auto' finds no candidates at all.
        """
        if name != 'auto':
            cls = self.get(name)
            return cls(**kwargs)

        # Auto mode — pick by priority, filtered by simulate flag.
        candidates = sorted(
            (e for e in self._entries.values() if e.is_simulator == simulate),
            key=lambda e: -e.priority,
        )

        if not candidates:
            raise ValueError(
                f"{self._kind} registry has no "
                f"{'simulator' if simulate else 'real'} drivers. "
                f"Auto-create cannot proceed."
            )

        # Try each candidate in priority order. Skip the null driver
        # until all real drivers have been attempted — null is the
        # explicit fallback, not a candidate.
        real_candidates = [e for e in candidates if e.priority > 0]
        null_candidates = [e for e in candidates if e.priority == 0]

        last_error: Exception | None = None
        for entry in real_candidates:
            try:
                instance = entry.cls(**kwargs)
                # Some drivers signal failure via a `.found` attribute
                # instead of raising (SerialBoard pattern). Honor that.
                if getattr(instance, 'found', True) is False:
                    logger.debug(
                        f'[registry] {self._kind}: {entry.cls.__name__} '
                        f'found=False, trying next candidate'
                    )
                    continue
                return instance
            except Exception as e:
                last_error = e
                logger.debug(
                    f'[registry] {self._kind}: {entry.cls.__name__} failed '
                    f'({type(e).__name__}: {e}), trying next candidate'
                )

        # All real drivers exhausted — fall back to null if one is registered.
        # Log at warning level with the last exception so the operator sees
        # WHY the scope came up with NullMotionBoard / NullLEDBoard (pre-B2
        # this was `logger.exception(...)` inside `Lumascope.__init__`, so
        # we preserve that visibility here).
        for entry in null_candidates:
            if last_error is not None:
                logger.warning(
                    f'[registry] {self._kind}: all real drivers failed, '
                    f'falling back to {entry.cls.__name__}. '
                    f'Last error: {type(last_error).__name__}: {last_error}',
                    exc_info=last_error,
                )
            else:
                logger.info(
                    f'[registry] {self._kind}: no real drivers attempted '
                    f'(none registered), falling back to {entry.cls.__name__}'
                )
            return entry.cls()

        # No null driver registered — raise with the last real-driver error.
        if last_error is not None:
            raise last_error
        raise ValueError(
            f"{self._kind} registry has no real drivers and no null fallback."
        )

    def registered_names(self) -> list[str]:
        """For tests and diagnostics."""
        return sorted(self._entries.keys())


# Three global registries — one per driver kind.
motor_registry = DriverRegistry('motor')
led_registry = DriverRegistry('led')
camera_registry = DriverRegistry('camera')
