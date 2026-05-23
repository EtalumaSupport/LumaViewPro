# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""LVP-A-13 -- pluggable Scheduler protocol for periodic background work.

Lumascope-side periodic work (metrics logging, motion monitor,
future Pylon-thread health checks, GC-pressure pollers) needs to fire
on a cadence regardless of whether the host process is a Kivy app, a
REST server, a headless soak harness, or a CLI tool. Each environment
brings its own scheduling primitive: Kivy uses ``Clock``, asyncio uses
``loop.call_later``, vanilla Python uses ``threading.Timer``.

This module formalizes the previously-informal "pass two callables
matching ``Clock.schedule_interval / Clock.unschedule``" pattern into
a real ``Scheduler`` protocol with two reference implementations:

- :class:`KivyClockScheduler` -- wraps Kivy ``Clock`` for the LVP App.
- :class:`ThreadingTimerScheduler` -- wraps ``threading.Timer`` for
  REST API, headless soak, future CLI tools.

Stays Rule-15-clean: ``Scheduler`` itself imports nothing GUI; the
Kivy variant only imports Kivy when constructed (so headless callers
never trigger a Kivy import path).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Protocol, runtime_checkable


# Type alias for a tick callback. Schedulers may invoke either
# ``cb()`` (no-arg) OR ``cb(dt)`` (Kivy convention). Implementations
# normalize so the user can pass a no-arg lambda safely.
TickCallback = Callable[..., None]


@runtime_checkable
class Scheduler(Protocol):
    """Pluggable periodic-task scheduler.

    Implementations let MetricsLogger (and any future Lumascope-side
    periodic work) fire on a cadence without coupling to a specific
    event loop. The protocol is intentionally tiny -- three methods --
    so the surface stays implementable from any host environment.

    All methods are safe to call from any thread. Cancellation must
    be idempotent (cancelling an already-cancelled handle is a no-op).
    """

    def schedule_interval(self, callback: TickCallback, interval_s: float) -> object:
        """Schedule ``callback`` to fire every ``interval_s`` seconds.

        Returns an opaque handle suitable for ``unschedule(handle)``.
        Implementations may invoke ``callback()`` or ``callback(dt)`` --
        wrap your callback to accept ``*args`` if you don't care which.
        """
        ...

    def unschedule(self, handle: object) -> None:
        """Cancel the schedule represented by ``handle``. Idempotent."""
        ...

    def shutdown(self) -> None:
        """Cancel every schedule this scheduler owns + release resources.

        Idempotent. After ``shutdown()`` the scheduler refuses new
        ``schedule_interval`` calls.
        """
        ...


class KivyClockScheduler:
    """Scheduler implementation backed by Kivy ``Clock``.

    Used by the Kivy App entry point. Kivy callbacks fire on the
    MainThread so the wrapped callback runs there too -- appropriate for
    UI-touching ticks; metrics logging is fine because it just calls
    ``logger.info`` which is thread-safe.
    """

    def __init__(self, clock=None):
        """Initialize.

        Args:
            clock: The Kivy Clock module. Defaults to importing
                ``kivy.clock.Clock`` lazily so importing this module
                from a headless context (which would never construct
                a ``KivyClockScheduler``) doesn't drag Kivy along.
        """
        if clock is None:
            from kivy.clock import Clock as _Clock

            clock = _Clock
        self._clock = clock
        self._handles: list[object] = []
        self._closed = False
        self._lock = threading.Lock()

    def schedule_interval(self, callback, interval_s):
        if self._closed:
            raise RuntimeError('KivyClockScheduler is shutdown; refusing new schedule')

        # Clock invokes callback(dt); accept callbacks that ignore it.
        def _wrapped(dt=0):
            try:
                callback()
            except TypeError:
                # Caller's callback wants the dt arg — pass it through.
                callback(dt)

        handle = self._clock.schedule_interval(_wrapped, interval_s)
        with self._lock:
            self._handles.append(handle)
        return handle

    def unschedule(self, handle):
        if handle is None:
            return
        try:
            self._clock.unschedule(handle)
        except Exception:
            pass
        with self._lock:
            try:
                self._handles.remove(handle)
            except ValueError:
                pass

    def shutdown(self):
        with self._lock:
            handles = list(self._handles)
            self._handles.clear()
            self._closed = True
        for h in handles:
            try:
                self._clock.unschedule(h)
            except Exception:
                pass


class _PeriodicTimer:
    """Internal: re-arming wrapper around stdlib ``threading.Timer``.

    ``threading.Timer`` is one-shot. To get an interval, the timer's
    callback re-arms itself. This class encapsulates the re-arm logic
    + cancellation so :class:`ThreadingTimerScheduler` can hand each
    scheduled interval a single opaque handle.
    """

    def __init__(
        self,
        callback: TickCallback,
        interval_s: float,
        on_error: Optional[Callable[[BaseException], None]] = None,
        name: str = 'PeriodicTimer',
    ):
        self._callback = callback
        self._interval_s = interval_s
        self._on_error = on_error
        self._name = name
        self._timer: Optional[threading.Timer] = None
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

    def _tick(self):
        if self._cancelled.is_set():
            return
        try:
            # Match Kivy's "callback may take dt" convention: pass the
            # interval as dt so callbacks that expect it get a sensible
            # value. Callers that don't take an arg use the same
            # try/except shim as KivyClockScheduler does internally.
            try:
                self._callback()
            except TypeError:
                self._callback(self._interval_s)
        except Exception as e:
            if self._on_error is not None:
                try:
                    self._on_error(e)
                except Exception:
                    pass
        finally:
            # Re-arm only if not cancelled while we were running.
            if not self._cancelled.is_set():
                self._arm()

    def _arm(self):
        with self._lock:
            if self._cancelled.is_set():
                return
            self._timer = threading.Timer(self._interval_s, self._tick)
            self._timer.daemon = True
            self._timer.name = self._name
            self._timer.start()

    def start(self):
        self._arm()
        return self  # Returning self so the scheduler can use it as the handle.

    def cancel(self):
        self._cancelled.set()
        with self._lock:
            t = self._timer
            self._timer = None
        if t is not None:
            t.cancel()


class ThreadingTimerScheduler:
    """Scheduler implementation backed by stdlib ``threading.Timer``.

    Used by REST API, headless soak harness, CLI tools -- any
    environment without a Kivy Clock. Each scheduled interval gets its
    own daemon timer thread that re-arms after every callback. Daemon
    threading is intentional: an unexpected interpreter exit must not
    hang on a pending timer.

    Safe to construct from any thread. Callbacks fire on the timer
    thread, NOT on the calling thread -- callbacks that need to
    serialize with other work should use their own lock.
    """

    def __init__(
        self,
        name_prefix: str = 'LVP-MetricsTimer',
        on_callback_error: Optional[Callable[[BaseException], None]] = None,
    ):
        """Initialize.

        Args:
            name_prefix: Thread-name prefix for daemon timers; helps
                ``threading.enumerate()`` / debugger output identify
                the source.
            on_callback_error: Optional hook called with the exception
                when a callback raises. Default behavior is to swallow
                the exception silently (matches Kivy's Clock behavior
                for unhandled exceptions in scheduled callbacks).
        """
        self._name_prefix = name_prefix
        self._on_callback_error = on_callback_error
        self._timers: list[_PeriodicTimer] = []
        self._closed = False
        self._lock = threading.Lock()
        self._next_id = 0

    def schedule_interval(self, callback, interval_s):
        if self._closed:
            raise RuntimeError('ThreadingTimerScheduler is shutdown; refusing new schedule')
        with self._lock:
            self._next_id += 1
            tid = self._next_id
        timer = _PeriodicTimer(
            callback=callback,
            interval_s=interval_s,
            on_error=self._on_callback_error,
            name=f'{self._name_prefix}-{tid}',
        )
        with self._lock:
            self._timers.append(timer)
        timer.start()
        return timer

    def unschedule(self, handle):
        if handle is None:
            return
        if not isinstance(handle, _PeriodicTimer):
            return
        handle.cancel()
        with self._lock:
            try:
                self._timers.remove(handle)
            except ValueError:
                pass

    def shutdown(self):
        with self._lock:
            timers = list(self._timers)
            self._timers.clear()
            self._closed = True
        for t in timers:
            try:
                t.cancel()
            except Exception:
                pass


# --------------- backwards-compat callable adapter ---------------
#
# Pre-LVP-A-13 callers passed two callables (``Clock.schedule_interval,
# Clock.unschedule``) directly into ``MetricsLogger.start``. The new
# Scheduler-based start() accepts a Scheduler instance, but we keep
# the two-callable form working via this adapter so a callsite that
# hasn't migrated yet keeps building. Delete after the last caller
# migrates.


class _CallablePairScheduler:
    """Wraps a (schedule_interval_fn, unschedule_fn) pair as a Scheduler.

    Used by the backwards-compat path inside MetricsLogger.start; not
    expected to be constructed by user code (use KivyClockScheduler /
    ThreadingTimerScheduler instead).
    """

    def __init__(self, schedule_interval_fn, unschedule_fn):
        self._schedule = schedule_interval_fn
        self._unschedule = unschedule_fn
        self._handles: list[object] = []
        self._closed = False
        self._lock = threading.Lock()

    def schedule_interval(self, callback, interval_s):
        if self._closed:
            raise RuntimeError('_CallablePairScheduler is shutdown; refusing new schedule')

        def _wrapped(dt=0):
            try:
                callback()
            except TypeError:
                callback(dt)

        h = self._schedule(_wrapped, interval_s)
        with self._lock:
            self._handles.append(h)
        return h

    def unschedule(self, handle):
        if handle is None:
            return
        try:
            self._unschedule(handle)
        except Exception:
            pass
        with self._lock:
            try:
                self._handles.remove(handle)
            except ValueError:
                pass

    def shutdown(self):
        with self._lock:
            handles = list(self._handles)
            self._handles.clear()
            self._closed = True
        for h in handles:
            try:
                self._unschedule(h)
            except Exception:
                pass
