# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""UI dispatch utilities for module-layer code.

Module-layer code must not import Kivy. This module holds the process's
one UI dispatcher, which a GUI host sets at startup through
``ScopeSession.set_ui_dispatcher``; every lane, run delivery and listener
that hands a callback to the UI thread reads it here at the moment it
dispatches. A process has one UI thread, so one store. With none set
(tests, headless, REST) a callback is called directly.
"""

import dataclasses
import threading
from collections.abc import Callable, Hashable

from modules.api_surface import api_fields


@api_fields('schedule', 'thread')
@dataclasses.dataclass(frozen=True)
class UiDispatcher:
    """How the GUI delivers a callback, and the thread it delivers on.

    One value, so the thread can never be stale for the dispatcher beside
    it: a wait on that thread for something only a later delivery there can
    do would never return, and a run's waits refuse it.

    Attributes:
        schedule: A function with signature (func, timeout) that schedules
            func on the GUI thread. Typically Clock.schedule_once.
        thread: The thread ``schedule`` delivers on; None for a dispatcher
            that calls inline on the caller's thread.
    """

    schedule: Callable[[Callable[[float], object], float], object]
    thread: threading.Thread | None


# Global UI dispatcher -- set by the GUI host at startup to
# Clock.schedule_once on the main thread. Default is direct invocation.
_ui_dispatcher: UiDispatcher | None = None

# schedule_ui_latest's pending entries: key -> [token, func]. An entry is
# here from its post until the UI thread takes it, so a key has at most one.
_latest_lock = threading.Lock()
_latest_pending: dict[Hashable, list] = {}


def _set_ui_dispatcher(dispatcher: UiDispatcher | None) -> None:
    """Set the global UI dispatcher; ``ScopeSession.set_ui_dispatcher`` is its one caller.

    None restores direct invocation. The pending latest-wins entries were
    posted through the dispatcher being replaced, so they go with it.
    """
    global _ui_dispatcher
    with _latest_lock:
        _latest_pending.clear()
    _ui_dispatcher = dispatcher


def ui_thread() -> threading.Thread | None:
    """The thread UI callbacks are delivered on, or None when they run inline."""
    dispatcher = _ui_dispatcher
    return None if dispatcher is None else dispatcher.thread


def schedule_ui(func: Callable, timeout: float = 0) -> None:
    """Schedule a function on the UI thread, or call directly if no GUI.

    Same signature as Clock.schedule_once -- func receives dt argument.
    """
    dispatcher = _ui_dispatcher
    if dispatcher is not None:
        dispatcher.schedule(func, timeout)
    else:
        # No GUI -- call directly (tests, headless, REST API)
        if callable(func):
            try:
                func(0)
            except Exception as ex:
                # Deliberately more forgiving than the GUI branch, which
                # re-raises: a REST or headless caller must not be killed
                # by one bad UI callback. The failure still has to be
                # visible -- discarding it made a throwing protocol or
                # recording callback produce no record at all, so the run
                # looked like it had succeeded -- and it is reported, not
                # only logged, since no caller waits on a scheduled one.
                from modules.notification_center import notifications

                notifications.report_outcome(ex, solicited=False, category='UI')


def schedule_ui_latest(key: Hashable, func: Callable) -> None:
    """Schedule ``func`` on the UI thread, keeping only the newest per ``key``.

    For an update that shows a state, posted faster than a held-up UI thread
    may run it: one callback per post held every post in Kivy's queue until
    the thread ran again, which for a display posting at 42 fps over a
    four-day stall was 8.7 GB, replayed in full on the return. Here a key has
    at most one pending callback, and a post while it is pending replaces the
    function it will run, so the thread shows the newest state once.

    Each entry carries a token, and a callback runs only the entry it was
    posted for: a callback left queued after its entry was emptied (a
    dispatcher swap) finds a newer entry's token and runs nothing, since
    that entry's own callback is queued behind it. The dispatcher is called
    outside the lock, so an inline dispatcher, or a function that posts
    under its own key, does not deadlock. With no GUI it runs inline, as
    ``schedule_ui`` does.

    Args:
        key: Names what the update shows; carry its owner, ``(owner, name)``.
        func: Called with the Clock's ``dt``, as for ``schedule_ui``.
    """
    dispatcher = _ui_dispatcher
    if dispatcher is None:
        schedule_ui(func)
        return
    with _latest_lock:
        entry = _latest_pending.get(key)
        if entry is not None:
            entry[1] = func
            return
        token = object()
        _latest_pending[key] = [token, func]
    dispatcher.schedule(lambda dt: _run_latest(key, token, dt), 0)


def _run_latest(key: Hashable, token: object, dt: float) -> None:
    """The UI thread's side of ``schedule_ui_latest``: take this callback's entry and run it."""
    with _latest_lock:
        entry = _latest_pending.get(key)
        if entry is None or entry[0] is not token:
            return
        del _latest_pending[key]
    entry[1](dt)
