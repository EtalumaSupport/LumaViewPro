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
from collections.abc import Callable

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


def _set_ui_dispatcher(dispatcher: UiDispatcher | None) -> None:
    """Set the global UI dispatcher; ``ScopeSession.set_ui_dispatcher`` is its one caller.

    None restores direct invocation.
    """
    global _ui_dispatcher
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
