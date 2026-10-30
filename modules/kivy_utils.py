# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""UI dispatch utilities for module-layer code.

Module-layer code must not import Kivy. This module holds the process's
one UI dispatcher, which a GUI host sets at startup through
``ScopeSession.set_ui_dispatcher``; every lane, run delivery and listener
that hands a callback to the UI thread reads it here at the moment it
dispatches. A process has one UI thread, so one store. With none set
(tests, headless, REST) a callback is called directly.

Every post from another thread goes through one ordered queue here, drained
on the UI thread: an event (``schedule_ui``) runs once, in the order it was
posted; a state (``schedule_ui_latest``) keeps one entry, moved to the back
with its newest value, so a held-up UI thread holds one entry per state
however long it waits. The queue is LVP's, not the toolkit's: a Qt GUI gives
a dispatcher and reuses it.
"""

import dataclasses
import threading
from collections.abc import Callable, Hashable

from modules.api_surface import api_fields
from modules.exceptions import BlockingCallRefusedError


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

# The entries waiting for the UI thread, in the order they run: key -> func.
# An event's key is a fresh object, so each event is an entry of its own; a
# state's key is its caller's, so a state has at most one entry.
_queue_lock = threading.Lock()
_queue: dict[Hashable, Callable] = {}
# One drain is asked of the dispatcher at a time: from the first post into an
# empty queue until that drain has run, a post only joins the queue.
_drain_asked = False
# Bumped by a swap: a drain callback left in the old dispatcher's queue
# carries the old value and runs nothing.
_generation = 0
# Held while a drain runs its entries, so a swap waits for it to finish.
_drain_lock = threading.Lock()
_draining_thread: threading.Thread | None = None


def _set_ui_dispatcher(dispatcher: UiDispatcher | None) -> None:
    """Set the global UI dispatcher; ``ScopeSession.set_ui_dispatcher`` is its one caller.

    The swap waits for a drain in progress, so no entry runs on two threads
    at once, then hands the entries still waiting to the new dispatcher in
    their order, as one drain. None restores direct invocation and runs the
    waiting entries here, each raise reported as a post with no GUI reports
    one, so nothing is left waiting and no raise escapes the swap.

    Raises:
        BlockingCallRefusedError: ``'swap_in_delivery'``, called from inside a
            drain, which the swap would wait on.
    """
    global _ui_dispatcher, _drain_asked, _generation
    if _draining_thread is threading.current_thread():
        raise BlockingCallRefusedError('swap_in_delivery', 'set_ui_dispatcher')
    with _drain_lock, _queue_lock:
        _generation += 1
        generation = _generation
        _ui_dispatcher = dispatcher
        pending = list(_queue.values()) if dispatcher is None else []
        if dispatcher is None:
            _queue.clear()
        _drain_asked = asked = bool(_queue)
    for func in pending:
        _run_inline(func)
    if asked:
        dispatcher.schedule(lambda dt: _drain(generation, dt), 0)


def ui_thread() -> threading.Thread | None:
    """The thread UI callbacks are delivered on, or None when they run inline."""
    dispatcher = _ui_dispatcher
    return None if dispatcher is None else dispatcher.thread


def ui_backlog() -> int:
    """How many entries wait for the UI thread: one per event, one per state key."""
    with _queue_lock:
        return len(_queue)


def schedule_ui(func: Callable) -> None:
    """Run ``func`` once on the UI thread, after everything posted before it; with no GUI, now.

    For an event: a finished task, a run event, an outcome, a popup opening
    or closing. ``func`` receives the Clock's ``dt``. A delay is a UI-thread
    deferral, posted there to the toolkit's own clock.
    """
    if not _post(object(), func):
        _run_inline(func)


def schedule_ui_latest(key: Hashable, func: Callable) -> None:
    """Show the newest state posted under ``key`` on the UI thread; with no GUI, now.

    For an update that shows a state, posted faster than a held-up UI thread
    may run it: one callback per post held every post in Kivy's queue until
    the thread ran again, which for a display posting at 42 fps over a
    four-day stall was 8.7 GB, replayed in full on the return. Here a key has
    at most one waiting entry: a post moves it to the back of the queue with
    the newest function, so the state is shown once, after every event
    posted before it, and an older value waiting when a newer one arrives
    is never shown.

    Args:
        key: Names what the update shows; carry its owner, ``(owner, name)``.
        func: Called with the Clock's ``dt``, as for ``schedule_ui``.
    """
    if not _post(key, func):
        _run_inline(func)


def _post(key: Hashable, func: Callable) -> bool:
    """Put ``func`` at the back of the queue under ``key``; False when there is no GUI.

    The dispatcher is called outside the lock, so an inline dispatcher, or
    an entry that posts again, does not deadlock.
    """
    global _drain_asked
    with _queue_lock:
        dispatcher = _ui_dispatcher
        if dispatcher is None:
            return False
        _queue.pop(key, None)
        _queue[key] = func
        if _drain_asked:
            return True
        _drain_asked = True
        generation = _generation
    dispatcher.schedule(lambda dt: _drain(generation, dt), 0)
    return True


def _drain(generation: int, dt: float) -> None:
    """The UI thread's side: run the entries waiting when the drain starts, in order.

    Entries posted while it runs wait for the next drain, asked as this one
    ends. A raising entry puts the entries behind it back at the head of the
    queue and re-raises, so the raise reaches the toolkit's handler as a lone
    callback's would and nothing behind it is lost. A requeued state whose
    key was posted again while the drain ran is dropped: the newer entry
    already stands behind every event posted before it.
    """
    global _draining_thread
    try:
        with _drain_lock:
            with _queue_lock:
                if generation != _generation:
                    return
                batch = list(_queue.items())
                _queue.clear()
            _draining_thread = threading.current_thread()
            try:
                for i, (_key, func) in enumerate(batch):
                    try:
                        func(dt)
                    except BaseException:
                        _requeue(batch[i + 1 :])
                        raise
            finally:
                _draining_thread = None
    finally:
        _ask_next_drain(generation)


def _requeue(rest: list[tuple[Hashable, Callable]]) -> None:
    """Put a raised drain's remaining entries back at the head of the queue."""
    with _queue_lock:
        newer = dict(_queue)
        _queue.clear()
        _queue.update((key, func) for key, func in rest if key not in newer)
        _queue.update(newer)


def _ask_next_drain(generation: int) -> None:
    """As a drain ends, ask for the next if entries wait; else the next post asks."""
    global _drain_asked
    with _queue_lock:
        if generation != _generation:
            return
        dispatcher = _ui_dispatcher
        _drain_asked = bool(_queue)
        if not _drain_asked:
            return
    dispatcher.schedule(lambda dt: _drain(generation, dt), 0)


def _run_inline(func: Callable) -> None:
    """Run a post with no GUI (tests, headless, REST), reporting a raise."""
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
