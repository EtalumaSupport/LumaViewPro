# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An operation the API started, as the caller holds it: one kind of handle for every one.

An ``Operation`` is a ``concurrent.futures.Future``, so ``result``,
``exception``, ``done``, ``add_done_callback``, ``concurrent.futures.wait``
and ``asyncio.wrap_future`` behave as a Future's do. What it adds:

- It is running from construction: the operation is under way by the time
  the caller holds it, so ``cancel()`` is False and only its producer
  settles it.
- Its done-callbacks run on one callback thread, one at a time in the order
  their operations settled -- never on the thread that settled it (the
  motion monitor, a lane, a run's thread), so a callback may itself wait on
  any operation without holding the producer that would settle it.
- ``stop()`` stops the operation's work while it is under way, and does
  nothing once it is done or when the operation has no stop.
- A producer whose end is a delivery (a run's ``run_ended``) settles it
  through ``settling``: inside that delivery, the delivering thread's
  ``result()``, ``exception()`` and ``done()`` answer at once, while every
  other observer sees it done only once the delivery has returned.
- A ``result()`` or ``exception()`` on the UI thread for an operation not
  yet done is refused (``BlockingCallRefusedError``, ``'wait_on_ui_thread'``).
"""

import concurrent.futures
import contextlib
import queue
import threading
from collections.abc import Callable, Iterator

from lvp_logger import logger
from modules import kivy_utils
from modules.exceptions import BlockingCallRefusedError


class _CallbackThread:
    """The one thread every operation's done-callbacks run on, started on first use."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None

    def hand(self, callback: Callable[['Operation'], object], op: 'Operation') -> None:
        with self._lock:
            if self._thread is None:
                self._queue = queue.SimpleQueue()
                self._thread = threading.Thread(
                    target=self._serve, args=(self._queue,), name='operation_callbacks', daemon=True
                )
                self._thread.start()
            self._queue.put((callback, op))

    @staticmethod
    def _serve(handed: queue.SimpleQueue) -> None:
        for callback, op in iter(handed.get, None):
            try:
                callback(op)
            except Exception as ex:
                # A callback is the caller's code and nobody waits on it here,
                # so its raise is reported, as a raising UI callback is.
                from modules.notification_center import notifications

                notifications.report_outcome(ex, solicited=False, category='Operation')

    def join(self, timeout_s: float) -> None:
        """Run every callback handed so far, then end the thread; the next hand starts another.

        Waits at most *timeout_s*: a callback is the caller's code, and one
        that does not return -- or that waits on the close that is joining
        it -- keeps its thread past the bound, logged, rather than holding
        the close.
        """
        with self._lock:
            thread, handed = self._thread, self._queue
            self._thread = self._queue = None
        if thread is None:
            return
        handed.put(None)
        # Called from a callback (a session closed by its own done-callback),
        # the thread ends once that callback returns; it cannot wait for itself.
        if thread is threading.current_thread():
            return
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            logger.warning(
                f'[Operation] a done-callback was still running {timeout_s:g} s into the '
                'close; its thread is left to finish it'
            )


_callbacks = _CallbackThread()


# As a lane's worker join is bounded: long enough for any callback that
# returns, short enough that one that never does cannot hold the close.
_JOIN_S = 5.0


def join_callback_thread() -> None:
    """Run every done-callback handed so far, then end the callback thread.

    A session's shutdown calls it, after the hardware teardown that may
    settle operations of its own.
    """
    _callbacks.join(_JOIN_S)


class Operation(concurrent.futures.Future):
    """One operation in flight: its outcome, its done-callbacks and its stop."""

    def __init__(self, *, stop: Callable[[], object] | None = None) -> None:
        super().__init__()
        self.set_running_or_notify_cancel()
        self._stop = stop
        # Set only while a producer delivers this operation's end on one
        # thread: that thread and the outcome it is delivering.
        self._own: tuple[threading.Thread, object, BaseException | None] | None = None

    def stop(self) -> None:
        """Stop this operation's work; nothing once it is done, or when it has no stop."""
        if self._stop is None or super().done():
            return
        self._stop()

    def add_done_callback(self, fn: Callable[['Operation'], object]) -> None:
        """Run ``fn(self)`` on the callback thread once this operation is done."""
        super().add_done_callback(lambda op: _callbacks.hand(fn, op))

    def done(self) -> bool:
        if self._delivering_here() is not None:
            return True
        return super().done()

    def result(self, timeout: float | None = None) -> object:
        own = self._delivering_here()
        if own is not None:
            result, exception = own
            if exception is not None:
                raise exception
            return result
        self._refuse_on_ui_thread('Operation.result')
        return super().result(timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        own = self._delivering_here()
        if own is not None:
            return own[1]
        self._refuse_on_ui_thread('Operation.exception')
        return super().exception(timeout)

    @contextlib.contextmanager
    def settling(
        self, *, result: object = None, exception: BaseException | None = None
    ) -> Iterator[None]:
        """Deliver this operation's end on this thread, then settle it with *result* or *exception*.

        For a producer whose end is a delivery: what runs inside reads the
        outcome at once, every other observer once the delivery has
        returned. It settles even when the delivery raises.
        """
        self._own = (threading.current_thread(), result, exception)
        try:
            yield
        finally:
            self._own = None
            if exception is None:
                self.set_result(result)
            else:
                self.set_exception(exception)

    def _delivering_here(self) -> tuple[object, BaseException | None] | None:
        own = self._own
        if own is None or own[0] is not threading.current_thread():
            return None
        return own[1], own[2]

    def _refuse_on_ui_thread(self, member: str) -> None:
        if not super().done() and threading.current_thread() is kivy_utils.ui_thread():
            raise BlockingCallRefusedError('wait_on_ui_thread', member)
