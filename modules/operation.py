# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An operation the API started, as the caller holds it: one kind of handle for every one.

An ``Operation`` is a ``concurrent.futures.Future``, so ``result``,
``exception``, ``done``, ``add_done_callback``, ``concurrent.futures.wait``
and ``asyncio.wrap_future`` behave as a Future's do. What it adds:

- It is running from construction: the operation is under way by the time
  the caller holds it, so ``cancel()`` is False and only its producer
  settles it: a holder's ``set_result`` or ``set_exception`` is refused
  (``OperationSettleRefusedError``), and a producer settles through the base
  class, ``concurrent.futures.Future.set_result(op, value)``.
- Its done-callbacks run on one callback thread, one at a time in the order
  their operations settled -- never on the thread that settled it (the
  motion monitor, a lane, a run's thread), so a callback may itself wait on
  any operation without holding the producer that would settle it.
- ``stop()`` stops the operation's work while it is under way, and does
  nothing once it is done or when the operation has no stop.
- ``progress()`` answers where it has got, for an operation that keeps a
  progress record (a run's ``RunProgress``); None for one that does not.
- ``files`` is a run's second operation, done once its files are written;
  None on every other operation.
- A producer whose end is a delivery (a run's ``run_ended``) settles it
  through ``settling(op, ...)``: inside that delivery, the delivering thread's
  ``result()``, ``exception()`` and ``done()`` answer at once, while every
  other observer sees it done only once the delivery has returned.
- A ``result()`` or ``exception()`` on the UI thread for an operation not
  yet done is refused (``BlockingCallRefusedError``, ``'wait_on_ui_thread'``).
  A waiting form, which starts an operation and then waits on it, is refused
  the same way before it starts (``refuse_waiting_on_ui_thread``).
"""

import concurrent.futures
import contextlib
import queue
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from lvp_logger import logger
from modules import kivy_utils
from modules.api_surface import api, api_fields
from modules.exceptions import BlockingCallRefusedError, OperationSettleRefusedError

if TYPE_CHECKING:
    from modules.sequenced_capture_runner import RunProgress


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


def refuse_waiting_on_ui_thread(member: str) -> None:
    """Refuse a wait on the UI thread, before anything is started.

    A waiting form calls it first: refused only at its wait, the caller
    would be told "refused" while the operation it had already started
    went on.

    Raises:
        BlockingCallRefusedError: ``'wait_on_ui_thread'``, this is the UI
            thread.
    """
    if threading.current_thread() is kivy_utils.ui_thread():
        raise BlockingCallRefusedError('wait_on_ui_thread', member)


@api_fields('files')
class Operation(concurrent.futures.Future):
    """One operation in flight: its outcome, its done-callbacks, its stop and its progress.

    Attributes:
        files: A run's files, as their own operation: done once the run's
            images are on disk or given up on, its hyperstack build (when it
            has one) has ended and its ``files_written`` has been delivered,
            with the run's ``RunFiles``. None on every other operation.
    """

    files: 'Operation | None'

    def __init__(
        self,
        *,
        stop: Callable[[], object] | None = None,
        progress: 'Callable[[], RunProgress] | None' = None,
        files: 'Operation | None' = None,
    ) -> None:
        super().__init__()
        self.set_running_or_notify_cancel()
        self._stop = stop
        self._progress = progress
        self.files = files
        # Set only while a producer delivers this operation's end on one
        # thread: that thread and the outcome it is delivering.
        self._own: tuple[threading.Thread, object, BaseException | None] | None = None

    @api
    def stop(self) -> None:
        """Stop this operation's work; only asks, and the operation ends on its own.

        Does nothing once the operation is done, or once its work has ended
        and it is still telling its listeners, or when it has no stop (a
        still): a stop that arrives late never touches another operation.

        Raises:
            HardwareCommandRefusedError: a move's or a home's stop, which is
                ``stop_motion``'s: ``'not_connected'`` or ``'axis_absent'``, no
                motor controller; ``'home_not_stoppable'``, a home drives on a
                board that cannot stop one. A run's and a recording's stop
                raise nothing.
            MotorStopFailedError: a move's or a home's STOP was not taken by
                the board, so the stage may still be moving.
        """
        if self._stop is None or super().done():
            return
        self._stop()

    @api(reads=True)
    def progress(self) -> 'RunProgress | None':
        """Where this operation has got, read now; None for one that keeps no progress record.

        Raises:
            Nothing.
        """
        return None if self._progress is None else self._progress()

    @classmethod
    def following(
        cls,
        waiter: concurrent.futures.Future,
        *,
        stop: Callable[[], object] | None = None,
        first: Callable[[], object] | None = None,
    ) -> 'Operation':
        """The operation a lane task's *waiter* settles: done with its outcome, once *first()* has run.

        For a producer whose task holds something it releases when the task
        ends: *first* runs on the thread that settles *waiter*, before this
        operation is done, so whoever reads the outcome finds it let go. A
        done-callback on the operation itself would run that release on the
        callback thread, after a waiting caller was already told.
        """
        op = cls(stop=stop)

        def settle(settled: concurrent.futures.Future) -> None:
            try:
                if first is not None:
                    first()
            finally:
                exception = settled.exception()
                if exception is None:
                    concurrent.futures.Future.set_result(op, settled.result())
                else:
                    concurrent.futures.Future.set_exception(op, exception)

        waiter.add_done_callback(settle)
        return op

    @classmethod
    def all_of(cls, operations: 'tuple[Operation, ...]') -> 'Operation':
        """One operation for several started together: done once every one is, raising the first one's raise.

        ``stop()`` stops each one still under way. With none, it is done at once.
        """

        def stop() -> None:
            for op in operations:
                op.stop()

        joined = cls(stop=stop)
        if not operations:
            concurrent.futures.Future.set_result(joined, None)
            return joined
        lock = threading.Lock()
        left = [len(operations)]

        def one_ended(_op: concurrent.futures.Future) -> None:
            with lock:
                left[0] -= 1
                if left[0]:
                    return
            raised = next((op.exception() for op in operations if op.exception() is not None), None)
            if raised is None:
                concurrent.futures.Future.set_result(joined, None)
            else:
                concurrent.futures.Future.set_exception(joined, raised)

        # Counted on each settling thread, not the callback thread: a
        # done-callback that waits on the joined operation would otherwise
        # hold the one thread its count needs.
        for op in operations:
            concurrent.futures.Future.add_done_callback(op, one_ended)
        return joined

    def set_result(self, result: object) -> None:
        """Refused: only the operation's producer settles it, through the base class.

        Raises:
            OperationSettleRefusedError: ``'settled_by_its_producer'``, always.
        """
        raise OperationSettleRefusedError('Operation.set_result')

    def set_exception(self, exception: BaseException | None) -> None:
        """Refused: only the operation's producer settles it, through the base class.

        Raises:
            OperationSettleRefusedError: ``'settled_by_its_producer'``, always.
        """
        raise OperationSettleRefusedError('Operation.set_exception')

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

    def _delivering_here(self) -> tuple[object, BaseException | None] | None:
        own = self._own
        if own is None or own[0] is not threading.current_thread():
            return None
        return own[1], own[2]

    def _refuse_on_ui_thread(self, member: str) -> None:
        if not super().done():
            refuse_waiting_on_ui_thread(member)


@contextlib.contextmanager
def settling(
    op: Operation, *, result: object = None, exception: BaseException | None = None
) -> Iterator[None]:
    """Deliver *op*'s end on this thread, then settle it with *result* or *exception*.

    For a producer whose end is a delivery: what runs inside reads the
    outcome at once, every other observer once the delivery has returned.
    It settles even when the delivery raises. A function here, not a member
    of the operation: a holder entering it would forge the outcome and
    overwrite the producer's delivery.
    """
    op._own = (threading.current_thread(), result, exception)
    try:
        yield
    finally:
        op._own = None
        if exception is None:
            concurrent.futures.Future.set_result(op, result)
        else:
            concurrent.futures.Future.set_exception(op, exception)
