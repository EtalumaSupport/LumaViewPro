# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
#
# GUI-agnostic: no Kivy imports.
"""ProtocolThread -- dedicated long-lived thread that runs a protocol scan loop.

Replaces the queue-of-1 SequentialIOExecutor pattern (`protocol_executor`)
that previously hosted protocol-run execution. The thread owns a generic
"run this callable" request queue; the caller (typically SCE.run()) submits
the protocol run_loop callable and receives a Future that resolves when the
run completes (success, abort, or failure).

Public API:
  start()                         -- spawn worker thread (idempotent).
  stop(timeout)                   -- signal stop, abort current run, join.
  run_protocol(callable, **kw)    -- enqueue a run; returns Future. Future
                                     resolves to whatever the callable
                                     returned, or carries its exception.
  abort()                         -- signal current run to unwind.
  is_running                      -- True if a run is in flight.
  aborted                         -- the abort threading.Event (read-only
                                     reference; readers like
                                     protocol_step_runner consult its
                                     is_set() directly each tick).

Concurrency contract: one protocol run at a time, admitted by the runner
that owns this thread, never here. A run the runner admits while the
previous run's loop is still returning -- its run already ended, its last
cleanup pass not yet done -- waits in the request slot and starts when that
loop returns. Refusing it here instead failed a run the runner had already
committed, and the failed start could not unwind while that last pass held
the runner's cleanup lock.
"""

import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future

logger = logging.getLogger('LVP.modules.protocol_thread')


# Sentinel posted to the request queue by stop() to wake the worker
# from queue.get() even when no protocol request is pending.
_SHUTDOWN_SENTINEL = object()


class ProtocolThread:
    """Owns the protocol-run execution thread.

    The thread idles on a request queue between runs; when run_protocol()
    is called it picks up the request, drives the supplied callable to
    completion, and resolves the request's Future. The scan-loop
    callable itself owns any UI dispatch it needs.
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._aborted = threading.Event()

        # One run waiting at most: the runner admits a run only once the
        # last one has ended, so the most this slot holds is the next run
        # while the last one's loop returns.
        self._request_queue: queue.Queue = queue.Queue(maxsize=1)

        self._state_lock = threading.Lock()
        self._current_future: Future | None = None

    # ---- lifecycle ----

    def start(self) -> None:
        """Spawn the worker thread. Idempotent: returns early if already
        running."""
        if self._thread is not None and self._thread.is_alive():
            logger.debug('protocol_thread already running')
            return
        self._stop_event.clear()
        self._aborted.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name='protocol_thread',
            daemon=True,
        )
        self._thread.start()
        logger.info('protocol_thread started')

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker to stop and join with bound timeout.

        Abort any in-flight run first; then wake the worker (via the
        shutdown sentinel) and join. If join times out the daemon=True
        flag means process exit will reap the thread.
        """
        self._stop_event.set()
        self._aborted.set()
        try:
            self._request_queue.put_nowait(_SHUTDOWN_SENTINEL)
        except queue.Full:
            # An in-flight request occupies the slot. The worker will
            # see _stop_event after it finishes the current request.
            pass
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning(
                    f'protocol_thread did not join within {timeout}s; '
                    f'daemon=True so process exit will reap it'
                )
        self._thread = None

    # ---- public API ----

    def run_protocol(
        self,
        run_loop_callable: Callable,
        **kwargs,
    ) -> Future:
        """Enqueue a protocol run. Returns a Future that resolves to the
        callable's return value on success, or carries the exception on
        failure or abort.

        Args:
            run_loop_callable: the function the thread should call; expected
                to be a long-running scan loop that polls self.aborted
                cooperatively. The callable receives no positional args
                from this method; any state it needs must already be set
                up on the object that owns it.
            **kwargs: forwarded verbatim to the callable.

        Returns:
            Future. Inspect with .result(timeout=...) to block; ignore to
            fire-and-forget.
        """
        future: Future = Future()
        with self._state_lock:
            previous = self._current_future
            self._current_future = future
            # Clear _aborted under the same lock that publishes
            # _current_future. Same-lock pairing makes the new-Future-with-
            # cleared-aborted publication atomic w.r.t. abort(); mirrors
            # AutofocusThread's race fix (autofocus_thread.py:160-168).
            # Cleared BEFORE the put: the worker may start the run the
            # moment it is queued, and must not find the last run's abort.
            self._aborted.clear()
            try:
                self._request_queue.put_nowait((run_loop_callable, kwargs, future))
            except queue.Full:
                # The slot is taken -- by stop()'s shutdown sentinel, or by a
                # second run the runner's own gate should have refused.
                self._current_future = previous
                future.set_exception(RuntimeError('Protocol request queue full'))
        return future

    def abort(self) -> None:
        """Signal the in-flight protocol run to unwind. The scan loop
        consults the abort event each tick and exits cooperatively. The
        Future from run_protocol() resolves when the callable returns.

        No-op if no run is in flight. Does NOT fire notifications --
        abort is a state transition; the caller owns user-facing
        notification (mirrors autofocus_thread.abort()).
        """
        if not self.is_running:
            return
        logger.info('protocol_thread abort requested')
        self._aborted.set()

    # ---- inspection ----

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            fut = self._current_future
        return fut is not None and not fut.done()

    @property
    def current_future(self) -> Future | None:
        with self._state_lock:
            return self._current_future

    @property
    def aborted(self) -> threading.Event:
        """Read-only reference to the abort event. Scan-loop readers
        consult this directly each tick; exposed so callers can compose
        their own abort propagation (e.g. PIW gets self.aborted via SCE)."""
        return self._aborted

    # ---- loop ----

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                req = self._request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if req is _SHUTDOWN_SENTINEL:
                return

            run_loop_callable, kwargs, future = req
            try:
                result = run_loop_callable(**kwargs)
                future.set_result(result)
            except Exception as ex:
                logger.exception(f'protocol run raised: {type(ex).__name__}: {ex}')
                future.set_exception(ex)
            finally:
                with self._state_lock:
                    # Only if it is still this run's: the next run may
                    # already wait in the slot, and its Future is current.
                    if self._current_future is future:
                        self._current_future = None

        logger.info('protocol_thread exiting')
