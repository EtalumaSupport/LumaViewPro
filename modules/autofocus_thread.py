# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
#
# GUI-agnostic: no Kivy imports. The GUI host injects ui_dispatcher
# (Clock.schedule_once in Kivy; direct invocation in headless tests).
"""AutofocusThread -- dedicated long-lived thread that runs AF requests.

Replaces the queue-of-1 SequentialIOExecutor pattern that previously
hosted AF execution. The thread owns the per-iteration loop; calls
AutofocusExecutor.run(**args, abort_event=...) to completion per
request; sets the request's Future when AF finishes (success, abort,
or failure).

Public API:
  start()                       -- spawn worker thread
  stop(timeout)                 -- signal stop, join with bound timeout
  run_autofocus(**kwargs)       -- enqueue an AF request; returns a Future.
                                   Future resolves to best_focus_position
                                   (float | None) on success, or carries
                                   the exception on failure / abort.
                                   AutofocusAborted is raised through the
                                   Future on caller-requested abort.
  abort()                       -- signal current run to unwind
  is_running                    -- True if an AF run is in flight
  current_future                -- the in-flight Future, or None

Concurrency contract: one AF at a time. A second run_autofocus()
invocation while the first is in flight returns a Future that resolves
immediately to a RuntimeError ("Autofocus already in progress"); the
in-flight run is not affected.
"""
import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from modules.exceptions import AutofocusAborted

logger = logging.getLogger('LVP.modules.autofocus_thread')


# Sentinel posted to the request queue by stop() to wake the worker
# from queue.get() even when no AF request is pending.
_SHUTDOWN_SENTINEL = object()


class AutofocusThread:
    """Owns the AF execution thread.

    The thread idles on a request queue between AF runs; when
    run_autofocus() is called it picks up the request, drives
    AutofocusExecutor.run(...) to completion, and resolves the
    request's Future.

    Args:
        afe: an AutofocusExecutor instance. The thread calls
            afe.run(**kwargs, abort_event=self._aborted) per request.
        ui_dispatcher: optional callable(callback, delay_seconds) for
            posting work back to the UI thread. Defaults to a direct
            inline call (headless mode).
    """

    def __init__(
        self,
        *,
        afe: Any,
        ui_dispatcher: Callable[[Callable, float], Any] | None = None,
    ):
        self._afe = afe
        self._ui_dispatcher = ui_dispatcher or self._direct_dispatch

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._aborted = threading.Event()

        # One outstanding AF request at a time; second arrivals while a
        # run is in flight fail-fast via run_autofocus(). The queue is
        # bounded at 1 to make that contract explicit.
        self._request_queue: queue.Queue = queue.Queue(maxsize=1)

        self._state_lock = threading.Lock()
        self._current_future: Future | None = None

    @staticmethod
    def _direct_dispatch(fn: Callable, delay: float) -> None:
        """Headless fallback for ui_dispatcher. Runs inline; delay is
        ignored (tests tick manually)."""
        try:
            fn(0)
        except Exception:
            logger.exception('direct_dispatch callback failed')

    # ---- lifecycle ----

    def start(self) -> None:
        """Spawn the worker thread. Idempotent: returns early if already
        running."""
        if self._thread is not None and self._thread.is_alive():
            logger.debug('autofocus_thread already running')
            return
        self._stop_event.clear()
        self._aborted.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name='autofocus_thread',
            daemon=True,
        )
        self._thread.start()
        logger.info('autofocus_thread started')

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker to stop and join with bound timeout.

        Abort any in-flight AF first; then wake the worker (via the
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
                    f'autofocus_thread did not join within {timeout}s; '
                    f'daemon=True so process exit will reap it'
                )
        self._thread = None

    # ---- public API ----

    def run_autofocus(self, **kwargs) -> Future:
        """Enqueue an AF request. Returns a Future that resolves to the
        best focus position (float | None) on success, or carries the
        exception on failure or abort.

        Args:
            **kwargs: forwarded verbatim to AutofocusExecutor.run().

        Returns:
            Future[float | None]. Inspect with .result(timeout=...) to
            block; ignore to fire-and-forget. Caller-requested abort
            surfaces as AutofocusAborted via the Future.
        """
        future: Future = Future()
        with self._state_lock:
            if self._current_future is not None and not self._current_future.done():
                future.set_exception(
                    RuntimeError('Autofocus already in progress')
                )
                return future
            self._current_future = future
        self._aborted.clear()
        try:
            self._request_queue.put_nowait((kwargs, future))
        except queue.Full:
            # Queue full despite the state lock guard above; should not
            # happen but degrade gracefully by failing the new Future.
            with self._state_lock:
                self._current_future = None
            future.set_exception(
                RuntimeError('Autofocus request queue full')
            )
        return future

    def abort(self) -> None:
        """Signal the in-flight AF run to unwind. AFE consults the
        abort_event each iteration and raises AutofocusAborted. The
        Future from run_autofocus() resolves with that exception.

        No-op if no run is in flight.
        """
        if not self.is_running:
            return
        logger.info('autofocus_thread abort requested')
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
        """Read-only reference to the abort event. AFE consults this
        directly each iteration; exposed so callers can compose their
        own abort propagation (e.g. protocol_thread.abort() chains
        autofocus_thread.abort())."""
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

            kwargs, future = req
            try:
                result = self._afe.run(**kwargs, abort_event=self._aborted)
                future.set_result(result)
            except AutofocusAborted as ex:
                logger.info(f'autofocus run aborted: {ex}')
                future.set_exception(ex)
            except Exception as ex:
                logger.exception(
                    f'autofocus run raised: {type(ex).__name__}: {ex}'
                )
                future.set_exception(ex)
            finally:
                with self._state_lock:
                    # Only clear current_future if it still points to
                    # this run; a second concurrent call could not have
                    # replaced it (run_autofocus rejects while we're
                    # running), but the explicit identity check costs
                    # nothing and survives future refactors.
                    if self._current_future is future:
                        self._current_future = None

        logger.info('autofocus_thread exiting')
