# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
#
# GUI-agnostic: no Kivy imports.
"""AutofocusThread -- dedicated long-lived thread that runs AF requests.

Replaces the queue-of-1 SequentialIOExecutor pattern that previously
hosted AF execution. The thread owns the per-iteration loop; calls
AutofocusRunner.run(**args, abort_event=...) to completion per
request; sets the request's Future when AF finishes (success, abort,
or failure).

Public API:
  start()                       -- spawn worker thread
  stop(timeout)                 -- signal stop, join with bound timeout
  run_autofocus(run_trigger_source=..., **kwargs)
                                -- enqueue an AF request; returns a Future.
                                   Future resolves to best_focus_position
                                   (float | None) on success, or carries
                                   the exception on failure / abort.
                                   AutofocusAborted is raised through the
                                   Future on caller-requested abort.
  abort()                       -- signal current run to unwind
  in_flight_sweep               -- the sweep running now (its Future and
                                   the run that dispatched it), or None
  is_running                    -- True if an AF run is in flight
  current_future                -- the recorded Future, or None

Concurrency contract: one AF at a time. A second run_autofocus()
invocation while the first is in flight returns a Future that resolves
immediately to a RuntimeError ("Autofocus already in progress"); the
in-flight run is not affected.
"""

import logging
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from modules.activity_claim import acting, current_taking
from modules.exceptions import AutofocusAborted

logger = logging.getLogger('LVP.modules.autofocus_thread')


# Sentinel posted to the request queue by stop() to wake the worker
# from queue.get() even when no AF request is pending.
_SHUTDOWN_SENTINEL = object()


@dataclass(frozen=True)
class AutofocusSweep:
    """A sweep in flight: its Future and the run that dispatched it.

    One object, written and cleared in one act, so "a sweep is running"
    and "whose sweep" cannot be read at two different instants and
    disagree -- the reader that sees the sweep sees the run that owns
    it. Naming only the sweep would report an autofocus when a whole
    protocol is the thing the user has to stop.
    """

    future: Future
    run_trigger_source: str


class AutofocusThread:
    """Owns the AF execution thread.

    The thread idles on a request queue between AF runs; when
    run_autofocus() is called it picks up the request, drives
    AutofocusRunner.run(...) to completion, and resolves the
    request's Future.

    Args:
        afe: an AutofocusRunner instance. The thread calls
            afe.run(**kwargs, abort_event=self._aborted) per request.
    """

    def __init__(
        self,
        *,
        afe: Any,
    ):
        self._afe = afe

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._aborted = threading.Event()

        # One outstanding AF request at a time; second arrivals while a
        # run is in flight fail-fast via run_autofocus(). The queue is
        # bounded at 1 to make that contract explicit.
        self._request_queue: queue.Queue = queue.Queue(maxsize=1)

        self._state_lock = threading.Lock()
        self._current_sweep: AutofocusSweep | None = None

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

    def run_autofocus(self, *, run_trigger_source: str, **kwargs) -> Future:
        """Enqueue an AF request. Returns a Future that resolves to the
        best focus position (float | None) on success, or carries the
        exception on failure or abort.

        Args:
            run_trigger_source: the trigger of the run dispatching this
                sweep. Required, because every dispatch has one and a
                sweep whose owner is unknown is precisely what the
                refusals downstream cannot describe. Recorded with the
                Future AND forwarded to the runner, which gates its own
                failure popups on it.
            **kwargs: forwarded verbatim to AutofocusRunner.run().

        Returns:
            Future[float | None]. Inspect with .result(timeout=...) to
            block; ignore to fire-and-forget. Caller-requested abort
            surfaces as AutofocusAborted via the Future.
        """
        request_kwargs = {**kwargs, 'run_trigger_source': run_trigger_source}
        future: Future = Future()
        with self._state_lock:
            in_flight = self._current_sweep
            if in_flight is not None and not in_flight.future.done():
                future.set_exception(
                    RuntimeError(
                        'Autofocus already in progress, dispatched by the '
                        f'{in_flight.run_trigger_source} run'
                    )
                )
                return future
            self._current_sweep = AutofocusSweep(
                future=future,
                run_trigger_source=run_trigger_source,
            )
            # Clear _aborted under the same lock that publishes the
            # sweep. A concurrent abort() reads is_running under this
            # lock; with the clear() outside the lock there was a
            # one-instruction window where abort() could see is_running
            # True (just-published Future), set _aborted, and then have
            # its bit cleared here -- silently dropping the abort.
            # Same-lock pairing makes the new-sweep-with-cleared-aborted
            # publication atomic w.r.t. abort().
            self._aborted.clear()
        # The sweep's moves, LED and camera writes are its caller's -- the run
        # that dispatched it -- so this thread acts under the caller's taking.
        try:
            self._request_queue.put_nowait((request_kwargs, future, current_taking()))
        except queue.Full:
            # Queue full despite the state lock guard above; should not
            # happen but degrade gracefully by failing the new Future.
            with self._state_lock:
                self._current_sweep = None
            future.set_exception(RuntimeError('Autofocus request queue full'))
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
    def in_flight_sweep(self) -> AutofocusSweep | None:
        """The sweep running NOW and the run that dispatched it, or None.

        The single read a gate needs: liveness and owner come from one
        snapshot, so a caller cannot pair "a sweep is running" with the
        name of a run that has since been replaced.
        """
        with self._state_lock:
            sweep = self._current_sweep
        if sweep is None or sweep.future.done():
            return None
        return sweep

    @property
    def is_running(self) -> bool:
        return self.in_flight_sweep is not None

    @property
    def current_future(self) -> Future | None:
        """The recorded Future, in flight or just finished and not yet
        cleared -- what a teardown waits on, unlike in_flight_sweep."""
        with self._state_lock:
            sweep = self._current_sweep
        return sweep.future if sweep is not None else None

    @property
    def aborted(self) -> threading.Event:
        """Read-only reference to the abort event. AFE consults this
        directly each iteration; exposed so callers can compose their
        own abort propagation. Note that protocol_thread.abort() does
        NOT chain to here -- it sets its own event only, and the
        autofocus is unwound by the run cleanup that the aborted scan
        loop falls into."""
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

            kwargs, future, taking = req
            try:
                with acting(taking):
                    result = self._afe.run(**kwargs, abort_event=self._aborted)
                future.set_result(result)
            except AutofocusAborted as ex:
                logger.info(f'autofocus run aborted: {ex}')
                future.set_exception(ex)
            except Exception as ex:
                logger.exception(f'autofocus run raised: {type(ex).__name__}: {ex}')
                future.set_exception(ex)
            finally:
                with self._state_lock:
                    # Only clear the sweep if it still points to this
                    # run; a second concurrent call could not have
                    # replaced it (run_autofocus rejects while we're
                    # running), but the explicit identity check costs
                    # nothing and survives future refactors.
                    if self._current_sweep is not None and self._current_sweep.future is future:
                        self._current_sweep = None

        logger.info('autofocus_thread exiting')
