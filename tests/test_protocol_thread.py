# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for ProtocolThread (Stage B3).

Covers the public API contract:
  - run_protocol(callable, **kwargs) -> Future that resolves to the
    callable's return value or carries its exception.
  - abort() sets the aborted Event; in-flight callables polling
    self.aborted unwind cooperatively.
  - One run at a time -- second concurrent run_protocol() rejects via
    a Future resolving to RuntimeError ("Protocol already in progress").
  - is_running flips True during a run, False after the Future resolves.
  - start() / stop() are idempotent and process-exit-safe (daemon=True).
  - C6 collapse: PIW's abort_fn (bound to protocol_thread.abort) drives
    abort propagation; standalone AF abort does not touch protocol state.
  - Reentrancy guard: rapid back-to-back run_protocol() rejects the
    second when the first is still in flight.
  - Daemon-reap: a hung callable exits cleanly when stop() is called.
"""

from __future__ import annotations

import threading
import time

import pytest

from modules.protocol_thread import ProtocolThread


@pytest.fixture
def pt():
    thread = ProtocolThread()
    thread.start()
    yield thread
    thread.stop(timeout=2.0)


class TestLifecycle:
    def test_start_idempotent(self):
        thread = ProtocolThread()
        thread.start()
        first = thread._thread
        thread.start()
        assert thread._thread is first
        thread.stop(timeout=1.0)

    def test_stop_joins_within_timeout(self):
        thread = ProtocolThread()
        thread.start()
        thread.stop(timeout=2.0)
        # daemon=True means process exit reaps even if join times out,
        # but a clean stop should leave the thread reference cleared.
        assert thread._thread is None

    def test_stop_when_not_started(self):
        thread = ProtocolThread()
        # No start() before stop() -- should not raise.
        thread.stop(timeout=0.5)

    def test_daemon_flag(self):
        thread = ProtocolThread()
        thread.start()
        try:
            assert thread._thread.daemon is True
        finally:
            thread.stop(timeout=1.0)


class TestRunProtocol:
    def test_simple_callable_runs_to_completion(self, pt):
        called = threading.Event()

        def cb():
            called.set()
            return 'done'

        future = pt.run_protocol(cb)
        result = future.result(timeout=2.0)
        assert called.is_set()
        assert result == 'done'

    def test_callable_exception_propagates_through_future(self, pt):
        def cb():
            raise ValueError('boom')

        future = pt.run_protocol(cb)
        with pytest.raises(ValueError, match='boom'):
            future.result(timeout=2.0)

    def test_kwargs_forwarded_to_callable(self, pt):
        received = {}

        def cb(**kwargs):
            received.update(kwargs)
            return None

        pt.run_protocol(cb, alpha=1, beta='two').result(timeout=2.0)
        assert received == {'alpha': 1, 'beta': 'two'}


class TestReentrancyGuard:
    """The single-slot queue + state-lock pair must reject a second
    run while the first is in flight."""

    def test_second_run_rejected_while_first_in_flight(self, pt):
        release = threading.Event()

        def slow_cb():
            release.wait(timeout=2.0)
            return 'first'

        first = pt.run_protocol(slow_cb)
        # Wait for is_running to flip True; small spin (no API exposes
        # an entered-run event for the generic case).
        deadline = time.monotonic() + 1.0
        while not pt.is_running and time.monotonic() < deadline:
            time.sleep(0.005)
        assert pt.is_running, 'first run did not start'

        # Second run while first is in flight -- Future fails immediately.
        second = pt.run_protocol(lambda: 'second')
        with pytest.raises(RuntimeError, match='already in progress'):
            second.result(timeout=1.0)

        # Release the first; it completes normally.
        release.set()
        assert first.result(timeout=2.0) == 'first'

    def test_serial_runs_after_first_completes(self, pt):
        a = pt.run_protocol(lambda: 'a')
        assert a.result(timeout=2.0) == 'a'
        b = pt.run_protocol(lambda: 'b')
        assert b.result(timeout=2.0) == 'b'


class TestAbort:
    """abort() sets the aborted Event; cooperative callables exit."""

    def test_abort_sets_aborted_event(self, pt):
        release = threading.Event()
        entered = threading.Event()

        def cb():
            entered.set()
            release.wait(timeout=2.0)
            return None

        future = pt.run_protocol(cb)
        assert entered.wait(timeout=1.0), 'callable did not start'
        assert pt.aborted.is_set() is False

        pt.abort()
        assert pt.aborted.is_set() is True

        release.set()
        future.result(timeout=2.0)

    def test_cooperative_callable_unwinds_on_abort(self, pt):
        def cb():
            # Mirror scan loop: poll aborted Event and exit when set.
            for _ in range(100):
                if pt.aborted.is_set():
                    return 'aborted'
                time.sleep(0.01)
            return 'completed'

        future = pt.run_protocol(cb)
        time.sleep(0.05)
        pt.abort()
        assert future.result(timeout=2.0) == 'aborted'

    def test_abort_noop_when_idle(self, pt):
        # No run in flight -- should not raise.
        pt.abort()
        assert pt.aborted.is_set() is False


class TestAbortClearedBetweenRuns:
    """run_protocol() clears _aborted under the same state lock that
    publishes _current_future. Mirrors the AutofocusThread race fix at
    autofocus_thread.py:160-168."""

    def test_aborted_cleared_at_start_of_new_run(self, pt):
        # First run, then abort, then start a new run -- the new run
        # must see _aborted cleared.
        def quick():
            return None

        first = pt.run_protocol(quick)
        first.result(timeout=1.0)
        pt.abort()  # no-op (first completed), but clear path matters

        # Second run; aborted should be cleared at run_protocol entry.
        observations = []

        def cb():
            observations.append(pt.aborted.is_set())
            return None

        second = pt.run_protocol(cb)
        second.result(timeout=1.0)
        assert observations == [False], (
            f'Second run saw aborted={observations}; run_protocol() should clear _aborted at entry.'
        )


class TestIsRunningProperty:
    def test_idle_when_no_run(self, pt):
        assert pt.is_running is False

    def test_true_during_run_false_after(self, pt):
        release = threading.Event()
        entered = threading.Event()

        def cb():
            entered.set()
            release.wait(timeout=2.0)
            return None

        future = pt.run_protocol(cb)
        assert entered.wait(timeout=1.0)
        assert pt.is_running is True
        release.set()
        future.result(timeout=2.0)
        assert pt.is_running is False


class TestC6Collapse:
    """C6 absorption: _protocol_ended Event retired; abort signal lives
    on protocol_thread._aborted. Verify the abort callable is the
    intended PIW failure path and is independent of any AF-side state."""

    def test_abort_fn_callable_bound_to_thread_abort(self, pt):
        # PIW gets abort_fn = pt.abort at SCE.run() time. Verify the
        # bound callable is the same Event-setter as the property.
        abort_fn = pt.abort

        # Start a long-running run that polls aborted.
        def cb():
            while not pt.aborted.is_set():
                time.sleep(0.005)
            return 'aborted-via-abort_fn'

        future = pt.run_protocol(cb)
        time.sleep(0.02)
        # PIW would call self._abort_fn() on capture failure.
        abort_fn()
        assert future.result(timeout=2.0) == 'aborted-via-abort_fn'

    def test_aborted_property_returns_event_reference(self, pt):
        # Readers (protocol_step_runner, protocol_run_loop) borrow
        # the Event reference and call is_set() each tick.
        ev = pt.aborted
        assert isinstance(ev, threading.Event)

        # Start a run so abort() is not a no-op. The same Event the
        # readers borrowed must be the one abort() flips.
        entered = threading.Event()
        release = threading.Event()

        def cb():
            entered.set()
            release.wait(timeout=2.0)
            return None

        future = pt.run_protocol(cb)
        assert entered.wait(timeout=1.0)
        pt.abort()
        assert ev.is_set() is True
        release.set()
        future.result(timeout=2.0)


class TestDaemonReap:
    """A hung callable must not block stop() past its timeout. daemon=True
    ensures process exit reaps the thread."""

    def test_stop_returns_within_timeout_even_with_running_task(self):
        thread = ProtocolThread()
        thread.start()
        try:

            def cb():
                # Cooperative shutdown: poll aborted; this exits cleanly
                # when stop() sets _aborted.
                for _ in range(2000):
                    if thread.aborted.is_set():
                        return None
                    time.sleep(0.005)
                return None

            thread.run_protocol(cb)
            time.sleep(0.05)
            t0 = time.monotonic()
            thread.stop(timeout=2.0)
            elapsed = time.monotonic() - t0
            assert elapsed < 2.5, f'stop() blocked for {elapsed:.2f}s past the 2.0s timeout'
        finally:
            # Belt-and-suspenders; stop() above should have done it.
            if thread._thread is not None and thread._thread.is_alive():
                thread.stop(timeout=0.5)
