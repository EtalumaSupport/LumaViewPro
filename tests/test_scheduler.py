# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""LVP-A-13 — Scheduler protocol + reference implementations.

ThreadingTimerScheduler is the load-bearing one for REST + headless
soak; KivyClockScheduler is a thin pass-through that's exercised in
the LVP app every launch. These tests focus on the threading variant
because Kivy isn't easily exercisable in a unit-test context.

The interesting properties:
- schedule_interval fires the callback periodically
- unschedule stops a single scheduled interval cleanly
- shutdown stops every scheduled interval AND refuses new schedules
- schedule_interval after shutdown raises RuntimeError
- callbacks that raise don't kill the timer (re-arms after error)
- daemon threads — interpreter exit doesn't hang on a pending Timer
"""

import threading
import time

import pytest

from modules.scheduler import (
    Scheduler,
    ThreadingTimerScheduler,
    _CallablePairScheduler,
)


class TestThreadingTimerScheduler:
    """Behavioral contract for the production REST scheduler."""

    def test_schedule_interval_fires_callback(self):
        sched = ThreadingTimerScheduler()
        try:
            counter = {'n': 0}
            evt = threading.Event()

            def _cb():
                counter['n'] += 1
                if counter['n'] >= 3:
                    evt.set()

            sched.schedule_interval(_cb, 0.05)
            assert evt.wait(timeout=2.0), "callback should have fired ≥3 times within 2s"
            assert counter['n'] >= 3
        finally:
            sched.shutdown()

    def test_unschedule_cancels_a_single_interval(self):
        sched = ThreadingTimerScheduler()
        try:
            counter = {'a': 0, 'b': 0}

            def _a():
                counter['a'] += 1

            def _b():
                counter['b'] += 1

            handle_a = sched.schedule_interval(_a, 0.05)
            sched.schedule_interval(_b, 0.05)

            time.sleep(0.2)  # let both fire ~3-4 times
            a_before = counter['a']
            sched.unschedule(handle_a)
            time.sleep(0.3)  # b should keep firing
            assert counter['a'] == a_before, "_a should NOT fire after unschedule"
            assert counter['b'] > a_before, "_b should keep firing after _a is unscheduled"
        finally:
            sched.shutdown()

    def test_shutdown_cancels_all_intervals(self):
        sched = ThreadingTimerScheduler()
        counter = {'n': 0}

        def _cb():
            counter['n'] += 1

        sched.schedule_interval(_cb, 0.05)
        sched.schedule_interval(_cb, 0.05)
        time.sleep(0.15)
        before = counter['n']
        sched.shutdown()
        time.sleep(0.3)
        # Allow up to one in-flight tick that started before shutdown took effect.
        assert counter['n'] <= before + 2, \
            f"shutdown should stop all intervals; counter went from {before} to {counter['n']}"

    def test_schedule_after_shutdown_raises(self):
        sched = ThreadingTimerScheduler()
        sched.shutdown()
        with pytest.raises(RuntimeError):
            sched.schedule_interval(lambda: None, 0.1)

    def test_callback_exception_doesnt_kill_timer(self):
        """A callback that raises must not stop the scheduler — the
        next tick should still fire (matches Kivy Clock semantics)."""
        sched = ThreadingTimerScheduler()
        try:
            counter = {'n': 0}
            evt = threading.Event()

            def _cb():
                counter['n'] += 1
                if counter['n'] == 1:
                    raise RuntimeError("simulated callback failure")
                if counter['n'] >= 3:
                    evt.set()

            sched.schedule_interval(_cb, 0.05)
            assert evt.wait(timeout=2.0), \
                "scheduler should keep firing after a callback raises"
            assert counter['n'] >= 3
        finally:
            sched.shutdown()

    def test_unschedule_idempotent(self):
        sched = ThreadingTimerScheduler()
        try:
            handle = sched.schedule_interval(lambda: None, 0.05)
            sched.unschedule(handle)
            sched.unschedule(handle)  # should not raise
            sched.unschedule(None)    # None is allowed
        finally:
            sched.shutdown()

    def test_callback_taking_dt_arg_works(self):
        """Both no-arg callbacks and ``cb(dt)`` callbacks should work
        (Kivy convention is one-arg; Threading is no-arg)."""
        sched = ThreadingTimerScheduler()
        try:
            received = {'dt': None}
            evt = threading.Event()

            def _cb(dt):
                received['dt'] = dt
                evt.set()

            sched.schedule_interval(_cb, 0.05)
            assert evt.wait(timeout=1.0)
            assert received['dt'] is not None
        finally:
            sched.shutdown()

    def test_satisfies_scheduler_protocol(self):
        sched = ThreadingTimerScheduler()
        try:
            assert isinstance(sched, Scheduler)
        finally:
            sched.shutdown()


class TestCallablePairScheduler:
    """Backwards-compat adapter for the legacy two-callable form."""

    def test_wraps_and_invokes(self):
        scheduled = []
        unscheduled = []

        def _sched(cb, interval):
            scheduled.append((cb, interval))
            return ('handle', len(scheduled))

        def _unsched(handle):
            unscheduled.append(handle)

        pair = _CallablePairScheduler(_sched, _unsched)
        h = pair.schedule_interval(lambda: None, 1.5)
        assert len(scheduled) == 1
        assert scheduled[0][1] == 1.5
        pair.unschedule(h)
        assert unscheduled == [h]
        pair.shutdown()

    def test_satisfies_scheduler_protocol(self):
        pair = _CallablePairScheduler(lambda cb, i: None, lambda h: None)
        assert isinstance(pair, Scheduler)
