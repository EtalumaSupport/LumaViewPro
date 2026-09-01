# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Scheduler protocol + reference implementation.

ThreadingTimerScheduler is the one scheduler every host uses (the API
layer deliberately carries no UI-clock scheduler).

The interesting properties:
- schedule_interval fires the callback periodically
- unschedule stops a single scheduled interval cleanly
- shutdown stops every scheduled interval AND refuses new schedules
- schedule_interval after shutdown raises RuntimeError
- callbacks that raise don't kill the timer (re-arms after error)
- daemon threads -- interpreter exit doesn't hang on a pending Timer
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
            assert evt.wait(timeout=2.0), 'callback should have fired >=3 times within 2s'
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
            # Wait for b to overtake instead of sleeping a fixed 300 ms: on a
            # loaded runner the 50 ms interval can slip enough that b has not
            # passed a's frozen count yet, which fails for scheduling reasons
            # rather than because unschedule() misbehaved.
            #
            # Still hold the floor at 300 ms. Polling alone can exit as soon
            # as b overtakes, which would SHRINK the window in which a stray
            # _a callback -- the thing the next assertion looks for -- has a
            # chance to fire. Keeping the original floor means this change
            # makes the b assertion robust without weakening the a assertion.
            started = time.monotonic()
            deadline = started + 5.0
            while time.monotonic() < deadline and counter['b'] <= a_before:
                time.sleep(0.01)
            remaining_floor = 0.3 - (time.monotonic() - started)
            if remaining_floor > 0:
                time.sleep(remaining_floor)
            assert counter['a'] == a_before, '_a should NOT fire after unschedule'
            assert counter['b'] > a_before, '_b should keep firing after _a is unscheduled'
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
        assert before > 0, 'intervals never fired, so this proves nothing about shutdown'

        sched.shutdown()
        # Asserted as "nothing fires once shutdown has SETTLED" rather than
        # "at most N more ticks", which is the same invariant stated in a way
        # that does not depend on runner speed. The old form allowed a fixed
        # +2 for in-flight callbacks, so a loaded runner that landed three
        # failed for scheduling reasons; this form tolerates any number of
        # in-flight ticks and is stricter afterwards -- exact equality, where
        # the old assertion permitted two extra forever.
        time.sleep(0.5)
        after_settle = counter['n']
        time.sleep(0.3)
        assert counter['n'] == after_settle, (
            f'a callback fired after shutdown settled: {after_settle} -> '
            f'{counter["n"]}; intervals were not cancelled'
        )

    def test_schedule_after_shutdown_raises(self):
        sched = ThreadingTimerScheduler()
        sched.shutdown()
        with pytest.raises(RuntimeError):
            sched.schedule_interval(lambda: None, 0.1)

    def test_callback_exception_doesnt_kill_timer(self):
        """A callback that raises must not stop the scheduler -- the
        next tick should still fire (matches Kivy Clock semantics)."""
        sched = ThreadingTimerScheduler()
        try:
            counter = {'n': 0}
            evt = threading.Event()

            def _cb():
                counter['n'] += 1
                if counter['n'] == 1:
                    raise RuntimeError('simulated callback failure')
                if counter['n'] >= 3:
                    evt.set()

            sched.schedule_interval(_cb, 0.05)
            assert evt.wait(timeout=2.0), 'scheduler should keep firing after a callback raises'
            assert counter['n'] >= 3
        finally:
            sched.shutdown()

    def test_unschedule_idempotent(self):
        sched = ThreadingTimerScheduler()
        try:
            handle = sched.schedule_interval(lambda: None, 0.05)
            sched.unschedule(handle)
            sched.unschedule(handle)  # should not raise
            sched.unschedule(None)  # None is allowed
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
