# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stage 3 tests: Lumascope fan polling thread + moving average.

The polling thread runs only while at least one subscriber has opted in
via ``enable_fan_polling(True, source=...)``. Refcounted by ``source``
string so multiple subscribers (the Microscope Settings tab, a perf-log
consumer, anything else) operate independently. Each tick calls
``get_fan_status()`` and fires fan listeners with a status dict enriched
by a 5-sample moving average of ``tach_rpm`` exposed as ``tach_rpm_avg``.

Firmware tach reports one fresh 1-second bucket per read; LVP's 5-sample
average is a 5-second trailing smoother. See ``_fan_monitor_loop`` in
``modules/lumascope_api.py``.
"""
import collections
import threading
import time
from unittest.mock import MagicMock

import pytest


def _make_scope(fan_status_seq=None, fan_status_static=None):
    """Build a minimal Lumascope sufficient for fan-polling tests.

    ``fan_status_seq`` is an iterable of dicts returned by successive
    ``get_fan_status`` calls. ``fan_status_static`` returns the same
    dict forever. Exactly one of the two should be provided.
    """
    from modules.lumascope_api import Lumascope
    s = Lumascope.__new__(Lumascope)

    m = MagicMock()
    if fan_status_seq is not None:
        seq = list(fan_status_seq)
        idx = {'i': 0}

        def _next():
            if idx['i'] < len(seq):
                v = seq[idx['i']]
                idx['i'] += 1
                return v
            return seq[-1]
        m.get_fan_status = MagicMock(side_effect=_next)
    else:
        m.get_fan_status = MagicMock(return_value=fan_status_static)
    s.motion = m

    # Listener infra (Stage 2)
    s._fan_listeners_lock = threading.Lock()
    s._fan_listeners = []

    # Polling infra (Stage 3)
    s._fan_monitor_lock = threading.Lock()
    s._fan_poll_sources = set()
    s._fan_poll_interval = 0.01  # 10 ms for test speed
    s._fan_monitor_stop = threading.Event()
    s._fan_monitor_wake = threading.Event()
    s._fan_monitor_thread = None
    s._fan_rpm_history = collections.deque(maxlen=5)
    return s


def _wait_for(predicate, timeout=2.0, interval=0.01):
    """Busy-wait on a predicate with a hard timeout. Returns True on
    success, False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# enable_fan_polling — refcount semantics
# ---------------------------------------------------------------------------

class TestEnableFanPollingRefcount:

    def test_enable_starts_thread_and_fires_listener(self):
        s = _make_scope(fan_status_static={'mode': 'HILO', 'state': 'HI',
                                           'fan_pct': None, 'tach_rpm': None})
        received = []
        s.add_fan_listener(lambda status: received.append(status))

        s.enable_fan_polling(True, source='test')

        assert _wait_for(lambda: len(received) >= 1, timeout=1.0)
        s.enable_fan_polling(False, source='test')
        s._stop_fan_monitor()

    def test_disable_last_source_quiesces_loop(self):
        s = _make_scope(fan_status_static={'mode': 'HILO', 'state': 'HI',
                                           'fan_pct': None, 'tach_rpm': None})
        fired = threading.Event()
        s.add_fan_listener(lambda _: fired.set())

        s.enable_fan_polling(True, source='test')
        assert fired.wait(1.0)

        s.enable_fan_polling(False, source='test')

        # After disable, sources should be empty; thread remains alive
        # but blocks on the wake event. A fresh listener should NOT fire
        # over the next poll interval.
        fired2 = threading.Event()
        s.add_fan_listener(lambda _: fired2.set())
        # Give the loop time to notice the empty source set and quiesce.
        time.sleep(0.1)
        assert not fired2.is_set()

        s._stop_fan_monitor()

    def test_refcount_multiple_sources(self):
        s = _make_scope(fan_status_static={'mode': 'HILO', 'state': 'HI',
                                           'fan_pct': None, 'tach_rpm': None})
        s.enable_fan_polling(True, source='ui_tab')
        s.enable_fan_polling(True, source='perf_log')
        assert s._fan_poll_sources == {'ui_tab', 'perf_log'}

        # Disabling one subscriber leaves the other active.
        s.enable_fan_polling(False, source='ui_tab')
        assert s._fan_poll_sources == {'perf_log'}

        # Disabling both quiesces.
        s.enable_fan_polling(False, source='perf_log')
        assert s._fan_poll_sources == set()

        s._stop_fan_monitor()

    def test_duplicate_enable_same_source_is_idempotent(self):
        s = _make_scope(fan_status_static={'mode': 'HILO', 'state': 'HI',
                                           'fan_pct': None, 'tach_rpm': None})
        s.enable_fan_polling(True, source='ui_tab')
        s.enable_fan_polling(True, source='ui_tab')
        assert s._fan_poll_sources == {'ui_tab'}
        s._stop_fan_monitor()

    def test_disable_unknown_source_is_noop(self):
        s = _make_scope(fan_status_static={'mode': 'HILO', 'state': 'HI',
                                           'fan_pct': None, 'tach_rpm': None})
        # Should not raise; should not start a thread.
        s.enable_fan_polling(False, source='never_enabled')
        assert s._fan_poll_sources == set()
        assert s._fan_monitor_thread is None


# ---------------------------------------------------------------------------
# Moving-average enrichment
# ---------------------------------------------------------------------------

class TestMovingAverage:

    def test_avg_equals_single_sample_on_first_tick(self):
        status = {'mode': 'PWM', 'state': None, 'fan_pct': 50, 'tach_rpm': 2000}
        s = _make_scope(fan_status_static=status)
        received = []
        s.add_fan_listener(lambda st: received.append(st))

        s.enable_fan_polling(True, source='test')
        assert _wait_for(lambda: len(received) >= 1, timeout=1.0)
        s.enable_fan_polling(False, source='test')
        s._stop_fan_monitor()

        assert received[0]['tach_rpm'] == 2000
        assert received[0]['tach_rpm_avg'] == 2000.0

    def test_avg_over_5_samples(self):
        # Five distinct RPM values; average should match arithmetic mean.
        rpms = [2000, 2100, 2050, 1950, 2200]
        seq = [{'mode': 'PWM', 'state': None, 'fan_pct': 50, 'tach_rpm': r}
               for r in rpms]
        s = _make_scope(fan_status_seq=seq)
        received = []
        s.add_fan_listener(lambda st: received.append(st))

        s.enable_fan_polling(True, source='test')
        assert _wait_for(lambda: len(received) >= 5, timeout=2.0)
        s.enable_fan_polling(False, source='test')
        s._stop_fan_monitor()

        # The 5th tick should average all 5 samples.
        last_avg = received[4]['tach_rpm_avg']
        assert abs(last_avg - sum(rpms) / 5) < 0.001

    def test_history_clears_on_disable(self):
        status = {'mode': 'PWM', 'state': None, 'fan_pct': 50, 'tach_rpm': 2000}
        s = _make_scope(fan_status_static=status)
        s.add_fan_listener(lambda _: None)

        s.enable_fan_polling(True, source='test')
        assert _wait_for(lambda: len(s._fan_rpm_history) >= 3, timeout=1.5)
        s.enable_fan_polling(False, source='test')
        # Loop quiesces + clears history.
        assert _wait_for(lambda: len(s._fan_rpm_history) == 0, timeout=1.0)
        s._stop_fan_monitor()

    def test_negative_tach_rpm_skipped_from_history(self):
        # Firmware returns -1 on tach timer crash; must not pollute the avg.
        seq = [
            {'mode': 'PWM', 'state': None, 'fan_pct': 50, 'tach_rpm': 2000},
            {'mode': 'PWM', 'state': None, 'fan_pct': 50, 'tach_rpm': -1},
            {'mode': 'PWM', 'state': None, 'fan_pct': 50, 'tach_rpm': 2100},
        ]
        s = _make_scope(fan_status_seq=seq)
        received = []
        s.add_fan_listener(lambda st: received.append(st))

        s.enable_fan_polling(True, source='test')
        assert _wait_for(lambda: len(received) >= 3, timeout=2.0)
        s.enable_fan_polling(False, source='test')
        s._stop_fan_monitor()

        # Third tick should average [2000, 2100] only — -1 excluded.
        assert abs(received[2]['tach_rpm_avg'] - 2050.0) < 0.001


# ---------------------------------------------------------------------------
# set_fan_poll_interval
# ---------------------------------------------------------------------------

class TestSetPollInterval:

    def test_clamps_to_minimum(self):
        s = _make_scope(fan_status_static={'mode': 'HILO', 'state': 'HI',
                                           'fan_pct': None, 'tach_rpm': None})
        s.set_fan_poll_interval(0.0)
        assert s._fan_poll_interval == 0.1

    def test_accepts_valid(self):
        s = _make_scope(fan_status_static={'mode': 'HILO', 'state': 'HI',
                                           'fan_pct': None, 'tach_rpm': None})
        s.set_fan_poll_interval(5.0)
        assert s._fan_poll_interval == 5.0


# ---------------------------------------------------------------------------
# Robustness: poll exceptions don't kill the thread
# ---------------------------------------------------------------------------

class TestRobustness:

    def test_exception_in_get_fan_status_does_not_kill_thread(self):
        from modules.lumascope_api import Lumascope
        s = Lumascope.__new__(Lumascope)

        calls = {'n': 0}

        def _flaky():
            calls['n'] += 1
            if calls['n'] == 1:
                raise RuntimeError('simulated firmware blip')
            return {'mode': 'HILO', 'state': 'HI',
                    'fan_pct': None, 'tach_rpm': None}

        m = MagicMock()
        m.get_fan_status = MagicMock(side_effect=_flaky)
        s.motion = m
        s._fan_listeners_lock = threading.Lock()
        s._fan_listeners = []
        s._fan_monitor_lock = threading.Lock()
        s._fan_poll_sources = set()
        s._fan_poll_interval = 0.01
        s._fan_monitor_stop = threading.Event()
        s._fan_monitor_wake = threading.Event()
        s._fan_monitor_thread = None
        s._fan_rpm_history = collections.deque(maxlen=5)

        received = []
        s.add_fan_listener(lambda st: received.append(st))

        s.enable_fan_polling(True, source='test')
        # Exception on first poll, success on subsequent ones.
        assert _wait_for(lambda: len(received) >= 2, timeout=1.5)
        s.enable_fan_polling(False, source='test')
        s._stop_fan_monitor()

        assert calls['n'] >= 2


# ---------------------------------------------------------------------------
# Diagnostic-scope compatibility (no fan infra)
# ---------------------------------------------------------------------------

class TestDiagnosticScopeCompat:

    def test_enable_on_diagnostic_scope_is_noop(self):
        from modules.lumascope_api import Lumascope
        s = Lumascope.__new__(Lumascope)
        # Deliberately omit the Stage 3 fan-monitor attrs.
        # enable_fan_polling must not raise.
        s.enable_fan_polling(True, source='test')
        s.set_fan_poll_interval(2.0)
        s._stop_fan_monitor()
