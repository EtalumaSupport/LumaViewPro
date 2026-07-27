# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression for issue #239: the second timelapse image arrived too soon
after the first.

Bug shape: the pacing anchor had two writers bound to two different
events. Scans 2..N re-anchor at gate-open, so their spacing is exactly
the period; scan 1's anchor was run() entry -- BEFORE run setup, the
initial go_to_step motion, and step-0 autofocus -- so interval 1 came up
short by that lead-in (toward zero at short periods).

Contract under test: the anchor is ESTABLISHED at the run's first
acquisition (scan 1), then MAINTAINED at gate-open for scans 2..N, whose
cadence must not change.
"""

import datetime
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from modules.protocol_run_loop import ProtocolRunLoop
from modules.protocol_state_machine import ProtocolState


class FakeClock:
    """Deterministic stand-in for time.monotonic/time.sleep."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def _make_parent(clock, n_scans, period_s, first_scan_lead_s=30.0, later_scan_lead_s=5.0):
    """Run-loop parent stub whose scan_loop consumes fake time like a real
    scan: a long lead-in on scan 1 (motion + AF), a short constant lead-in
    on pre-positioned scans 2..N, and a capture that records its time and
    the first-acquisition timestamp exactly as the step runner does."""
    p = SimpleNamespace()
    p._n_scans = n_scans
    p._scan_count = 0
    p.remaining_scans = lambda: p._n_scans - p._scan_count
    p._run_in_progress_event = threading.Event()
    p._run_in_progress_event.set()
    p._aborted = threading.Event()
    p._state = ProtocolState.RUNNING
    p._scope = mock.MagicMock()
    p._protocol = mock.MagicMock()
    p._protocol.period.return_value = datetime.timedelta(seconds=period_s)
    p._parent_dir = None
    p._start_t = clock.t  # run() entry semantics (the fallback anchor)
    p._curr_step = 0
    p._af_future = None
    p._scan_first_capture_t = None
    p.capture_times = []

    def _reset_scan_state():
        p._curr_step = 0
        p._af_future = None
        p._scan_first_capture_t = None

    def _advance_scan_count():
        p._scan_count += 1
        return p._scan_count

    def _scan_loop():
        lead = first_scan_lead_s if not p.capture_times else later_scan_lead_s
        clock.sleep(lead)
        p.capture_times.append(clock.t)
        if p._scan_first_capture_t is None:
            p._scan_first_capture_t = clock.t
        clock.sleep(10.0)  # remaining steps + save

    p._reset_scan_state = _reset_scan_state
    p.advance_scan_count = _advance_scan_count
    p._step_executor = mock.MagicMock()
    p._step_executor.scan_loop.side_effect = _scan_loop
    p._scan_in_progress = mock.MagicMock()
    p._set_state = mock.MagicMock()
    p._cleanup = mock.MagicMock()
    p._protocol_state_lock = threading.Lock()
    p._auto_gain_armed_step = -1
    p.LOGGER_NAME = 'TEST'
    p._callbacks = SimpleNamespace(
        protocol_iterate_pre=None,
        run_scan_pre=None,
        scan_iterate_post=None,
    )
    return p


def _run(clock, parent):
    loop = ProtocolRunLoop(parent)
    with (
        mock.patch('modules.protocol_run_loop.time.monotonic', clock.monotonic),
        mock.patch('modules.protocol_run_loop.time.sleep', clock.sleep),
        mock.patch('modules.protocol_run_loop._schedule_ui', lambda cb, *a, **k: None),
    ):
        loop._run_loop_inner()


def test_first_interval_is_never_shorter_than_period():
    """Period 100 s, scan-1 lead-in 30 s: pre-fix the anchor sat at run()
    entry, so capture 2 landed 100 s after t=0 -- interval 75 s < period.
    Post-fix the anchor is capture 1, so interval 1 >= period."""
    clock = FakeClock()
    p = _make_parent(clock, n_scans=2, period_s=100.0)
    _run(clock, p)

    assert len(p.capture_times) == 2
    interval_1 = p.capture_times[1] - p.capture_times[0]
    assert interval_1 >= p._protocol.period.return_value.total_seconds(), (
        f'first interval {interval_1:.1f}s is shorter than the period -- '
        f'the anchor must be the first ACQUISITION, not run() entry'
    )


def test_later_scan_cadence_unchanged():
    """Scans 2..N keep the gate-open anchor: with a constant pre-positioned
    lead-in, capture-to-capture spacing stays exactly the period (no
    per-scan drift from re-anchoring at acquisition)."""
    clock = FakeClock()
    p = _make_parent(clock, n_scans=3, period_s=100.0)
    _run(clock, p)

    assert len(p.capture_times) == 3
    interval_2 = p.capture_times[2] - p.capture_times[1]
    assert interval_2 == pytest.approx(100.0, abs=1.0), (
        f'scans 2..N must keep start-to-start cadence == period; saw {interval_2:.1f}s'
    )


def test_reset_scan_state_clears_first_capture_time():
    """The recorder is per-scan coupled data, not a latch: the real
    runner's _reset_scan_state must clear it every scan."""
    from tests.test_audit_fixes import _make_capture_runner

    runner = _make_capture_runner()
    runner._scan_first_capture_t = 123.0
    runner._reset_scan_state()
    assert runner._scan_first_capture_t is None
