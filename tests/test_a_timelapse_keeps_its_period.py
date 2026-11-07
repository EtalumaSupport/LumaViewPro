# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A time-lapse's scans start one period apart, with no drift.

The contract (``ProtocolRunLoop._run_loop_inner``'s pacing):

- The anchor is ESTABLISHED at the run's first acquisition (#239: anchored
  at run() entry, interval 1 came up short by the run's lead-in), then
  advanced by exactly one period for each scan that fits its period, so a
  timer that wakes late does not move the next scan's due time.
- A scan that overran its period is followed at once by the next, and a
  period's spacing resumes from that one (#239's overrun rule). Lateness is
  judged when the loop reaches its gate after the scan, never from when a
  wait woke.
- The wait is on the run's Stop event, so a Stop during it is seen at once;
  and it is cut at the hardware check's interval, so a board lost during a
  long period ends the run as a disconnect without waiting the period out.
"""

import threading
from types import MethodType, SimpleNamespace
from unittest import mock

import pytest

from _lumascope.modules import protocol_run_loop
from _lumascope.modules.lumascope_api.bring_up import Part
from _lumascope.modules.protocol_run_loop import ProtocolRunLoop
from _lumascope.modules.protocol_state_machine import ProtocolState
from _lumascope.modules.run_events import RunEvents
from _lumascope.modules.run_outcome import PendingRunOutcome
from _lumascope.modules.sequenced_capture_runner import SequencedCaptureRunner

# A timer on Windows wakes up to ~16 ms after it was asked to.
_TIMER_OVERSHOOT_S = 0.016


class FakeClock:
    """Deterministic stand-in for the loop's monotonic clock.

    Every way the loop can wait -- a sleep, or the Stop event's wait --
    advances it by what was asked plus ``overshoot_s``, as a late timer does.
    """

    def __init__(self, overshoot_s=0.0):
        self.t = 0.0
        self.overshoot_s = overshoot_s

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds + self.overshoot_s


class FakeStopEvent(threading.Event):
    """The run's Stop event on the fake clock: a wait passes fake time, never real.

    ``stop_at`` sets the event at that fake instant, as a Stop pressed
    during a wait would, and the wait returns there.
    """

    def __init__(self, clock, stop_at=None):
        super().__init__()
        self._clock = clock
        self._stop_at = stop_at

    def wait(self, timeout=None):
        if self.is_set():
            return True
        wake = self._clock.t + timeout + self._clock.overshoot_s
        if self._stop_at is not None and self._stop_at <= wake:
            self._clock.t = max(self._clock.t, self._stop_at)
            self.set()
            return True
        self._clock.t = wake
        return False


def _make_parent(
    clock,
    n_scans,
    period_s,
    first_scan_lead_s=30.0,
    later_scan_lead_s=5.0,
    scan_tail_s=10.0,
    long_scans=None,
    stop_at=None,
):
    """Run-loop parent stub whose scan_loop consumes fake time like a real
    scan: a long lead-in on scan 1 (motion + AF), a short constant lead-in
    on pre-positioned scans 2..N, and a capture that records its time and
    the first-acquisition timestamp exactly as the step runner does.

    ``long_scans`` maps a scan index to the tail it takes instead of
    ``scan_tail_s``, to make that scan overrun its period.
    """
    # A stand-in by design: the subject is the loop's pacing over many
    # periods of fake time; the simulated runner's scans take real time.
    p = SimpleNamespace()
    p._n_scans = n_scans
    p._scan_count = 0
    p._remaining_scans = lambda: p._n_scans - p._scan_count
    p._aborted = FakeStopEvent(clock, stop_at=stop_at)
    p._state = ProtocolState.RUNNING
    # The production predicate over the stub's own state, so the stub
    # cannot answer 'is a run live' differently from the runner.
    p._is_run_live = MethodType(SequencedCaptureRunner._is_run_live, p)
    # The loop takes the camera first; a stub with a mock scope has no lane to wait for.
    p._take_camera = lambda: None
    p._scope = mock.MagicMock()
    # The loop asks which parts are not connected; a bare mock's truthy
    # answer would end the run as a disconnect.
    p._scope.unconnected_parts.return_value = ()
    p._protocol = mock.MagicMock()
    p._protocol.period_s.return_value = period_s
    p._parent_dir = None
    p._start_t = clock.t  # run() entry semantics (the fallback anchor)
    p._curr_step = 0
    p._af_future = None
    p._scan_first_capture_t = None
    p.capture_times = []
    long_scans = long_scans or {}

    def _reset_scan_state():
        p._curr_step = 0
        p._af_future = None
        p._scan_first_capture_t = None

    def _advance_scan_count():
        p._scan_count += 1
        return p._scan_count

    def _scan_loop():
        lead = first_scan_lead_s if not p.capture_times else later_scan_lead_s
        clock.t += lead
        p.capture_times.append(clock.t)
        if p._scan_first_capture_t is None:
            p._scan_first_capture_t = clock.t
        clock.t += long_scans.get(len(p.capture_times) - 1, scan_tail_s)

    p._reset_scan_state = _reset_scan_state
    p.advance_scan_count = _advance_scan_count
    p._step_executor = mock.MagicMock()
    p._step_executor.scan_loop.side_effect = _scan_loop
    p._scan_in_progress = mock.MagicMock()
    p._set_state = mock.MagicMock()
    p.end_scan = mock.MagicMock()
    p.end_run_fatally = mock.MagicMock()
    p._cleanup = mock.MagicMock()
    p._protocol_state_lock = threading.Lock()
    p._auto_gain_armed_step = -1
    p.LOGGER_NAME = 'TEST'
    p._events = RunEvents()
    return p


def _run(clock, parent):
    loop = ProtocolRunLoop(parent)
    with (
        # The loop's own name for time, not the time module: patching
        # time.sleep through it would hand every thread in the process the
        # fake clock, and a leftover thread's sleep would advance it.
        mock.patch('_lumascope.modules.protocol_run_loop.time', clock),
    ):
        return loop._run_loop_inner(PendingRunOutcome())


def test_first_interval_is_never_shorter_than_period():
    """Period 100 s, scan-1 lead-in 30 s: pre-fix the anchor sat at run()
    entry, so capture 2 landed 100 s after t=0 -- interval 75 s < period.
    Post-fix the anchor is capture 1, so interval 1 >= period."""
    clock = FakeClock()
    p = _make_parent(clock, n_scans=2, period_s=100.0)
    _run(clock, p)

    assert len(p.capture_times) == 2
    interval_1 = p.capture_times[1] - p.capture_times[0]
    assert interval_1 >= p._protocol.period_s.return_value, (
        f'first interval {interval_1:.1f}s is shorter than the period -- '
        f'the anchor must be the first ACQUISITION, not run() entry'
    )


def test_later_scan_cadence_unchanged():
    """Scans 2..N are anchored at their due time: with a constant
    pre-positioned lead-in, capture-to-capture spacing stays exactly the
    period (no per-scan drift from re-anchoring at acquisition)."""
    clock = FakeClock()
    p = _make_parent(clock, n_scans=3, period_s=100.0)
    _run(clock, p)

    assert len(p.capture_times) == 3
    interval_2 = p.capture_times[2] - p.capture_times[1]
    assert interval_2 == pytest.approx(100.0, abs=1.0), (
        f'scans 2..N must keep start-to-start cadence == period; saw {interval_2:.1f}s'
    )


def test_twenty_periods_take_twenty_periods_with_a_late_timer():
    """Every wait wakes 16 ms late, and still scan k starts k-1 periods
    after scan 2, within one timer's lateness: the next due time is the
    last one plus a period, never the instant a wait woke. Pre-fix each
    poll's lateness re-anchored the next period, and twenty of them added up."""
    clock = FakeClock(overshoot_s=_TIMER_OVERSHOOT_S)
    p = _make_parent(clock, n_scans=21, period_s=100.0)
    _run(clock, p)

    assert len(p.capture_times) == 21
    behind = [
        round(p.capture_times[k] - p.capture_times[1] - (k - 1) * 100.0, 6) for k in range(1, 21)
    ]
    assert all(0.0 <= b <= _TIMER_OVERSHOOT_S for b in behind), (
        f'each scan must start one period after the last one was due; seconds behind: {behind}'
    )


def test_an_overrun_scan_is_followed_at_once_then_a_period_apart():
    """Scan 3 runs 150 s against a 100 s period: scan 4 starts the moment
    the loop is back at its gate, and scan 5 a period after scan 4."""
    clock = FakeClock()
    p = _make_parent(clock, n_scans=5, period_s=100.0, long_scans={2: 150.0})
    _run(clock, p)

    assert len(p.capture_times) == 5
    # Scan 3 captured 5 s after its start and ended 150 s later; scan 4's
    # capture is its own 5 s lead after that, with no wait between.
    assert p.capture_times[3] == pytest.approx(p.capture_times[2] + 150.0 + 5.0)
    assert p.capture_times[4] - p.capture_times[3] == pytest.approx(100.0)


def test_a_stop_during_the_wait_ends_the_run_where_it_was_pressed():
    """A Stop pressed 40 s into a 90 s wait ends the run at that instant:
    the wait is on the Stop event, not a poll that waits out its tick or
    the period."""
    clock = FakeClock()
    # Scan 1 captures at 30 s and ends at 40 s; scan 2 is due at 130 s.
    stop_at = 80.0
    p = _make_parent(clock, n_scans=2, period_s=100.0, stop_at=stop_at)
    ending = _run(clock, p)

    assert len(p.capture_times) == 1
    assert ending is protocol_run_loop.RUN_STOPPED
    assert clock.t == stop_at


def test_a_board_lost_during_a_long_wait_ends_the_run_at_the_next_check():
    """A one-hour period does not hold off the hardware check: a camera lost
    1 s into the wait ends the run as a disconnect within the check's
    interval, not at the next scan."""
    clock = FakeClock()
    # Scan 1 ends at 40 s.
    p = _make_parent(clock, n_scans=2, period_s=3600.0)
    lost_at = 41.0
    p._scope.unconnected_parts.side_effect = lambda: (Part.CAMERA,) if clock.t >= lost_at else ()
    ending = _run(clock, p)

    assert len(p.capture_times) == 1
    assert ending.reason == 'hardware_disconnected'
    assert clock.t <= lost_at + protocol_run_loop.HW_CHECK_INTERVAL_S + 1.0


def test_reset_scan_state_clears_first_capture_time():
    """The recorder is per-scan coupled data, not a latch: the real
    runner's _reset_scan_state must clear it every scan."""
    from tests.test_audit_fixes import _make_capture_runner

    runner = _make_capture_runner()
    runner._scan_first_capture_t = 123.0
    runner._reset_scan_state()
    assert runner._scan_first_capture_t is None
