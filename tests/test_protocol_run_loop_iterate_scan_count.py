"""Regression: the protocol_iterate_pre UI callback must report the
remaining-scan count from the scan that scheduled it, not whatever the run
loop has advanced to by the time Kivy runs the deferred callback.

The run loop schedules protocol_iterate_pre via schedule_ui (Kivy
Clock.schedule_once), which defers the call to the UI thread while the worker
loop continues into a multi-second scan. If the scheduled lambda closes over
the loop variable instead of binding its value, every deferred callback reads
the final remaining-scan count, so a multi-scan run reports the same number
(or zero) for every scan instead of counting down.

This test defers the scheduled callbacks (captures them without running them),
drives two scans, then runs the callbacks and asserts they counted down.
"""

import datetime
import threading
import time
from types import SimpleNamespace
from unittest import mock

from modules.protocol_run_loop import ProtocolRunLoop
from modules.protocol_state_machine import ProtocolState


def _make_two_scan_parent():
    p = SimpleNamespace()
    p._n_scans = 2
    p._scan_count = 0
    # Real remaining-scan semantics: n_scans - scan_count, counting down as
    # scans complete (scan_count is incremented inside the loop).
    p.remaining_scans = lambda: p._n_scans - p._scan_count
    p._run_in_progress_event = threading.Event()
    p._run_in_progress_event.set()
    p._aborted = threading.Event()
    p._state = ProtocolState.RUNNING
    p._scope = mock.MagicMock()
    p._protocol = mock.MagicMock()
    p._protocol.period.return_value = datetime.timedelta(0)
    p._parent_dir = None  # skip the disk-space branch
    # _start_t is a monotonic timestamp (seconds); set it an hour in the past so
    # the period-0 pacing check fires immediately.
    p._start_t = time.monotonic() - 3600.0
    p._curr_step = 0
    p._af_future = None
    p._step_executor = mock.MagicMock()
    p._scan_in_progress = mock.MagicMock()
    p._set_state = mock.MagicMock()
    p._cleanup = mock.MagicMock()
    p._protocol_state_lock = threading.Lock()
    p._auto_gain_armed_step = -1
    p.LOGGER_NAME = 'TEST'
    p._callbacks = SimpleNamespace(
        protocol_iterate_pre=mock.MagicMock(),
        run_scan_pre=None,
        scan_iterate_post=None,
    )
    # The run loop itself increments _scan_count once per completed scan, so
    # remaining_scans() counts down across iterations without extra help here.
    return p


def test_iterate_callback_reports_per_scan_remaining_count():
    p = _make_two_scan_parent()
    loop = ProtocolRunLoop(p)

    deferred = []

    def _capture(callback, *_a, **_k):
        # Simulate Kivy deferral: stash the callback, run it later.
        deferred.append(callback)

    with mock.patch('modules.protocol_run_loop._schedule_ui', _capture):
        loop._run_loop_inner()

    # Two scans ran, so two iterate-pre callbacks were scheduled.
    assert len(deferred) == 2

    # Run the deferred callbacks now (as the Kivy clock would, after the loop
    # advanced) and capture what remaining-scan count each reported.
    for cb in deferred:
        cb(0)

    reported = [
        call.kwargs['remaining_scans'] for call in p._callbacks.protocol_iterate_pre.call_args_list
    ]
    assert reported == [2, 1]


def test_scan_pacing_waits_on_monotonic_period():
    """Inter-scan pacing compares a MONOTONIC elapsed against the period in
    seconds. With _start_t set to 'now' and a 100 s period, the loop must WAIT on
    the second scan (only ~0 s of monotonic time has elapsed) rather than firing
    it. A wall-clock _start_t in the past, or a backward clock jump, must not be
    able to rush or stall the schedule.
    """
    p = _make_two_scan_parent()
    p._protocol.period.return_value = datetime.timedelta(seconds=100)
    p._start_t = time.monotonic()  # just started: ~0 s elapsed

    loop = ProtocolRunLoop(p)

    # The pacing branch sleeps then continues while it waits; abort on the first
    # sleep so the loop exits instead of spinning on the unmet period.
    def _abort_on_sleep(_seconds):
        p._aborted.set()

    deferred = []
    with (
        mock.patch(
            'modules.protocol_run_loop._schedule_ui', lambda cb, *a, **k: deferred.append(cb)
        ),
        mock.patch('modules.protocol_run_loop.time.sleep', _abort_on_sleep),
    ):
        loop._run_loop_inner()

    for cb in deferred:
        cb(0)

    # Scan 0 fired (its iterate-pre was scheduled); scan 1 hit the pacing wait
    # and aborted, so exactly one iterate-pre callback fired -- the period gated
    # on monotonic time, not on a stale wall-clock _start_t (which sat an hour in
    # the past in the helper and would have fired immediately).
    assert p._callbacks.protocol_iterate_pre.call_count == 1
