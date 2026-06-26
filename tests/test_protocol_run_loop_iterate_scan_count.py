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
    p._start_t = datetime.datetime.now() - datetime.timedelta(hours=1)
    p._curr_step = 0
    p._af_future = None

    # The run loop drives per-scan state + the scan-count increment through the
    # runner's own methods now (single-owner counters); the stub implements them
    # to match. advance_scan_count increments and returns the new count.
    def _reset_scan_state():
        p._curr_step = 0
        p._af_future = None

    def _advance_scan_count():
        p._scan_count += 1
        return p._scan_count

    p._reset_scan_state = _reset_scan_state
    p.advance_scan_count = _advance_scan_count
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
