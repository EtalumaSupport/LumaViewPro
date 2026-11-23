# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's progress states whether a scan is running and when its next one starts.

The contract (``RunProgress.is_scanning`` / ``next_scan_at``, from the
engine's one schedule, ``SequencedCaptureRunner._scan_schedule``, which the
run loop's gate waits on too):

- A scan is running from its start until it ends or fails; nothing is
  running between scans, or once the run has an ending.
- The next start is the schedule's own: scan 1 a period after its first
  capture (the run's start until it lands), each later scan a period after
  its due time; past after a scan that overran (at once), the run's start
  before the first scan and before a failed first scan's re-run (at once);
  none while the last scan runs.
- It is published as a wall instant made at each read, to the whole second,
  so a reader polling a waiting run reads one instant, not one that moves.
"""

import datetime
import threading
from unittest import mock

from _lumascope.drivers import host_clock
from _lumascope.drivers.exceptions import CameraReadFailedError
from _lumascope.modules import wall_time
from tests.test_a_timelapse_keeps_its_period import FakeClock, _make_parent, _run
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_the_run_handle import WAIT_S, headless_settings, open_composite_session

_PERIOD_S = 100.0


def _observed_run(clock, fail_scans=(), **parent_kwargs):
    """Run a time-lapse on the fake clock, reading the schedule where a reader would.

    The stand-in parent by design: the subject is the schedule over many
    periods of fake time, which the simulated runner's real-time scans
    cannot give. Read as a scan's steps begin (after its first step's
    move), after its first capture, and at its end (counted, or failed),
    and before and after the run; *fail_scans* are the
    scan attempts, by index from 0, that fail after their capture as a
    camera read would.
    """
    p = _make_parent(clock, period_s=_PERIOD_S, **parent_kwargs)
    p._scope.motion.axes_without_position = ()
    seen = [('before the run', clock.t, p._scan_schedule())]
    attempts = []
    first_scan_lead_s = parent_kwargs.get('first_scan_lead_s', 30.0)
    long_scans = parent_kwargs.get('long_scans', {})

    def _scan_loop():
        attempt = len(attempts)
        attempts.append(clock.t)
        seen.append((f'attempt {attempt} starts', clock.t, p._scan_schedule()))
        clock.t += first_scan_lead_s if attempt == 0 else 5.0
        p.capture_times.append(clock.t)
        p._note_first_capture(clock.t)
        seen.append((f'attempt {attempt} captured', clock.t, p._scan_schedule()))
        clock.t += long_scans.get(attempt, 10.0)
        if attempt in fail_scans:
            raise CameraReadFailedError('the camera did not answer')

    p._step_executor.scan_loop.side_effect = _scan_loop
    p.end_scan.side_effect = lambda: seen.append(
        (f'attempt {len(attempts) - 1} ended', clock.t, p._scan_schedule())
    )
    _run(clock, p)
    seen.append(('after the run', clock.t, p._scan_schedule()))
    return seen, attempts


def test_a_fitting_time_lapse_names_each_next_start():
    seen, attempts = _observed_run(FakeClock(), n_scans=3)

    # Scan 1 captures at 30 s, so scan 2 is due at 130 s and scan 3 at 230 s.
    assert attempts == [0.0, 130.0, 230.0]
    assert [(at, t, schedule) for at, t, schedule in seen] == [
        ('before the run', 0.0, (False, 0.0)),
        ('attempt 0 starts', 0.0, (True, 100.0)),
        ('attempt 0 captured', 30.0, (True, 130.0)),
        ('attempt 0 ended', 40.0, (False, 130.0)),
        ('attempt 1 starts', 130.0, (True, 230.0)),
        ('attempt 1 captured', 135.0, (True, 230.0)),
        ('attempt 1 ended', 145.0, (False, 230.0)),
        ('attempt 2 starts', 230.0, (True, None)),
        ('attempt 2 captured', 235.0, (True, None)),
        ('attempt 2 ended', 245.0, (False, None)),
        ('after the run', 245.0, (False, None)),
    ]


def test_a_scan_that_overran_names_a_past_start_and_the_spacing_resumes():
    # Scan 2 starts at 130 s and ends at 335 s, past the 230 s it made due.
    seen, attempts = _observed_run(FakeClock(), n_scans=4, long_scans={1: 200.0})

    assert attempts == [0.0, 130.0, 335.0, 435.0]
    by_name = {at: (t, schedule) for at, t, schedule in seen}
    assert by_name['attempt 1 ended'] == (335.0, (False, 230.0))
    assert by_name['attempt 2 starts'] == (335.0, (True, 435.0))


def test_a_failed_scan_names_the_start_the_gate_retries_it_at():
    # The first attempt of scan 1 fails, and its re-run starts at once; the
    # first attempt of scan 2 fails, and is retried a period after its due.
    seen, attempts = _observed_run(FakeClock(), n_scans=2, fail_scans={0, 2})

    by_name = {at: (t, schedule) for at, t, schedule in seen}
    assert by_name['attempt 0 ended'] == (40.0, (False, 0.0))
    assert attempts[1] == 40.0
    # The re-run states no capture of the attempt that failed.
    assert by_name['attempt 1 starts'] == (40.0, (True, 100.0))
    assert by_name['attempt 1 captured'] == (45.0, (True, 145.0))
    assert attempts[2] == 145.0
    assert by_name['attempt 2 ended'] == (160.0, (False, 245.0))
    assert attempts[3] == 245.0


def test_the_published_instant_is_a_whole_second_and_holds_across_reads():
    clock = FakeClock()
    epoch_ns = 1_800_000_000_123_456_789

    def _instant(due):
        with mock.patch.object(host_clock, 'wall_ns', lambda: epoch_ns + round(clock.t * 1e9)):
            return wall_time.whole_second_after(due - clock.t)

    clock.t = 40.25
    first = _instant(130.0)
    clock.t = 97.75
    later = _instant(130.0)

    assert first == later
    assert first.microsecond == 0
    assert first.utcoffset() is not None
    # 130 s from the run's start, wherever the wall clock stood at it.
    exact = datetime.datetime.fromtimestamp(epoch_ns / 1e9 + 130.0, datetime.UTC)
    assert abs((first - exact).total_seconds()) <= 0.5


def test_a_live_run_states_its_scan_and_next_start_and_an_ended_one_neither(tmp_path, monkeypatch):
    """The record through the real runner on the simulator: the fields the GUI and REST read.

    A one-minute time-lapse is held in scan 1 just after its first capture,
    read, then stopped.
    """
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        engine = session._sequenced_capture_runner
        note = engine._note_first_capture
        captured = threading.Event()
        let_go = threading.Event()

        def held_note(t):
            note(t)
            captured.set()
            let_go.wait(WAIT_S)

        monkeypatch.setattr(engine, '_note_first_capture', held_note)
        run = runner.start_protocol(
            protocol=_protocol([_step('C1', 0, x=20.0, gain=1.0)]),
            parent_dir=str(tmp_path / 'runs'),
        )
        try:
            assert captured.wait(WAIT_S), 'the run never captured'
            during = run.progress()
            read_at = wall_time.now()
            run.stop()
        finally:
            let_go.set()
        assert run.result(timeout=WAIT_S) is not None
        ended = run.progress()

    assert (during.is_live, during.is_scanning) == (True, True)
    assert during.next_scan_at.microsecond == 0
    assert during.next_scan_at.utcoffset() is not None
    # A period after the capture just made, to the second.
    assert 58.0 <= (during.next_scan_at - read_at).total_seconds() <= 61.0
    assert (ended.is_scanning, ended.next_scan_at) == (False, None)
