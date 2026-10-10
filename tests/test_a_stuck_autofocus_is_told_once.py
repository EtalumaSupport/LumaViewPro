# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An autofocus sweep that will not stop is told once, and then never again.

A Stop aborts a run's sweep and its cleanup waits, bounded, for the sweep to
unwind. Before, a sweep still inside a call at the bound was only logged:
the run ended as if it had handed the scope back cleanly, the next run was
refused "Stop it or let it finish" with nothing left to stop, and when the
sweep finally woke it reported its own refused restore and its refused
capture as faults of a run that was over. Now the stuck sweep is the run's
cleanup failure, told once in the cleanup summary; the run keeps the ending
the person gave it; the next run is refused with the truth; and the sweep,
once free, ends without a word.

The sweep is wedged inside its real capture call, on the real engine over
simulated hardware, and released by the test.
"""

import threading
import time

import pytest

import modules.protocol_cleanup as protocol_cleanup
from modules.exceptions import (
    AutofocusFailedError,
    AutofocusZNotRestoredError,
    ProtocolRunRefusedError,
    RunCleanupFailedError,
)
from tests.test_composite_run_e2e import headless_settings, open_composite_session

RESULT_TIMEOUT_S = 30.0


@pytest.fixture
def af_session(tmp_path, monkeypatch):
    # The bound is the house's 30 s wedge threshold; a test cannot wait it
    # out, so it is shortened at the one name cleanup reads.
    monkeypatch.setattr(protocol_cleanup, '_AF_UNWIND_WAIT_S', 0.5)
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        # Clear of the travel limit the home leaves Z at, so the sweep's
        # window fits and it reaches its first capture.
        session.scope.motion.move_absolute('Z', 3000.0)
        yield session, runner


class _WedgedSweep:
    """Hold the autofocus thread inside its capture until released, whatever it is told."""

    def __init__(self, monkeypatch, session, *, stops_when_aborted=False):
        self.inside = threading.Event()
        self.release = threading.Event()
        imaging_type = type(session.scope.imaging)
        real = imaging_type.capture_and_wait
        af_thread = session.autofocus_thread

        def wedged(imaging, *args, **kwargs):
            if threading.current_thread().name == 'autofocus_thread' and not self.release.is_set():
                self.inside.set()
                deadline = time.monotonic() + RESULT_TIMEOUT_S
                while not self.release.is_set() and time.monotonic() < deadline:
                    if stops_when_aborted and af_thread.aborted.is_set():
                        break
                    time.sleep(0.01)
            return real(imaging, *args, **kwargs)

        monkeypatch.setattr(imaging_type, 'capture_and_wait', wedged)


def _record_reports(monkeypatch):
    from modules.notification_center import notifications

    reported = []
    real = notifications.report_outcome

    def recorded(outcome, *args, **kwargs):
        reported.append(outcome)
        return real(outcome, *args, **kwargs)

    monkeypatch.setattr(notifications, 'report_outcome', recorded)
    return reported


def _wait_until(predicate, timeout_s=RESULT_TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, 'the condition never came true'
        time.sleep(0.02)


def test_a_sweep_that_will_not_stop_is_one_cleanup_failure_and_then_silent(af_session, monkeypatch):
    session, runner = af_session
    sweep = _WedgedSweep(monkeypatch, session)
    reported = _record_reports(monkeypatch)

    handle = runner.run_autofocus(layer='BF')
    assert sweep.inside.wait(RESULT_TIMEOUT_S), 'the sweep never reached its capture'
    handle.stop()
    outcome = handle.wait(timeout_s=RESULT_TIMEOUT_S)

    assert outcome is not None, 'the stopped run never let go of the scope'
    assert (outcome.status, outcome.reason) == ('aborted', 'stopped'), outcome
    assert outcome.cleanup_failures == ('Stop autofocus',), outcome.cleanup_failures
    told = [r for r in reported if isinstance(r, RunCleanupFailedError)]
    assert [r.steps for r in told] == [['Stop autofocus']], told

    with pytest.raises(ProtocolRunRefusedError) as refused:
        runner.run_autofocus(layer='BF')
    assert refused.value.reason == 'autofocus_running'
    assert 'did not stop' in str(refused.value)
    assert 'Stop it' not in str(refused.value)

    reported.clear()
    sweep.release.set()
    _wait_until(lambda: session.autofocus_thread.in_flight_sweep is None)

    late = [
        r for r in reported if isinstance(r, (AutofocusFailedError, AutofocusZNotRestoredError))
    ]
    assert late == [], 'the freed sweep reported faults of a run that was over'

    again = runner.run_autofocus(layer='BF').wait(timeout_s=RESULT_TIMEOUT_S)
    assert again is not None and again.status == 'completed', again


def test_a_sweep_that_stops_in_time_is_no_cleanup_failure(af_session, monkeypatch):
    session, runner = af_session
    sweep = _WedgedSweep(monkeypatch, session, stops_when_aborted=True)
    reported = _record_reports(monkeypatch)

    handle = runner.run_autofocus(layer='BF')
    assert sweep.inside.wait(RESULT_TIMEOUT_S), 'the sweep never reached its capture'
    handle.stop()
    outcome = handle.wait(timeout_s=RESULT_TIMEOUT_S)

    assert (outcome.status, outcome.reason) == ('aborted', 'stopped'), outcome
    assert outcome.cleanup_failures == (), outcome.cleanup_failures
    assert not [r for r in reported if isinstance(r, RunCleanupFailedError)]
