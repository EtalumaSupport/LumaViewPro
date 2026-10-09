# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A lane task's outcome is reported once, by the executor or by its waiter, never both.

The rule: a silent task's outcome is its waiter's if a waiter is registered
for it when the outcome lands; otherwise it gets the log half only. Any other
task is logged and shown. Whose it is gets decided where the registration is
popped, under the lock a timed-out waiter takes to pop it back, so exactly one
of the two finds it -- a fault is never handed to a waiter that has already
left, and never logged by both.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import CancelledError
from concurrent.futures import TimeoutError as FutureTimeoutError

import pytest

import modules.sequential_io_executor as sio
from modules.activity_claim import ActivityClaim
from modules.exceptions import HardwareCommandRefusedError, RunAlreadyEndedError
from modules.notification_center import Severity, notifications
from modules.sequential_io_executor import IOTask, SequentialIOExecutor

_WAIT_S = 2.0
OUTCOMES = 'LVP.outcomes'


@pytest.fixture(autouse=True)
def _executor_log(monkeypatch):
    # The suite mocks lvp_logger, so the executor's own lines go nowhere;
    # a real logger lets these tests read what it writes.
    monkeypatch.setattr(sio, 'logger', logging.getLogger('LVP.test_executor'))


@pytest.fixture
def lane():
    ex = SequentialIOExecutor(name='TEST_IO')
    ex.start()
    yield ex
    ex.shutdown(wait=False)


def _outcome_records(caplog):
    return [r for r in caplog.records if r.name == OUTCOMES]


def _drain(lane):
    lane.put(IOTask(action=lambda: None), return_future=True).result(timeout=_WAIT_S)


def _raise(exc):
    raise exc


class TestWhoReports:
    def test_a_non_silent_fault_is_logged_and_shown_once_by_the_executor(
        self, lane, centre_posts, caplog
    ):
        with caplog.at_level(logging.DEBUG):
            lane.put(IOTask(action=_raise, args=(RuntimeError('board gone'),)))
            _drain(lane)

        assert [(n.severity, n.title) for n in centre_posts] == [
            (Severity.ERROR, 'Background operation failed')
        ]
        records = _outcome_records(caplog)
        assert [(r.levelno, bool(r.exc_info)) for r in records] == [(logging.ERROR, True)]

    def test_a_call_tasks_fault_reaches_its_caller_and_the_executor_reports_nothing(
        self, lane, centre_posts, caplog
    ):
        boom = RuntimeError('board gone')
        with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError) as raised:
            lane.call(IOTask(action=_raise, args=(boom,)), 'move', timeout_s=_WAIT_S)

        assert raised.value is boom
        assert centre_posts == []
        assert _outcome_records(caplog) == []

    @pytest.mark.parametrize('with_callback', [True, False], ids=['callback', 'no_callback'])
    def test_a_silent_task_with_no_waiter_is_logged_and_not_shown(
        self, lane, centre_posts, caplog, with_callback
    ):
        # A GUI callback task, the composite Stop (no callback, no future) and
        # the image writer's retry all have this shape.
        callback = (lambda *a, **k: None) if with_callback else None
        with caplog.at_level(logging.DEBUG):
            lane.put(
                IOTask(
                    action=_raise,
                    args=(RuntimeError('teardown failed'),),
                    callback=callback,
                    silent_on_failure=True,
                )
            )
            _drain(lane)

        assert centre_posts == []
        records = _outcome_records(caplog)
        assert [(r.levelno, bool(r.exc_info)) for r in records] == [(logging.ERROR, True)]

    def test_a_waiter_that_reports_what_the_executor_already_showed_adds_nothing(
        self, lane, centre_posts, caplog
    ):
        refusal = HardwareCommandRefusedError('exclusive_activity_running', 'move', 'protocol')
        with caplog.at_level(logging.DEBUG):
            fut = lane.put(IOTask(action=_raise, args=(refusal,)), return_future=True)
            with pytest.raises(HardwareCommandRefusedError) as raised:
                fut.result(timeout=_WAIT_S)
            sio.notifications.report_outcome(raised.value, solicited=True, category='UI:TEST')

        assert [n.title for n in centre_posts] == ['Microscope Busy']
        assert _outcome_records(caplog) == []

    def test_a_quiet_outcome_is_logged_at_info_and_never_shown(self, lane, centre_posts, caplog):
        with caplog.at_level(logging.DEBUG):
            lane.put(IOTask(action=_raise, args=(RunAlreadyEndedError('the run has ended'),)))
            _drain(lane)

        assert centre_posts == []
        assert [r.levelno for r in _outcome_records(caplog)] == [logging.INFO]


class TestATimedOutCall:
    def test_a_fault_after_the_waiter_left_is_logged_not_shown(self, lane, centre_posts, caplog):
        release = threading.Event()

        def _slow_then_fail():
            release.wait(_WAIT_S)
            raise RuntimeError('late fault')

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(FutureTimeoutError):
                lane.call(IOTask(action=_slow_then_fail), 'move', timeout_s=0.05)
            release.set()
            _drain(lane)

        assert centre_posts == []
        records = _outcome_records(caplog)
        assert [(r.levelno, bool(r.exc_info)) for r in records] == [(logging.ERROR, True)]
        assert 'late fault' in records[0].getMessage()

    def test_an_outcome_claimed_before_the_waiter_left_is_the_callers(
        self, lane, centre_posts, caplog, monkeypatch
    ):
        # Hold the epilogue between claiming the waiter's registration and
        # completing its future, well past the caller's timeout: the caller
        # times out, finds its registration already taken, and receives the
        # outcome instead of a timeout.
        claimed = threading.Event()
        original = lane._report_task_failure

        def _held(task, exception, owned):
            claimed.set()
            time.sleep(0.5)
            original(task, exception, owned)

        monkeypatch.setattr(lane, '_report_task_failure', _held)
        boom = RuntimeError('landed first')

        with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError) as raised:
            lane.call(IOTask(action=_raise, args=(boom,)), 'move', timeout_s=0.05)

        assert claimed.is_set()
        assert raised.value is boom
        assert centre_posts == []
        assert _outcome_records(caplog) == []


class TestRefusalsDecidedWhereTheWaiterIsPopped:
    @pytest.fixture
    def claim(self):
        return ActivityClaim()

    @pytest.fixture
    def lane_and_key(self, claim):
        ex = SequentialIOExecutor(name='TEST_IO')
        key = ex.ask_claim(claim)
        ex.start()
        yield ex, key
        ex.shutdown(wait=False)

    @pytest.fixture
    def held_lane(self, lane_and_key):
        return lane_and_key[0]

    @pytest.mark.parametrize('waited', [True, False], ids=['waiter', 'no_waiter'])
    def test_a_silent_task_refused_while_queued(
        self, claim, lane_and_key, centre_posts, caplog, waited
    ):
        held_lane, key = lane_and_key
        gate = threading.Event()
        running = threading.Event()

        def _hold_the_worker():
            running.set()
            gate.wait(_WAIT_S)

        held_lane.put(IOTask(action=_hold_the_worker))
        assert running.wait(_WAIT_S)
        fut = held_lane.put(
            IOTask(action=lambda: None, silent_on_failure=True), return_future=waited
        )
        with caplog.at_level(logging.DEBUG):
            held = claim.try_claim('diagnostic')
            try:
                gate.set()
                if waited:
                    with pytest.raises(HardwareCommandRefusedError):
                        fut.result(timeout=_WAIT_S)
                # The lane runs its queue in order: once a task queued after
                # the silent one has run, that one was popped and decided
                # under the claim. The override key admits it past the holder.
                held_lane.call(IOTask(action=lambda: None), 'settle', _WAIT_S, override=key)
            finally:
                held.release()

        assert centre_posts == []
        levels = [r.levelno for r in _outcome_records(caplog)]
        assert levels == ([] if waited else [logging.WARNING])

    @pytest.mark.parametrize('waited', [True, False], ids=['waiter', 'no_waiter'])
    def test_a_silent_task_refused_at_submit(self, claim, held_lane, centre_posts, caplog, waited):
        held = claim.try_claim('diagnostic')
        try:
            box = {}

            def _submit():
                box['r'] = held_lane.put(
                    IOTask(action=lambda: None, silent_on_failure=True), return_future=waited
                )

            with caplog.at_level(logging.DEBUG):
                t = threading.Thread(target=_submit)
                t.start()
                t.join(_WAIT_S)
        finally:
            held.release()

        assert centre_posts == []
        levels = [r.levelno for r in _outcome_records(caplog)]
        assert levels == ([] if waited else [logging.WARNING])


def test_shutdown_completes_a_waiter_whose_task_is_still_running(centre_posts):
    lane = SequentialIOExecutor(name='TEST_IO')
    lane.start()
    release = threading.Event()
    started = threading.Event()

    def _running():
        started.set()
        release.wait(_WAIT_S * 5)

    box = {}

    def _wait():
        try:
            lane.call(IOTask(action=_running), 'move', timeout_s=None)
        except BaseException as e:  # the outcome under test, whatever it is
            box['outcome'] = e

    # A daemon: before the fix this waiter never returns.
    waiter = threading.Thread(target=_wait, daemon=True)
    waiter.start()
    assert started.wait(_WAIT_S)
    lane.shutdown(wait=False)
    waiter.join(_WAIT_S)
    release.set()

    assert not waiter.is_alive(), 'the waiter was left waiting for an outcome no one will set'
    assert isinstance(box.get('outcome'), CancelledError)


def test_a_listener_that_submits_from_inside_a_report_does_not_deadlock(lane):
    submitted = []

    def _listener(_notification):
        submitted.append(lane.put(IOTask(action=lambda: None), return_future=True))
        lane.caller_futures_stats()

    notifications.add_listener(_listener, min_severity=Severity.INFO)
    done = threading.Event()

    def _run():
        lane.put(IOTask(action=_raise, args=(RuntimeError('board gone'),)))
        _drain(lane)
        done.set()

    try:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        assert done.wait(_WAIT_S), 'the epilogue held a lock while the report ran its listener'
    finally:
        notifications.remove_listener(_listener)
    assert len(submitted) == 1


def test_an_abandoned_worker_names_what_its_task_raised(caplog):
    lane = SequentialIOExecutor(name='TEST_IO')
    lane.start()
    started = threading.Event()
    release = threading.Event()

    def _stuck_then_fail():
        started.set()
        release.wait(_WAIT_S)
        raise RuntimeError('finished after recovery')

    try:
        with caplog.at_level(logging.DEBUG):
            lane.put(IOTask(action=_stuck_then_fail))
            assert started.wait(_WAIT_S)
            worker = lane._worker_thread
            lane._worker_generation += 1  # what wedge recovery does to the stuck worker
            release.set()
            worker.join(_WAIT_S)

        lines = [r for r in caplog.records if 'Abandoned worker finished' in r.getMessage()]
        assert len(lines) == 1
        assert 'RuntimeError: finished after recovery' in lines[0].getMessage()
    finally:
        lane.shutdown(wait=False)
