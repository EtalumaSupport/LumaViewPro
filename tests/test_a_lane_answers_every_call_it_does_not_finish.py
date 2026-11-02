# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A call put on a lane gets its outcome, a refusal saying why it never ran, or "still running".

The executor's contract (``SequentialIOExecutor.call`` and ``submit``): a
task it does not finish is answered, never left waiting and never settled
with a bare cancel that names no cause. A run's end touches only the run's
own work, and never the next run; an override call (the camera temperature
read) waiting when a run starts or ends is served, not stranded or dropped.
What "closed" means is stated by each executor's closer: the scope's lanes
answer ``scope_disconnected``, the session's ``SessionClosingError``.
"""

import logging
import threading

import pytest

from modules.activity_claim import ActivityClaim, acting
from modules.exceptions import (
    FileWriterReplacedError,
    HardwareCommandRefusedError,
    LaneCallStillRunningError,
    SessionClosingError,
)
from modules.executor_registry import session_lane_closed
from modules.lumascope_api._lumascope import scope_lane_closed
from modules.sequential_io_executor import (
    PRIORITY_HIGH,
    PRIORITY_MED,
    IOTask,
    SequentialIOExecutor,
)
from tests.protocol_drives import run_identity

_WAIT_S = 5.0


@pytest.fixture
def lane():
    """A started scope lane that asks no claim."""
    ex = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
    ex.start()
    yield ex
    ex.shutdown()


@pytest.fixture
def claim():
    return ActivityClaim()


@pytest.fixture
def claimed_lane(claim):
    """A started scope lane that asks ``claim``, and the override key it returned."""
    ex = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_CAMERA')
    key = ex.ask_claim(claim)
    ex.start()
    yield ex, key
    ex.shutdown()


def _blocking_task(started: threading.Event, release: threading.Event, **kwargs) -> IOTask:
    """A task that says it has started, then runs until ``release`` is set."""

    def hold():
        started.set()
        release.wait(_WAIT_S)
        return 'held'

    return IOTask(action=hold, **kwargs)


def _told(box: list) -> dict:
    """Callback kwargs that record each (result, exception) the task's callback hears."""
    return {
        'callback': lambda result, exception: box.append((result, exception)),
        'pass_result': True,
    }


class TestARunsEnd:
    def test_it_leaves_nothing_that_drops_the_next_runs_queued_work(self, lane):
        lane.protocol_start()
        lane.protocol_end()
        lane.clear_protocol_pending()

        lane.protocol_start()
        started, release = threading.Event(), threading.Event()
        first = lane.protocol_put(_blocking_task(started, release), return_future=True)
        second = lane.protocol_put(IOTask(action=lambda: 'second'), return_future=True)
        release.set()

        assert first.result(timeout=_WAIT_S) == 'held'
        assert second.result(timeout=_WAIT_S) == 'second', (
            "the previous run's end must not drop the work queued behind this run's first task"
        )

    def test_the_runs_own_task_still_queued_is_refused_activity_ended_and_its_callback_told(
        self, claim, claimed_lane
    ):
        lane, _key = claimed_lane
        taking = claim.try_claim('protocol', run=run_identity())
        lane.protocol_start(taking)
        started, release = threading.Event(), threading.Event()
        heard = []
        ran = threading.Event()
        with acting(taking):
            lane.protocol_put(_blocking_task(started, release))
            left = lane.protocol_put(
                IOTask(action=ran.set, silent_on_failure=True, **_told(heard)), return_future=True
            )
        assert started.wait(_WAIT_S)

        lane.protocol_end()
        lane.clear_protocol_pending()
        release.set()

        with pytest.raises(HardwareCommandRefusedError) as refused:
            left.result(timeout=_WAIT_S)
        assert refused.value.reason == 'activity_ended'
        assert heard == [(None, refused.value)]
        assert not ran.is_set()
        taking.release()


class TestAnOverrideRead:
    def test_one_waiting_as_a_run_starts_is_served_during_the_run(self, claim, claimed_lane):
        lane, key = claimed_lane
        started, release = threading.Event(), threading.Event()
        lane.put(_blocking_task(started, release))
        assert started.wait(_WAIT_S)
        read = lane.put(IOTask(action=lambda: 31.5), return_future=True, override=key)

        taking = claim.try_claim('protocol', run=run_identity())
        lane.protocol_start(taking)
        release.set()

        assert read.result(timeout=_WAIT_S) == 31.5, (
            'a read put just before the run, behind a running task, must not wait out the run '
            'on the queue protocol mode does not serve'
        )
        lane.protocol_end()
        lane.clear_protocol_pending()
        taking.release()

    def test_one_made_during_a_run_takes_its_turn_among_the_runs_work(self, claim, claimed_lane):
        lane, key = claimed_lane
        taking = claim.try_claim('protocol', run=run_identity())
        lane.protocol_start(taking)
        order = []
        started, release = threading.Event(), threading.Event()
        with acting(taking):
            lane.protocol_put(_blocking_task(started, release))
        assert started.wait(_WAIT_S)
        read = lane.submit(IOTask(action=lambda: order.append('read')), 'read', override=key)
        with acting(taking):
            last = lane.protocol_put(IOTask(action=lambda: order.append('run')), return_future=True)
        release.set()

        last.result(timeout=_WAIT_S)
        read.result(timeout=_WAIT_S)
        assert order == ['read', 'run']
        lane.protocol_end()
        lane.clear_protocol_pending()
        taking.release()

    def test_one_still_waiting_when_the_run_ends_is_served_after_it(self, claim, claimed_lane):
        lane, key = claimed_lane
        taking = claim.try_claim('protocol', run=run_identity())
        lane.protocol_start(taking)
        started, release = threading.Event(), threading.Event()
        with acting(taking):
            lane.protocol_put(_blocking_task(started, release))
        assert started.wait(_WAIT_S)
        read = lane.submit(IOTask(action=lambda: 31.5), 'read', override=key)

        lane.protocol_end()
        lane.clear_protocol_pending()
        taking.release()
        release.set()

        assert read.result(timeout=_WAIT_S) == 31.5


class TestTheScopeLanesClosing:
    def test_a_queued_call_is_refused_scope_disconnected_and_never_runs(self, lane):
        started, release = threading.Event(), threading.Event()
        lane.put(_blocking_task(started, release))
        assert started.wait(_WAIT_S)
        heard = []
        ran = threading.Event()
        queued = lane.submit(IOTask(action=ran.set, **_told(heard)), 'set_gain_db')

        lane.shutdown(wait=False)
        release.set()

        with pytest.raises(HardwareCommandRefusedError) as refused:
            queued.result(timeout=_WAIT_S)
        assert refused.value.reason == 'scope_disconnected'
        assert refused.value.member == 'set_gain_db'
        assert heard == [(None, refused.value)]
        assert not ran.is_set()

    def test_a_running_call_is_told_it_is_still_running(self, lane):
        started, release = threading.Event(), threading.Event()
        running = lane.submit(_blocking_task(started, release), 'move_z')
        assert started.wait(_WAIT_S)

        lane.shutdown(wait=False)

        with pytest.raises(LaneCallStillRunningError) as still:
            running.result(timeout=_WAIT_S)
        assert still.value.reason == 'lane_closed'
        assert still.value.member == 'move_z'
        release.set()

    def test_the_queued_calls_are_answered_in_the_order_they_would_have_run(self):
        ex = SequentialIOExecutor(
            closed_answer=scope_lane_closed, name='TEST_POOL', priority_aware=True
        )
        ex.start()
        started, release = threading.Event(), threading.Event()
        ex.put(_blocking_task(started, release))
        assert started.wait(_WAIT_S)
        answered = []
        for label, priority in (
            ('med-a', PRIORITY_MED),
            ('high', PRIORITY_HIGH),
            ('med-b', PRIORITY_MED),
        ):
            ex.put(
                IOTask(
                    action=lambda: None,
                    priority=priority,
                    callback=lambda label=label: answered.append(label),
                )
            )

        ex.shutdown(wait=False)
        release.set()

        assert answered == ['high', 'med-a', 'med-b']

    def test_a_call_after_the_close_is_refused_scope_disconnected(self, lane):
        lane.shutdown()
        with pytest.raises(HardwareCommandRefusedError) as refused:
            lane.submit(IOTask(action=lambda: None), 'set_gain_db')
        assert refused.value.reason == 'scope_disconnected'

    def test_a_task_taken_off_the_queue_unstarted_is_refused_and_never_runs(self):
        ex = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
        taken, resume = threading.Event(), threading.Event()
        real_get = ex.queue.get

        # a stand-in by design: widens the real window between the worker
        # taking a task off the queue and starting it, which no timing of
        # the simulator holds open long enough to close the lane in.
        def get_then_pause(*args, **kwargs):
            task = real_get(*args, **kwargs)
            taken.set()
            resume.wait(_WAIT_S)
            return task

        ex.queue.get = get_then_pause
        ex.start()
        ran = threading.Event()
        fut = ex.submit(IOTask(action=ran.set), 'set_gain_db')
        assert taken.wait(_WAIT_S)

        ex.shutdown(wait=False)
        resume.set()

        with pytest.raises(HardwareCommandRefusedError) as refused:
            fut.result(timeout=_WAIT_S)
        assert refused.value.reason == 'scope_disconnected'
        ex._worker_thread.join(_WAIT_S)
        assert not ran.is_set(), 'a task answered as never run must never run'

    def test_a_waiter_asked_for_after_the_close_swept_is_refused_not_left_waiting(self):
        ex = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
        ex.start()
        ex.shutdown()
        # a stand-in by design: the first ask of the gate answers as it would
        # have before the close, the window a close can land in; no timing
        # holds it open long enough to close the lane in. Every later ask is
        # the real one.
        real_gate = ex.accepts_work

        def open_once():
            ex.accepts_work = real_gate
            return True

        ex.accepts_work = open_once

        with pytest.raises(HardwareCommandRefusedError) as refused:
            ex.submit(IOTask(action=lambda: None), 'set_gain_db')
        assert refused.value.reason == 'scope_disconnected'
        assert ex.caller_futures_stats()[2] == 0, 'a waiter registered after the sweep'

    def test_a_put_racing_the_close_is_refused_and_never_runs(self):
        ex = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
        ex.start()
        entered, resume = threading.Event(), threading.Event()
        real_put = ex.queue.put

        # a stand-in by design: holds the put between registering its waiter
        # and queueing the task, the window a close can land in.
        def pause_then_put(task, *args, **kwargs):
            entered.set()
            resume.wait(_WAIT_S)
            return real_put(task, *args, **kwargs)

        ex.queue.put = pause_then_put
        ran = threading.Event()
        box = {}
        putter = threading.Thread(
            target=lambda: box.update(fut=ex.submit(IOTask(action=ran.set), 'set_gain_db'))
        )
        putter.start()
        assert entered.wait(_WAIT_S)

        ex.shutdown(wait=False)
        resume.set()
        putter.join(_WAIT_S)

        with pytest.raises(HardwareCommandRefusedError) as refused:
            box['fut'].result(timeout=_WAIT_S)
        assert refused.value.reason == 'scope_disconnected'
        ex._worker_thread.join(_WAIT_S)
        assert not ran.is_set()


class TestAnOverrun:
    def test_it_says_what_the_call_waited_behind(self, lane, caplog):
        started, release = threading.Event(), threading.Event()
        lane.put(_blocking_task(started, release))
        assert started.wait(_WAIT_S)
        lane.put(IOTask(action=lambda: None))

        with caplog.at_level(logging.INFO), pytest.raises(LaneCallStillRunningError) as overran:
            lane.call(IOTask(action=lambda: None), 'get_temperature', timeout_s=0.05)
        release.set()

        assert overran.value.reason == 'call_overran'
        # The log is the product here: what the 4 unexplained overruns lacked.
        [line] = [r.getMessage() for r in caplog.records if 'overran' in r.getMessage()]
        assert '2 queued, 0 on the run queue, running hold' in line


class TestTheSessionsLanesClosing:
    def test_a_queued_file_write_and_one_after_the_close_are_not_done(self, sim_session):
        file_lane = sim_session.file_io_executor
        started, release = threading.Event(), threading.Event()
        file_lane.put(_blocking_task(started, release))
        assert started.wait(_WAIT_S)
        wrote = threading.Event()
        queued = file_lane.submit(IOTask(action=wrote.set), 'write_tiff')

        sim_session.shutdown()
        release.set()

        with pytest.raises(SessionClosingError) as not_done:
            queued.result(timeout=_WAIT_S)
        assert not_done.value.work == 'write_tiff'
        with pytest.raises(SessionClosingError):
            file_lane.submit(IOTask(action=wrote.set), 'write_tiff')
        assert not wrote.is_set()


class TestAStuckWorkerReplaced:
    def test_its_waiter_is_told_what_its_caller_states(self):
        ex = SequentialIOExecutor(closed_answer=session_lane_closed, name='TEST_FILE')
        ex.start()
        started, release = threading.Event(), threading.Event()
        stuck = ex.submit(_blocking_task(started, release), 'write_tiff')
        assert started.wait(_WAIT_S)

        ex.replace_stuck_worker(FileWriterReplacedError)

        with pytest.raises(FileWriterReplacedError) as replaced:
            stuck.result(timeout=_WAIT_S)
        assert replaced.value.reason == 'file_writer_replaced'
        assert ex.submit(IOTask(action=lambda: 'next'), 'write_tiff').result(_WAIT_S) == 'next'
        release.set()
        ex.shutdown()


def test_runs_complete_while_clients_poll_the_camera_temperature(tmp_path):
    """Every run ends, and every read answers, while clients poll the override read.

    A run ending left the next run's first finished task to drop the run's
    queued work (``run_loop_crashed``), and a read waiting as a run started
    hung the run in its wait for the camera lane. The bound is sized from
    an unpolled run (about 0.5 s), so a design that starves the run fails
    here as a hung one does.
    """
    from concurrent.futures import TimeoutError as FutureTimeoutError

    from tests.test_composite_run_e2e import headless_settings, open_composite_session
    from tests.test_going_to_a_step_is_the_sessions_move import _PLATE, _protocol, _step

    run_bound_s = 10.0
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        stop = threading.Event()
        failed = []

        def poll():
            while not stop.is_set():
                try:
                    session.scope.diagnostics.get_camera_temperatures_degc()
                except Exception as ex:
                    failed.append(ex)

        # Four clients, as the execution check that hung a run used.
        readers = [threading.Thread(target=poll, daemon=True) for _ in range(4)]
        for reader in readers:
            reader.start()
        objective = session.scope.runtime_state.get_current_objective_id()
        outcomes = []
        try:
            for i in range(5):
                parent = tmp_path / f'run{i}'
                parent.mkdir()
                op = runner.start_single_scan(
                    _protocol(_PLATE, _step(20.0, 20.0, 5000.0, objective)),
                    parent_dir=str(parent),
                    enable_image_saving=False,
                )
                try:
                    outcomes.append(op.result(timeout=run_bound_s).reason)
                except FutureTimeoutError:
                    op.stop()
                    op.result(timeout=run_bound_s)
                    outcomes.append(f'did not end within {run_bound_s} s')
        finally:
            stop.set()
            for reader in readers:
                reader.join(_WAIT_S)

    assert outcomes == ['completed'] * 5
    assert failed == []
