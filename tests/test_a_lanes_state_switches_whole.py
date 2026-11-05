# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A lane's state switches in one step no submit and no worker decision lands inside.

The lane's state -- closing, in a run, a run finishing -- was three fields
that every door and the worker read and then acted on a queue as a separate
step, while every switch wrote them and then moved or cleared queues as a
separate step. A switch landing between the read and the act stranded an
override read on a queue the worker did not serve (a run hung until a Stop),
refused it as if the scope were held, accepted a task across the close and
never answered it, or let a finishing run's end end the next run.

Each window test forces its window with a hook at the read and asserts the
hook fired. Where the switch lands inside the lane's state lock, it runs on
another thread (on the same thread the re-entrant lock lets it in, and the
race would reproduce on the fixed code) and the test asserts it waited;
where the window is before the lock, the switch runs whole in the hook and
the lock's second ask of the gate is what refuses.
"""

import queue
import threading

import pytest

from _lumascope.modules.activity_claim import ActivityClaim
from _lumascope.modules.lumascope_api._lumascope import scope_lane_closed
from _lumascope.modules.sequential_io_executor import ENQUEUED, IOTask, SequentialIOExecutor
from tests.protocol_drives import run_identity

_WAIT_S = 2.0
# How long a switch is given to show it is held at the lock. Only the
# fixed lane's switch is asserted still waiting, and it waits until the
# window closes, so a slow host cannot pass a broken lane on this bound.
_BLOCKED_S = 0.2


@pytest.fixture
def claim():
    return ActivityClaim()


@pytest.fixture
def lane(claim):
    """A started lane that asks ``claim``, with its override key."""
    ex = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
    key = ex.ask_claim(claim)
    ex.start()
    yield ex, key
    ex.protocol_end()
    ex.clear_protocol_pending()
    ex.shutdown()


def _busy(lane):
    """Run a task that holds the worker until the returned event is set.

    With a task running, the next one put is not first in line, so protocol
    mode does not serve it from the default queue.
    """
    started = threading.Event()
    release = threading.Event()
    lane.put(IOTask(action=lambda: (started.set(), release.wait(_WAIT_S * 3))))
    assert started.wait(_WAIT_S)
    return release


def _switch_on_another_thread(switch):
    """Start ``switch`` on its own thread; returns the thread and whether it was held."""
    t = threading.Thread(target=switch, daemon=True)
    t.start()
    t.join(_BLOCKED_S)
    return t, t.is_alive()


def test_an_override_read_put_as_a_run_starts_is_served_on_the_runs_queue(lane, claim):
    """W1: the run's start lands between put's gate and its enqueue.

    The read went on the default queue after the start's carry had run, and
    protocol mode never served it: the run's wait for an idle lane waited on
    it until a Stop.
    """
    lane, key = lane
    release = _busy(lane)
    held = claim.try_claim('protocol', run=run_identity())
    real_put = lane.queue.put
    seen = {}

    def put_as_the_run_starts(task):
        lane.queue.put = real_put
        seen['start'], seen['held'] = _switch_on_another_thread(lambda: lane.protocol_start(held))
        real_put(task)

    lane.queue.put = put_as_the_run_starts
    try:
        fut = lane.submit(IOTask(action=lambda: 'read'), 'read', override=key)
        seen['start'].join(_WAIT_S)
        release.set()
        assert fut.result(timeout=_WAIT_S) == 'read'
        assert lane.is_protocol_running()
        assert seen['held'], "the run's start did not wait for the read's enqueue"
    finally:
        release.set()
        held.release()


def test_an_override_read_asked_as_a_run_starts_is_not_refused(lane, claim):
    """W3: the run's start lands between submit's choice of door and the default door's gate.

    submit found no run and chose ``put``; the start then fenced the
    default door, and the read was refused ``exclusive_activity_running``,
    an answer an override never gives.
    """
    lane, key = lane
    release = _busy(lane)
    held = claim.try_claim('protocol', run=run_identity())
    real_ask = lane.is_protocol_running
    seen = {}

    def asked_as_the_run_starts():
        answer = real_ask()
        lane.is_protocol_running = real_ask
        seen['start'], seen['held'] = _switch_on_another_thread(lambda: lane.protocol_start(held))
        return answer

    lane.is_protocol_running = asked_as_the_run_starts
    try:
        fut = lane.submit(IOTask(action=lambda: 'read'), 'read', override=key)
        seen['start'].join(_WAIT_S)
        release.set()
        assert fut.result(timeout=_WAIT_S) == 'read'
        assert seen['held'], "the run's start did not wait for submit's door"
    finally:
        release.set()
        held.release()


def test_an_override_read_is_not_cleared_off_a_run_starting_as_the_worker_looks(lane, claim):
    """W2: the run's start lands between the worker's mode read and its clear of an ended run.

    The worker read no run, the start carried the waiting read onto the
    run's queue, and the worker then cleared that queue as an ended run's:
    the read went back to the default queue and waited out the run.
    """
    lane, key = lane
    release = _busy(lane)
    fut = lane.submit(IOTask(action=lambda: 'read'), 'read', override=key)
    held = claim.try_claim('protocol', run=run_identity())
    pq = lane.protocol_queue
    real_empty = pq.empty
    at_clear = threading.Event()
    go = threading.Event()

    def worker_looks_at_the_runs_queue():
        if threading.current_thread() is lane._worker_thread and not at_clear.is_set():
            at_clear.set()
            go.wait(_WAIT_S)
        return real_empty()

    pq.empty = worker_looks_at_the_runs_queue
    try:
        release.set()
        assert at_clear.wait(_WAIT_S), 'the worker never read the mode'
        start, start_held = _switch_on_another_thread(lambda: lane.protocol_start(held))
        go.set()
        start.join(_WAIT_S)
        assert fut.result(timeout=_WAIT_S) == 'read'
        assert start_held, "the run's start did not wait for the worker's decision"
    finally:
        go.set()
        pq.empty = real_empty
        held.release()


def test_an_override_read_is_not_cleared_off_a_run_starting_after_the_worker_takes_a_task(
    lane, claim
):
    """W2, the second look: the run's start lands between the worker's dequeue and its clear.

    After taking a task off the default queue the worker asked again
    whether an ended run had left work; a start in between carried the
    waiting read onto the run's queue, and the worker carried it back.
    """
    lane, key = lane
    release = _busy(lane)
    lane.put(IOTask(action=lambda: None))
    fut = lane.submit(IOTask(action=lambda: 'read'), 'read', override=key)
    pq = lane.protocol_queue
    real_empty = pq.empty
    seen = {}

    def worker_looks_after_taking_a_task():
        on_worker = threading.current_thread() is lane._worker_thread
        if on_worker and lane.running_task is not None and not seen:
            seen['claim'] = claim.try_claim('protocol', run=run_identity())
            seen['start'], seen['held'] = _switch_on_another_thread(
                lambda: lane.protocol_start(seen['claim'])
            )
        return real_empty()

    pq.empty = worker_looks_after_taking_a_task
    try:
        release.set()
        assert fut.result(timeout=_WAIT_S) == 'read'
        assert seen['held'], "the run's start did not wait for the worker's decision"
    finally:
        pq.empty = real_empty
        if seen.get('claim') is not None:
            seen['claim'].release()


def test_a_put_the_close_lands_behind_is_refused_not_left_on_the_queue():
    """W4: the close lands between a fire-and-forget put's gate and its enqueue.

    The put passed the gate, the close swept the queue, and the task was
    then put on it: ``ENQUEUED`` returned, the task never run or answered.
    """
    lane = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
    real_gate = lane.accepts_work
    closed = []

    def asked_as_the_lane_closes():
        answer = real_gate()
        lane.accepts_work = real_gate
        lane.shutdown()
        closed.append(True)
        return answer

    lane.accepts_work = asked_as_the_lane_closes

    outcome = lane.put(IOTask(action=lambda: 'ran'))

    assert closed, 'the close never landed in the window'
    assert outcome is None, 'the closed lane accepted the task'
    assert lane.queue_size() == 0, 'the task was left on the closed lane'


def test_a_put_the_close_lands_inside_is_answered_by_the_close():
    """W4 at the enqueue: a close on another thread waits for the put, then answers its task.

    The close swept the queue between the put's gate and its enqueue, and
    the task was put behind the sweep, never run or answered.
    """
    lane = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
    real_put = lane.queue.put
    seen = {}

    def put_as_the_lane_closes(task):
        lane.queue.put = real_put
        seen['close'], seen['held'] = _switch_on_another_thread(lane.shutdown)
        real_put(task)

    lane.queue.put = put_as_the_lane_closes
    told = []
    task = IOTask(
        action=lambda: 'ran',
        callback=lambda **kw: told.append(kw['exception'].reason),
        pass_result=True,
    )

    lane.put(task)
    seen['close'].join(_WAIT_S)

    assert told == ['scope_disconnected'], 'the task the close landed behind was never answered'
    assert lane.queue_size() == 0, 'the task was left on the closed lane'
    assert seen['held'], 'the close did not wait for the enqueue'


def test_a_run_step_the_close_lands_inside_is_answered_by_the_close():
    """W4 at the run's door: a close on another thread waits for the step, then answers it.

    The close swept the run's queue between the door's gate and its
    enqueue, and the step was put behind the sweep on a lane with no worker.
    """
    lane = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
    lane.protocol_start()
    pq = lane.protocol_queue
    real_put = pq.put
    seen = {}

    def put_as_the_lane_closes(task, *args, **kwargs):
        pq.put = real_put
        seen['close'], seen['held'] = _switch_on_another_thread(lane.shutdown)
        real_put(task, *args, **kwargs)

    pq.put = put_as_the_lane_closes
    told = []
    task = IOTask(
        action=lambda: 'step',
        callback=lambda **kw: told.append(kw['exception'].reason),
        pass_result=True,
    )

    lane.protocol_put(task)
    seen['close'].join(_WAIT_S)

    assert told == ['scope_disconnected'], 'the step the close landed behind was never answered'
    assert seen['held'], "the close did not wait for the step's enqueue"


def test_a_step_put_as_a_run_finishes_is_run_before_the_mode_ends():
    """A finishing run's mode ends only once its queue is drained, asked where the step is put.

    The worker's get found the run's queue empty, a step was put, and the
    worker then ended the mode: the step was answered as an ended run's.
    """
    lane = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
    lane.start()
    lane.protocol_start()
    pq = lane.protocol_queue
    real_get = pq.get
    in_get = threading.Event()
    go = threading.Event()

    def get_comes_back_empty(block=True, timeout=None):
        if threading.current_thread() is lane._worker_thread and not in_get.is_set():
            in_get.set()
            go.wait(_WAIT_S)
            raise queue.Empty
        return real_get(block, timeout)

    pq.get = get_comes_back_empty
    try:
        assert in_get.wait(_WAIT_S), "the worker never waited on the run's queue"
        lane.end_protocol_mode()
        fut = lane.protocol_put(IOTask(action=lambda: 'step'), return_future=True)
        go.set()
        assert fut.result(timeout=_WAIT_S) == 'step'
    finally:
        go.set()
        pq.get = real_get
        lane.shutdown()


def test_a_run_started_on_a_closed_lane_queues_nothing():
    """The run's door is shut once the lane closes, whatever its mode.

    A run started after the close sets the mode again; its work went on a
    queue no worker serves and no close answers -- the fire-and-forget put
    told ``PROTOCOL_ENQUEUED``, the waited one left waiting.
    """
    lane = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
    lane.shutdown()
    lane.protocol_start()

    assert lane.protocol_put(IOTask(action=lambda: 'step')) is None
    assert lane.protocol_put(IOTask(action=lambda: 'step'), return_future=True) is None
    _allocs, _pops, live = lane.caller_futures_stats()
    assert live == 0, 'a waiter was registered on the closed lane'


def test_a_finishing_runs_end_does_not_end_the_next_run():
    """W5: the next run's start lands between the worker's finish check and its end of the old run.

    The worker saw the old run finishing and ended protocol mode after the
    new run had set it, so the new run's steps were refused at its door.
    """
    lane = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
    lane.start()
    lane.protocol_start()
    finish = lane.protocol_finish
    real_is_set = finish.is_set
    seen = {}
    hooked = threading.Event()

    def worker_sees_the_run_finishing():
        answer = real_is_set()
        if answer and threading.current_thread() is lane._worker_thread and not seen:
            seen['start'], seen['held'] = _switch_on_another_thread(lane.protocol_start)
            hooked.set()
        return answer

    finish.is_set = worker_sees_the_run_finishing
    try:
        lane.end_protocol_mode()
        # end_protocol_mode's own ask is on this thread; the worker's comes
        # off its next empty get.
        assert hooked.wait(_WAIT_S), 'the worker never saw the run finishing'
        seen['start'].join(_WAIT_S)
        fut = lane.protocol_put(IOTask(action=lambda: 'step'), return_future=True)
        assert fut is not None, "the next run's step was refused: its mode was ended"
        assert fut.result(timeout=_WAIT_S) == 'step'
        assert lane.is_protocol_running()
        assert seen['held'], "the next run's start did not wait for the worker's end"
    finally:
        finish.is_set = real_is_set
        lane.shutdown()


def test_an_answer_given_as_a_run_ends_runs_with_the_state_lock_released():
    """A run's task answered at its end is answered after the switch, not inside it.

    An answer settles a waiter whose done-callbacks, and a callback with no
    UI, run on the switching thread; one that waits on work another thread
    puts on this lane would never be served inside the lock.
    """
    lane = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
    lane.protocol_start()
    put_from_elsewhere = []

    def on_answer(**_kw):
        other = threading.Thread(
            target=lambda: put_from_elsewhere.append(lane.put(IOTask(action=lambda: None)))
        )
        other.start()
        other.join(_WAIT_S)

    lane.protocol_put(IOTask(action=lambda: None, callback=on_answer, pass_result=True))
    lane.protocol_end()
    lane.clear_protocol_pending()

    assert put_from_elsewhere == [ENQUEUED]
    lane.shutdown()
