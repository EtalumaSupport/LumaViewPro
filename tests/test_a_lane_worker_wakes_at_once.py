# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A lane's worker starts the work it is given, and stops when closed, at once.

The worker waits for work and for a change of the lane's state -- a run
starting, ending or finishing, the lane closing -- and is woken by each, so
the first task after a switch starts within a few milliseconds and a close
returns as soon as the worker is out of its wait. Each test lets the worker
finish a task first, so it is in its wait when the switch lands.

Timing-subject tests: each asserts a start or a return within ``_PROMPT_S``,
a quarter of the 200 ms a worker that polled its queues would take from the
moment it began waiting.
"""

import threading
import time

import pytest

from _lumascope.modules.lumascope_api._lumascope import scope_lane_closed
from _lumascope.modules.sequential_io_executor import IOTask, SequentialIOExecutor

_WAIT_S = 2.0
_PROMPT_S = 0.05

pytestmark = pytest.mark.slow


@pytest.fixture
def lane():
    ex = SequentialIOExecutor(closed_answer=scope_lane_closed, name='TEST_IO')
    ex.start()
    yield ex
    ex.shutdown()


def _settle(lane) -> None:
    """Run one task to its end, so the worker goes back into its wait."""
    lane.put(IOTask(action=lambda: None), return_future=True).result(timeout=_WAIT_S)


def test_the_first_step_after_a_run_starts_starts_at_once(lane):
    _settle(lane)
    started = {}
    done = threading.Event()

    def mark():
        started['at'] = time.monotonic()
        done.set()

    switched_at = time.monotonic()
    lane.protocol_start()
    lane.protocol_put(IOTask(action=mark))
    assert done.wait(_WAIT_S), 'the step never started'
    assert started['at'] - switched_at < _PROMPT_S


def _waiting_through_a_run(lane):
    """Leave a task on the default queue that the run does not serve, the worker waiting.

    The task is put behind a running one, so it is not first in line, and
    the run starts before the running task ends: the worker then waits on
    the run's empty queue, and only the run's end can bring it to the task.
    No put follows the switch, so the switch alone must wake the worker.
    Returns the task's start mark and the event its start sets.
    """
    started, release = threading.Event(), threading.Event()
    lane.put(IOTask(action=lambda: (started.set(), release.wait(_WAIT_S))))
    assert started.wait(_WAIT_S)
    marks = {}
    done = threading.Event()

    def mark():
        marks['at'] = time.monotonic()
        done.set()

    lane.put(IOTask(action=mark))
    lane.protocol_start()
    release.set()
    assert lane.wait_for_idle(_WAIT_S)
    assert not done.is_set(), 'the run served a task put before it'
    return marks, done


def test_a_task_left_waiting_through_a_run_starts_at_once_when_the_run_ends(lane):
    marks, done = _waiting_through_a_run(lane)
    switched_at = time.monotonic()
    lane.protocol_end()
    assert done.wait(_WAIT_S), 'the task never started'
    assert marks['at'] - switched_at < _PROMPT_S


def test_a_task_left_waiting_through_a_run_starts_at_once_when_the_run_finishes(lane):
    marks, done = _waiting_through_a_run(lane)
    switched_at = time.monotonic()
    lane.protocol_finish_then_end()
    assert done.wait(_WAIT_S), 'the task never started'
    assert marks['at'] - switched_at < _PROMPT_S
    assert not lane.is_protocol_running()


def test_a_close_returns_at_once(lane):
    _settle(lane)
    closed_at = time.monotonic()
    lane.shutdown()
    assert time.monotonic() - closed_at < _PROMPT_S
    assert not lane.worker_alive
