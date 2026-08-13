# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests: a task carries its executor's name before it is queued.

IOTask.name starts empty and the adopting executor fills it in -- the caller
constructing the task cannot supply it, because it does not know which
executor will run it. That leaves a window, and the executor used to close it
one statement too late: the task went onto the queue first and was named
after. A single worker drains that queue continuously, so it can dequeue and
start running a task in between, and IOTask.run's first act is to rename its
own thread to task.name -- the empty string.

The cost lands on the one report allowed to say an exception escaped its
thread: lvp_logger's threading.excepthook prints args.thread.name, so an
unnamed worker degrades exactly the line everything else defers to.

All three enqueue paths are covered; each names the task at entry, before any
guard can return early and before any queue insertion.
"""

import queue

from modules.sequential_io_executor import IOTask, SequentialIOExecutor

EXPECTED_NAME = 'TEST_WORKER'


class _NamingSpyQueue:
    """Wraps a real queue and records task.name at the moment of insertion."""

    def __init__(self, real):
        self._real = real
        self.names_at_insert = []

    def put(self, task, *args, **kwargs):
        self.names_at_insert.append(task.name)
        return self._real.put(task, *args, **kwargs)

    def put_nowait(self, task):
        self.names_at_insert.append(task.name)
        return self._real.put_nowait(task)

    def __getattr__(self, attr):
        return getattr(self._real, attr)


def _executor_with_spy(attr: str) -> tuple[SequentialIOExecutor, _NamingSpyQueue]:
    executor = SequentialIOExecutor(name='TEST')
    spy = _NamingSpyQueue(getattr(executor, attr))
    setattr(executor, attr, spy)
    return executor, spy


def test_default_queue_task_is_named_before_insertion():
    executor, spy = _executor_with_spy('queue')
    executor.put(IOTask(lambda: None))
    assert spy.names_at_insert == [EXPECTED_NAME], (
        'The task reached the queue unnamed; a worker dequeuing it here would '
        f'rename its thread to the empty string. Saw {spy.names_at_insert!r}'
    )


def test_protocol_queue_task_is_named_before_insertion():
    executor, spy = _executor_with_spy('protocol_queue')
    executor.protocol_running.set()
    executor.protocol_put(IOTask(lambda: None))
    assert spy.names_at_insert == [EXPECTED_NAME], (
        f'protocol_put queued an unnamed task; saw {spy.names_at_insert!r}'
    )


def test_blocking_protocol_queue_task_is_named_before_insertion():
    executor, spy = _executor_with_spy('protocol_queue')
    executor.protocol_running.set()
    executor.protocol_put_wait(
        IOTask(lambda: None), should_abort=lambda: False, stall_timeout_s=1.0
    )
    assert spy.names_at_insert == [EXPECTED_NAME], (
        f'protocol_put_wait queued an unnamed task; saw {spy.names_at_insert!r}'
    )


def test_name_is_assigned_even_when_the_enqueue_is_refused():
    """Naming at entry, not at insertion, is what makes the window closed.

    A task refused by a guard is never queued, so this asserts the placement
    rather than the queue state: a later reader moving the call back down to
    the insertion sites reintroduces the race these tests exist to stop.
    """
    executor = SequentialIOExecutor(name='TEST')
    executor.disable()
    task = IOTask(lambda: None)
    assert executor.put(task) is None
    assert task.name == EXPECTED_NAME


def test_spy_queue_forwards_to_the_real_queue():
    """Guards the harness: a spy that silently swallowed puts would make
    every assertion above pass against a broken executor."""
    executor, spy = _executor_with_spy('queue')
    executor.put(IOTask(lambda: None))
    assert isinstance(spy._real, queue.Queue)
    assert spy._real.qsize() == 1
