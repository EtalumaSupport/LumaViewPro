# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The protocol-drain predicate never reports idle with work in flight.

``is_protocol_queue_active()`` is what every drain waiter asks before
declaring a run's files written: the hyperstack build, the run-start
``files_writing`` refusal, the wedged-writer recovery offer, and the GUI's
completion poll all gate on it. It answers from two facts -- the queue is
non-empty, or a protocol task is the running task.

Between those two facts sits a window. The worker takes a task off the
queue and only later records it as the running task; in between, a
single remaining task belongs to neither fact and the predicate answers
idle while that task's write has not happened yet. A waiter that samples
there proceeds early: the stack build reads a directory missing its last
plane, and the next run's refusal gate lets a run start on top of a write
still in flight.

The predicate is therefore conservative by construction -- the worker
claims a task as it takes it, so "in flight" covers dequeued-but-not-yet
-started as well as running. The cost is that a waiter can be held for at
most one task's runtime longer than strictly necessary, which is the
honest direction to err.
"""

import threading
import time

import pytest

from modules.sequential_io_executor import IOTask, SequentialIOExecutor


class _ProbingTask(IOTask):
    """An IOTask that samples a probe when the worker wires its dispatcher.

    The worker sets ``_ui_dispatch`` on a task it has already taken off
    the queue and is about to run -- the exact moment a drain waiter must
    not be told the queue is idle.
    """

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == '_ui_dispatch':
            probe = self.__dict__.get('_drain_probe')
            if probe is not None:
                probe()


@pytest.fixture
def executor():
    ex = SequentialIOExecutor(name='DRAIN_PREDICATE')
    ex.start()
    yield ex
    # Unguarded on purpose: a teardown that swallows a shutdown failure
    # hides exactly the worker-wedge these tests exist to detect.
    ex.shutdown()


def _run_one_probed_protocol_task(executor):
    """Run a single protocol task, sampling the predicate mid-handoff."""
    observations = []
    ran = threading.Event()

    task = _ProbingTask(action=ran.set)
    task.__dict__['_drain_probe'] = lambda: observations.append(executor.is_protocol_queue_active())

    executor.protocol_start()
    executor.protocol_put(task)
    assert ran.wait(timeout=5.0), 'the probed protocol task never ran'
    return observations


class TestDrainPredicateCoversTheHandoff:
    def test_predicate_is_true_while_a_dequeued_task_waits_to_start(self, executor):
        observations = _run_one_probed_protocol_task(executor)

        assert observations, 'the probe never fired; the worker handoff moved'
        assert all(observations), (
            'is_protocol_queue_active() reported idle while the last protocol '
            'task was already off the queue and about to run. Every drain '
            'waiter that samples here proceeds on an incomplete run: the '
            'hyperstack build stacks a directory missing its final plane, and '
            "the next run's files_writing gate lets a run start over a write "
            f'still in flight. Samples: {observations}'
        )

    def test_predicate_is_true_while_a_protocol_task_runs(self, executor):
        """The half that already held -- kept so the fix cannot trade one
        window for the other."""
        release = threading.Event()
        observed = []

        def _blocking_action():
            observed.append(executor.is_protocol_queue_active())
            release.wait(timeout=5.0)

        executor.protocol_start()
        executor.protocol_put(IOTask(action=_blocking_action))
        deadline = time.monotonic() + 5.0
        while not observed and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()

        assert observed == [True], f'expected a busy predicate while running; got {observed}'

    def test_predicate_is_false_once_the_queue_has_drained(self, executor):
        """The other polarity: the fix must not wedge the predicate true,
        which would hang every waiter instead of releasing it early."""
        ran = threading.Event()
        executor.protocol_start()
        executor.protocol_put(IOTask(action=ran.set))
        assert ran.wait(timeout=5.0)

        deadline = time.monotonic() + 5.0
        while executor.is_protocol_queue_active() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not executor.is_protocol_queue_active(), (
            'the predicate stayed true after the queue drained; a claim that '
            'is never released hangs every drain waiter forever'
        )
