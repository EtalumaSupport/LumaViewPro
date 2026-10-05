# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every run is torn down exactly once, by the thread that owns it.

The run loop's body decides how the run ended and returns it; the loop's
finally is the one teardown. Before that, a normal run reached its
teardown three times and two guards turned the extra passes away. Pinned
here for each way a run ends: it completes, it is stopped, the loop
crashes, a fault ends it.
"""

import threading
import time

from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    COMPLETION_TIMEOUT,
    executor,
    executors,
    scope,
)
from tests.test_run_teardown_authority import _start_run


def _count_teardowns(executor, monkeypatch):
    teardowns = []
    teardown = executor._cleanup_inner

    def counted(ending, run):
        teardowns.append(ending)
        return teardown(ending, run)

    monkeypatch.setattr(executor, '_cleanup_inner', counted)
    return teardowns


def _ended(executor, run):
    outcome = run.wait(timeout_s=COMPLETION_TIMEOUT)
    assert outcome is not None, 'the run never reported'
    assert executor.wait_for_run_idle(COMPLETION_TIMEOUT), 'the run never went idle'
    deadline = time.monotonic() + COMPLETION_TIMEOUT
    while executor.protocol_thread.is_running:
        assert time.monotonic() < deadline, 'the run loop never finished'
        time.sleep(0.01)
    return outcome


def test_a_completed_run_is_torn_down_once(executor, tmp_path, monkeypatch):
    teardowns = _count_teardowns(executor, monkeypatch)
    run = _start_run(executor, tmp_path, threading.Event())

    outcome = _ended(executor, run)

    assert outcome.status == 'completed'
    assert len(teardowns) == 1


def test_a_stopped_run_is_torn_down_once(executor, tmp_path, monkeypatch):
    teardowns = _count_teardowns(executor, monkeypatch)
    run = _start_run(executor, tmp_path, threading.Event())

    executor._reset(run)
    outcome = _ended(executor, run)

    assert (outcome.status, outcome.reason) == ('aborted', 'stopped')
    assert len(teardowns) == 1


def test_a_run_whose_loop_crashed_is_torn_down_once(executor, tmp_path, monkeypatch):
    teardowns = _count_teardowns(executor, monkeypatch)

    def crash(run):
        raise RuntimeError('the loop died here')

    monkeypatch.setattr(executor._run_loop_executor, '_run_loop_inner', crash)
    run = _start_run(executor, tmp_path, threading.Event())

    outcome = _ended(executor, run)

    assert (outcome.status, outcome.reason) == ('failed', 'run_loop_crashed')
    assert len(teardowns) == 1


def test_a_run_a_fault_ended_is_torn_down_once(executor, scope, tmp_path, monkeypatch):
    teardowns = _count_teardowns(executor, monkeypatch)

    unplugged = threading.Event()
    connected = executor._scope.are_all_connected

    def the_move_fails(*args, **kwargs):
        unplugged.set()
        raise RuntimeError('the stage did not answer')

    monkeypatch.setattr(executor._step_executor, 'go_to_step', the_move_fails)
    monkeypatch.setattr(
        executor._scope, 'are_all_connected', lambda: not unplugged.is_set() and connected()
    )
    run = _start_run(executor, tmp_path, threading.Event())

    outcome = _ended(executor, run)

    assert (outcome.status, outcome.reason) == ('failed', 'hardware_disconnected')
    assert len(teardowns) == 1
