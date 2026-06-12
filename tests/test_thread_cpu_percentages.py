# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for common_utils.thread_cpu_percentages.

Locks in the per-thread CPU sampler that feeds the [THREAD CPU] metrics
block. The sampler is driven through an injected fake psutil-process so the
rate math is deterministic on every platform -- macOS psutil only reports a
single aggregate thread, so real per-thread enumeration can't be exercised
here. Behaviors pinned:

  * first call returns {} (no baseline yet) but records state
  * a thread with a CPU delta reports delta / interval as a percentage
  * OS thread ids map to Python thread names via native_id
  * same-named threads collapse into one summed label
  * exited threads are evicted from the module-level state cache
  * a sub-0.5s interval is skipped (divide-by-near-zero guard)
"""

import threading

import pytest

from modules import common_utils


class _FakePthread:
    """Mirrors the psutil pthread namedtuple shape used by the sampler."""

    def __init__(self, tid, user_time, system_time):
        self.id = tid
        self.user_time = user_time
        self.system_time = system_time


class _FakeProc:
    """Stands in for psutil.Process -- returns a scripted threads() list."""

    def __init__(self, threads):
        self._threads = threads

    def threads(self):
        return self._threads


@pytest.fixture(autouse=True)
def _clear_state():
    common_utils._thread_cpu_state.clear()
    yield
    common_utils._thread_cpu_state.clear()


def test_first_call_returns_empty_but_records_baseline():
    proc = _FakeProc([_FakePthread(100, 1.0, 0.0)])
    assert common_utils.thread_cpu_percentages(proc=proc, now=0.0) == {}
    # Baseline is cached so the next call can compute a delta.
    assert 100 in common_utils._thread_cpu_state


def test_delta_over_interval_is_percentage():
    proc1 = _FakeProc([_FakePthread(100, 1.0, 0.0)])
    common_utils.thread_cpu_percentages(proc=proc1, now=0.0)  # baseline
    # +0.5s CPU over a 1.0s wall interval -> 50% of one core.
    proc2 = _FakeProc([_FakePthread(100, 1.4, 0.1)])
    result = common_utils.thread_cpu_percentages(proc=proc2, now=1.0)
    assert result == {'tid_100': pytest.approx(50.0)}


def test_thread_id_maps_to_python_name():
    # The thread must stay ALIVE during sampling so threading.enumerate()
    # still resolves its native_id -> name. Park it on an Event.
    gate = threading.Event()
    real = threading.Thread(target=gate.wait, name='named_worker')
    real.start()
    try:
        tid = real.native_id
        proc1 = _FakeProc([_FakePthread(tid, 0.0, 0.0)])
        common_utils.thread_cpu_percentages(proc=proc1, now=0.0)
        proc2 = _FakeProc([_FakePthread(tid, 0.5, 0.0)])
        result = common_utils.thread_cpu_percentages(proc=proc2, now=1.0)
    finally:
        gate.set()
        real.join()
    assert 'named_worker' in result
    assert result['named_worker'] == pytest.approx(50.0)


def test_same_named_threads_collapse_into_one_label():
    gate = threading.Event()
    a = threading.Thread(target=gate.wait, name='dup')
    b = threading.Thread(target=gate.wait, name='dup')
    a.start(); b.start()
    try:
        tid_a, tid_b = a.native_id, b.native_id
        proc1 = _FakeProc([_FakePthread(tid_a, 0.0, 0.0), _FakePthread(tid_b, 0.0, 0.0)])
        common_utils.thread_cpu_percentages(proc=proc1, now=0.0)
        # Each adds 0.3s over 1.0s -> 30% apiece -> 60% summed under "dup".
        proc2 = _FakeProc([_FakePthread(tid_a, 0.3, 0.0), _FakePthread(tid_b, 0.3, 0.0)])
        result = common_utils.thread_cpu_percentages(proc=proc2, now=1.0)
    finally:
        gate.set()
        a.join(); b.join()
    assert list(result.keys()) == ['dup']
    assert result['dup'] == pytest.approx(60.0)


def test_exited_threads_evicted_from_state():
    proc1 = _FakeProc([_FakePthread(100, 1.0, 0.0), _FakePthread(200, 1.0, 0.0)])
    common_utils.thread_cpu_percentages(proc=proc1, now=0.0)
    assert 100 in common_utils._thread_cpu_state
    assert 200 in common_utils._thread_cpu_state
    # tid 200 is gone on the next sample -> evicted.
    proc2 = _FakeProc([_FakePthread(100, 1.0, 0.0)])
    common_utils.thread_cpu_percentages(proc=proc2, now=1.0)
    assert 100 in common_utils._thread_cpu_state
    assert 200 not in common_utils._thread_cpu_state


def test_subsecond_interval_skipped():
    proc1 = _FakeProc([_FakePthread(100, 1.0, 0.0)])
    common_utils.thread_cpu_percentages(proc=proc1, now=0.0)
    proc2 = _FakeProc([_FakePthread(100, 5.0, 0.0)])
    # 0.2s interval is below the 0.5s guard -> no sample emitted.
    result = common_utils.thread_cpu_percentages(proc=proc2, now=0.2)
    assert result == {}
