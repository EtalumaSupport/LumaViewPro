# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the selective live-frame backpressure on the default
SequentialIOExecutor queue.

The default queue is unbounded, and an IOTask holds its args/kwargs (and thus
any frame ndarray) by reference -- so a single worker falling behind could pin
GBs of ~3.5 MB frame buffers (the manual-record RAM balloon). The fix bounds
ONLY frame-carrying tasks (tagged droppable_live), latest-wins, with drop
accounting; must-execute tasks (config / motor / save) are never dropped.

These tests lock in:
  * put() drops droppable_live tasks over the cap (the backstop) and never
    drops must-execute tasks,
  * the worker frees an in-flight slot when it dequeues a droppable task.

The producer-side drop-before-reserve gate that once sat in front of put()
is gone with the caller that needed it: it existed so a producer holding a
reserved slot could decide to drop BEFORE producing, and the only such
producer was the memmap record path the engine replaced.
"""

import threading
import time

from modules.sequential_io_executor import (
    ENQUEUED,
    LIVE_FRAME_DROPPED,
    _LIVE_FRAME_MAXSIZE,
    IOTask,
    SequentialIOExecutor,
)


def _ex():
    # Constructed but NOT started: no worker drains, so put()/inflight state is
    # observable. _disable defaults False and protocol_running is clear, so
    # put() enqueues.
    return SequentialIOExecutor(name='TEST')


def test_put_bounds_droppable_live_frames():
    ex = _ex()
    for _ in range(_LIVE_FRAME_MAXSIZE):
        assert ex.put(IOTask(lambda: None, droppable_live=True)) is ENQUEUED
    assert ex._live_inflight == _LIVE_FRAME_MAXSIZE
    assert ex.queue.qsize() == _LIVE_FRAME_MAXSIZE

    # One more over the cap -> dropped (backstop), not enqueued.
    assert ex.put(IOTask(lambda: None, droppable_live=True)) is LIVE_FRAME_DROPPED
    assert ex._live_inflight == _LIVE_FRAME_MAXSIZE
    assert ex.queue.qsize() == _LIVE_FRAME_MAXSIZE
    assert ex._live_dropped_count == 1


def test_must_execute_tasks_are_never_bounded_or_dropped():
    ex = _ex()
    n = _LIVE_FRAME_MAXSIZE + 10
    for _ in range(n):
        ex.put(IOTask(lambda: None))  # droppable_live defaults False
    assert ex._live_inflight == 0  # never touched by must-execute tasks
    assert ex.queue.qsize() == n  # all enqueued, none dropped
    assert ex._live_dropped_count == 0


def test_worker_frees_inflight_slot_on_dequeue():
    ex = _ex()
    ex.start()
    try:
        ran = threading.Event()
        assert ex.put(IOTask(lambda: ran.set(), droppable_live=True)) is ENQUEUED
        assert ran.wait(2.0)  # the task actually ran
        # The decrement happens right after get(), before run; allow a beat.
        deadline = time.monotonic() + 1.0
        while ex._live_inflight != 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ex._live_inflight == 0
    finally:
        ex.shutdown(wait=True)
