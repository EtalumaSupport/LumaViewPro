# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for protocol-mode teardown on the SequentialIOExecutor.

While ``protocol_running`` is set, the executor worker pulls only from
``protocol_queue`` and ``put`` rejects normal-queue tasks. If a protocol aborts
or tears down without the normal completion path ending the executor, it stays
stuck in protocol-mode forever: the worker blocks on ``protocol_queue.get`` and
every normal file op (composite, video, z-projection, manual save) is refused,
so the post-processing operation hangs with its progress popup stuck open.

``end_protocol_mode()`` is the idempotent safety net -- it drains any remaining
protocol items then returns the executor to normal-queue service. The run's
own cleanup calls it on every path out, before the run ends.
"""

import threading
import time

from modules.sequential_io_executor import IOTask, SequentialIOExecutor


def test_end_protocol_mode_restores_normal_queue_service():
    """Stuck-in-protocol-mode refuses normal file ops; the safety net exits
    protocol-mode so normal-queue service resumes and tasks run again."""
    ex = SequentialIOExecutor(name='TEST')
    ex.start()
    try:
        ex.protocol_start()  # worker serves only protocol_queue; put() refuses

        ran_during = threading.Event()
        ex.put(IOTask(lambda: ran_during.set()))
        assert not ran_during.wait(0.5), 'normal task ran while stuck in protocol-mode'

        ex.end_protocol_mode()

        deadline = time.monotonic() + 2.0
        while ex.protocol_running.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not ex.protocol_running.is_set(), 'protocol_running still set after drain'
        assert not ex.protocol_finish.is_set(), 'protocol_finish still set after drain'

        ran_after = threading.Event()
        ex.put(IOTask(lambda: ran_after.set()))
        assert ran_after.wait(2.0), 'normal-queue service not restored after end_protocol_mode'
    finally:
        ex.shutdown(wait=True)


def test_end_protocol_mode_flushes_queued_protocol_writes():
    """The safety net is graceful: a protocol write already queued when
    end_protocol_mode fires still flushes (it is not dropped) before the
    executor leaves protocol-mode.

    A first blocking task holds the worker so the write under test is sitting
    in the queue -- not racing a concurrent enqueue -- when the safety net
    fires, which mirrors the real teardown (pending writes already queued, no
    producer still running)."""
    ex = SequentialIOExecutor(name='TEST')
    ex.start()
    try:
        ex.protocol_start()

        started = threading.Event()
        release = threading.Event()

        def _block():
            started.set()
            release.wait(2.0)

        ex.protocol_put(IOTask(_block))
        assert started.wait(2.0), 'worker did not pick up the first protocol task'

        wrote = threading.Event()
        ex.protocol_put(IOTask(lambda: wrote.set()))  # queued behind the blocker

        ex.end_protocol_mode()  # signal graceful exit while a write is still queued
        release.set()  # let the blocker finish; worker drains the write

        assert wrote.wait(2.0), 'queued protocol write was dropped instead of flushed'

        deadline = time.monotonic() + 2.0
        while ex.protocol_running.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not ex.protocol_running.is_set()
    finally:
        ex.shutdown(wait=True)


def test_end_protocol_mode_is_noop_when_not_in_protocol_mode():
    """Safe to call on every teardown: a no-op when not in protocol-mode."""
    ex = SequentialIOExecutor(name='TEST')  # not started, protocol_running clear
    ex.end_protocol_mode()
    assert not ex.protocol_running.is_set()
    assert not ex.protocol_finish.is_set()
