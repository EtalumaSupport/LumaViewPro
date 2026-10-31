# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the enqueue-outcome contract of the default queue.

`put` returned None both for a successful fire-and-forget enqueue and for a
task the executor refused (disabled, or fenced by a running protocol), so a
caller could not tell the two apart. The fire-and-forget LED submit, which
had to, logged '<name> dropped: the io executor is not accepting work' and
returned False on EVERY successful LED submit. On
hardware the warning fired 109 times in one session while the I2C writes for
those same commands went out milliseconds later.

These tests lock the outcome vocabulary in: enqueued is ENQUEUED, refused is
None, and a caller asking for a waiter still gets a waiter.
"""

import logging
import threading

import modules.sequential_io_executor as sio
from modules.sequential_io_executor import (
    ENQUEUED,
    IOTask,
    SequentialIOExecutor,
)
from tests.log_capture import records_of


def test_fire_and_forget_success_is_distinguishable_from_a_drop():
    ex = SequentialIOExecutor(name='TEST')
    ex.start()
    try:
        ran = threading.Event()
        assert ex.put(IOTask(action=ran.set)) is ENQUEUED
        assert ran.wait(2.0)  # the task really ran, not just reported enqueued
    finally:
        ex.shutdown()


def test_a_protocol_fence_still_returns_none():
    ex = SequentialIOExecutor(name='TEST')
    # protocol_start fences the default queue: the protocol owns the worker
    # until protocol_finish, and work submitted meanwhile is refused.
    ex.protocol_start()
    assert ex.put(IOTask(action=lambda: None)) is None


def test_a_waiter_caller_still_gets_a_waiter_not_the_sentinel():
    ex = SequentialIOExecutor(name='TEST')
    waiter = ex.put(IOTask(action=lambda: None), return_future=True)
    assert waiter is not ENQUEUED
    assert hasattr(waiter, 'result')


# --- Episode-scoped narration of refused submits -------------------------
#
# The four state-refusal exits returned a bare None in silence while the two
# queue-pressure exits next to them logged. A run refuses for its whole
# duration -- the camera lane is disabled from run start to cleanup -- so
# per-task logging would inflate: the display's auto-gain submit alone lands
# ~3/s, which is ~21,600 refusals across a two-hour scan. These pin the
# episode bound instead: one line when work is first lost, one when the
# executor accepts again, regardless of how many were refused between.


def _warning_lines(caplog):
    return [r.getMessage() for r in records_of(caplog, sio) if r.levelno == logging.WARNING]


def _warnings(caplog):
    return ' '.join(_warning_lines(caplog))


def _episode_lines(caplog):
    """Only the episode-narration warnings.

    The executor emits unrelated warnings on these paths -- notably the
    protocol queue-depth warning, which fires every 10 tasks past depth 20
    whenever the worker is not draining. Counting every warning would measure
    those instead of the episode bound under test.
    """
    return [
        line
        for line in _warning_lines(caplog)
        if 'REFUSED' in line or 'Accepting work again' in line
    ]


def test_a_fenced_put_warns_once_and_still_returns_none(caplog):
    ex = SequentialIOExecutor(name='TEST')
    ex.protocol_start()
    with caplog.at_level(logging.WARNING):
        assert ex.put(IOTask(action=lambda: None)) is None
        assert len(_episode_lines(caplog)) == 1
        warned = _warnings(caplog)
    assert 'REFUSED' in warned and 'fenced' in warned
    assert 'TEST' in warned


def test_protocol_put_with_no_run_in_session_warns_and_returns_none(caplog):
    ex = SequentialIOExecutor(name='TEST')
    with caplog.at_level(logging.WARNING):
        assert ex.protocol_put(IOTask(action=lambda: None)) is None
        assert len(_episode_lines(caplog)) == 1
        assert 'no protocol run is in session' in _warnings(caplog)


def test_a_sustained_refusal_does_not_inflate_the_log(caplog):
    """The case that killed the per-drop design: a fenced lane under a
    sustained submitter emits a bounded number of lines, not one per task."""
    ex = SequentialIOExecutor(name='TEST')
    ex.protocol_start()
    with caplog.at_level(logging.WARNING):
        for _ in range(5000):
            assert ex.put(IOTask(action=lambda: None)) is None
        lines = _episode_lines(caplog)
        assert len(lines) == 1, f'{len(lines)} lines for 5000 refusals'


def test_the_episode_closes_with_a_count_when_work_is_accepted_again(caplog):
    ex = SequentialIOExecutor(name='TEST')
    ex.protocol_start()
    for _ in range(7):
        ex.put(IOTask(action=lambda: None))
    ex.protocol_end()
    ex.start()
    caplog.clear()  # the line opening the episode is not the window's
    try:
        with caplog.at_level(logging.WARNING):
            ran = threading.Event()
            assert ex.put(IOTask(action=ran.set)) is ENQUEUED
            assert ran.wait(2.0)
            warned = _warnings(caplog)
            assert len(_episode_lines(caplog)) == 1
    finally:
        ex.shutdown()
    assert '7 task(s) were refused' in warned


def test_the_two_lanes_do_not_close_each_others_episodes(caplog):
    """A run fences the default lane while the protocol lane accepts, so the
    two interleave. A shared tracker would reopen an episode per submit."""
    ex = SequentialIOExecutor(name='TEST')
    ex.protocol_start()
    with caplog.at_level(logging.WARNING):
        for _ in range(50):
            assert ex.put(IOTask(action=lambda: None)) is None  # fenced
            ex.protocol_put(IOTask(action=lambda: None))  # accepted
        lines = _episode_lines(caplog)
        assert len(lines) == 1, f"the accepting lane reopened the fenced lane's episode: {lines}"


class TestWorkerAlive:
    """A lane with no live worker never services a submission, so a caller
    that would wait on one reads the fact first, publicly."""

    def test_false_before_start_true_while_running_false_after_shutdown(self):
        lane = SequentialIOExecutor(name='LIVENESS_TEST')
        assert lane.worker_alive is False
        lane.start()
        try:
            assert lane.worker_alive is True
        finally:
            lane.shutdown()
        assert lane.worker_alive is False
