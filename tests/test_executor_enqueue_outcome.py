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

import threading
import time
from unittest.mock import patch

from modules.sequential_io_executor import (
    ENQUEUED,
    IOTask,
    SequentialIOExecutor,
)


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


def _warnings(mock_logger):
    return ' '.join(str(c) for c in mock_logger.warning.call_args_list)


def _episode_lines(mock_logger):
    """Only the episode-narration warnings.

    The executor emits unrelated warnings on these paths -- notably the
    protocol queue-depth warning, which fires every 10 tasks past depth 20
    whenever the worker is not draining. Counting every warning would measure
    those instead of the episode bound under test.
    """
    return [
        str(c)
        for c in mock_logger.warning.call_args_list
        if 'REFUSED' in str(c) or 'Accepting work again' in str(c)
    ]


def test_a_fenced_put_warns_once_and_still_returns_none():
    ex = SequentialIOExecutor(name='TEST')
    ex.protocol_start()
    with patch('modules.sequential_io_executor.logger') as mock_logger:
        assert ex.put(IOTask(action=lambda: None)) is None
        assert len(_episode_lines(mock_logger)) == 1
        warned = _warnings(mock_logger)
    assert 'REFUSED' in warned and 'fenced' in warned
    assert 'TEST' in warned


def test_protocol_put_with_no_run_in_session_warns_and_returns_none():
    ex = SequentialIOExecutor(name='TEST')
    with patch('modules.sequential_io_executor.logger') as mock_logger:
        assert ex.protocol_put(IOTask(action=lambda: None)) is None
        assert len(_episode_lines(mock_logger)) == 1
        assert 'no protocol run is in session' in _warnings(mock_logger)


def test_a_sustained_refusal_does_not_inflate_the_log():
    """The case that killed the per-drop design: a disabled lane under a
    sustained submitter emits a bounded number of lines, not one per task."""
    ex = SequentialIOExecutor(name='TEST')
    ex.disable()
    with patch('modules.sequential_io_executor.logger') as mock_logger:
        for _ in range(5000):
            assert ex.put(IOTask(action=lambda: None)) is None
        lines = _episode_lines(mock_logger)
        assert len(lines) == 1, f'{len(lines)} lines for 5000 refusals'


def test_the_episode_closes_with_a_count_when_work_is_accepted_again():
    ex = SequentialIOExecutor(name='TEST')
    ex.disable()
    for _ in range(7):
        ex.put(IOTask(action=lambda: None))
    ex.enable()
    ex.start()
    try:
        with patch('modules.sequential_io_executor.logger') as mock_logger:
            ran = threading.Event()
            assert ex.put(IOTask(action=ran.set)) is ENQUEUED
            assert ran.wait(2.0)
            warned = _warnings(mock_logger)
            assert len(_episode_lines(mock_logger)) == 1
    finally:
        ex.shutdown()
    assert '7 task(s) were refused' in warned


def test_the_two_lanes_do_not_close_each_others_episodes():
    """A run fences the default lane while the protocol lane accepts, so the
    two interleave. A shared tracker would reopen an episode per submit."""
    ex = SequentialIOExecutor(name='TEST')
    ex.protocol_start()
    with patch('modules.sequential_io_executor.logger') as mock_logger:
        for _ in range(50):
            assert ex.put(IOTask(action=lambda: None)) is None  # fenced
            ex.protocol_put(IOTask(action=lambda: None))  # accepted
        lines = _episode_lines(mock_logger)
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


def test_a_task_queued_before_disable_still_runs_and_the_lane_goes_idle():
    """A run closes the camera lane and then waits for it to go idle; what the
    lane already held must run to completion, not sit parked until the run
    ends holding its caller with it."""
    ex = SequentialIOExecutor(name='TEST')
    ex.start()
    release = threading.Event()
    ran = threading.Event()
    try:
        ex.put(IOTask(action=lambda: release.wait(5.0)))
        ex.put(IOTask(action=ran.set))
        ex.disable()
        assert ex.is_busy(), 'the lane reported idle with two tasks on it'
        release.set()
        assert ran.wait(5.0), 'a task queued before disable() never ran'
        deadline = time.monotonic() + 5.0
        while ex.is_busy() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not ex.is_busy(), 'the lane never went idle after draining'
        assert ex.put(IOTask(action=lambda: None)) is None, (
            'a new submit was accepted while disabled'
        )
    finally:
        ex.enable()
        ex.shutdown()
