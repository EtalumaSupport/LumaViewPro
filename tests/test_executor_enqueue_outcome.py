# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the enqueue-outcome contract of the default queue.

`put` returned None both for a successful fire-and-forget enqueue and for a
task the executor refused (disabled, or fenced by a running protocol), so a
caller could not tell the two apart. `IlluminationAPI._submit_io` is the one
consumer that must: it logged '<name> dropped: the io executor is not
accepting work' and returned False on EVERY successful async LED submit. On
hardware the warning fired 109 times in one session while the I2C writes for
those same commands went out milliseconds later.

These tests lock the outcome vocabulary in: enqueued is ENQUEUED, refused is
None, and a caller asking for a waiter still gets a waiter.
"""

import threading
from unittest.mock import patch

import pytest

from modules.sequential_io_executor import (
    ENQUEUED,
    LIVE_FRAME_DROPPED,
    IOTask,
    SequentialIOExecutor,
)


@pytest.fixture
def led_executors(sim_scope):
    """Real started executors registered on the scope, as production has.

    Real SequentialIOExecutors rather than doubles: the point of these tests
    is what put() actually returns, which a double would only assert about
    itself.
    """
    io = SequentialIOExecutor(name='TEST_IO')
    camera = SequentialIOExecutor(name='TEST_CAMERA')
    file_io = SequentialIOExecutor(name='TEST_FILE')
    for ex in (io, camera, file_io):
        ex.start()
    sim_scope.register_executors(camera_executor=camera, io_executor=io, file_io_executor=file_io)
    yield {'io': io, 'camera': camera, 'file_io': file_io}
    for ex in (io, camera, file_io):
        ex.shutdown()


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


def test_submit_io_does_not_report_a_successful_led_submit_as_dropped(sim_scope, led_executors):
    """The success half of the drop contract.

    The drop half -- that a refused submit DOES warn -- is already pinned by
    test_dispatch_contract.test_async_warns_when_the_executor_drops_it.

    Asserts through a patched module logger rather than caplog: conftest
    installs lvp_logger as a MagicMock, so `from lvp_logger import logger`
    never produces logging records and a caplog assertion would pass whether
    or not the warning fired.
    """
    sub = sim_scope.illumination
    module = type(sub).__module__
    with patch(f'{module}.logger') as mock_logger:
        sub.leds_off_async()
        warned = ' '.join(str(c) for c in mock_logger.warning.call_args_list)
    assert 'dropped' not in warned, (
        f'a successful leds_off_async submit was reported as dropped: {warned!r}'
    )


def test_submit_io_treats_a_truthy_refusal_as_a_refusal(sim_scope, led_executors):
    """Success is an ENQUEUED identity check, not `is not None`.

    put() has two refusal values and only one of them is falsy: a task refused
    by the live-frame cap comes back as LIVE_FRAME_DROPPED, which is truthy.
    A consumer testing the negative would report that refusal as a success.
    """
    sub = sim_scope.illumination
    module = type(sub).__module__
    with (
        patch.object(led_executors['io'], 'put', return_value=LIVE_FRAME_DROPPED),
        patch(f'{module}.logger') as mock_logger,
    ):
        sub.leds_off_async()
        warned = ' '.join(str(c) for c in mock_logger.warning.call_args_list)
    assert 'dropped' in warned, (
        f'a refused submit was reported as delivered; warnings seen: {warned!r}'
    )
