# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A lane keeps nothing of a finished task once its caller has the outcome.

A lane's worker kept its last task's result and exception as locals through
the wait for the next task, and each calling thread's cached waiter kept them
until that thread's next lane call. A failure keeps every frame of its
traceback alive, so a failed stitch's tiles and stitched image, 62 MB for a
3x3 one, stayed held until the post-processing lane next had work.

Each test sends one task and no other: the object it made is freed once the
caller has let go of the outcome, with nothing after it to push it out.
"""

from __future__ import annotations

import threading
import weakref

import pytest

from _lumascope.modules.lumascope_api._lumascope import scope_lane_closed
from _lumascope.modules.sequential_io_executor import IOTask, SequentialIOExecutor


class _Tile:
    """Stands for a build's data: held by the task's frame, or returned by it."""


def _freed(obj) -> threading.Event:
    gone = threading.Event()
    weakref.finalize(obj, gone.set)
    return gone


@pytest.fixture
def lane():
    lane = SequentialIOExecutor(closed_answer=scope_lane_closed, name='KEEPS_NOTHING_TEST')
    lane.start()
    yield lane
    lane.shutdown()


def test_a_failed_tasks_frames_are_freed_once_its_caller_has_the_failure(lane):
    freed = []

    def fails():
        tile = _Tile()
        freed.append(_freed(tile))
        raise OSError(28, 'No space left on device')

    with pytest.raises(OSError):
        lane.call(IOTask(action=fails, silent_on_failure=True), 'fails', 5.0)

    assert freed[0].wait(5.0), "the failed task's frames are still held"


def test_a_tasks_result_is_freed_once_its_caller_lets_it_go(lane):
    freed = []

    def makes():
        tile = _Tile()
        freed.append(_freed(tile))
        return tile

    tile = lane.call(IOTask(action=makes), 'makes', 5.0)
    del tile

    assert freed[0].wait(5.0), "the task's result is still held"
