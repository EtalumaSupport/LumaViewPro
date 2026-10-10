# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A post-processing build never sits in front of a run's image writes.

A run paces its writes on the file lane: with its backlog full, a capture
waits for room, and once it has waited past the stall threshold while the
task in flight has run past it too, the run judges its writer stuck --
the frame is counted lost and the run aborts. The lane cannot tell a
run's write from anything else it runs, so a stitch of a large folder on
that lane read as a wedged writer. The builds now run on their own lane.

The first test shows the hazard is real on a shared lane, so the second
is known able to fail.
"""

from __future__ import annotations

import threading
import time

import pytest

import modules.protocol_image_writer as piw
from modules.sequential_io_executor import IOTask, SequentialIOExecutor

STALL_S = 0.5
BOUND = 4


@pytest.fixture
def fast_pacing(monkeypatch):
    monkeypatch.setattr(piw, 'WRITE_STALL_FATAL_S', STALL_S)
    monkeypatch.setattr(piw, 'WRITE_BACKLOG_BOUND', BOUND)
    monkeypatch.setattr(piw, '_PACING_POLL_S', 0.02)
    monkeypatch.setattr(SequentialIOExecutor, '_stall_threshold_s', lambda self, floor_s: floor_s)


def _hold(lane, release: threading.Event) -> None:
    """Occupy *lane*'s worker until *release*, the way a long build does."""
    started = threading.Event()

    def build():
        started.set()
        release.wait(10.0)

    lane.put(IOTask(action=build))
    assert started.wait(5.0)


def _paced_write(file_lane):
    """Fill a run's backlog to its bound, then make one paced write; its answer."""
    batch = piw.RunWriteBatch(file_lane)
    for i in range(BOUND):
        batch.submit(lambda: None, {}, what=f'frame {i}', pace_until=lambda: False)
    return batch.submit(lambda: None, {}, what='paced frame', pace_until=lambda: False)


def test_a_long_task_sharing_the_file_lane_wedges_a_runs_paced_write(fast_pacing):
    file_lane = SequentialIOExecutor(name='FILE_SHARED')
    file_lane.start()
    release = threading.Event()
    try:
        _hold(file_lane, release)
        time.sleep(STALL_S)
        assert _paced_write(file_lane) is piw.WRITER_WEDGED
    finally:
        release.set()
        file_lane.shutdown(wait=False)


def test_a_build_on_the_post_processing_lane_leaves_a_runs_writes_free(fast_pacing):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(), simulate=True)
    release = threading.Event()
    try:
        assert session.post_processing.lane is not session.file_io_executor
        _hold(session.post_processing.lane, release)
        time.sleep(STALL_S)
        started = time.monotonic()
        answer = _paced_write(session.file_io_executor)
        assert answer is not piw.WRITER_WEDGED
        assert time.monotonic() - started < STALL_S, 'the run waited behind the build'
    finally:
        release.set()
        session.shutdown()
