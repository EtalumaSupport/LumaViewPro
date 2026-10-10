# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A finished run's stalled file writer is reported by the API, once per stall, with its remedy.

Nobody is waiting on a run's files once the run has ended, so a writer
stuck on one of them was told only to the GUI, by its own drain tick, in
words of its own. A REST or headless client heard nothing until its next
run was refused. The run engine now watches the draining batch on the
session's scheduler and reports the stall itself, as a fault carrying the
recover-file-writer remedy -- the same remedy and words the run refusal
carries. A declined offer is not repeated for the same stuck write; a
different write found stuck is a new stall.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from modules.exceptions import FileWriterStalledError, ProtocolRunRefusedError
from modules.image_mode import ImageCaptureConfig
from modules.notification_center import OutcomeKind
from modules.protocol_image_writer import RunWriteBatch
from modules.scope_session import ScopeSession
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.sequential_io_executor import ENQUEUED
from tests.scope_fakes import real_executor_bundle, spec_scope


class _Scheduler:
    """Holds what was scheduled; the test ticks it."""

    def __init__(self):
        self.callbacks = []

    def schedule_interval(self, callback, interval_s):
        self.callbacks.append(callback)
        return callback

    def unschedule(self, handle):
        self.callbacks.remove(handle)

    def tick(self):
        for callback in list(self.callbacks):
            callback()


@pytest.fixture
def session():
    bundle = real_executor_bundle(file_io_executor=MagicMock())
    session = ScopeSession(settings={}, scope=spec_scope(), executor_bundle=bundle)
    yield session
    # The planted writes went to a scripted lane that never runs them, so
    # nothing would ever land them; the session cannot close until they settle.
    batch = session.sequenced_capture_runner.write_batch()
    if batch is not None:
        batch.abandon('the test is over')
        if batch.outcome is None:
            # Planted as a live run's: end it as the run's cleanup would.
            batch.close()


@pytest.fixture
def scheduler(session):
    scheduler = _Scheduler()
    session.sequenced_capture_runner.start_file_writer_check(scheduler)
    return scheduler


def _batch(
    session, *, stalled: bool, writes: int = 3, closed: bool = True, task: object | None = None
) -> RunWriteBatch:
    """The last run handed *writes* to the file lane; *closed* once the run has ended.

    *task* is the write in flight on the lane; a new one unless given.
    """
    lane = session.file_io_executor
    lane.put.return_value = ENQUEUED
    lane.in_flight_task_stalled.return_value = stalled
    lane.describe_running_task.return_value = "write_capture 'B2_BF' 45s in flight"
    lane.running_task = object() if task is None else task
    batch = RunWriteBatch(lane)
    for i in range(writes):
        batch.submit(lambda: None, {}, what=f'The image {i}', pace_until=None)
    if closed:
        batch.close()
    session.sequenced_capture_runner._write_batch = batch
    return batch


def _stalls(centre_posts):
    return [n for n in centre_posts if n.reason == 'files_writing_stalled']


class TestTheStallIsReported:
    @pytest.mark.slow
    def test_once_as_a_fault_carrying_its_remedy(self, session, centre_posts, scheduler):
        _batch(session, stalled=True)

        scheduler.tick()
        scheduler.tick()

        stalls = _stalls(centre_posts)
        assert len(stalls) == 1, 'one stall is one report, not one per tick'
        stall = stalls[0]
        assert (stall.kind, stall.shown, stall.solicited) == (OutcomeKind.FAULT, True, False)
        assert stall.title == 'File Writer Stalled'
        assert stall.remedy is not None and stall.remedy.member == 'recover_file_writer'
        assert '3 unsaved image(s)' in stall.message
        assert "write_capture 'B2_BF'" in stall.message

    @pytest.mark.slow
    def test_a_different_stuck_write_is_a_new_stall(self, session, centre_posts, scheduler):
        _batch(session, stalled=True)
        scheduler.tick()

        session.file_io_executor.running_task = object()
        scheduler.tick()

        assert len(_stalls(centre_posts)) == 2

    @pytest.mark.slow
    def test_a_new_run_s_batch_starts_unreported(self, session, centre_posts, scheduler):
        _batch(session, stalled=True)
        scheduler.tick()

        # The same write still in flight: only the batch is new.
        _batch(session, stalled=True, task=session.file_io_executor.running_task)
        scheduler.tick()

        assert len(_stalls(centre_posts)) == 2


class TestNothingIsReported:
    @pytest.mark.slow
    def test_while_the_writer_is_moving(self, session, centre_posts, scheduler):
        _batch(session, stalled=False)
        scheduler.tick()
        assert _stalls(centre_posts) == []

    def test_while_the_run_is_live(self, session, centre_posts, scheduler):
        # A live run's own writes answer for a stuck writer; the drain has not begun.
        _batch(session, stalled=True, closed=False)
        scheduler.tick()
        assert _stalls(centre_posts) == []

    def test_with_no_run_yet(self, session, centre_posts, scheduler):
        scheduler.tick()
        assert _stalls(centre_posts) == []


@pytest.mark.slow
def test_the_refusal_and_the_report_offer_one_remedy_in_one_set_of_words(
    session, centre_posts, scheduler, tmp_path
):
    _batch(session, stalled=True)
    scheduler.tick()
    report = _stalls(centre_posts)[0]

    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.sequenced_capture_runner.prepare(
            protocol=MagicMock(),
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='t',
            image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
            autogain_settings={},
            parent_dir=tmp_path,
        )

    assert refused.value.remedy == report.remedy
    cost = FileWriterStalledError.cost_sentence(3)
    assert cost in refused.value.message and cost in report.message
