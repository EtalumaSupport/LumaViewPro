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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.notification_center as nc
from modules.exceptions import FileWriterStalledError, ProtocolRunRefusedError
from modules.image_mode import ImageCaptureConfig
from modules.notification_center import NotificationCenter, OutcomeKind, Severity
from modules.protocol_image_writer import RunWriteBatch
from modules.scope_session import ScopeSession
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.sequential_io_executor import ENQUEUED
from tests.protocol_drives import autofocus_snapshot
from tests.scope_fakes import spec_scope
from tests.settings_fixtures import complete_settings


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
            callback(0)


@pytest.fixture
def session():
    bundle = SimpleNamespace(
        file_io_executor=MagicMock(),
        post_processing_executor=MagicMock(),
        protocol_thread=MagicMock(),
        shutdown=lambda: None,
    )
    return ScopeSession(settings={}, scope=spec_scope(), executor_bundle=bundle)


@pytest.fixture
def heard(monkeypatch):
    centre = NotificationCenter(dedup_window_s=0)
    posts = []
    centre.add_listener(posts.append, min_severity=Severity.DEBUG)
    monkeypatch.setattr(nc, 'notifications', centre)
    return posts


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
        batch.close(lambda outcome: None)
    session.sequenced_capture_runner._write_batch = batch
    return batch


def _stalls(heard):
    return [n for n in heard if n.reason == 'files_writing_stalled']


class TestTheStallIsReported:
    @pytest.mark.slow
    def test_once_as_a_fault_carrying_its_remedy(self, session, heard, scheduler):
        _batch(session, stalled=True)

        scheduler.tick()
        scheduler.tick()

        stalls = _stalls(heard)
        assert len(stalls) == 1, 'one stall is one report, not one per tick'
        stall = stalls[0]
        assert (stall.kind, stall.shown, stall.solicited) == (OutcomeKind.FAULT, True, False)
        assert stall.title == 'File Writer Stalled'
        assert stall.remedy is not None and stall.remedy.member == 'recover_file_writer'
        assert '3 unsaved image(s)' in stall.message
        assert "write_capture 'B2_BF'" in stall.message

    def test_its_remedy_recovers_the_writer(self, session, heard, scheduler):
        batch = _batch(session, stalled=True)
        scheduler.tick()

        session.apply_remedy(_stalls(heard)[0].remedy)

        assert batch.not_written_reason == 'write_batch_abandoned'
        session.file_io_executor.replace_stuck_worker.assert_called_once()

    @pytest.mark.slow
    def test_a_different_stuck_write_is_a_new_stall(self, session, heard, scheduler):
        _batch(session, stalled=True)
        scheduler.tick()

        session.file_io_executor.running_task = object()
        scheduler.tick()

        assert len(_stalls(heard)) == 2

    @pytest.mark.slow
    def test_a_new_run_s_batch_starts_unreported(self, session, heard, scheduler):
        _batch(session, stalled=True)
        scheduler.tick()

        # The same write still in flight: only the batch is new.
        _batch(session, stalled=True, task=session.file_io_executor.running_task)
        scheduler.tick()

        assert len(_stalls(heard)) == 2


class TestNothingIsReported:
    @pytest.mark.slow
    def test_while_the_writer_is_moving(self, session, heard, scheduler):
        _batch(session, stalled=False)
        scheduler.tick()
        assert _stalls(heard) == []

    def test_while_the_run_is_live(self, session, heard, scheduler):
        # A live run's own writes answer for a stuck writer; the drain has not begun.
        _batch(session, stalled=True, closed=False)
        scheduler.tick()
        assert _stalls(heard) == []

    def test_with_no_run_yet(self, session, heard, scheduler):
        scheduler.tick()
        assert _stalls(heard) == []


@pytest.mark.slow
def test_the_refusal_and_the_report_offer_one_remedy_in_one_set_of_words(
    session, heard, scheduler, tmp_path
):
    _batch(session, stalled=True)
    scheduler.tick()
    report = _stalls(heard)[0]

    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.sequenced_capture_runner.prepare(
            protocol=MagicMock(),
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='t',
            image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
            autogain_settings={},
            parent_dir=tmp_path,
            autofocus_snapshot=autofocus_snapshot(),
        )

    assert refused.value.remedy == report.remedy
    cost = FileWriterStalledError.cost_sentence(3)
    assert cost in refused.value.message and cost in report.message


def test_bring_up_arms_the_check(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        assert session.sequenced_capture_runner._file_writer_check_handle is not None
    finally:
        session.shutdown()
