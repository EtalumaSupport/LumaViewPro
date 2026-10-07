# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated session can hold its file lane, as a save drive that stops answering would.

A run's writes can stall without anything failing: the drive stops
answering and the file lane's one worker sits inside a write that does not
return. What LVP does about that -- the stalled-writer report and its
recovery -- has to be shown without hardware, so a simulated session takes
a stall for its file lane (``--sim-file-stall=AFTER,FOR``): AFTER seconds
in, one task holds the lane's worker for FOR seconds. Nothing in the write
path knows it is simulated; the lane simply has a worker stuck in a task.
"""

from __future__ import annotations

import time

import pytest

import modules.notification_center as nc
import modules.sequenced_capture_runner as runner_module
from drivers.simulated_camera import SimulatedStall
from modules.notification_center import NotificationCenter, Severity
from modules.protocol_image_writer import RunWriteBatch
from modules.scope_session import ScopeSession
from tests.scope_fakes import build_scope
from tests.settings_fixtures import complete_settings


def _wait_for(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _held(session) -> bool:
    task = session.file_io_executor.running_task
    return task is not None and getattr(task.action, '__name__', '') == 'simulated_stuck_write'


class TestAStallIsRefusedWhereItCannotHappen:
    def test_a_real_session_refuses_a_file_stall(self, tmp_path):
        with pytest.raises(ValueError, match='sim_file_stall needs a simulated scope'):
            ScopeSession.create(
                complete_settings(live_folder=str(tmp_path)),
                simulate=False,
                sim_file_stall=SimulatedStall(0.0, 1.0),
            )

    def test_a_session_refuses_a_file_stall_beside_a_scope_it_was_given(self):
        scope = build_scope(simulate=True)
        try:
            with pytest.raises(ValueError, match='sim_file_stall is refused beside a scope'):
                ScopeSession.create(
                    complete_settings(), scope=scope, sim_file_stall=SimulatedStall(0.0, 1.0)
                )
        finally:
            scope.disconnect()


class TestTheLaneIsHeld:
    def test_after_the_delay_for_the_stall_and_then_freed(self, tmp_path):
        session = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path)),
            simulate=True,
            sim_file_stall=SimulatedStall(after_s=0.3, for_s=0.8),
        )
        try:
            assert not _held(session), 'held before its delay'
            assert _wait_for(lambda: _held(session), 3.0), 'never held'
            assert _wait_for(lambda: not _held(session), 3.0), 'never freed'
        finally:
            session.shutdown()

    @pytest.mark.slow
    def test_a_drain_behind_the_hold_is_reported_as_a_stalled_writer(self, tmp_path, monkeypatch):
        centre = NotificationCenter(dedup_window_s=0)
        heard = []
        centre.add_listener(heard.append, min_severity=Severity.DEBUG)
        monkeypatch.setattr(nc, 'notifications', centre)
        # The real threshold is 30 s; the judgement is the same at any threshold.
        monkeypatch.setattr(runner_module, 'WRITE_STALL_FATAL_S', 0.3)
        session = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path)),
            simulate=True,
            sim_file_stall=SimulatedStall(after_s=0.0, for_s=5.0),
        )
        try:
            assert _wait_for(lambda: _held(session), 3.0)
            # A run's writes, queued behind the stuck one, and the run over.
            batch = RunWriteBatch(session.file_io_executor)
            batch.submit(lambda: None, {}, what='The image', pace_until=None)
            batch.close()
            session.sequenced_capture_runner._write_batch = batch

            assert _wait_for(
                lambda: any(n.reason == 'files_writing_stalled' for n in heard), 5.0
            ), 'the stall behind the held lane was not reported'
            stall = next(n for n in heard if n.reason == 'files_writing_stalled')
            assert 'simulated_stuck_write' in stall.message
        finally:
            session.shutdown()
