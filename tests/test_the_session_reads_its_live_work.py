# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a session is still doing is one read, from every place work runs.

A close has to wait for all of the session's work, and a closing host has
to show it; the work runs in seven places (the claim; a recording's and a
run's video finish; a finished run's image writes; its post-run builds;
post-processing builds; reports; a still), and before ``live_work`` only
some of them were readable at all -- a run's post-run builds and every
post-processing build were not, so a close returned while they still
wrote. Each source is put into its working state here and read back
through the one record, with its count or progress, and gone once it ends.
"""

import pathlib
import threading
from unittest.mock import MagicMock

import pytest

from modules import live_work
from modules.post_processing_api import BuildResult
from modules.sequential_io_executor import IOTask

SETTLE_S = 10.0


@pytest.fixture
def sim_session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings
    from tests.test_a_run_needs_every_axis_position import _settings

    session = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(session.scope)
        yield session
    finally:
        session.shutdown()


def _kinds(session):
    return [item.kind for item in session.live_work.work]


def _item(session, kind):
    items = [item for item in session.live_work.work if item.kind == kind]
    assert len(items) == 1, f'expected one {kind!r} item, read {session.live_work.work}'
    return items[0]


def _until(predicate, what):
    import time

    deadline = time.monotonic() + SETTLE_S
    while not predicate():
        assert time.monotonic() < deadline, f'never saw {what}'
        time.sleep(0.01)


def test_an_idle_session_is_doing_nothing_and_says_closed_once_shut_down(sim_session):
    assert sim_session.live_work.work == ()
    assert sim_session.live_work.closed is False

    sim_session.shutdown()

    assert sim_session.live_work.closed is True


def test_what_holds_the_scope_is_named(sim_session):
    held = sim_session.activity_claim.try_claim('diagnostic')
    try:
        item = _item(sim_session, live_work.DIAGNOSTIC)
        assert item.name == 'A diagnostic'
    finally:
        held.release()
    assert sim_session.live_work.work == ()


def test_a_run_unwinding_after_its_claim_is_still_work(sim_session, monkeypatch):
    from modules.protocol_state_machine import ProtocolState

    monkeypatch.setattr(sim_session.sequenced_capture_runner, '_state', ProtocolState.COMPLETING)
    assert _kinds(sim_session) == [live_work.PROTOCOL]


def test_a_recordings_finish_counts_its_frames_down(sim_session):
    engine = MagicMock(is_recording=False, is_draining=True, pending_writes=5)
    sim_session.manual_recording._engine = engine
    try:
        assert _item(sim_session, live_work.RECORDING_FINISH).left == 5
        engine.pending_writes = 2
        assert _item(sim_session, live_work.RECORDING_FINISH).left == 2
    finally:
        sim_session.manual_recording._engine = None


def test_a_live_recording_is_its_claim_not_its_finish(sim_session):
    held = sim_session.activity_claim.try_claim('recording')
    sim_session.manual_recording._engine = MagicMock(is_recording=True, is_draining=False)
    try:
        assert _kinds(sim_session) == [live_work.RECORDING]
    finally:
        sim_session.manual_recording._engine = None
        held.release()


def test_a_runs_video_finish_counts_its_frames(sim_session, monkeypatch):
    writer = MagicMock(video_busy=True, video_pending_writes=7)
    monkeypatch.setattr(sim_session.sequenced_capture_runner, '_image_writer', writer)
    assert _item(sim_session, live_work.RUN_VIDEO_FINISH).left == 7


def test_a_finished_runs_images_count_down(sim_session):
    from modules.protocol_image_writer import RunWriteBatch

    gate = threading.Event()
    batch = RunWriteBatch(sim_session.file_io_executor)
    for _ in range(3):
        batch.submit(lambda: gate.wait(SETTLE_S), {}, what='an image', pace_until=None)
    sim_session.sequenced_capture_runner._write_batch = batch
    batch.close()
    try:
        assert _item(sim_session, live_work.RUN_FILES).left == 3
    finally:
        gate.set()
    _until(lambda: live_work.RUN_FILES not in _kinds(sim_session), 'the images land')


def test_a_runs_post_run_step_is_work_until_its_thread_ends(sim_session):
    gate = threading.Event()
    step = sim_session.sequenced_capture_runner._spawn_post_run_step(
        name='hyperstack-build', build_fn=lambda: gate.wait(SETTLE_S)
    )
    assert _item(sim_session, live_work.POST_RUN_STEP).name == "A run's hyperstack-build"
    gate.set()
    step.join(SETTLE_S)
    assert sim_session.live_work.work == ()


def test_a_build_reports_its_progress_and_the_builds_behind_it_wait(sim_session):
    post = sim_session.post_processing
    said, go_on, finish = threading.Event(), threading.Event(), threading.Event()
    seen_by_caller = []
    answers = {}

    def built(folder):
        return BuildResult('built', 0, pathlib.Path(folder), (), ())

    def slow_build(folder, *, on_progress):
        on_progress(10, 'Image 1 of 10')
        said.set()
        go_on.wait(SETTLE_S)
        on_progress(60, 'Image 6 of 10')
        finish.wait(SETTLE_S)
        return built(folder)

    def quick_build(folder, *, on_progress):
        return built(folder)

    first = threading.Thread(
        target=lambda: answers.__setitem__(
            'stitch',
            post._run(
                slow_build, 'stitch', 'folder-a', on_progress=lambda p, d: seen_by_caller.append(p)
            ),
        )
    )
    second = threading.Thread(
        target=lambda: answers.__setitem__(
            'zproject', post._run(quick_build, 'zproject', 'folder-b', on_progress=None)
        )
    )
    first.start()
    assert said.wait(SETTLE_S)
    second.start()
    _until(lambda: live_work.POST_PROCESSING_QUEUED in _kinds(sim_session), 'the queued build')

    running = _item(sim_session, live_work.POST_PROCESSING)
    assert running.name == 'stitch of folder-a'
    assert running.percent == 10
    assert _item(sim_session, live_work.POST_PROCESSING_QUEUED).left == 1

    go_on.set()
    _until(lambda: _item(sim_session, live_work.POST_PROCESSING).percent == 60, 'progress move')
    finish.set()
    first.join(SETTLE_S)
    second.join(SETTLE_S)

    assert seen_by_caller == [10, 60], "the caller's own progress callback is still called"
    # Each build answered its caller: one that raised on the lane has no answer.
    assert answers == {'stitch': built('folder-a'), 'zproject': built('folder-b')}
    assert sim_session.live_work.work == ()


def test_a_protocols_processors_on_the_lane_are_named_without_progress(sim_session):
    gate = threading.Event()
    lane = sim_session.post_processing.lane
    waiter = lane.submit(IOTask(action=lambda: gate.wait(SETTLE_S)), 'processors')
    try:
        _until(lambda: live_work.POST_PROCESSING in _kinds(sim_session), 'the processors')
        item = _item(sim_session, live_work.POST_PROCESSING)
        assert item.name == "a protocol's post-processing"
        assert item.percent is None
    finally:
        gate.set()
    waiter.result(SETTLE_S)


def test_a_scripts_logs_zip_is_seen_on_its_own_thread(sim_session, monkeypatch, tmp_path):
    from modules.tech_support_report import TechSupportReport

    started, gate = threading.Event(), threading.Event()

    def slow_zip(self, callback=None, output_dir=None):
        started.set()
        gate.wait(SETTLE_S)
        return str(tmp_path / 'logs.zip')

    monkeypatch.setattr(TechSupportReport, 'generate_logs_only', slow_zip)
    script = threading.Thread(target=sim_session.make_logs_zip, kwargs={'output_dir': tmp_path})
    script.start()
    assert started.wait(SETTLE_S)
    try:
        assert _item(sim_session, live_work.LOGS_ZIP).left == 1
    finally:
        gate.set()
    script.join(SETTLE_S)
    assert sim_session.live_work.work == ()


def test_a_still_being_saved_is_work(sim_session):
    sim_session.manual_capture._in_flight.acquire()
    try:
        assert _kinds(sim_session) == [live_work.STILL]
    finally:
        sim_session.manual_capture._in_flight.release()
