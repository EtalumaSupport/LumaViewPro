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
import concurrent.futures
import threading
from unittest.mock import MagicMock

import pytest

from modules import live_work
from modules.exceptions import PostProcessingOperation
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


def test_a_run_lent_a_diagnostics_claim_is_named_beside_it(sim_session):
    from tests.protocol_drives import run_identity

    held = sim_session.activity_claim.try_claim('diagnostic')
    try:
        run = held.lend().try_claim('protocol', run=run_identity())
        try:
            work = sim_session.live_work.work
        finally:
            run.release()
        after = _kinds(sim_session)
    finally:
        held.release()

    assert [item.kind for item in work] == [live_work.DIAGNOSTIC, live_work.PROTOCOL]
    assert work[1].name == 'The scan run'
    assert after == [live_work.DIAGNOSTIC]


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


def test_only_the_video_finishes_are_discardable(sim_session, monkeypatch):
    """The discard drops a recording's file and a run's video frames, and
    those two items alone say so, so a host offers the discard only while
    there is something for it to drop."""
    held = sim_session.activity_claim.try_claim('diagnostic')
    # Stand-ins by design: the simulator cannot yet hold a recording's or a
    # run's video drain open, as the counting tests above.
    sim_session.manual_recording._engine = MagicMock(
        is_recording=False, is_draining=True, pending_writes=3
    )
    writer = MagicMock(video_busy=True, video_pending_writes=4)
    monkeypatch.setattr(sim_session.sequenced_capture_runner, '_image_writer', writer)
    try:
        work = sim_session.live_work.work
        assert {item.kind: item.discardable for item in work} == {
            live_work.DIAGNOSTIC: False,
            live_work.RECORDING_FINISH: True,
            live_work.RUN_VIDEO_FINISH: True,
        }
    finally:
        sim_session.manual_recording._engine = None
        held.release()


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
    assert batch.wait_complete(SETTLE_S), 'the images never landed'
    assert live_work.RUN_FILES not in _kinds(sim_session)


def test_a_runs_hyperstack_is_post_processing_work_until_it_ends(sim_session, tmp_path):
    # A run's own build is the lane's work, named as the lane's: there is no
    # second kind of work for a build a run asked for.
    started, gate = threading.Event(), threading.Event()

    def written():
        started.set()
        gate.wait(SETTLE_S)

    build = sim_session.post_processing.run_hyperstack(
        tmp_path,
        save_encoding='8bit',
        channel_colors=sim_session.channel_colors(),
        written=written,
    )
    assert started.wait(SETTLE_S), 'the lane never began the build'
    assert _item(sim_session, live_work.POST_PROCESSING).name == f'run_hyperstack of {tmp_path}'
    gate.set()
    concurrent.futures.wait([build], SETTLE_S)
    assert build.done()
    assert sim_session.live_work.work == ()


def test_a_build_reports_its_progress_and_the_builds_behind_it_wait(sim_session, tmp_path):
    post = sim_session.post_processing
    # The lane refuses a folder that is not there.
    folder_a = tmp_path / 'folder-a'
    folder_a.mkdir()
    said, go_on, finish = threading.Event(), threading.Event(), threading.Event()
    said_sixty = threading.Event()
    seen_by_caller = []
    answers = {}

    def built(folder):
        return BuildResult('built', 0, pathlib.Path(folder), (), (), ())

    def slow_build(folder, *, on_progress):
        on_progress(10, 'Image 1 of 10')
        said.set()
        go_on.wait(SETTLE_S)
        on_progress(60, 'Image 6 of 10')
        said_sixty.set()
        finish.wait(SETTLE_S)
        return built(folder)

    first = threading.Thread(
        target=lambda: answers.__setitem__(
            'stitch',
            post._run(
                slow_build,
                'stitch',
                folder_a,
                operation=PostProcessingOperation.STITCH,
                on_progress=lambda p, d: seen_by_caller.append(p),
            ),
        )
    )
    first.start()
    assert said.wait(SETTLE_S)
    # A build behind it: submit returns once the task is on the lane, so the
    # read after it finds it waiting, whoever queued it.
    behind = sim_session.post_processing.lane.submit(IOTask(action=lambda: None), 'queued build')

    running = _item(sim_session, live_work.POST_PROCESSING)
    assert running.name == f'stitch of {folder_a}'
    assert running.progress_pct == 10
    assert _item(sim_session, live_work.POST_PROCESSING_QUEUED).left == 1

    go_on.set()
    # The build's own report has returned, so the lane has recorded it.
    assert said_sixty.wait(SETTLE_S), 'the build never reported 60'
    assert _item(sim_session, live_work.POST_PROCESSING).progress_pct == 60
    finish.set()
    first.join(SETTLE_S)
    behind.result(SETTLE_S)

    assert seen_by_caller == [10, 60], "the caller's own progress callback is still called"
    # The build answered its caller: one that raised on the lane has no answer.
    assert answers == {'stitch': built(folder_a)}
    assert sim_session.live_work.work == ()


def test_a_protocols_processors_on_the_lane_are_named_without_progress(sim_session):
    gate, started = threading.Event(), threading.Event()
    lane = sim_session.post_processing.lane

    def processors():
        started.set()
        gate.wait(SETTLE_S)

    waiter = lane.submit(IOTask(action=processors), 'processors')
    try:
        assert started.wait(SETTLE_S), 'the processors never started'
        assert live_work.POST_PROCESSING in _kinds(sim_session)
        item = _item(sim_session, live_work.POST_PROCESSING)
        assert item.name == "a protocol's post-processing"
        assert item.progress_pct is None
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


def test_a_plugin_run_is_work_counted_until_each_ends(sim_session):
    with sim_session.plugin_run_in_flight():
        assert _item(sim_session, live_work.PLUGIN_RUN).left == 1
        with sim_session.plugin_run_in_flight():
            assert _item(sim_session, live_work.PLUGIN_RUN).left == 2
        assert _item(sim_session, live_work.PLUGIN_RUN).left == 1
    assert sim_session.live_work.work == ()
