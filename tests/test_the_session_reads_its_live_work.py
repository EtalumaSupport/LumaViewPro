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

from _lumascope.modules import live_work
from _lumascope.modules.exceptions import ArgumentRefusedError, PostProcessingOperation
from _lumascope.modules.post_processing_api import BuildResult
from _lumascope.modules.sequential_io_executor import IOTask

SETTLE_S = 10.0


@pytest.fixture
def sim_session(tmp_path):
    from _lumascope.modules.scope_session import ScopeSession
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
    held = sim_session._activity_claim.try_claim('diagnostic')
    try:
        item = _item(sim_session, live_work.WorkKind.DIAGNOSTIC)
        assert item.name == 'A diagnostic'
        assert type(item.kind) is live_work.WorkKind
    finally:
        held.release()
    assert sim_session.live_work.work == ()


def test_a_work_item_is_of_a_kind_of_the_set():
    assert type(live_work.WorkItem('run_files', "A run's images").kind) is live_work.WorkKind
    with pytest.raises(ArgumentRefusedError) as refused:
        live_work.WorkItem('nap', 'A nap')
    assert (refused.value.reason, refused.value.argument) == ('not_offered', 'WorkItem.kind')


def test_a_run_lent_a_diagnostics_claim_is_named_beside_it(sim_session):
    from tests.protocol_drives import run_identity

    held = sim_session._activity_claim.try_claim('diagnostic')
    try:
        run = held.lend().try_claim('protocol', run=run_identity())
        try:
            work = sim_session.live_work.work
        finally:
            run.release()
        after = _kinds(sim_session)
    finally:
        held.release()

    assert [item.kind for item in work] == [
        live_work.WorkKind.DIAGNOSTIC,
        live_work.WorkKind.PROTOCOL,
    ]
    assert work[1].name == 'The scan run'
    assert after == [live_work.WorkKind.DIAGNOSTIC]


def test_a_run_unwinding_after_its_claim_is_still_work(sim_session, monkeypatch):
    from _lumascope.modules.protocol_state_machine import ProtocolState

    monkeypatch.setattr(sim_session._sequenced_capture_runner, '_state', ProtocolState.COMPLETING)
    assert _kinds(sim_session) == [live_work.WorkKind.PROTOCOL]


def test_a_recordings_finish_counts_its_frames_down(sim_session):
    engine = MagicMock(is_recording=False, is_draining=True, pending_writes=5)
    sim_session.manual_recording._engine = engine
    try:
        assert _item(sim_session, live_work.WorkKind.RECORDING_FINISH).left == 5
        engine.pending_writes = 2
        assert _item(sim_session, live_work.WorkKind.RECORDING_FINISH).left == 2
    finally:
        sim_session.manual_recording._engine = None


def test_a_live_recording_is_its_claim_not_its_finish(sim_session):
    held = sim_session._activity_claim.try_claim('recording')
    sim_session.manual_recording._engine = MagicMock(is_recording=True, is_draining=False)
    try:
        assert _kinds(sim_session) == [live_work.WorkKind.RECORDING]
    finally:
        sim_session.manual_recording._engine = None
        held.release()


def test_a_runs_video_finish_counts_its_frames(sim_session, monkeypatch):
    writer = MagicMock(video_busy=True, video_draining=True, video_pending_writes=7)
    monkeypatch.setattr(sim_session._sequenced_capture_runner, '_image_writer', writer)
    assert _item(sim_session, live_work.WorkKind.RUN_VIDEO_FINISH).left == 7


def test_only_the_video_finishes_are_discardable(sim_session, monkeypatch):
    """The discard drops a recording's file and a run's video frames, and
    those two items alone say so, so a host offers the discard only while
    there is something for it to drop."""
    held = sim_session._activity_claim.try_claim('diagnostic')
    # Stand-ins by design: the simulator cannot yet hold a recording's or a
    # run's video drain open, as the counting tests above.
    sim_session.manual_recording._engine = MagicMock(
        is_recording=False, is_draining=True, pending_writes=3
    )
    writer = MagicMock(video_busy=True, video_draining=True, video_pending_writes=4)
    monkeypatch.setattr(sim_session._sequenced_capture_runner, '_image_writer', writer)
    try:
        work = sim_session.live_work.work
        assert {item.kind: item.discardable for item in work} == {
            live_work.WorkKind.DIAGNOSTIC: False,
            live_work.WorkKind.RECORDING_FINISH: True,
            live_work.WorkKind.RUN_VIDEO_FINISH: True,
        }
    finally:
        sim_session.manual_recording._engine = None
        held.release()


def test_a_finish_past_its_drain_is_listed_with_nothing_left_to_discard(sim_session, monkeypatch):
    """A recording closing its file or building its stack, and a run's video
    step finishing, are still work the close waits on, but have no frames
    left, so they say none and offer no discard (a discard offered then
    dropped nothing and stopped nothing, bundle 'logs-LVP wont close')."""
    # Stand-ins by design: the simulator cannot yet hold a recording's or a
    # run's video finish open past its drain.
    sim_session.manual_recording._engine = MagicMock(
        is_recording=False, is_draining=False, pending_writes=0
    )
    monkeypatch.setattr(sim_session.manual_recording, '_finishing', True)
    writer = MagicMock(video_busy=True, video_draining=False, video_pending_writes=0)
    monkeypatch.setattr(sim_session._sequenced_capture_runner, '_image_writer', writer)
    try:
        work = {item.kind: item for item in sim_session.live_work.work}
    finally:
        sim_session.manual_recording._engine = None

    said = {
        kind: (item.name, item.left, item.discardable)
        for kind, item in work.items()
        if kind in (live_work.WorkKind.RECORDING_FINISH, live_work.WorkKind.RUN_VIDEO_FINISH)
    }
    assert said == {
        live_work.WorkKind.RECORDING_FINISH: ('A recording, finishing', None, False),
        live_work.WorkKind.RUN_VIDEO_FINISH: ("A run's video, finishing", None, False),
    }


def test_a_manual_recording_says_how_long_it_has_recorded(sim_session):
    recording = sim_session.manual_recording.start()
    try:
        item = _item(sim_session, live_work.WorkKind.RECORDING)
    finally:
        sim_session.manual_recording.stop()
        recording.result(timeout=SETTLE_S)

    assert item.elapsed_s is not None and item.elapsed_s >= 0, item
    assert item.total_s is None, 'a manual recording records until it is stopped'


def test_a_runs_recording_step_says_how_long_it_has_recorded_of_its_length(
    sim_session, monkeypatch
):
    from types import SimpleNamespace

    from tests.protocol_drives import run_identity

    # Stand-ins by design: the subject is the session's reading of the step
    # in flight, which the run's own video tests record for real.
    step = SimpleNamespace(elapsed_s=3.0, total_s=10.0)
    writer = MagicMock(video_busy=False, recording_video_step=step)
    monkeypatch.setattr(sim_session._sequenced_capture_runner, '_image_writer', writer)
    held = sim_session._activity_claim.try_claim('protocol', run=run_identity())
    try:
        item = _item(sim_session, live_work.WorkKind.PROTOCOL)
    finally:
        held.release()

    assert (item.elapsed_s, item.total_s) == (3.0, 10.0)


@pytest.mark.slow
def test_a_runs_video_step_is_read_while_it_records(tmp_path):
    """A real run's video step, read from inside its recording: the run's item
    carries the step's seconds so far and its length."""
    from _lumascope.modules.run_events import RunEvents, VideoPhase
    from tests.test_composite_run_e2e import headless_settings, open_composite_session
    from tests.test_what_a_run_keeps_is_bounded_by_its_protocol import _protocol, _run, _videos

    read = []
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):

        def video_progress(progress):
            # Delivered on the run's thread from inside the recording.
            if progress.phase == VideoPhase.RECORDING and not read:
                read.append(_item(session, live_work.WorkKind.PROTOCOL))

        outcome = _run(
            runner, tmp_path, _protocol(_videos(1), 1), RunEvents(video_progress=video_progress)
        )

    assert outcome.status == 'completed', outcome
    (item,) = read
    assert item.elapsed_s is not None and item.elapsed_s >= 0, item
    assert item.total_s == 0.4, item


def test_a_finished_runs_images_count_down(sim_session):
    from _lumascope.modules.protocol_image_writer import RunWriteBatch

    gate = threading.Event()
    batch = RunWriteBatch(sim_session.file_io_executor)
    for _ in range(3):
        batch.submit(lambda: gate.wait(SETTLE_S), {}, what='an image', pace_until=None)
    sim_session._sequenced_capture_runner._write_batch = batch
    batch.close()
    try:
        assert _item(sim_session, live_work.WorkKind.RUN_FILES).left == 3
    finally:
        gate.set()
    assert batch.wait_complete(SETTLE_S), 'the images never landed'
    assert live_work.WorkKind.RUN_FILES not in _kinds(sim_session)


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
        channel_colors=sim_session.channel_colors,
        written=written,
        stills=True,
        recordings=False,
    )
    assert started.wait(SETTLE_S), 'the lane never began the build'
    assert (
        _item(sim_session, live_work.WorkKind.POST_PROCESSING).name
        == f"Building a run's hyperstacks in {tmp_path.name}"
    )
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
                writes_beside=False,
                saved_in=lambda result: None,
                on_progress=lambda p, d: seen_by_caller.append(p),
            ),
        )
    )
    first.start()
    assert said.wait(SETTLE_S)
    # A build behind it: submit returns once the task is on the lane, so the
    # read after it finds it waiting, whoever queued it.
    behind = sim_session.post_processing.lane.submit(IOTask(action=lambda: None), 'queued build')

    running = _item(sim_session, live_work.WorkKind.POST_PROCESSING)
    # A person's words and the folder's own name, never a member or a path.
    assert running.name == f'Stitching in {folder_a.name}'
    assert running.progress_pct == 10
    assert _item(sim_session, live_work.WorkKind.POST_PROCESSING_QUEUED).left == 1

    go_on.set()
    # The build's own report has returned, so the lane has recorded it.
    assert said_sixty.wait(SETTLE_S), 'the build never reported 60'
    assert _item(sim_session, live_work.WorkKind.POST_PROCESSING).progress_pct == 60
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
        assert live_work.WorkKind.POST_PROCESSING in _kinds(sim_session)
        item = _item(sim_session, live_work.WorkKind.POST_PROCESSING)
        assert item.name == "a protocol's post-processing"
        assert item.progress_pct is None
    finally:
        gate.set()
    waiter.result(SETTLE_S)


def test_a_scripts_logs_zip_is_seen_on_its_own_thread(sim_session, monkeypatch, tmp_path):
    from _lumascope.modules.tech_support_report import TechSupportReport

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
        assert _item(sim_session, live_work.WorkKind.LOGS_ZIP).left == 1
    finally:
        gate.set()
    script.join(SETTLE_S)
    assert sim_session.live_work.work == ()


def test_a_still_being_saved_is_work(sim_session):
    sim_session.manual_capture._in_flight.acquire()
    try:
        assert _kinds(sim_session) == [live_work.WorkKind.STILL]
    finally:
        sim_session.manual_capture._in_flight.release()


def _still_read_at_each_summed_frame(session, also=None):
    """The session's still item as read after each frame a still's sum keeps, on the lane.

    The real lane capture, observed: its own per-frame callback is called
    first, then the session is read, then ``also`` (handed the progress).
    """
    seen = []
    imaging = session.scope.imaging
    real = imaging.capture_holding_channel_on_lane

    def observed(channel, member, **capture):
        own = capture['sum_iteration_callback']

        def each_frame(progress):
            own(progress)
            seen.append(_item(session, live_work.WorkKind.STILL))
            if also is not None:
                also(progress)

        capture['sum_iteration_callback'] = each_frame
        return real(channel, member, **capture)

    imaging.capture_holding_channel_on_lane = observed
    return seen


def _still_session(tmp_path, sum_count):
    from tests.test_manual_capture_member import _open_session, _settings

    settings = _settings(tmp_path)
    settings['BF']['sum'] = sum_count
    return _open_session(settings)


def _read_as_the_still_is_shown(session, read):
    from _lumascope.modules.run_events import RunEvents

    return RunEvents(
        frame_captured=lambda *_frame: read.append(_item(session, live_work.WorkKind.STILL))
    )


def test_a_summed_still_counts_its_frames_down_and_is_gone_once_saved(tmp_path):
    with _still_session(tmp_path, sum_count=4) as session:
        seen = _still_read_at_each_summed_frame(session)
        shown = []

        session.manual_capture.start_save_still(
            events=_read_as_the_still_is_shown(session, shown)
        ).result(timeout=30)

        assert [(item.name, item.left) for item in seen] == [
            ('Capturing a summed still', 3),
            ('Capturing a summed still', 2),
            ('Capturing a summed still', 1),
            ('Capturing a summed still', None),
        ]
        assert [(item.name, item.left) for item in shown] == [('Capturing a summed still', None)]
        assert live_work.WorkKind.STILL not in _kinds(session)
        assert session.manual_capture.sum_progress is None


def test_a_still_of_one_frame_is_named_with_no_count(tmp_path):
    with _still_session(tmp_path, sum_count=1) as session:
        shown = []

        session.manual_capture.start_save_still(
            events=_read_as_the_still_is_shown(session, shown)
        ).result(timeout=30)

        assert [(item.name, item.left) for item in shown] == [('Capturing a still', None)]


def test_a_change_mid_sum_starts_the_count_again_and_never_runs_past_the_sum(tmp_path):
    from _lumascope.modules.frame_validity import FrameValiditySource

    with _still_session(tmp_path, sum_count=3) as session:

        def change_once(progress):
            # A gain change landing while the sum is taken: the sum is taken again.
            if progress.kept == 1 and not changed:
                changed.append(True)
                session.scope.imaging._frame_validity.invalidate(
                    FrameValiditySource.GAIN, began_ns=None
                )

        changed = []
        seen = _still_read_at_each_summed_frame(session, also=change_once)

        session.manual_capture.start_save_still().result(timeout=30)

        lefts = [item.left for item in seen]
        assert lefts == [2, 1, None, 2, 1, None], lefts


def test_a_camera_removed_mid_sum_leaves_no_still_and_no_count(tmp_path):
    from _lumascope.modules.exceptions import HardwareCommandRefusedError

    with _still_session(tmp_path, sum_count=3) as session:
        _still_read_at_each_summed_frame(
            session, also=lambda _progress: session.scope.simulation.remove_camera()
        )

        with pytest.raises(HardwareCommandRefusedError) as refused:
            session.manual_capture.start_save_still().result(timeout=30)

        assert refused.value.reason == 'not_connected'
        assert live_work.WorkKind.STILL not in _kinds(session)
        assert session.manual_capture.sum_progress is None


def test_a_plugin_run_is_work_counted_until_each_ends(sim_session):
    with sim_session.plugin_run_in_flight():
        assert _item(sim_session, live_work.WorkKind.PLUGIN_RUN).left == 1
        with sim_session.plugin_run_in_flight():
            assert _item(sim_session, live_work.WorkKind.PLUGIN_RUN).left == 2
        assert _item(sim_session, live_work.WorkKind.PLUGIN_RUN).left == 1
    assert sim_session.live_work.work == ()
