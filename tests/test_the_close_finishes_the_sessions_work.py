# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Closing a session finishes what it is doing, then tears down, for every host.

``shutdown`` tore down without ending or waiting for the session's work,
and the GUI ended some of it itself before calling it, so every other host
-- a script, the headless REST server -- got a close that cut its files
(the close probes, at the plan's pin): a live recording left a 48-byte MP4
(F1); a live run's complete images were recorded ``write_batch_abandoned``
(F2); a run started on another thread during the close stayed live for ever
(F3); a post-processing build and a run's post-run build kept writing after
it returned (F4, F5); two closes both tore down (F6). Each case is run here
through ``shutdown`` alone, as a script or the REST server calls it.
"""

import datetime
import pathlib
import threading
import time

import pytest

from modules.exceptions import PluginProcessorSkippedError, SessionClosingError
from modules.post_processing_api import BuildResult

SETTLE_S = 30.0


def _session(tmp_path, model='LS850'):
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings
    from tests.test_a_run_needs_every_axis_position import _settings

    session = ScopeSession.create(
        complete_settings(**_settings(tmp_path), microscope=model), simulate=True
    )
    home_sim_scope(session.scope)
    return session


@pytest.fixture
def sim_session(tmp_path):
    session = _session(tmp_path)
    try:
        yield session
    finally:
        session.shutdown()


def _one_step_protocol(session):
    session.set_layer_acquire('BF', 'image')
    return session.new_protocol(duration=datetime.timedelta(0), period=datetime.timedelta(0))


@pytest.mark.slow
def test_a_live_recording_is_stopped_and_its_file_finished(sim_session, tmp_path):
    recording = sim_session.manual_recording
    write = recording._write_mp4_frame
    recording._write_mp4_frame = lambda *a, **k: (time.sleep(0.05), write(*a, **k))[1]
    recording.start(layer='BF')
    time.sleep(2.0)
    assert sim_session.manual_recording.pending_writes > 0, 'no backlog for the close to finish'

    sim_session.shutdown()

    assert not recording.is_busy
    videos = sorted(tmp_path.rglob('*.mp4'))
    assert len(videos) == 1, videos
    data = videos[0].read_bytes()
    assert b'moov' in data, 'the MP4 was cut before its index was written'
    assert (videos[0].parent / f'{videos[0].stem}_manifest.json').exists(), 'no manifest'


def test_a_live_runs_captured_images_are_written_not_called_incomplete(sim_session, tmp_path):
    protocol = sim_session.create_empty_protocol()
    sim_session.set_layer_acquire('BF', 'image')
    for _ in range(40):
        sim_session.add_step(protocol)
    run = sim_session.create_protocol_runner().run_single_scan(
        protocol, parent_dir=str(tmp_path / 'runs')
    )
    time.sleep(1.0)

    sim_session.shutdown()

    outcome = run.wait(timeout_s=SETTLE_S)
    assert (outcome.status, outcome.reason) == ('aborted', 'shutdown')
    files = run.wait_for_files(timeout_s=SETTLE_S)
    assert files.not_written_reason != 'write_batch_abandoned', files
    assert files.written == outcome.captures.captured, files


def _race_a_start_against_the_close(session, close_delay_s):
    protocol = _one_step_protocol(session)
    runner = session.create_protocol_runner()
    go = threading.Barrier(2)
    box = {}

    def starter():
        go.wait()
        try:
            box['run'] = runner.run_single_scan(protocol)
        except Exception as refused:
            box['refused'] = refused

    def closer():
        go.wait()
        time.sleep(close_delay_s)
        session.shutdown()

    threads = [threading.Thread(target=starter), threading.Thread(target=closer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(SETTLE_S)
    return box


def test_a_run_started_while_the_close_runs_is_never_left_live(tmp_path):
    for trial in range(10):
        session = _session(tmp_path / f'trial{trial}')
        box = _race_a_start_against_the_close(session, 0.002 * (trial % 3))

        assert session.activity_claim.holder is None, f'trial {trial}: the scope is still held'
        assert not session.sequenced_capture_runner.run_live, f'trial {trial}: a run is live'
        if 'run' in box:
            assert box['run'].wait(timeout_s=SETTLE_S) is not None


def test_two_closes_at_once_tear_down_once(tmp_path):
    session = _session(tmp_path)
    teardowns = []
    real = session.executor_bundle.shutdown
    session.executor_bundle.shutdown = lambda: (teardowns.append(1), real())[1]
    go = threading.Barrier(2)
    returned = []

    def closer():
        go.wait()
        session.shutdown()
        returned.append(session.live_work.closed)

    threads = [threading.Thread(target=closer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(SETTLE_S)

    assert teardowns == [1]
    assert returned == [True, True], 'each close returned only once the session was closed'


def test_a_runs_post_run_build_ends_before_the_close_returns(sim_session):
    built = threading.Event()

    def build():
        time.sleep(0.5)
        built.set()

    sim_session.sequenced_capture_runner._spawn_post_run_step(name='hyperstack', build_fn=build)

    sim_session.shutdown()

    assert built.is_set(), 'the close returned while the post-run build still ran'


def test_a_running_and_a_queued_build_both_finish(sim_session, tmp_path):
    post = sim_session.post_processing
    results = {}

    def slow(folder, *, on_progress):
        time.sleep(0.5)
        return BuildResult(
            message=f'built {folder}',
            new_count=0,
            output_root=pathlib.Path(folder),
            artifact_paths=(),
            degraded_outputs=(),
        )

    def build(name):
        results[name] = post._run(slow, 'stitch', name).message

    threads = [threading.Thread(target=build, args=(name,)) for name in ('a', 'b')]
    threads[0].start()
    time.sleep(0.1)
    threads[1].start()
    time.sleep(0.1)

    sim_session.shutdown()

    assert results == {'a': 'built a', 'b': 'built b'}
    for thread in threads:
        thread.join(SETTLE_S)


def test_a_scripts_logs_zip_is_waited_for(sim_session, monkeypatch, tmp_path):
    from modules.tech_support_report import TechSupportReport

    started = threading.Event()
    finished = threading.Event()

    def slow_zip(self, callback=None, output_dir=None):
        started.set()
        time.sleep(0.5)
        finished.set()
        return str(tmp_path / 'logs.zip')

    monkeypatch.setattr(TechSupportReport, 'generate_logs_only', slow_zip)
    script = threading.Thread(target=sim_session.make_logs_zip, kwargs={'output_dir': tmp_path})
    script.start()
    assert started.wait(SETTLE_S)

    sim_session.shutdown()

    assert finished.is_set(), 'the close returned while the logs zip was still being written'
    script.join(SETTLE_S)


def test_discard_ends_the_wait_on_a_recordings_frames(sim_session, tmp_path):
    recording = sim_session.manual_recording
    write = recording._write_mp4_frame
    recording._write_mp4_frame = lambda *a, **k: (time.sleep(0.2), write(*a, **k))[1]
    recording.start(layer='BF')
    time.sleep(2.0)
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (sim_session.shutdown(), closed.set()))
    closer.start()
    time.sleep(0.5)
    assert not closed.is_set(), 'nothing was left for the close to wait on'

    sim_session.discard_close_drain()

    assert closed.wait(SETTLE_S), 'Discard did not end the wait'
    videos = sorted(tmp_path.rglob('*.mp4'))
    assert len(videos) == 1 and b'moov' in videos[0].read_bytes()


def test_a_runs_processor_skipped_by_the_unload_is_reported(monkeypatch):
    from unittest.mock import MagicMock

    from modules.notification_center import notifications
    from modules.plugins import PluginRegistry

    reported = []
    monkeypatch.setattr(notifications, 'report_outcome', lambda ex, **kw: reported.append(ex))
    registry = PluginRegistry()
    registry._track('stitcher', MagicMock(unregister=lambda host: None))
    spec = MagicMock(auto_run_on_protocol_complete=True)
    spec.name = 'stitcher'
    processor = MagicMock()
    registry.post_processing.handlers = MagicMock(return_value=[(spec, processor)])

    registry.unload(host=None)
    registry.run_protocol_complete_processors(
        input_dir='/runs/plate1',
        manifest={'protocol_name': 'plate1'},
        output_dir='',
        files='written',
    )

    processor.assert_not_called()
    assert len(reported) == 1 and isinstance(reported[0], PluginProcessorSkippedError)
    assert reported[0].plugin_name == 'stitcher' and reported[0].run == 'plate1'


def test_after_the_close_a_start_is_refused_by_what_is_true(tmp_path):
    session = _session(tmp_path)
    session.shutdown()

    assert session.live_work.closing is False
    assert session.live_work.closed is True
    with pytest.raises(Exception) as refused:
        session.scope.motion.home('Z')
    assert not isinstance(refused.value, SessionClosingError)
    assert getattr(refused.value, 'reason', None) == 'scope_disconnected'


def test_a_close_from_run_ended_with_no_dispatcher_completes(tmp_path):
    """With no UI dispatcher, ``run_ended`` runs on the thread that sends it,
    which a script's or the REST server's handler may close the session
    from; the close must not wait on, or join, the thread it runs on."""
    from modules.run_events import RunEvents

    session = _session(tmp_path)
    protocol = session.create_empty_protocol()
    session.set_layer_acquire('BF', 'image')
    session.add_step(protocol)
    closed, failed, took, ran_on = threading.Event(), [], [], []

    def close_the_session(*_args):
        ran_on.append(threading.current_thread())
        began = time.monotonic()
        try:
            session.shutdown()
        except Exception as ex:
            failed.append(ex)
        took.append(time.monotonic() - began)
        closed.set()

    session.create_protocol_runner().run_single_scan(
        protocol,
        parent_dir=str(tmp_path / 'runs'),
        events=RunEvents(run_ended=close_the_session),
    )

    assert closed.wait(SETTLE_S), 'the close from run_ended did not return'
    assert failed == [], failed
    assert session.live_work.closed is True
    assert took[0] < 2.0, f'the close from run_ended took {took[0]:.1f} s'
    ran_on[0].join(SETTLE_S)
    assert not ran_on[0].is_alive(), 'the thread the close ran on never ended'
