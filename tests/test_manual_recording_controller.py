# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Contract tests for the manual recording controller.

The controller is the caller-shaped half of manual recording: settings
snapshot, per-leg write edges (real production writes into tmp_path),
typed start refusals, the rolling disk floor, the duration cap, and the
post-drain finish. The engine underneath is the real one; the camera is
a stub delivering frames straight into the registered listener.
"""

import itertools
import json
import threading

import numpy as np
import pytest

import modules.manual_recording as manual_recording_module
from modules.exceptions import RecordingRefusedError
from modules.manual_recording import ManualRecordingController
from modules.recording_frames import MANUAL_HYPERSTACK_FILENAME
from modules.video_cadence import INTERIM_DELIVERY_BOUND_FPS
from tests.video_engine_harness import ClaimStub, FakeClock, NotifyRecorder

TICK_HZ = 1_000_000_000


@pytest.fixture(autouse=True)
def _healthy_disk(monkeypatch):
    # conftest mocks psutil, so the real probe returns MagicMocks; the
    # default here reports ample free disk and the floor tests re-patch.
    monkeypatch.setattr(
        manual_recording_module, 'check_disk_space_ok', lambda *_: (True, 1_000_000.0)
    )


class _FakeImaging:
    def __init__(self, exposure_ms=100.0, width=32, height=24):
        self.active_cached = True
        self.exposure_ms_cached = exposure_ms
        self._frame_size = {'width': width, 'height': height}
        self.listener = None
        # Read at recording start to resolve the frames' image scale.
        self._binning_size = 1

    @property
    def frame_size_cached(self):
        return dict(self._frame_size)

    @property
    def camera_identity(self):
        return {
            'model': 'testcam-9000',
            'serial': 'TC9K-001',
            'timestamp_tick_frequency_hz': TICK_HZ,
        }

    def add_frame_listener(self, cb, name=None):
        self.listener = cb

    def remove_frame_listener(self, cb):
        # Equality, not identity. A bound method is a fresh object on every
        # attribute access, so `is` never matches one that was stored
        # earlier; the production API keys its wrapper dict by the callable
        # and removes by equality, and a fake that removes by identity
        # would report a leaked listener the real path does not have.
        if self.listener == cb:
            self.listener = None


class _FakeMotion:
    def get_current_position(self):
        return {'X': 1.0, 'Y': 2.0, 'Z': 3.0}

    def has_turret(self):
        return False


class _FakeIllumination:
    """Commanded LED state, the shape get_led_states() returns.

    Empty dict models a scope with no LED board, which is what the real
    accessor returns in that case.
    """

    def __init__(self, lit=None, board=True):
        self._board = board
        self._lit = lit

    def get_led_states(self):
        if not self._board:
            return {}
        return {
            color: {'enabled': color == self._lit, 'illumination_ma': None, 'owner': ''}
            for color in ('Blue', 'Green', 'Red', 'BF', 'PC', 'DF')
        }


class _FakeRuntimeState:
    """The objective store the scale resolver reads at recording start.

    Without an app context these tests resolve to no scale, which is the
    honest-degradation path; the attribute still has to exist because the
    recording start reads it the same way the still-save path does.
    """

    def __init__(self, focal_length=9.0):
        self._objective = {'focal_length': focal_length}

    def get_current_objective(self):
        return self._objective


class _FakeScope:
    def __init__(self, lit=None, board=True, **imaging_kwargs):
        self.imaging = _FakeImaging(**imaging_kwargs)
        self.motion = _FakeMotion()
        self.illumination = _FakeIllumination(lit=lit, board=board)
        self.runtime_state = _FakeRuntimeState()


def make_settings(tmp_path, *, video_as_frames=True, max_fps=0, duration_s=60, hyperstack=False):
    settings = {
        'live_folder': str(tmp_path),
        'video_as_frames': video_as_frames,
        'video': {
            'max_fps': max_fps,
            'max_duration_seconds': duration_s,
            'timestamp_overlay': True,
        },
    }
    if hyperstack:
        settings['image_output_format'] = {'sequenced': 'OME-TIFF Hyperstack'}
    return settings


def make_controller(tmp_path, *, scope=None, clock=None, lit=None, **settings_kwargs):
    scope = scope or _FakeScope(lit=lit)
    clock = clock or FakeClock()
    controller = ManualRecordingController(
        scope=scope,
        settings=make_settings(tmp_path, **settings_kwargs),
        activity_claim=ClaimStub(),
        clock=clock,
    )
    return controller, scope, clock


def feed_frames(scope, clock, n, *, fps=10.0, width=32, height=24, jitter=None):
    """Deliver n frames through the registered listener at the given rate.

    Ticks advance uniformly at the camera's own clock; the host float
    timestamp optionally jitters around the same instants.
    """
    step = 1.0 / fps
    for i in range(n):
        clock.advance(step)
        listener = scope.imaging.listener
        if listener is None:
            # The recording ended mid-feed and deregistered. Production
            # delivers nothing to a removed listener, so neither does this
            # -- the write lane's own disk-floor breach ends a recording
            # exactly this way, from a thread racing this loop.
            return
        host_ts = clock() + (jitter[i % len(jitter)] if jitter else 0.0)
        chunks = {'Timestamp': int(clock() * TICK_HZ), 'FrameID': i}
        image = np.full((height, width), i % 256, dtype=np.uint8)
        listener(image, host_ts, chunks)


def finish(controller, timeout=15.0):
    thread = controller._finish_thread
    if thread is not None:
        thread.join(timeout)
        assert not thread.is_alive(), 'finish thread did not complete'


class TestStartRefusals:
    def test_inactive_camera_refused(self, tmp_path):
        controller, scope, _ = make_controller(tmp_path)
        scope.imaging.active_cached = False
        with pytest.raises(RecordingRefusedError) as e:
            controller.start()
        assert e.value.reason == 'camera_inactive'

    def test_unknown_exposure_refused(self, tmp_path):
        controller, scope, _ = make_controller(tmp_path)
        scope.imaging.exposure_ms_cached = 0.0
        with pytest.raises(RecordingRefusedError) as e:
            controller.start()
        assert e.value.reason == 'camera_exposure_unknown'

    def test_low_disk_refused(self, tmp_path, monkeypatch):
        controller, _, _ = make_controller(tmp_path)
        monkeypatch.setattr(
            manual_recording_module, 'check_disk_space_ok', lambda *_: (False, 100.0)
        )
        with pytest.raises(RecordingRefusedError) as e:
            controller.start()
        assert e.value.reason == 'insufficient_disk'

    def test_refusal_commits_nothing(self, tmp_path):
        controller, scope, _ = make_controller(tmp_path)
        scope.imaging.active_cached = False
        with pytest.raises(RecordingRefusedError):
            controller.start()
        assert scope.imaging.listener is None
        assert not controller.is_recording and not controller.is_draining


class TestRateClamp:
    def test_exposure_bounds_the_rate(self, tmp_path):
        controller, _, _ = make_controller(tmp_path)  # 100 ms -> 10 fps
        controller.start()
        assert controller._config.fps == pytest.approx(10.0)
        controller.stop()
        finish(controller)

    def test_user_cap_applies(self, tmp_path):
        controller, _, _ = make_controller(tmp_path, max_fps=5)
        controller.start()
        assert controller._config.fps == pytest.approx(5.0)
        controller.stop()
        finish(controller)

    def test_uncapped_fast_exposure_bounded_by_delivery_constant(self, tmp_path):
        scope = _FakeScope(exposure_ms=1.0)  # 1000 fps by exposure alone
        controller, _, _ = make_controller(tmp_path, scope=scope)
        controller.start()
        assert controller._config.fps == pytest.approx(INTERIM_DELIVERY_BOUND_FPS)
        controller.stop()
        finish(controller)

    def test_uncapped_never_fires_fps_budget_warning(self, tmp_path, monkeypatch):
        # max_fps == 0 means uncapped: a fresh install must not see the
        # FPS-budget warning at every long exposure (the regression the
        # legacy _user_requested_fps_limit flag closed).
        recorder = NotifyRecorder()
        monkeypatch.setattr(manual_recording_module, 'notifications', recorder)
        controller, _, _ = make_controller(tmp_path, max_fps=0)
        controller.start()
        assert 'warning' not in recorder.severities()
        controller.stop()
        finish(controller)

    def test_missing_video_settings_dict_tolerated(self, tmp_path):
        # A partially-edited settings file without the video dict must
        # start with defaults, never KeyError.
        scope = _FakeScope()
        settings = make_settings(tmp_path)
        del settings['video']
        controller = ManualRecordingController(
            scope=scope,
            settings=settings,
            activity_claim=ClaimStub(),
            clock=FakeClock(),
        )
        controller.start()
        assert controller.is_recording
        controller.stop()
        finish(controller)


class TestFramesLeg:
    def test_end_to_end_frames_and_manifest(self, tmp_path):
        controller, scope, clock = make_controller(tmp_path, lit='Blue')
        controller.start(layer='Blue', false_color_on=True)
        feed_frames(scope, clock, 5, fps=10.0)
        controller.stop()
        finish(controller)

        folders = list((tmp_path / 'Manual').glob('Video_*'))
        assert len(folders) == 1
        tiffs = sorted(folders[0].glob('ManualVideo_Frame_*.tiff'))
        assert len(tiffs) == 5
        assert tiffs[0].name.startswith('ManualVideo_Frame_0000_')

        manifest = json.loads((folders[0] / 'recording_manifest.json').read_text())
        assert manifest['frames_written'] == 5
        assert manifest['write_failures'] == 0
        assert manifest['channel_color'] == 'Blue'
        assert manifest['camera']['model'] == 'testcam-9000'
        assert manifest['provenance']['software']['lvp_version'] is not None
        assert manifest['timestamp_grade'] == 'camera'
        assert all(entry['chunks'] is not None for entry in manifest['frame_index'])

    def test_two_recordings_in_one_second_get_their_own_folders(self, tmp_path):
        # The folder name comes from a one-second-resolution timestamp, and a
        # full start-stop-finish-start cycle runs in about 250 ms, so pressing
        # Record twice quickly derived the same name twice. The second used to
        # join the first via mkdir(exist_ok=True): measured on the bench as 19
        # recordings landing in 6 folders, frame numbers restarting per
        # recording so a rebuild interleaved them into one scrambled video.
        controller, scope, clock = make_controller(tmp_path, lit='Blue')

        controller.start(layer='Blue', false_color_on=True)
        feed_frames(scope, clock, 3, fps=10.0)
        controller.stop()
        finish(controller)

        # Same wall-clock second: the derived name collides.
        controller.start(layer='Blue', false_color_on=True)
        feed_frames(scope, clock, 4, fps=10.0)
        controller.stop()
        finish(controller)

        folders = sorted((tmp_path / 'Manual').glob('Video_*'))
        assert len(folders) == 2, f'each recording needs its own folder, got {folders}'

        # Neither recording's frames leaked into the other's folder, and each
        # manifest counts only its own.
        counts = sorted(len(list(f.glob('ManualVideo_Frame_*.tiff'))) for f in folders)
        assert counts == [3, 4], counts
        written = sorted(
            json.loads((f / 'recording_manifest.json').read_text())['frames_written']
            for f in folders
        )
        assert written == [3, 4], written

    def test_camera_ticks_smooth_host_jitter(self, tmp_path):
        # Host arrival stamps jitter around the true instants; the camera
        # ticks are uniform. The manifest timeline must follow the ticks.
        controller, scope, clock = make_controller(tmp_path)
        controller.start()
        feed_frames(scope, clock, 5, fps=10.0, jitter=[0.0, 0.03, -0.02, 0.04, 0.01])
        controller.stop()
        finish(controller)

        folder = next((tmp_path / 'Manual').glob('Video_*'))
        manifest = json.loads((folder / 'recording_manifest.json').read_text())
        times = [entry['ts_s'] for entry in manifest['frame_index']]
        intervals = [b - a for a, b in itertools.pairwise(times)]
        assert all(dt == pytest.approx(0.1, abs=1e-6) for dt in intervals)


class TestMp4Leg:
    def test_end_to_end_mp4_and_manifest(self, tmp_path):
        controller, scope, clock = make_controller(tmp_path, video_as_frames=False)
        controller.start()
        feed_frames(scope, clock, 5, fps=10.0)
        controller.stop()
        finish(controller)

        manual = tmp_path / 'Manual'
        mp4s = list(manual.glob('Video_*.mp4'))
        assert len(mp4s) == 1
        manifests = list(manual.glob('Video_*_manifest.json'))
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text())
        assert manifest['frames_written'] == 5


def _capture_engine_and_writer(monkeypatch):
    """Hold the per-recording engine and writer the unwind detaches.

    A failed start clears ``_engine`` and ``_writer``, so a test that
    reads them off the controller afterwards sees None and can prove
    nothing about the drain or the container. Both subclass the real
    types, so the production path is unchanged.
    """
    made = {}
    real_engine = manual_recording_module.VideoRecordingEngine
    real_writer = manual_recording_module.VideoWriter

    class _CapturingEngine(real_engine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            made['engine'] = self

    class _CapturingWriter(real_writer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            made['writer'] = self
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            super().close()

    monkeypatch.setattr(manual_recording_module, 'VideoRecordingEngine', _CapturingEngine)
    monkeypatch.setattr(manual_recording_module, 'VideoWriter', _CapturingWriter)
    return made


def _fail_the_finish_thread(monkeypatch):
    """Raise at the last statement of start(), after every commit.

    Thread exhaustion at ``_finish_thread.start()`` is the only reachable
    raise site at or after the listener registration, so it is the one
    that exercises the unwind's listener removal. Keyed on the thread's
    production name: the engine's writer lane is constructed in the same
    window and must still start, or the claim could never be released.
    """
    real_thread = threading.Thread

    def _thread(*args, **kwargs):
        if kwargs.get('name') == 'ManualRecordingFinish':
            raise RuntimeError('scripted thread exhaustion')
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(manual_recording_module.threading, 'Thread', _thread)


class TestFailedStartUnwind:
    """A start that raises after the engine committed must leave nothing
    behind: no claim, no listener, no open container, no empty artifact.

    These pin the controller half of the unwind. The engine half has its
    own contract tests; the two frames hold different resources, so
    neither set covers the other.
    """

    def test_post_commit_raise_releases_claim(self, tmp_path, monkeypatch):
        made = _capture_engine_and_writer(monkeypatch)
        controller, _scope, _clock = make_controller(tmp_path)
        _fail_the_finish_thread(monkeypatch)

        with pytest.raises(RuntimeError):
            controller.start()

        # The unwind ends the recording without waiting for the drain, so
        # the release lands on the writer lane rather than in start().
        assert made['engine'].wait_for_drain(timeout=5)
        assert controller._claim.owner is None
        assert not controller.is_busy

    def test_post_commit_raise_removes_listener(self, tmp_path, monkeypatch):
        made = _capture_engine_and_writer(monkeypatch)
        controller, scope, _clock = make_controller(tmp_path)
        _fail_the_finish_thread(monkeypatch)

        with pytest.raises(RuntimeError):
            controller.start()

        assert scope.imaging.listener is None
        # The unwind deliberately does not wait for the drain, so the lane
        # outlives start(). Join it here or it runs on into later tests.
        assert made['engine'].wait_for_drain(timeout=5)

    def test_pre_registration_raise_unwinds_without_a_listener(self, tmp_path, monkeypatch):
        # The removal must tolerate a callable that was never added: the
        # registration sits late in start(), so an earlier raise reaches
        # the unwind with nothing to remove.
        made = _capture_engine_and_writer(monkeypatch)
        controller, scope, _clock = make_controller(tmp_path)

        def _explode(*_args, **_kwargs):
            raise ValueError('scripted stall-watch failure')

        monkeypatch.setattr(manual_recording_module, 'StallWatch', _explode)

        with pytest.raises(ValueError):
            controller.start()

        assert scope.imaging.listener is None
        assert made['engine'].wait_for_drain(timeout=5)
        assert controller._claim.owner is None

    def test_failed_start_closes_container(self, tmp_path, monkeypatch):
        made = _capture_engine_and_writer(monkeypatch)
        controller, _scope, _clock = make_controller(tmp_path, video_as_frames=False)
        _fail_the_finish_thread(monkeypatch)

        with pytest.raises(RuntimeError):
            controller.start()

        assert made['writer'].close_calls == 1
        assert made['engine'].wait_for_drain(timeout=5)

    def test_failed_start_leaves_no_zero_frame_mp4(self, tmp_path, monkeypatch):
        # False color ON is the discriminator, not decoration. The writer
        # eager-opens its container at construction only when the label
        # applies a chromatic map, so this is the configuration where a
        # failed start has a real file to leave behind -- with the toggle
        # off the encoder init defers and the assertion would hold for a
        # reason that has nothing to do with the unwind.
        made = _capture_engine_and_writer(monkeypatch)
        controller, _scope, _clock = make_controller(tmp_path, video_as_frames=False, lit='Blue')
        _fail_the_finish_thread(monkeypatch)

        with pytest.raises(RuntimeError):
            controller.start(layer='Blue', false_color_on=True)

        # The file existed: close() alone would flush and close it, and
        # leave it on disk for the next recording to rename around.
        assert made['writer'].close_calls == 1
        assert not list((tmp_path / 'Manual').glob('*.mp4'))
        assert made['engine'].wait_for_drain(timeout=5)

    def test_read_only_save_location_refuses_before_the_commit(self, tmp_path):
        # Reserving the per-recording folder is what makes two same-second
        # recordings land in two folders, and it happens ahead of the commit
        # point so the config can carry the name actually taken. That moves an
        # unwritable save location from a post-commit raise to a clean refusal:
        # the user gets a message and the button back, with nothing started.
        controller, scope, _clock = make_controller(tmp_path)
        tmp_path.chmod(0o500)
        try:
            with pytest.raises(RecordingRefusedError) as excinfo:
                controller.start()
        finally:
            tmp_path.chmod(0o700)

        assert excinfo.value.reason == 'capture_location_unusable'
        # Nothing was started: no listener, no claim, and the button is free.
        assert scope.imaging.listener is None
        assert controller._claim.owner is None
        assert not controller.is_busy
        # A refused recording leaves no folder behind.
        assert not (tmp_path / 'Manual').exists()

    def test_listener_is_not_registered_before_the_commit_point(self, tmp_path, monkeypatch):
        # Registering the listener ahead of engine.start was proposed and
        # refused: the frame gate reads self._engine, assigned after the
        # commit, so an early listener routes frames into a writer that
        # does not exist yet. Pin the ordering so it cannot return.
        seen = {}
        real_start = manual_recording_module.VideoRecordingEngine.start
        controller, scope, _clock = make_controller(tmp_path)

        def _spy(engine_self, config):
            seen['listener_at_commit'] = scope.imaging.listener
            return real_start(engine_self, config)

        monkeypatch.setattr(manual_recording_module.VideoRecordingEngine, 'start', _spy)

        controller.start()
        assert seen['listener_at_commit'] is None
        assert scope.imaging.listener is not None

        controller.stop()
        finish(controller)


def _block_the_finish(monkeypatch):
    """Park the finish thread inside the hyperstack build.

    This is the window the guard exists for: selection is closed, the
    drain is over and the engine has already released its claim, so the
    engine's own second-capture guard would admit a new recording -- while
    the finish thread is still reading this controller's per-recording
    state.
    """
    entered = threading.Event()
    release = threading.Event()

    class _BlockingBuilder:
        def __init__(self, **kwargs):
            pass

        def create_single_recording_stack(self, df, path, output_file_loc):
            entered.set()
            release.wait(timeout=15)
            return {'status': True, 'error': None, 'metadata': {}}

    monkeypatch.setattr(manual_recording_module, 'StackBuilder', _BlockingBuilder)
    return entered, release


class TestExclusivitySpansTheFinish:
    def test_second_start_during_finish_is_refused(self, tmp_path, monkeypatch):
        entered, release = _block_the_finish(monkeypatch)
        controller, scope, clock = make_controller(tmp_path, hyperstack=True, lit='Blue')
        try:
            controller.start(layer='Blue', false_color_on=True)
            feed_frames(scope, clock, 3, fps=10.0)
            controller.stop()
            assert entered.wait(timeout=10)

            # The engine would NOT refuse here -- selection is closed, the
            # drain is done and the claim is free. Only the controller
            # knows the recording is not over.
            assert not controller.is_recording
            assert not controller.is_draining
            assert controller._claim.owner is None
            assert controller.is_busy

            with pytest.raises(RecordingRefusedError) as e:
                controller.start()
            assert e.value.reason == 'recording_active'
        finally:
            release.set()
        finish(controller)

    def test_refusal_during_finish_emits_no_fps_warning(self, tmp_path, monkeypatch):
        # The guard has to precede the rate clamp, which pops a warning
        # before the last refusal: a refused start that still nags the
        # user about their FPS budget has already run half of start().
        entered, release = _block_the_finish(monkeypatch)
        controller, scope, clock = make_controller(
            tmp_path, hyperstack=True, lit='Blue', max_fps=20
        )
        try:
            controller.start(layer='Blue', false_color_on=True)
            feed_frames(scope, clock, 3, fps=10.0)
            controller.stop()
            assert entered.wait(timeout=10)

            # Installed only now: the first start legitimately warns.
            recorder = NotifyRecorder()
            monkeypatch.setattr(manual_recording_module, 'notifications', recorder)
            with pytest.raises(RecordingRefusedError):
                controller.start()
            assert recorder.calls == []
        finally:
            release.set()
        finish(controller)


class TestFinishThreadResilience:
    def test_finish_thread_survives_a_finalize_failure(self, tmp_path, monkeypatch):
        # The finish thread must complete even when the engine never
        # produced a result. result() raises in exactly that case, and it
        # used to sit upstream of the container close and the completion
        # callback -- so a failed finalize left an unclosed MP4 and a UI
        # stuck in its recording state, with the drain already over.
        made = _capture_engine_and_writer(monkeypatch)
        recorder = NotifyRecorder()
        monkeypatch.setattr(manual_recording_module, 'notifications', recorder)
        controller, scope, clock = make_controller(tmp_path, video_as_frames=False)

        def _explode(_self):
            raise RuntimeError('scripted finalize failure')

        monkeypatch.setattr(
            manual_recording_module.VideoRecordingEngine, '_finalize_locked', _explode
        )

        completed = threading.Event()
        controller.start(on_complete=completed.set)
        feed_frames(scope, clock, 3, fps=10.0)
        controller.stop()
        finish(controller)

        assert completed.is_set()
        assert made['writer'].close_calls == 1
        assert 'error' in recorder.severities()
        assert not controller.is_busy


class TestDurationCap:
    def test_tick_stops_at_max_duration(self, tmp_path):
        controller, scope, clock = make_controller(tmp_path, duration_s=2)
        controller.start()
        feed_frames(scope, clock, 3, fps=10.0)
        assert controller.is_recording
        clock.advance(2.0)
        controller.tick()
        assert not controller.is_recording
        finish(controller)


class TestLossIsNotified:
    def test_write_failure_notifies_short_video(self, tmp_path, monkeypatch):
        # A frame lost to a write error costs that frame and the finish
        # must say so -- the invariant the legacy finalize guards pinned.
        controller, scope, clock = make_controller(tmp_path)
        recorder = NotifyRecorder()
        monkeypatch.setattr(manual_recording_module, 'notifications', recorder)

        real_write = manual_recording_module.image_save.write_video_frame
        calls = {'n': 0}

        def _flaky_write(**kwargs):
            calls['n'] += 1
            if calls['n'] == 2:
                raise OSError('scripted write failure')
            return real_write(**kwargs)

        monkeypatch.setattr(manual_recording_module.image_save, 'write_video_frame', _flaky_write)
        controller.start()
        feed_frames(scope, clock, 5, fps=10.0)
        controller.stop()
        finish(controller)

        folder = next((tmp_path / 'Manual').glob('Video_*'))
        assert len(list(folder.glob('ManualVideo_Frame_*.tiff'))) == 4
        manifest = json.loads((folder / 'recording_manifest.json').read_text())
        assert manifest['write_failures'] == 1
        assert 'warning' in recorder.severities()

    def test_finish_failure_notifies(self, tmp_path, monkeypatch):
        # A post-drain finish failure (here the hyperstack build) must
        # reach the user, never vanish behind a log line.
        controller, scope, clock = make_controller(tmp_path, hyperstack=True)
        recorder = NotifyRecorder()
        monkeypatch.setattr(manual_recording_module, 'notifications', recorder)

        class _ExplodingBuilder:
            def __init__(self, **kwargs):
                raise RuntimeError('scripted hyperstack failure')

        monkeypatch.setattr(manual_recording_module, 'StackBuilder', _ExplodingBuilder)
        controller.start()
        feed_frames(scope, clock, 3, fps=10.0)
        controller.stop()
        finish(controller)
        assert 'error' in recorder.severities()


def _capture_hyperstack_df(monkeypatch):
    """Capture the DataFrame the hyperstack build receives.

    The rows carry channel IDENTITY; asserting on them is what separates
    identity from the false-color toggle, which is the whole defect.
    """
    captured = {}

    class _CapturingBuilder:
        def __init__(self, **kwargs):
            pass

        def create_single_recording_stack(self, df, path, output_file_loc):
            captured['df'] = df
            return {'status': True, 'error': None, 'metadata': {}}

    monkeypatch.setattr(manual_recording_module, 'StackBuilder', _CapturingBuilder)
    return captured


class TestChannelIdentity:
    """Identity is what was imaged; the false-color toggle is how it is
    displayed. They are independent, and the hyperstack needs identity
    whether or not the toggle is on.
    """

    def test_brightfield_toggle_off_writes_bf_identity(self, tmp_path, monkeypatch):
        # The common case and the default: BF lit, false color off. Every
        # row must carry 'BF', not None -- a null Color makes nunique() 0
        # and the hyperstack never builds.
        captured = _capture_hyperstack_df(monkeypatch)
        controller, scope, clock = make_controller(tmp_path, hyperstack=True, lit='BF')
        controller.start(layer='BF', false_color_on=False)
        feed_frames(scope, clock, 3, fps=10.0)
        controller.stop()
        finish(controller)

        assert list(captured['df']['Color']) == ['BF', 'BF', 'BF']

    def test_lit_channel_wins_over_the_open_accordion(self, tmp_path, monkeypatch):
        # A channel can be lit with a different accordion open -- the LED
        # is the evidence of what was imaged, so it wins.
        captured = _capture_hyperstack_df(monkeypatch)
        controller, scope, clock = make_controller(tmp_path, hyperstack=True, lit='Blue')
        controller.start(layer='BF', false_color_on=False)
        feed_frames(scope, clock, 2, fps=10.0)
        controller.stop()
        finish(controller)

        assert list(captured['df']['Color']) == ['Blue', 'Blue']

    def test_luminescence_records_with_zero_lit_channels(self, tmp_path, monkeypatch):
        # Lumi is legitimately unlit; the open layer names it.
        captured = _capture_hyperstack_df(monkeypatch)
        controller, scope, clock = make_controller(tmp_path, hyperstack=True, lit=None)
        controller.start(layer='Lumi', false_color_on=False)
        feed_frames(scope, clock, 2, fps=10.0)
        controller.stop()
        finish(controller)

        assert list(captured['df']['Color']) == ['Lumi', 'Lumi']

    def test_startup_state_defaults_to_brightfield(self, tmp_path, monkeypatch):
        # Nothing lit and no layer selected is LumaViewPro's startup state.
        # It records as BF -- it is not refused and never writes a null.
        captured = _capture_hyperstack_df(monkeypatch)
        controller, scope, clock = make_controller(tmp_path, hyperstack=True, lit=None)
        controller.start()
        feed_frames(scope, clock, 2, fps=10.0)
        controller.stop()
        finish(controller)

        assert list(captured['df']['Color']) == ['BF', 'BF']

    def test_no_led_board_falls_back_to_the_open_layer(self, tmp_path, monkeypatch):
        # get_led_states() is empty with no board, so the open layer stands.
        captured = _capture_hyperstack_df(monkeypatch)
        scope = _FakeScope(board=False)
        controller, scope, clock = make_controller(tmp_path, scope=scope, hyperstack=True)
        controller.start(layer='Green', false_color_on=True)
        feed_frames(scope, clock, 2, fps=10.0)
        controller.stop()
        finish(controller)

        assert list(captured['df']['Color']) == ['Green', 'Green']

    def test_brightfield_hyperstack_is_actually_written(self, tmp_path):
        # End to end with the REAL builder: the defect this fix exists for
        # is a brightfield recording producing no hyperstack file at all,
        # and every other test here substitutes the builder. Nothing else
        # exercises _create_stack on manual-shaped rows.
        controller, scope, clock = make_controller(tmp_path, hyperstack=True, lit='BF')
        controller.start(layer='BF', false_color_on=False)
        feed_frames(scope, clock, 3, fps=10.0)
        controller.stop()
        finish(controller)

        folder = next((tmp_path / 'Manual').glob('Video_*'))
        assert (folder / MANUAL_HYPERSTACK_FILENAME).exists()

    def test_toggle_off_keeps_manifest_channel_color_null(self, tmp_path):
        # REGRESSION PIN. channel_color is a RENDERING field: video_builder
        # reads it back to decide whether to colorize a rebuild. Writing
        # identity here would false-color a recording saved mono.
        controller, scope, clock = make_controller(tmp_path, lit='Blue')
        controller.start(layer='Blue', false_color_on=False)
        feed_frames(scope, clock, 2, fps=10.0)
        controller.stop()
        finish(controller)

        folder = next((tmp_path / 'Manual').glob('Video_*'))
        manifest = json.loads((folder / 'recording_manifest.json').read_text())
        assert manifest['channel_color'] is None

    def test_toggle_on_still_carries_channel_color(self, tmp_path):
        # The other half of the pin: rendering on still records the color.
        controller, scope, clock = make_controller(tmp_path, lit='Blue')
        controller.start(layer='Blue', false_color_on=True)
        feed_frames(scope, clock, 2, fps=10.0)
        controller.stop()
        finish(controller)

        folder = next((tmp_path / 'Manual').glob('Video_*'))
        manifest = json.loads((folder / 'recording_manifest.json').read_text())
        assert manifest['channel_color'] == 'Blue'


class TestCloseWithProgress:
    def test_is_busy_spans_recording_drain_and_finish(self, tmp_path):
        controller, scope, clock = make_controller(tmp_path)
        assert not controller.is_busy
        controller.start()
        assert controller.is_busy
        feed_frames(scope, clock, 3, fps=10.0)
        controller.stop()
        finish(controller)
        assert not controller.is_busy

    def test_discard_pending_releases_busy(self, tmp_path):
        controller, scope, clock = make_controller(tmp_path)
        controller.start()
        feed_frames(scope, clock, 3, fps=10.0)
        controller.discard_pending()
        finish(controller)
        assert not controller.is_busy

    def test_app_close_gate_reads_the_controller(self):
        # The close hook must consult the recording controller and route
        # through the progress-with-discard flow; kv/Window plumbing has
        # no headless seam, so pin the wiring on source.
        repo = manual_recording_module.Path(__file__).resolve().parent.parent
        app_src = (repo / 'lumaviewpro.py').read_text()
        assert 'recording.is_busy' in app_src
        assert '_close_with_drain_progress' in app_src
        assert 'show_blocking_progress_popup' in app_src
        assert 'Discard Remaining Frames' in app_src


class TestScratchSweep:
    def test_leftover_scratch_deleted(self, tmp_path):
        scratch = tmp_path / 'recording_temp.dat'
        scratch.write_bytes(b'x' * 4096)
        manual_recording_module.sweep_recording_scratch(tmp_path)
        assert not scratch.exists()

    def test_no_scratch_is_a_quiet_no_op(self, tmp_path):
        manual_recording_module.sweep_recording_scratch(tmp_path)


class TestDiskFloor:
    def test_floor_breach_stops_selection_keeps_frames(self, tmp_path, monkeypatch):
        controller, scope, clock = make_controller(tmp_path)
        recorder = NotifyRecorder()
        monkeypatch.setattr(manual_recording_module, 'notifications', recorder)

        checks = {'n': 0}

        def _fake_check(path, required_mb):
            checks['n'] += 1
            # Pre-flight passes; every rolling probe reports a full disk.
            return (checks['n'] == 1, 100.0)

        monkeypatch.setattr(manual_recording_module, 'check_disk_space_ok', _fake_check)
        controller.start()
        feed_frames(scope, clock, 5, fps=10.0)
        controller._engine.wait_for_drain(timeout=10)
        finish(controller)

        assert not controller.is_recording
        assert 'error' in recorder.severities()
        folder = next((tmp_path / 'Manual').glob('Video_*'))
        assert list(folder.glob('ManualVideo_Frame_*.tiff')), 'drained frames must stay'


class TestD15LiveSettingsBackDoor:
    """Loading a protocol writes video config into LIVE settings while a
    recording can be running (the layer-control load path). The running
    recording is immune by construction: its RecordingConfig snapshot was
    baked at start and neither the controller nor the engine re-reads
    live settings mid-recording."""

    def test_mid_recording_settings_mutation_is_harmless(self, tmp_path):
        controller, scope, clock = make_controller(tmp_path, max_fps=0, duration_s=60)
        controller.start()
        baked = controller._config

        # The back door: a protocol load rewrites the live video settings
        # (rate cap, duration, overlay) under the running recording.
        controller._settings['video'].update(
            {'max_fps': 1, 'max_duration_seconds': 1, 'timestamp_overlay': False}
        )
        controller._settings['video_as_frames'] = False

        feed_frames(scope, clock, 5, fps=10.0)
        clock.advance(2.0)  # past the mutated 1 s duration, far under the baked 60 s
        controller.tick()
        assert controller.is_recording, (
            'the duration cap must read the baked snapshot, not the mutated setting'
        )

        controller.stop()
        finish(controller)
        result = controller._engine.result()
        assert controller._config is baked, 'the snapshot object must never be rebuilt mid-run'
        assert result.configured_fps == baked.fps, (
            'the manifest must carry the rate baked at start, not the mutated cap'
        )
        assert result.frames_selected == 5, (
            'every delivered frame within the baked budget must stay selected'
        )


class TestFeedLossDetection:
    """tick() ends a recording whose camera died -- by event or silently.

    The disconnect latch is the fast path; the stall watch catches the
    feed that stops delivering while active_cached stays True. Either
    way the user gets one loss notification and kept frames stay.
    """

    def _recording_controller(self, tmp_path, monkeypatch):
        notify = NotifyRecorder()
        monkeypatch.setattr(manual_recording_module, 'notifications', notify)
        controller, scope, clock = make_controller(tmp_path)  # 100 ms -> 10 fps
        controller.start()
        feed_frames(scope, clock, 3)
        return controller, scope, clock, notify

    def test_silently_stalled_feed_stops_with_a_loss_notification(self, tmp_path, monkeypatch):
        controller, _scope, clock, notify = self._recording_controller(tmp_path, monkeypatch)
        # Threshold is the 5 s floor (10 fps, 0.1 s exposure). Ticks
        # before it: recording continues; past it with no new frames:
        # the feed is dead.
        controller.tick()
        clock.advance(4.0)
        controller.tick()
        assert controller.is_recording
        clock.advance(1.5)
        controller.tick()
        assert not controller.is_recording
        errors = [c for c in notify.calls if c[0] == 'error']
        assert len(errors) == 1
        assert 'Recording Stopped' in errors[0][1]
        finish(controller)
        manifest = json.loads(
            next((controller.save_folder).glob('recording_manifest.json')).read_text()
        )
        assert manifest['end_reason'] == 'camera_stalled'

    def test_disconnect_stops_immediately_without_waiting_for_the_threshold(
        self, tmp_path, monkeypatch
    ):
        controller, scope, clock, notify = self._recording_controller(tmp_path, monkeypatch)
        scope.imaging.active_cached = False
        clock.advance(0.2)
        controller.tick()
        assert not controller.is_recording
        errors = [c for c in notify.calls if c[0] == 'error']
        assert len(errors) == 1
        finish(controller)

    def test_healthy_feed_keeps_recording_through_ticks(self, tmp_path, monkeypatch):
        # Preservation guard: frames arriving between ticks never trip
        # the watch, and the duration cap still owns the normal end.
        controller, scope, clock, notify = self._recording_controller(tmp_path, monkeypatch)
        for _ in range(4):
            clock.advance(3.0)
            feed_frames(scope, clock, 2)
            controller.tick()
        assert controller.is_recording
        assert not [c for c in notify.calls if c[0] == 'error']
        controller.stop()
        finish(controller)
