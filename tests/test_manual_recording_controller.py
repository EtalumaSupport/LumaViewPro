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

import numpy as np
import pytest

import modules.manual_recording as manual_recording_module
from modules.exceptions import RecordingRefusedError
from modules.manual_recording import ManualRecordingController
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
        self.camera_active = True
        self.camera_exposure_ms = exposure_ms
        self._frame_size = {'width': width, 'height': height}
        self.listener = None

    @property
    def camera_frame_size(self):
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
        if self.listener is cb:
            self.listener = None


class _FakeMotion:
    def get_current_position(self):
        return {'X': 1.0, 'Y': 2.0, 'Z': 3.0}

    def has_turret(self):
        return False


class _FakeScope:
    def __init__(self, **imaging_kwargs):
        self.imaging = _FakeImaging(**imaging_kwargs)
        self.motion = _FakeMotion()


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


def make_controller(tmp_path, *, scope=None, clock=None, **settings_kwargs):
    scope = scope or _FakeScope()
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
        host_ts = clock() + (jitter[i % len(jitter)] if jitter else 0.0)
        chunks = {'Timestamp': int(clock() * TICK_HZ), 'FrameID': i}
        image = np.full((height, width), i % 256, dtype=np.uint8)
        scope.imaging.listener(image, host_ts, chunks)


def finish(controller, timeout=15.0):
    thread = controller._finish_thread
    if thread is not None:
        thread.join(timeout)
        assert not thread.is_alive(), 'finish thread did not complete'


class TestStartRefusals:
    def test_inactive_camera_refused(self, tmp_path):
        controller, scope, _ = make_controller(tmp_path)
        scope.imaging.camera_active = False
        with pytest.raises(RecordingRefusedError) as e:
            controller.start()
        assert e.value.reason == 'camera_inactive'

    def test_unknown_exposure_refused(self, tmp_path):
        controller, scope, _ = make_controller(tmp_path)
        scope.imaging.camera_exposure_ms = 0.0
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
        scope.imaging.camera_active = False
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
        controller, scope, clock = make_controller(tmp_path)
        controller.start(false_color='Blue')
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
