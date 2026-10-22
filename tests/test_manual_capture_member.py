# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A still is captured and saved by one Session member, headless.

The GUI's Capture button used to decide the channel, folder, name,
summing, format, depth and overlay copy itself, reaching past the API to
do it, so a script or REST caller had no way to make the same file.
`session.manual_capture.capture` makes every one of those decisions from
the session's settings and its shown channel; the caller names only its
false-colour state and the overlays it wants.

Run against the simulator with the lanes the factory builds and starts, so
the capture goes through the real camera lane -- started once: a second
start puts a second worker on a lane, and a still would no longer be
serialized against the camera writes these tests queue beside it.
"""

import concurrent.futures
import contextlib
import pathlib
import threading
import time

import numpy as np
import pytest

import modules.image_mode as image_mode
from modules.exceptions import (
    CaptureFailedError,
    HardwareCommandRefusedError,
)
from modules.image_utils import read_postproc_input_metadata, read_tiff_with_legacy_collapse
from modules.sequential_io_executor import IOTask
from tests.scope_fakes import home_sim_scope
from tests.test_composite_run_e2e import headless_settings
from tests.protocol_drives import run_identity

RESULT_TIMEOUT_S = 30.0


def _settings(tmp_path, **overrides):
    settings = headless_settings(tmp_path, **overrides)
    settings['separate_folder_per_channel'] = False
    return settings


@contextlib.contextmanager
def _open_session(settings):
    from modules.scope_session import ScopeSession

    session = ScopeSession.create(settings, simulate=True)
    try:
        scope = session.scope
        scope._led_driver.set_timing_mode('fast')
        scope._motion_driver.set_timing_mode('fast')
        scope._camera_driver.set_timing_mode('fast')
        home_sim_scope(scope)
        scope.illumination.leds_off()
        yield session
    finally:
        session.shutdown()


@pytest.fixture
def still_session(tmp_path):
    with _open_session(_settings(tmp_path)) as session:
        yield session, tmp_path


def over_the_first_well(session) -> str:
    """Move the stage over the plate's first well and return its label.

    A homed stage is off the plate's wells, and a position off them names
    no well, so a test about the well a file records puts the stage over one.
    """
    scope = session.scope
    x_mm, y_mm = scope.runtime_state.get_labware().get_well_position(0, 0)
    stage = scope.runtime_state.plate_to_stage(x_mm, y_mm)
    scope.motion.move_xy(x_um=stage.x_um)
    scope.motion.move_xy(y_um=stage.y_um)
    label = scope.runtime_state.get_well_label()
    assert label, 'the stage was moved over a well'
    return label


def _capture(session, **kwargs):
    kwargs.setdefault('false_color_on', False)
    return session.manual_capture.capture(**kwargs).result(timeout=RESULT_TIMEOUT_S)


def _files(session, **kwargs):
    return [still.file for still in _capture(session, **kwargs)]


class _LaneBlocker:
    """Holds the camera lane with a task of its own until released."""

    def __init__(self, session):
        self._release = threading.Event()
        self._running = threading.Event()
        session.camera_executor.put(IOTask(action=self._hold))
        assert self._running.wait(5.0), 'the blocker never reached the camera lane'

    def _hold(self):
        self._running.set()
        self._release.wait(RESULT_TIMEOUT_S)

    def release(self):
        self._release.set()


def _wait_until_queued(session, depth):
    deadline = time.monotonic() + 5.0
    while session.camera_executor.queue_size() < depth:
        assert time.monotonic() < deadline, 'the still never reached the camera lane'
        time.sleep(0.01)


class TestWhereTheFileLands:
    def test_a_fresh_session_shows_brightfield_and_names_it_in_manual(self, still_session):
        session, tmp_path = still_session
        well = over_the_first_well(session)

        stills = _capture(session)

        assert len(stills) == 1
        path = stills[0].file
        assert isinstance(path, pathlib.Path)
        assert path.parent == tmp_path / 'Manual'
        assert path.name == f'live_{well}_BF_000001.tiff'
        assert path.is_file()

    def test_a_lost_position_saves_the_image_without_a_well_and_says_so_once(
        self, still_session, monkeypatch
    ):
        # After a failed home the target cache keeps its last well, real and
        # stale. The image is still real, so it is saved -- without the well.
        from modules.lumascope_api import AxisState
        from modules.notification_center import notifications

        session, _ = still_session
        shown = []
        monkeypatch.setattr(
            notifications,
            'report_outcome',
            lambda outcome, **kwargs: shown.append(outcome.title),
        )
        with session.scope.motion._axis_state_lock:
            session.scope.motion._axis_state['X'] = AxisState.UNKNOWN

        stills = _capture(session)

        assert [s.file.name for s in stills] == ['live_BF_000001.tiff']
        assert stills[0].file.is_file()
        assert stills[0].well_label is None
        assert shown == ['Position Not Recorded']

    def test_the_shown_channel_names_the_still_when_nothing_is_lit(self, still_session):
        session, _ = still_session
        session.show_channel('Blue')

        (path,) = _files(session)

        assert '_Blue_' in path.name
        assert read_postproc_input_metadata(path)['channel'] == 'Blue'

    def test_a_lit_led_outranks_the_shown_channel(self, still_session):
        session, _ = still_session
        session.show_channel('Blue')
        session.scope.illumination.led_on('Green', 50.0)

        (path,) = _files(session)

        assert '_Green_' in path.name
        assert read_postproc_input_metadata(path)['channel'] == 'Green'

    def test_a_second_still_takes_the_next_number(self, still_session):
        session, _ = still_session

        (first,) = _files(session)
        (second,) = _files(session)

        assert first.name.endswith('_000001.tiff')
        assert second.name.endswith('_000002.tiff')

    def test_a_folder_per_channel(self, tmp_path):
        settings = _settings(tmp_path)
        settings['separate_folder_per_channel'] = True
        with _open_session(settings) as session:
            session.show_channel('Red')
            (path,) = _files(session)

        assert path.parent == tmp_path / 'Manual' / 'Red'

    def test_the_live_format_is_the_settings(self, tmp_path):
        with _open_session(_settings(tmp_path, live_format='JPG')) as session:
            (path,) = _files(session)

        assert path.suffix == '.jpg'
        assert path.read_bytes()[:2] == b'\xff\xd8'


class TestOverlays:
    def test_an_overlay_is_a_second_file_from_the_same_capture(self, still_session):
        session, _ = still_session

        raw, overlay = _files(session, crosshairs=True)

        assert overlay.name == raw.name.replace('.tiff', '_overlay.tiff')
        raw_pixels = read_tiff_with_legacy_collapse(raw)
        overlay_pixels = read_tiff_with_legacy_collapse(overlay)
        center = raw_pixels.shape[1] // 2
        assert overlay_pixels[:, center - 1 : center + 1].min() == 255
        assert raw_pixels[:, center - 1 : center + 1].min() < np.iinfo(raw_pixels.dtype).max, (
            'the unmarked file carries the crosshairs'
        )

    def test_a_summed_capture_saves_its_overlay(self, tmp_path):
        """A lit 12-bit frame summed three times peaks above 4095; the
        overlay's 8-bit rendering must scale against the summed depth, since
        scaling against the per-frame depth refuses the frame."""
        settings = _settings(tmp_path)
        settings['image_mode'] = image_mode.IMAGE_MODE_12BIT_SCIENTIFIC
        settings['BF']['sum'] = 3
        with _open_session(settings) as session:
            # At the sample: homing leaves Z at the floor, where the
            # simulated specimen is a defocused blur with no bright detail.
            sample_z = session.scope._camera_driver.get_nominal_focal_z()
            session.scope.motion.move_z(sample_z)
            session.scope.illumination.led_on('BF', 200.0)
            paths = _files(session, bullseye=True, crosshairs=True)
            peak = int(read_tiff_with_legacy_collapse(paths[0]).max())

        assert peak > 4095, f'the summed frame never left the per-frame range (peak {peak})'

        assert len(paths) == 2
        assert all(p.is_file() for p in paths)


class TestRefusalsAndFailures:
    def test_a_still_has_no_stop_so_stop_leaves_it_to_finish(self, still_session):
        """A still is an ``Operation`` with no stop: ``stop()`` while it is
        on the lane does nothing, and the still is written (Eric 2026-10-13:
        refusing it "seems rude")."""
        session, _ = still_session
        blocker = _LaneBlocker(session)
        try:
            still = session.manual_capture.capture(false_color_on=False)
            assert still.stop() is None
            assert not still.done()
        finally:
            blocker.release()
        assert still.result(timeout=RESULT_TIMEOUT_S)[0].file.is_file()

    def test_a_second_still_while_one_is_in_flight_is_refused(self, still_session):
        session, _ = still_session
        blocker = _LaneBlocker(session)
        try:
            first = session.manual_capture.capture(false_color_on=False)
            with pytest.raises(HardwareCommandRefusedError) as refused:
                session.manual_capture.capture(false_color_on=False)
            assert refused.value.reason == 'capture_in_flight'
        finally:
            blocker.release()
        assert first.result(timeout=RESULT_TIMEOUT_S)[0].file.is_file()
        assert not session.manual_capture.in_flight

    def test_the_guard_holds_until_the_lane_body_ends_not_the_callers_wait(self, tmp_path):
        """A slow still -- summed frames, each settled for its exposure --
        is still running on the lane when its caller stops waiting; a second
        press then must not start a still beside it."""
        settings = _settings(tmp_path)
        settings['BF']['sum'] = 3
        settings['BF']['exposure_ms'] = 400.0
        with _open_session(settings) as session:
            body_running = threading.Event()
            original = session.manual_capture._capture_and_save

            def observed(request):
                body_running.set()
                return original(request)

            session.manual_capture._capture_and_save = observed
            first = session.manual_capture.capture(false_color_on=False)
            assert body_running.wait(5.0), 'the still never started on the lane'
            with pytest.raises(concurrent.futures.TimeoutError):
                first.result(timeout=0.1)

            assert session.manual_capture.in_flight
            with pytest.raises(HardwareCommandRefusedError) as refused:
                session.manual_capture.capture(false_color_on=False)
            assert refused.value.reason == 'capture_in_flight'
            first.result(timeout=RESULT_TIMEOUT_S)
            assert not session.manual_capture.in_flight

    def test_a_still_while_a_run_holds_the_camera_is_refused_at_the_call(self, still_session):
        session, _ = still_session
        held = session.activity_claim.try_claim('protocol', run=run_identity())
        assert held is not None
        try:
            with pytest.raises(HardwareCommandRefusedError) as refused:
                session.manual_capture.capture(false_color_on=False)
        finally:
            held.release()
        assert refused.value.reason == 'exclusive_activity_running'
        assert not session.manual_capture.in_flight, 'a refused still kept the guard'

    def test_a_still_the_lane_drops_unrun_settles_and_frees_the_guard(self, still_session):
        """The lane drops what it holds when the scope goes away; a still it
        dropped must say so, not leave its caller and the guard waiting."""
        session, _ = still_session
        blocker = _LaneBlocker(session)
        try:
            future = session.manual_capture.capture(false_color_on=False)
            _wait_until_queued(session, 1)
            session.camera_executor.clear_pending()
        finally:
            blocker.release()
        with pytest.raises(concurrent.futures.CancelledError):
            future.result(timeout=RESULT_TIMEOUT_S)
        assert not session.manual_capture.in_flight, 'a dropped still kept the guard'

    def test_no_frame_raises_the_captures_own_failure(self, still_session):
        session, tmp_path = still_session
        session.scope.simulation.stall_stream(0.0, 600.0)

        future = session.manual_capture.capture(false_color_on=False)

        with pytest.raises(CaptureFailedError) as failed:
            future.result(timeout=RESULT_TIMEOUT_S)
        assert failed.value.reason == 'no_frame', (
            "the still raises the capture's own reason, not a save-time placeholder"
        )
        assert not list((tmp_path / 'Manual').glob('*.tiff'))
        assert not session.manual_capture.in_flight

    def test_a_stopped_feed_is_refused_as_not_streaming(self, still_session):
        session, tmp_path = still_session
        session.scope._camera_driver.stop_grabbing()

        future = session.manual_capture.capture(false_color_on=False)

        with pytest.raises(HardwareCommandRefusedError) as refused:
            future.result(timeout=RESULT_TIMEOUT_S)
        assert refused.value.reason == 'camera_not_streaming'
        assert not list((tmp_path / 'Manual').glob('*.tiff'))
        assert not session.manual_capture.in_flight


class TestTheRecordIsTheCapturesMoment:
    def test_a_gain_write_queued_during_the_still_does_not_reach_its_record(self, still_session):
        session, _ = still_session
        session.scope.camera.set_gain_db(1.0)
        blocker = _LaneBlocker(session)
        try:
            future = session.manual_capture.capture(false_color_on=False)
            # The still is queued behind the blocker before the write is sent.
            _wait_until_queued(session, 1)
            writer = threading.Thread(target=session.scope.camera.set_gain_db, args=(20.0,))
            writer.start()
        finally:
            blocker.release()
        (still,) = future.result(timeout=RESULT_TIMEOUT_S)
        path = still.file
        writer.join(RESULT_TIMEOUT_S)

        assert read_postproc_input_metadata(path)['gain_db'] == pytest.approx(1.0)


class TestTheStillIsMadeAfterThePendingWrites:
    def test_a_still_after_an_led_write_is_drained_past_it(self, still_session):
        """A manual still saves a frame made after every pending change: an
        LED change (the owner's ``invalidate('led')``, which every LED write
        makes) leaves 'led' pending for its skip count, and the capture
        drains that many frames before the one it saves. A still grabbed
        without the drain (a bare get_image) would save a frame that may
        have integrated before the LED changed, and records no drain."""
        session, _ = still_session
        imaging = session.scope.imaging
        # A first still drains whatever bring-up left pending, so the
        # second has the LED change alone to wait out.
        _capture(session)
        imaging.frame_validity.invalidate('led', began_ns=None)

        _capture(session)

        info = imaging.last_capture_info
        assert info is not None, 'the still was not captured through the drain'
        assert info['drained'] >= imaging.frame_validity.SKIP_FRAMES['led'], (
            f'the still was not drained past the LED write: {info}'
        )


class TestAStillReturnsItsRecord:
    """A still's result is its file with the instrument's record of it,
    built at the capture, so a JPG -- whose file embeds none of it -- says
    where and how it was taken as a TIFF does."""

    # parametrize-justified: each format is its own writer branch, and a JPG
    # embeds no metadata, so its record is the capture's alone.
    @pytest.mark.parametrize('live_format', ['TIFF', 'OME-TIFF', 'JPG'])
    def test_each_format_returns_its_file_with_the_record(self, tmp_path, live_format):
        with _open_session(_settings(tmp_path, live_format=live_format)) as session:
            well = over_the_first_well(session)
            x_mm, y_mm = session.scope.runtime_state.get_labware().get_well_position(0, 0)
            objective_id, _ = session.scope.runtime_state.resolve_current_objective()

            (still,) = _capture(session)

        assert still.file.is_file()
        assert still.output_format == live_format
        assert still.channel == 'BF'
        assert still.objective_id == objective_id
        assert still.well_label == well
        assert still.plate_x_mm == pytest.approx(x_mm, abs=0.01)
        assert still.plate_y_mm == pytest.approx(y_mm, abs=0.01)
        assert still.stage_z_um is not None
        assert still.frame.frames_summed == 1
        assert still.significant_bits == 8
        assert 0 <= still.statistics.minimum <= still.statistics.maximum <= 255

    def test_the_statistics_are_the_saved_pixels(self, still_session):
        from modules.image_utils import image_statistics, load_pixels

        session, _ = still_session

        (still,) = _capture(session)

        pixels, bits, _ = load_pixels(still.file)
        assert bits == still.significant_bits
        assert still.statistics == image_statistics(pixels)

    def test_an_overlay_copy_states_its_own_pixels_and_its_frames_record(self, still_session):
        session, _ = still_session

        raw, overlay = _capture(session, crosshairs=True)

        assert overlay.frame == raw.frame
        assert overlay.significant_bits == 8
        # The crosshairs are drawn white, so the copy reaches full scale.
        assert overlay.statistics.maximum == 255
        assert overlay.statistics != raw.statistics

    def test_the_record_is_the_jobs_result_on_the_wire(self, still_session):
        from fastapi.testclient import TestClient

        from rest.app import build_app

        session, _ = still_session
        with TestClient(build_app(session)) as client:
            job = client.post(
                '/api/v1/manual_capture/capture', json={'false_color_on': False}
            ).json()
            ended = client.get(f'/api/v1/jobs/{job["id"]}', headers={'Prefer': 'wait=30'}).json()

        assert ended['status'] == 'completed', ended
        (still,) = ended['result']
        assert still['file']['name'] == 'Manual/live_BF_000001.tiff'
        assert still['channel'] == 'BF'
        assert still['frame']['frames_summed'] == 1
        assert set(still['statistics']) == {
            'minimum',
            'maximum',
            'mean',
            'standard_deviation',
            'percentile_1',
            'median',
            'percentile_99',
        }
