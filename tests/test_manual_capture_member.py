# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A still is captured and saved by one Session member, headless.

The GUI's Capture button used to decide the channel, folder, name,
summing, format, depth and overlay copy itself, reaching past the API to
do it, so a script or REST caller had no way to make the same file.
`session.manual_capture.capture` makes every one of those decisions from
the session's settings; the caller names only the open drawer, its
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
from modules.exceptions import CaptureError, HardwareCommandRefusedError
from modules.image_utils import read_postproc_input_metadata, read_tiff_with_legacy_collapse
from modules.lumascope_api.imaging import capture_failure_cause
from modules.sequential_io_executor import IOTask
from tests.scope_fakes import home_sim_scope
from tests.test_composite_run_e2e import headless_settings

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


def _capture(session, **kwargs):
    kwargs.setdefault('layer', None)
    kwargs.setdefault('false_color_on', False)
    return session.manual_capture.capture(**kwargs).result(timeout=RESULT_TIMEOUT_S)


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
    def test_no_drawer_and_no_led_is_brightfield_in_manual(self, still_session):
        session, tmp_path = still_session

        paths = _capture(session)

        assert len(paths) == 1
        path = paths[0]
        assert isinstance(path, pathlib.Path)
        assert path.parent == tmp_path / 'Manual'
        assert path.name == f'live_{session.scope.runtime_state.get_well_label()}_BF_000001.tiff'
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
            'warning',
            lambda category, title, message, **kwargs: shown.append(title),
        )
        with session.scope.motion._axis_state_lock:
            session.scope.motion._axis_state['X'] = AxisState.UNKNOWN

        paths = _capture(session)

        assert [p.name for p in paths] == ['live_BF_000001.tiff']
        assert paths[0].is_file()
        assert shown == ['Position Not Recorded']

    def test_the_open_drawer_names_the_channel_when_nothing_is_lit(self, still_session):
        session, _ = still_session

        (path,) = _capture(session, layer='Blue')

        assert '_Blue_' in path.name
        assert read_postproc_input_metadata(path)['channel'] == 'Blue'

    def test_a_lit_led_outranks_the_open_drawer(self, still_session):
        session, _ = still_session
        session.scope.illumination.led_on('Green', 50.0)

        (path,) = _capture(session, layer='Blue')

        assert '_Green_' in path.name
        assert read_postproc_input_metadata(path)['channel'] == 'Green'

    def test_a_second_still_takes_the_next_number_and_is_a_path(self, still_session):
        session, _ = still_session

        (first,) = _capture(session)
        (second,) = _capture(session)

        assert first.name.endswith('_000001.tiff')
        assert second.name.endswith('_000002.tiff')
        assert isinstance(second, pathlib.Path)

    def test_a_folder_per_channel(self, tmp_path):
        settings = _settings(tmp_path)
        settings['separate_folder_per_channel'] = True
        with _open_session(settings) as session:
            (path,) = _capture(session, layer='Red')

        assert path.parent == tmp_path / 'Manual' / 'Red'

    def test_the_live_format_is_the_settings(self, tmp_path):
        with _open_session(_settings(tmp_path, live_format='JPG')) as session:
            (path,) = _capture(session)

        assert path.suffix == '.jpg'
        assert path.read_bytes()[:2] == b'\xff\xd8'


class TestOverlays:
    def test_an_overlay_is_a_second_file_from_the_same_capture(self, still_session):
        session, _ = still_session

        raw, overlay = _capture(session, crosshairs=True)

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
            session.scope.illumination.led_on('BF', 200.0)
            paths = _capture(session, bullseye=True, crosshairs=True)
            peak = int(read_tiff_with_legacy_collapse(paths[0]).max())

        assert peak > 4095, f'the summed frame never left the per-frame range (peak {peak})'

        assert len(paths) == 2
        assert all(p.is_file() for p in paths)


class TestRefusalsAndFailures:
    def test_a_layer_that_is_not_a_channel_is_refused_before_anything_happens(self, tmp_path):
        settings = _settings(tmp_path)
        settings['separate_folder_per_channel'] = True
        with _open_session(settings) as session:
            with pytest.raises(ValueError, match='Foo'):
                session.manual_capture.capture(layer='Foo', false_color_on=False)
            assert not session.manual_capture.in_flight

        assert not (tmp_path / 'Manual').exists()

    def test_a_second_still_while_one_is_in_flight_is_refused(self, still_session):
        session, _ = still_session
        blocker = _LaneBlocker(session)
        try:
            first = session.manual_capture.capture(layer=None, false_color_on=False)
            with pytest.raises(HardwareCommandRefusedError) as refused:
                session.manual_capture.capture(layer=None, false_color_on=False)
            assert refused.value.reason == 'capture_in_flight'
        finally:
            blocker.release()
        assert first.result(timeout=RESULT_TIMEOUT_S)[0].is_file()
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
            first = session.manual_capture.capture(layer=None, false_color_on=False)
            assert body_running.wait(5.0), 'the still never started on the lane'
            with pytest.raises(concurrent.futures.TimeoutError):
                first.result(timeout=0.1)

            assert session.manual_capture.in_flight
            with pytest.raises(HardwareCommandRefusedError) as refused:
                session.manual_capture.capture(layer=None, false_color_on=False)
            assert refused.value.reason == 'capture_in_flight'
            first.result(timeout=RESULT_TIMEOUT_S)
            assert not session.manual_capture.in_flight

    def test_a_still_while_a_run_holds_the_camera_gets_the_lanes_refusal(self, still_session):
        session, _ = still_session
        session.camera_executor.disable()
        try:
            future = session.manual_capture.capture(layer=None, false_color_on=False)
            with pytest.raises(HardwareCommandRefusedError) as refused:
                future.result(timeout=RESULT_TIMEOUT_S)
        finally:
            session.camera_executor.enable()
        assert refused.value.reason == 'exclusive_activity_running'
        assert not session.manual_capture.in_flight, 'a refused still kept the guard'

    def test_no_frame_raises_with_the_engines_cause(self, still_session):
        session, tmp_path = still_session
        session.scope._camera_driver.stop_grabbing()

        future = session.manual_capture.capture(layer=None, false_color_on=False)

        with pytest.raises(CaptureError) as failed:
            future.result(timeout=RESULT_TIMEOUT_S)
        assert failed.value.reason == 'no_frame_returned'
        assert str(failed.value) == capture_failure_cause(
            session.scope.imaging.last_capture_info
        ), "the failure must carry the capture engine's cause, not a save-time placeholder"
        assert not list((tmp_path / 'Manual').glob('*.tiff'))
        assert not session.manual_capture.in_flight


class TestTheRecordIsTheCapturesMoment:
    def test_a_gain_write_queued_during_the_still_does_not_reach_its_record(self, still_session):
        session, _ = still_session
        session.scope.imaging.set_gain_db(1.0)
        blocker = _LaneBlocker(session)
        try:
            future = session.manual_capture.capture(layer=None, false_color_on=False)
            # The still is queued behind the blocker before the write is sent.
            _wait_until_queued(session, 1)
            writer = threading.Thread(target=session.scope.imaging.set_gain_db, args=(20.0,))
            writer.start()
        finally:
            blocker.release()
        (path,) = future.result(timeout=RESULT_TIMEOUT_S)
        writer.join(RESULT_TIMEOUT_S)

        assert read_postproc_input_metadata(path)['gain_db'] == pytest.approx(1.0)


def test_a_reconnect_rewires_the_capture_controller():
    """Left on the discarded scope, a still after a reconnect would grab
    from a camera that is gone."""
    from unittest.mock import MagicMock

    from modules.scope_session import ScopeSession
    from tests.scope_fakes import spec_scope

    old_scope = spec_scope()
    new_scope = spec_scope()
    session = ScopeSession(
        settings={},
        scope=old_scope,
        io_executor=MagicMock(),
        camera_executor=MagicMock(),
    )

    session.set_scope(new_scope)

    assert session.manual_capture._scope is new_scope
