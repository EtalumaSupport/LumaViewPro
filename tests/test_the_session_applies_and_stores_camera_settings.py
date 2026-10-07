# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Session applies a camera setting and stores it only once the camera took it.

The image mode, the binning and the frame each have one writer for every
host: ScopeSession.set_image_mode / set_binning_size / set_frame_size. The
GUI used to write the settings store before the camera answered and undo it
on failure; a REST or SDK caller had no writer at all. Each member applies
through the imaging API, stores on success, and raises on failure with the
store untouched, so no caller can record a value the camera never took.

Run against the simulated camera: 3840x2160 native, binning 1/2/4, a 48x4
grid.
"""

from __future__ import annotations

import pytest

from modules.exceptions import (
    CameraSettingRejected,
    CameraSettingUnsupportedError,
    ConfigError,
    HardwareCommandRefusedError,
    MissingPart,
    Refusal,
)


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield s
    finally:
        s.shutdown()


def _stored_frame(session) -> dict:
    return dict(session.settings['frame'])


def _spy(monkeypatch, obj, name):
    calls = []
    real = getattr(obj, name)

    def _recording(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(obj, name, _recording)
    return calls


class TestTheFrame:
    def test_it_stores_what_the_camera_delivered_and_the_region_it_came_from(self, session):
        delivered = session.set_frame_size(1000, 800)

        assert delivered == session.scope.imaging.frame_size_cached
        frame = _stored_frame(session)
        assert {'width': frame['width'], 'height': frame['height']} == delivered
        assert (frame['native_width'], frame['native_height']) == (1000, 800), (
            'the unbinned region is what was asked for at 1x, capped at the sensor'
        )

    def test_a_size_the_camera_already_delivers_is_not_written_again(self, session, monkeypatch):
        delivered = session.set_frame_size(1000, 800)
        writes = _spy(monkeypatch, session.scope.imaging, 'set_frame_size')

        assert session.set_frame_size(delivered['width'], delivered['height']) == delivered
        assert writes == []

    def test_a_refused_frame_raises_and_stores_nothing(self, session, monkeypatch):
        before = _stored_frame(session)
        monkeypatch.setattr(session.scope._camera_driver, 'set_frame_size', lambda w, h: False)

        with pytest.raises(CameraSettingRejected):
            session.set_frame_size(1000, 800)
        assert _stored_frame(session) == before

    def test_no_camera_stores_nothing(self, session, monkeypatch):
        before = _stored_frame(session)
        monkeypatch.setattr(session.scope.imaging, 'set_frame_size', lambda w, h: None)

        assert session.set_frame_size(1000, 800) is None
        assert _stored_frame(session) == before

    def test_the_size_is_read_at_the_stored_binning(self, session):
        session.set_binning_size(2)
        delivered = session.set_frame_size(480, 400)

        frame = _stored_frame(session)
        assert (frame['native_width'], frame['native_height']) == (960, 800)
        assert {'width': frame['width'], 'height': frame['height']} == delivered

    def test_a_refused_frame_does_not_block_a_retry_of_the_same_size(self, session, monkeypatch):
        # The camera still holds its prior size after a refusal, so the retry
        # is not taken for a size already in force.
        driver = session.scope._camera_driver
        real = driver.set_frame_size
        refusing = [True]
        requests = []

        def _driver(w, h):
            requests.append((w, h))
            return False if refusing[0] else real(w, h)

        monkeypatch.setattr(driver, 'set_frame_size', _driver)
        with pytest.raises(CameraSettingRejected):
            session.set_frame_size(1200, 800)
        refusing[0] = False

        delivered = session.set_frame_size(1200, 800)

        assert requests == [(1200, 800), (1200, 800)], 'the retry must reach the camera'
        frame = _stored_frame(session)
        assert {'width': frame['width'], 'height': frame['height']} == delivered

    def test_a_disconnected_camera_is_never_reached_and_the_reconnect_applies(
        self, session, monkeypatch
    ):
        # The reconnect window: the camera is gone. The frame is refused,
        # naming the camera -- nothing reaches the driver and nothing is
        # stored -- and once it is back the identical size reaches it.
        session.set_frame_size(1000, 800)
        before = _stored_frame(session)
        driver = session.scope._camera_driver
        requests = _spy(monkeypatch, driver, 'set_frame_size')
        monkeypatch.setattr(driver, 'active', False)

        with pytest.raises(HardwareCommandRefusedError) as exc:
            session.set_frame_size(1200, 800)
        assert exc.value.missing == MissingPart.CAMERA
        assert requests == [], 'the driver must never be reached without a camera'
        assert _stored_frame(session) == before

        monkeypatch.setattr(driver, 'active', True)
        delivered = session.set_frame_size(1200, 800)

        assert requests == [(1200, 800)], (
            'the post-reconnect apply of the identical size must reach the camera'
        )
        frame = _stored_frame(session)
        assert {'width': frame['width'], 'height': frame['height']} == delivered

    def test_a_retype_of_a_clamped_request_reaches_the_camera(self, session, monkeypatch):
        # What the camera delivered is what is in force: a repeat of the
        # delivered size is not written again, but the original request,
        # typed again after the camera clamped it, is a real request.
        driver = session.scope._camera_driver
        real = driver.set_frame_size
        requests = []

        def _clamping(w, h):
            requests.append((w, h))
            return real(w - 48, h)

        monkeypatch.setattr(driver, 'set_frame_size', _clamping)
        delivered = session.set_frame_size(1200, 800)
        assert delivered == {'width': 1152, 'height': 800}
        requests.clear()

        assert session.set_frame_size(1152, 800) == delivered
        assert requests == [], 'the delivered size is what the camera holds'

        session.set_frame_size(1200, 800)
        assert requests == [(1200, 800)], 'the retyped request differs from what was delivered'


class TestTheBinning:
    def test_a_binning_the_camera_does_not_offer_is_refused_before_the_camera(
        self, session, monkeypatch
    ):
        writes = _spy(monkeypatch, session.scope.imaging, 'set_binning_size')
        before = (session.settings['binning']['size'], _stored_frame(session))

        with pytest.raises(CameraSettingUnsupportedError) as refused:
            session.set_binning_size(8)

        assert isinstance(refused.value, Refusal)
        assert refused.value.title == 'Binning not supported'
        assert '8x8' in str(refused.value)
        assert refused.value.reason == 'binning_unsupported'
        assert writes == []
        assert (session.settings['binning']['size'], _stored_frame(session)) == before

    def test_it_keeps_the_framed_region_and_divides_it(self, session):
        session.set_frame_size(1200, 800)
        delivered = session.set_binning_size(2)

        assert session.settings['binning']['size'] == '2x2'
        frame = _stored_frame(session)
        assert (frame['native_width'], frame['native_height']) == (1200, 800)
        assert delivered == {'width': frame['width'], 'height': frame['height']}
        assert delivered == session.scope.imaging.frame_size_cached
        assert delivered['width'] <= 600 and delivered['height'] == 400

    def test_cycling_the_binning_comes_back_to_the_same_frame(self, session):
        first = session.set_frame_size(1200, 800)
        session.set_binning_size(4)
        session.set_binning_size(2)

        assert session.set_binning_size(1) == first

    def test_a_refused_binning_stores_nothing(self, session, monkeypatch):
        before = (session.settings['binning']['size'], _stored_frame(session))
        monkeypatch.setattr(session.scope._camera_driver, 'set_binning_size', lambda size: False)

        with pytest.raises(CameraSettingRejected):
            session.set_binning_size(2)
        assert (session.settings['binning']['size'], _stored_frame(session)) == before

    def test_no_camera_stores_nothing(self, session, monkeypatch):
        before = (session.settings['binning']['size'], _stored_frame(session))
        monkeypatch.setattr(session.scope.imaging, 'set_binning_size', lambda size: False)

        assert session.set_binning_size(2) is None
        assert (session.settings['binning']['size'], _stored_frame(session)) == before

    def test_a_binning_change_stores_the_region_a_settings_file_lacked(self, session):
        # Settings saved before the unbinned pair existed hold only the
        # displayed size. A binning change must store the pair, or every
        # later change rebuilds it from an already-floored displayed size and
        # the cycle drifts (#683).
        first = session.set_frame_size(1200, 800)
        del session.settings['frame']['native_width']
        del session.settings['frame']['native_height']

        session.set_binning_size(2)

        frame = _stored_frame(session)
        assert (frame['native_width'], frame['native_height']) == (1200, 800)
        session.set_binning_size(4)
        session.set_binning_size(2)
        assert session.set_binning_size(1) == first

    def test_a_missing_region_is_rebuilt_at_the_binning_the_frame_was_stored_at(
        self, session, monkeypatch
    ):
        # The displayed size was stored at the OLD binning, so that is the
        # factor that rebuilds the region -- not the new one, and not a
        # hardware read that can lag the store (the IDS non-square 2x bug).
        session.set_binning_size(2)
        session.set_frame_size(480, 400)
        del session.settings['frame']['native_width']
        del session.settings['frame']['native_height']
        monkeypatch.setattr(session.scope.imaging, 'get_binning_size', lambda: 4)

        session.set_binning_size(1)

        frame = _stored_frame(session)
        assert (frame['native_width'], frame['native_height']) == (960, 800)

    def test_the_binning_reaches_the_camera_before_the_frame(self, session, monkeypatch):
        order = []
        imaging = session.scope.imaging
        real_binning, real_frame = imaging.set_binning_size, imaging.set_frame_size
        monkeypatch.setattr(
            imaging, 'set_binning_size', lambda size: order.append('binning') or real_binning(size)
        )
        monkeypatch.setattr(
            imaging, 'set_frame_size', lambda w, h: order.append('frame') or real_frame(w, h)
        )
        session.set_frame_size(1200, 800)
        order.clear()

        session.set_binning_size(2)
        assert order == ['binning', 'frame']

    def test_the_preview_names_the_frame_the_apply_delivers_and_applies_nothing(
        self, session, monkeypatch
    ):
        # A display shows the preview beside the new binning while the
        # apply runs; an edit typed meanwhile must be read against it.
        session.set_frame_size(1200, 800)
        before = (session.settings['binning']['size'], _stored_frame(session))
        writes = _spy(monkeypatch, session.scope.imaging, 'set_binning_size')

        preview = session.frame_at_binning(2)

        assert writes == [], 'a preview must not reach the camera'
        assert (session.settings['binning']['size'], _stored_frame(session)) == before
        assert session.set_binning_size(2) == preview


class TestTheImageMode:
    def test_it_stores_the_mode_once_the_format_is_applied(self, session, monkeypatch):
        applied = _spy(monkeypatch, session.scope.imaging, 'set_pixel_format')

        assert session.set_image_mode('12bit_scientific') is True
        assert session.settings['image_mode'] == '12bit_scientific'
        assert len(applied) == 1

    def test_a_refused_format_stores_nothing(self, session, monkeypatch):
        before = session.settings['image_mode']
        monkeypatch.setattr(session.scope._camera_driver, 'set_pixel_format', lambda fmt: False)

        with pytest.raises(CameraSettingRejected):
            session.set_image_mode('12bit_scientific')
        assert session.settings['image_mode'] == before

    def test_a_camera_that_went_away_stores_nothing(self, session, monkeypatch):
        before = session.settings['image_mode']
        monkeypatch.setattr(session.scope.imaging, 'set_pixel_format', lambda fmt: False)

        assert session.set_image_mode('12bit_scientific') is False
        assert session.settings['image_mode'] == before

    def test_an_unknown_mode_stores_nothing(self, session):
        before = session.settings['image_mode']

        with pytest.raises(ConfigError):
            session.set_image_mode('16bit')
        assert session.settings['image_mode'] == before
