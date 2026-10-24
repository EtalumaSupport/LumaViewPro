# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The camera API applies a camera setting and stores it only once the camera took it.

The image mode, the binning and the frame each have one writer for every
host: CameraAPI.set_image_mode / set_binning / set_frame_size, reached
as ``session.camera``. The GUI used to write the settings store before the
camera answered and undo it on failure; a REST or SDK caller had no writer
at all. Each member applies to the camera, stores on success, and raises on
failure with the store untouched, so no caller can record a value the camera
never took.

Run against the simulated camera: 3840x2160 native, binning 1/2/4, a 48x4
grid.
"""

from __future__ import annotations

import pytest

from modules.exceptions import (
    ArgumentRefusedError,
    CameraSettingRejected,
    CameraSettingUnsupportedError,
    HardwareCommandRefusedError,
    MissingPart,
    Refusal,
)
from modules.image_mode import ImageMode


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield s
    finally:
        s.shutdown()


def _gone_before_the_body(member):
    """A camera write body whose camera went away after the lane asked for it."""

    def refused(*_args):
        part = MissingPart.CAMERA
        raise HardwareCommandRefusedError(part.reason, member, missing=part)

    return refused


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
        delivered = session.camera.set_frame_size(1000, 800)

        assert delivered == session.scope.camera.frame_size
        frame = _stored_frame(session)
        assert {'width': frame['width'], 'height': frame['height']} == delivered.to_dict()
        assert (frame['native_width'], frame['native_height']) == (1000, 800), (
            'the unbinned region is what was asked for at 1x, capped at the sensor'
        )

    def test_a_size_the_camera_already_delivers_is_not_written_again(self, session, monkeypatch):
        delivered = session.camera.set_frame_size(1000, 800)
        writes = _spy(monkeypatch, session.scope.camera, '_set_frame_size_impl')

        assert session.camera.set_frame_size(delivered.width, delivered.height) == delivered
        assert writes == []

    def test_a_refused_frame_raises_and_stores_nothing(self, session, monkeypatch):
        before = _stored_frame(session)
        monkeypatch.setattr(session.scope._camera_driver, 'set_frame_size', lambda w, h: False)

        with pytest.raises(CameraSettingRejected):
            session.camera.set_frame_size(1000, 800)
        assert _stored_frame(session) == before

    def test_no_camera_stores_nothing(self, session, monkeypatch):
        before = _stored_frame(session)
        monkeypatch.setattr(session.scope.camera, '_set_frame_size_impl', lambda w, h: None)

        assert session.camera.set_frame_size(1000, 800) is None
        assert _stored_frame(session) == before

    def test_the_size_is_read_at_the_binning_the_camera_holds(self, session):
        session.camera.set_binning(2)
        delivered = session.camera.set_frame_size(480, 400)

        frame = _stored_frame(session)
        assert (frame['native_width'], frame['native_height']) == (960, 800)
        assert {'width': frame['width'], 'height': frame['height']} == delivered.to_dict()

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
            session.camera.set_frame_size(1200, 800)
        refusing[0] = False

        delivered = session.camera.set_frame_size(1200, 800)

        assert requests == [(1200, 800), (1200, 800)], 'the retry must reach the camera'
        frame = _stored_frame(session)
        assert {'width': frame['width'], 'height': frame['height']} == delivered.to_dict()

    def test_a_disconnected_camera_is_never_reached_and_the_reconnect_applies(
        self, session, monkeypatch
    ):
        # The reconnect window: the camera is gone. The frame is refused,
        # naming the camera -- nothing reaches the driver and nothing is
        # stored -- and once it is back the identical size reaches it.
        session.camera.set_frame_size(1000, 800)
        before = _stored_frame(session)
        driver = session.scope._camera_driver
        requests = _spy(monkeypatch, driver, 'set_frame_size')
        monkeypatch.setattr(driver, 'active', False)

        with pytest.raises(HardwareCommandRefusedError) as exc:
            session.camera.set_frame_size(1200, 800)
        assert exc.value.missing == MissingPart.CAMERA
        assert requests == [], 'the driver must never be reached without a camera'
        assert _stored_frame(session) == before

        monkeypatch.setattr(driver, 'active', True)
        delivered = session.camera.set_frame_size(1200, 800)

        assert requests == [(1200, 800)], (
            'the post-reconnect apply of the identical size must reach the camera'
        )
        frame = _stored_frame(session)
        assert {'width': frame['width'], 'height': frame['height']} == delivered.to_dict()

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
        delivered = session.camera.set_frame_size(1200, 800)
        assert delivered.to_dict() == {'width': 1152, 'height': 800}
        requests.clear()

        assert session.camera.set_frame_size(1152, 800) == delivered
        assert requests == [], 'the delivered size is what the camera holds'

        session.camera.set_frame_size(1200, 800)
        assert requests == [(1200, 800)], 'the retyped request differs from what was delivered'


class TestTheBinning:
    def test_a_binning_the_camera_does_not_offer_is_refused_before_the_camera(
        self, session, monkeypatch
    ):
        writes = _spy(monkeypatch, session.scope.camera, '_set_binning_impl')
        before = (session.settings['binning']['size'], _stored_frame(session))

        with pytest.raises(CameraSettingUnsupportedError) as refused:
            session.camera.set_binning(8)

        assert isinstance(refused.value, Refusal)
        assert refused.value.title == 'Not Available on This Camera'
        assert '8x8' in str(refused.value)
        assert refused.value.reason == 'binning_unsupported'
        assert writes == []
        assert (session.settings['binning']['size'], _stored_frame(session)) == before

    def test_it_keeps_the_framed_region_and_divides_it(self, session):
        session.camera.set_frame_size(1200, 800)
        delivered = session.camera.set_binning(2)

        assert session.settings['binning']['size'] == '2x2'
        frame = _stored_frame(session)
        assert (frame['native_width'], frame['native_height']) == (1200, 800)
        assert delivered.to_dict() == {'width': frame['width'], 'height': frame['height']}
        assert delivered == session.scope.camera.frame_size
        assert delivered.width <= 600 and delivered.height == 400

    def test_cycling_the_binning_comes_back_to_the_same_frame(self, session):
        first = session.camera.set_frame_size(1200, 800)
        session.camera.set_binning(4)
        session.camera.set_binning(2)

        assert session.camera.set_binning(1) == first

    def test_a_refused_binning_stores_nothing(self, session, monkeypatch):
        before = (session.settings['binning']['size'], _stored_frame(session))
        monkeypatch.setattr(session.scope._camera_driver, 'set_binning_size', lambda size: False)

        with pytest.raises(CameraSettingRejected):
            session.camera.set_binning(2)
        assert (session.settings['binning']['size'], _stored_frame(session)) == before

    def test_no_camera_stores_nothing(self, session, monkeypatch):
        before = (session.settings['binning']['size'], _stored_frame(session))
        monkeypatch.setattr(
            session.scope.camera,
            '_set_binning_impl',
            _gone_before_the_body('set_binning'),
        )

        with pytest.raises(HardwareCommandRefusedError) as caught:
            session.camera.set_binning(2)
        assert caught.value.reason == 'not_connected'
        assert (session.settings['binning']['size'], _stored_frame(session)) == before

    def test_the_region_is_read_as_saved_never_from_a_hardware_binning_read(
        self, session, monkeypatch
    ):
        # The saved region is the intent. A hardware read that lags the
        # store (the IDS non-square 2x bug) must not re-derive it; the saved
        # frame always carries it, so nothing is rebuilt from the frame.
        session.camera.set_binning(2)
        session.camera.set_frame_size(480, 400)
        monkeypatch.setattr(session.scope.camera, 'read_binning', lambda: 4)

        session.camera.set_binning(1)

        frame = _stored_frame(session)
        assert (frame['native_width'], frame['native_height']) == (960, 800)

    def test_the_binning_reaches_the_camera_before_the_frame(self, session, monkeypatch):
        order = []
        camera = session.scope.camera
        real_binning, real_frame = camera._set_binning_impl, camera._set_frame_size_impl
        monkeypatch.setattr(
            camera,
            '_set_binning_impl',
            lambda size: order.append('binning') or real_binning(size),
        )
        monkeypatch.setattr(
            camera, '_set_frame_size_impl', lambda w, h: order.append('frame') or real_frame(w, h)
        )
        session.camera.set_frame_size(1200, 800)
        order.clear()

        session.camera.set_binning(2)
        assert order == ['binning', 'frame']

    def test_the_preview_names_the_frame_the_apply_delivers_and_applies_nothing(
        self, session, monkeypatch
    ):
        # A display shows the preview beside the new binning while the
        # apply runs; an edit typed meanwhile must be read against it.
        session.camera.set_frame_size(1200, 800)
        before = (session.settings['binning']['size'], _stored_frame(session))
        writes = _spy(monkeypatch, session.scope.camera, '_set_binning_impl')

        preview = session.camera.frame_at_binning(2)

        assert writes == [], 'a preview must not reach the camera'
        assert (session.settings['binning']['size'], _stored_frame(session)) == before
        assert session.camera.set_binning(2).to_dict() == preview


class TestTheImageMode:
    def test_it_stores_the_mode_once_the_format_is_applied(self, session, monkeypatch):
        applied = _spy(monkeypatch, session.scope.camera, '_set_pixel_format_impl')

        session.camera.set_image_mode('12bit_scientific')
        assert session.settings['image_mode'] == '12bit_scientific'
        assert len(applied) == 1

    def test_a_refused_format_stores_nothing(self, session, monkeypatch):
        before = session.settings['image_mode']
        monkeypatch.setattr(session.scope._camera_driver, 'set_pixel_format', lambda fmt: False)

        with pytest.raises(CameraSettingRejected):
            session.camera.set_image_mode('12bit_scientific')
        assert session.settings['image_mode'] == before

    def test_a_camera_that_went_away_stores_nothing(self, session, monkeypatch):
        before = session.settings['image_mode']
        monkeypatch.setattr(
            session.scope.camera, '_set_pixel_format_impl', _gone_before_the_body('set_image_mode')
        )

        with pytest.raises(HardwareCommandRefusedError) as caught:
            session.camera.set_image_mode('12bit_scientific')
        assert caught.value.reason == 'not_connected'
        assert session.settings['image_mode'] == before

    # A dict cannot be looked up in the vocabulary; it is refused, not a TypeError.
    @pytest.mark.parametrize('mode', ['16bit'])
    def test_an_unknown_mode_is_refused_and_nothing_is_applied_or_stored(self, session, mode):
        before = session.settings['image_mode']
        pixel_format = session.scope.camera.pixel_format

        with pytest.raises(ArgumentRefusedError) as refused:
            session.camera.set_image_mode(mode)
        assert (refused.value.reason, refused.value.argument) == ('not_offered', 'mode')
        assert session.settings['image_mode'] == before
        assert session.scope.camera.pixel_format == pixel_format

    def test_a_mode_set_by_its_member_is_stored_as_its_string_and_read_as_the_member(self, session):
        session.camera.set_image_mode(ImageMode.SCALED)

        stored = session.settings['image_mode']
        assert (type(stored), stored) == (str, '12bit_scaled')
        read = session.camera.image_mode
        assert (type(read), read) == (ImageMode, '12bit_scaled')


class TestTheCameraToggles:
    """The simulated camera has neither toggle: on is refused, off is its state."""

    @pytest.mark.parametrize(
        ('member', 'path'),
        [
            ('set_high_conversion_gain', 'high_conversion_gain'),
            ('set_line_noise_reduction', 'line_noise_reduction'),
        ],
    )
    def test_on_without_the_mode_is_refused_and_stores_nothing(self, session, member, path):
        session.settings['camera'][path] = False

        with pytest.raises(CameraSettingUnsupportedError):
            getattr(session.camera, member)(True)
        assert session.settings['camera'][path] is False

    @pytest.mark.parametrize(
        ('member', 'path'),
        [
            ('set_high_conversion_gain', 'high_conversion_gain'),
            ('set_line_noise_reduction', 'line_noise_reduction'),
        ],
    )
    def test_off_without_the_mode_is_stored(self, session, member, path):
        session.settings['camera'][path] = True

        getattr(session.camera, member)(False)
        assert session.settings['camera'][path] is False
