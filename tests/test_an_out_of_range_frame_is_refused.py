# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame outside what the camera can deliver is refused, never capped.

``ScopeSession.set_frame_size`` capped a request at the sensor and raised
one below the camera's minimum up to it, so a caller asking for 5000 px got
the sensor's width with nothing said, and one asking for 10 px got the
minimum. The frame is now checked where the other camera setting bounds are
-- the public ``ImagingAPI.set_frame_size`` -- and a width or height outside
[the camera's minimum, the sensor at the current binning] is refused with
the range, before anything reaches the camera or the settings. Snapping to
the camera's grid stays: it is alignment, not a range, and it floors to a
whole grid step, never below one.
"""

import copy

import pytest

from modules.exceptions import CameraSettingOutOfRangeError


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850'), simulate=True
    )
    try:
        yield s
    finally:
        s.shutdown()


def _camera_writes(monkeypatch, imaging):
    writes = []
    real = imaging._set_frame_size_impl

    def _recording(w, h):
        writes.append((w, h))
        return real(w, h)

    monkeypatch.setattr(imaging, '_set_frame_size_impl', _recording)
    return writes


class TestTheSession:
    @pytest.mark.parametrize(
        ('width', 'height', 'setting'),
        [(5000, 800, 'frame_width'), (1000, 5000, 'frame_height'), (1000, 20, 'frame_height')],
        ids=['over the sensor width', 'over the sensor height', 'under the minimum height'],
    )
    def test_it_is_refused_with_the_range_and_nothing_moves(
        self, session, monkeypatch, width, height, setting
    ):
        imaging = session.scope.imaging
        # A minimum above the camera's 4 px grid, as a Pylon camera declares:
        # the sim's minimum is one grid step, which the grid snap alone meets.
        monkeypatch.setattr(
            type(imaging), 'min_frame_size_cached', property(lambda _: {'width': 48, 'height': 64})
        )
        before = copy.deepcopy(session.settings['frame'])
        delivered_before = imaging.frame_size_cached
        writes = _camera_writes(monkeypatch, imaging)

        with pytest.raises(CameraSettingOutOfRangeError) as refused:
            session.set_frame_size(width, height)

        assert refused.value.setting == setting
        assert refused.value.minimum is not None and refused.value.maximum is not None
        assert str(refused.value.maximum) in str(refused.value) or (
            f'{refused.value.maximum:g}' in str(refused.value)
        ), 'the refusal states the range'
        assert writes == []
        assert session.settings['frame'] == before
        assert imaging.frame_size_cached == delivered_before

    def test_the_ceiling_follows_the_binning(self, session):
        session.set_binning_size(2)
        sensor = session.scope.imaging.get_native_resolution()
        with pytest.raises(CameraSettingOutOfRangeError) as refused:
            session.set_frame_size(sensor['width'], 400)
        assert refused.value.maximum == sensor['width'] // 2

    def test_an_in_range_size_is_applied_on_the_grid(self, session):
        delivered = session.set_frame_size(1000, 802)
        assert delivered == session.scope.imaging.frame_size_cached
        assert delivered['width'] % 48 == 0 and delivered['height'] % 4 == 0

    def test_an_undeclared_sensor_leaves_the_ceiling_unchecked(self, session, monkeypatch):
        imaging = session.scope.imaging
        monkeypatch.setattr(imaging, 'get_native_resolution', lambda: {})
        writes = _camera_writes(monkeypatch, imaging)
        try:
            session.set_frame_size(5000, 800)
        except CameraSettingOutOfRangeError:
            pytest.fail('with no declared sensor there is no ceiling to refuse against')
        except Exception:
            pass  # what the camera itself makes of it is the driver's answer
        assert writes, 'the request reaches the camera when no ceiling is declared'
