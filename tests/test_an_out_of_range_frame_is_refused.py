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


class TestBringUpStoresTheDeliveredFrame:
    """Bring-up applies the stored frame to the camera directly, and the
    camera snaps it to its grid and caps it at its sensor. The settings then
    hold what the camera delivered, not the request -- the frame fields and
    every reader of the store show the geometry the camera actually holds.
    """

    def _create(self, tmp_path, frame, binning='1x1', **overrides):
        from modules.scope_session import ScopeSession
        from tests.settings_fixtures import complete_settings

        settings = complete_settings(live_folder=str(tmp_path), microscope='LS850', **overrides)
        settings['frame']['width'], settings['frame']['height'] = frame
        settings['binning']['size'] = binning
        return ScopeSession.create(settings, simulate=True)

    def test_a_frame_the_camera_snaps_is_stored_as_delivered(self, tmp_path):
        # The shipped template's 1900x1900 on the simulated 1920x1200 sensor.
        s = self._create(tmp_path, (1900, 1900))
        try:
            delivered = s.scope.imaging.frame_size_cached
            assert (delivered['width'], delivered['height']) != (1900, 1900)
            frame = s.settings['frame']
            assert {'width': frame['width'], 'height': frame['height']} == delivered
        finally:
            s.shutdown()

    def test_a_frame_already_on_the_grid_is_unchanged(self, tmp_path):
        s = self._create(tmp_path, (960, 600))
        try:
            frame = s.settings['frame']
            assert (frame['width'], frame['height']) == (960, 600)
            assert s.scope.imaging.frame_size_cached == {'width': 960, 'height': 600}
        finally:
            s.shutdown()

    def test_a_saved_binning_the_camera_lacks_is_stored_as_delivered(self, tmp_path):
        # A saved 3x3 on the simulated camera, which offers 1, 2 and 4: bring-up
        # runs at the camera's 1x1, and the store says so beside the delivered
        # frame. The Session's own frame writer then works its native region
        # out at the binning in force, not at the one the file carried.
        s = self._create(tmp_path, (1900, 1900), binning='3x3')
        try:
            imaging = s.scope.imaging
            assert imaging.get_binning_size() == 1
            assert s.settings['binning']['size'] == '1x1'
            frame = s.settings['frame']
            assert {'width': frame['width'], 'height': frame['height']} == imaging.frame_size_cached
            assert s.set_frame_size(960, 600) == {'width': 960, 'height': 600}
            assert (frame['native_width'], frame['native_height']) == (960, 600)
            assert s.frame_at_binning(2) == {'width': 480, 'height': 300}
        finally:
            s.shutdown()

    def test_a_grid_built_from_the_store_is_spaced_at_the_cameras_field_of_view(self, tmp_path):
        # Two sessions whose cameras hold the same geometry build the same
        # 3x3 grid, whatever binning their files saved: the grid is spaced by
        # the field of view the camera has, which the store now describes.
        grids = []
        for saved in ('1x1', '3x3'):
            s = self._create(tmp_path, (1920, 1200), binning=saved, BF={'acquire': 'image'})
            try:
                assert s.scope.imaging.get_binning_size() == 1
                protocol = s.scope.protocols.create_protocol(
                    input_config=s.get_sequenced_capture_config(tiling='3x3')
                )
                grids.append(sorted({round(x, 4) for x in protocol.steps()['X']}))
            finally:
                s.shutdown()
        # Every well of the plate, three tiles across each: the pitch inside a
        # well is the camera's field of view, three times wider from a store
        # that still said 3x3.
        assert len(grids[0]) > 1, grids
        assert grids[0] == grids[1], grids
