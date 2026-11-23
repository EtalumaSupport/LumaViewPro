# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A frame side past what the camera can deliver is set at the limit and told.

The frame writer refused a width or height outside [the camera's minimum,
the scope's maximum at the binning held] with ``CameraSettingOutOfRangeError``,
while a gain, an exposure, an LED current and a sum past their limits were set
at them and told: one kind of fact met two rules by the setting it was. Now
``CameraAPI.set_frame_size`` holds each side to those limits through the one
camera routine (``_held_to_range``), frames and saves the held size, and
reports one ``ValueClampedNotice``; a side of 0 or below is still refused,
with nothing written or saved. A size inside the range is delivered exactly:
the camera acquires the next window up on its grid and crops back.
"""

import copy

import pytest

from _lumascope.modules.exceptions import ArgumentRefusedError, ValueClampedNotice
from _lumascope.modules.scope_capabilities import FrameSize
from tests.scope_fakes import home_sim_scope, set_camera_limits


@pytest.fixture
def session(tmp_path):
    from _lumascope.modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850'), simulate=True
    )
    try:
        yield s
    finally:
        s.shutdown()


def _camera_writes(monkeypatch, camera):
    writes = []
    real = camera._set_frame_size_impl

    def _recording(w, h):
        writes.append((w, h))
        return real(w, h)

    monkeypatch.setattr(camera, '_set_frame_size_impl', _recording)
    return writes


def _clamp_notices(centre_posts):
    return [post.message for post in centre_posts if post.reason == ValueClampedNotice.reason]


# A minimum above the camera's 4 px grid, as a Pylon camera declares: the
# sim's minimum is one grid step, which the grid snap alone meets.
_MINIMUM = FrameSize(width=48, height=64)


class TestTheFrameWriter:
    @pytest.mark.parametrize(
        ('asked', 'held', 'clamped'),
        [
            ((5000, 800), (3840, 800), [('frame_width', 5000.0, 3840)]),
            ((1000, 5000), (1000, 2160), [('frame_height', 5000.0, 2160)]),
            ((1000, 20), (1000, 64), [('frame_height', 20.0, 64)]),
        ],
        ids=['over the sensor width', 'over the sensor height', 'under the minimum height'],
    )
    def test_a_side_past_a_limit_is_set_at_it_saved_and_told(
        self, session, monkeypatch, centre_posts, asked, held, clamped
    ):
        camera = session.scope.camera
        set_camera_limits(monkeypatch, session.scope, frame_min=_MINIMUM)
        writes = _camera_writes(monkeypatch, camera)

        delivered = camera.set_frame_size(*asked)

        assert delivered == FrameSize(*held) == camera.frame_size
        assert writes == [held]
        frame = session.settings['frame']
        assert (frame['width'], frame['height']) == held
        assert (frame['native_width'], frame['native_height']) == held
        assert _clamp_notices(centre_posts) == [
            str(ValueClampedNotice([('camera', *row) for row in clamped]))
        ]

    @pytest.mark.parametrize('width', [0, -2], ids=['zero', 'negative'])
    def test_a_side_with_no_size_is_refused_and_nothing_moves(
        self, session, monkeypatch, centre_posts, width
    ):
        camera = session.scope.camera
        before = copy.deepcopy(session.settings['frame'])
        delivered_before = camera.frame_size.to_dict()
        writes = _camera_writes(monkeypatch, camera)

        with pytest.raises(ArgumentRefusedError) as refused:
            camera.set_frame_size(width, 800)

        assert (refused.value.reason, refused.value.argument) == ('wrong_kind', 'width')
        assert writes == []
        assert session.settings['frame'] == before
        assert camera.frame_size.to_dict() == delivered_before
        assert _clamp_notices(centre_posts) == []

    def test_the_ceiling_follows_the_binning(self, session, centre_posts):
        session.scope.camera.set_binning(2)
        width = session.scope.capabilities.camera_limits.frame_max.width
        delivered = session.scope.camera.set_frame_size(width, 400)
        assert delivered == FrameSize(width // 2, 400)
        assert _clamp_notices(centre_posts) == [
            str(ValueClampedNotice([('camera', 'frame_width', float(width), width // 2)]))
        ]

    def test_an_in_range_size_is_delivered_exactly_and_told_nothing(self, session, centre_posts):
        delivered = session.scope.camera.set_frame_size(1000, 802)
        assert delivered == session.scope.camera.frame_size
        assert delivered.to_dict() == {'width': 1000, 'height': 802}
        assert _clamp_notices(centre_posts) == []

    def test_an_unknown_maximum_leaves_the_ceiling_unheld(self, session, monkeypatch, centre_posts):
        camera = session.scope.camera
        set_camera_limits(monkeypatch, session.scope, frame_max=None)
        writes = _camera_writes(monkeypatch, camera)

        camera.set_frame_size(5000, 800)

        # What the camera itself makes of 5000 is the driver's answer.
        assert writes == [(5000, 800)]
        assert _clamp_notices(centre_posts) == []


def test_a_frame_under_the_hardware_window_minimum_is_set_as_asked(tmp_path, centre_posts):
    """The smallest frame published is the smallest delivered: the camera
    profile's alignment, since the camera acquires its smallest window and
    crops back. The Classic's sensor window is at least 100 x 100, and a
    64 x 64 frame is delivered as asked, with nothing told."""
    from _lumascope.modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS620'), simulate=True
    )
    try:
        assert s.scope.capabilities.camera_limits.frame_min == FrameSize(2, 2)
        assert s.scope.camera.set_frame_size(64, 64) == FrameSize(64, 64)
        assert _clamp_notices(centre_posts) == []
    finally:
        s.shutdown()


class TestBringUpStoresTheDeliveredFrame:
    """Bring-up applies the stored frame to the camera directly, and the
    camera snaps it to its grid and caps it at its sensor. The settings then
    hold what the camera delivered, not the request -- the frame fields and
    every reader of the store show the geometry the camera actually holds.
    """

    def _create(self, tmp_path, frame, binning='1x1', **overrides):
        from _lumascope.modules.scope_session import ScopeSession
        from tests.settings_fixtures import complete_settings

        settings = complete_settings(live_folder=str(tmp_path), microscope='LS850', **overrides)
        settings['frame']['width'], settings['frame']['height'] = frame
        settings['binning']['size'] = binning
        return ScopeSession.create(settings, simulate=True)

    def test_a_frame_the_camera_snaps_is_stored_as_delivered(self, tmp_path):
        # A frame wider than the simulated 3840x2160 sensor.
        s = self._create(tmp_path, (4000, 1900))
        try:
            delivered = s.scope.camera.frame_size.to_dict()
            assert (delivered['width'], delivered['height']) != (4000, 1900)
            frame = s.settings['frame']
            assert {'width': frame['width'], 'height': frame['height']} == delivered
        finally:
            s.shutdown()

    def test_a_frame_already_on_the_grid_is_unchanged(self, tmp_path):
        s = self._create(tmp_path, (960, 600))
        try:
            frame = s.settings['frame']
            assert (frame['width'], frame['height']) == (960, 600)
            assert s.scope.camera.frame_size.to_dict() == {'width': 960, 'height': 600}
        finally:
            s.shutdown()

    def test_a_saved_binning_the_camera_lacks_is_stored_as_delivered(self, tmp_path):
        # A saved 3x3 on the simulated camera, which offers 1, 2 and 4: bring-up
        # runs at the camera's 1x1, and the store says so beside the delivered
        # frame. The camera's own frame writer then works its native region
        # out at the binning in force, not at the one the file carried.
        s = self._create(tmp_path, (1900, 1900), binning='3x3')
        try:
            camera = s.scope.camera
            assert camera.read_binning() == 1
            assert s.settings['binning']['size'] == '1x1'
            frame = s.settings['frame']
            assert {
                'width': frame['width'],
                'height': frame['height'],
            } == camera.frame_size.to_dict()
            assert s.scope.camera.set_frame_size(960, 600) == FrameSize(960, 600)
            assert (frame['native_width'], frame['native_height']) == (960, 600)
            assert s.scope.camera.frame_at_binning(2) == FrameSize(480, 300)
        finally:
            s.shutdown()

    def test_a_grid_built_from_the_store_is_spaced_at_the_cameras_field_of_view(self, tmp_path):
        # Two sessions whose cameras hold the same geometry build the same
        # 3x3 grid, whatever binning their files saved: the grid is spaced by
        # the field of view the camera has, which the store now describes.
        grids = []
        for saved in ('1x1', '3x3'):
            s = self._create(tmp_path, (3840, 2160), binning=saved, BF={'acquire': 'image'})
            try:
                # The unfocused BF steps take the current Z, which needs a home.
                home_sim_scope(s.scope)
                assert s.scope.camera.read_binning() == 1
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
