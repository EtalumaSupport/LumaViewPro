# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The camera saves what it took, whoever changes it, so the saved settings and the camera agree.

The camera is one object, ``scope.camera`` (``CameraAPI``; ``session.scope.camera``
in-process). Each setting it saves -- the binning, the frame and the unbinned
region it was framed from, the image mode, high conversion gain and
line-noise reduction -- has one door, which saves what the camera took. A raw
write left the saved settings describing a camera that did not exist: a
pixel format set below the image mode left ``'8bit'`` saved beside a camera
delivering Mono12, and a run then saved Mono12 frames by the 8-bit policy.

Run on the simulated LS850: the daA3840's simulated body, binning 1/2/4, no
high conversion gain and no line-noise reduction.
"""

from __future__ import annotations

import pytest

from modules import image_mode, sequential_io_executor
from drivers.exceptions import CameraReadFailedError
from modules.exceptions import (
    CameraSettingRejected,
    ConfigurationDefectError,
    HardwareCommandRefusedError,
    SettingRefusedError,
)
from modules.lumascope_api.camera import CameraAPI
from modules.lumascope_api.imaging import ImagingAPI
from modules.scope_init_config import ScopeInitConfig
from tests.scope_fakes import bind_settings_like_a_session, build_scope


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        yield s
    finally:
        s.shutdown()


def _saved_geometry(session) -> dict:
    return {
        'image_mode': session.settings['image_mode'],
        'binning.size': session.settings['binning']['size'],
        **{f'frame.{key}': value for key, value in session.settings['frame'].items()},
    }


def test_the_session_camera_is_the_same_object(session):
    assert session.scope.camera is session.scope.camera


def test_no_door_writes_a_saved_camera_setting_without_saving_it():
    """The unsaved twins are gone, and a pixel format has no door of its own.

    Fails if any of them is put back on either class.
    """
    for cls, retired in (
        (
            ImagingAPI,
            (
                'set_frame_size',
                'set_binning_size',
                'set_pixel_format',
                'set_conversion_gain_mode',
                'set_line_noise_reduction',
            ),
        ),
        (CameraAPI, ('set_pixel_format', 'set_conversion_gain_mode')),
    ):
        present = [name for name in retired if hasattr(cls, name)]
        assert present == [], f'{cls.__name__} still has {present}'


def test_the_image_mode_and_the_format_the_camera_delivers_agree(session):
    camera = session.scope.camera
    expected = image_mode.select_capture_pixel_format(
        12, session.scope.capabilities.camera_pixel_formats
    )
    assert expected is not None and expected != camera.pixel_format

    camera.set_image_mode('12bit_scientific')

    assert session.settings['image_mode'] == '12bit_scientific'
    assert camera.pixel_format == expected


def test_a_restore_puts_back_the_saved_image_mode_with_the_format(session):
    """Three 12-bit modes share one format, so the mode is saved in the snapshot."""
    camera = session.scope.camera
    camera.set_image_mode('12bit_scaled')
    format_before = camera.pixel_format
    snapshot = camera.save_camera_state('test')

    camera.set_image_mode('8bit')
    camera.restore_camera_state(snapshot)

    assert session.settings['image_mode'] == '12bit_scaled'
    assert camera.pixel_format == format_before


def test_a_restore_puts_back_the_saved_frame_and_the_region_it_came_from(session):
    """The plugin's sequence: snapshot, a storing frame change, restore.

    The region the frame change saved (800x600) is put back as the snapshot
    had it, so the next binning change divides the snapshot's region, not
    the plugin's (it delivered 400x300 when only the frame came back).
    """
    camera = session.scope.camera
    before = _saved_geometry(session)
    snapshot = camera.save_camera_state('test')

    camera.set_frame_size(800, 600)
    assert session.settings['frame']['native_width'] == 800
    camera.restore_camera_state(snapshot)

    assert _saved_geometry(session) == before
    assert camera.set_binning(2).to_dict() == {
        'width': before['frame.native_width'] // 2,
        'height': before['frame.native_height'] // 2,
    }


def test_every_saving_setter_saves_on_the_camera_lane_inside_its_own_task(session, monkeypatch):
    """A setter called off the lane writes the camera and saves in one lane task.

    A save on the caller's thread after the task let a second caller's
    command land between them: binning 2 and binning 4 asked at once left 2x2
    saved beside a camera at 4x. On the lane, nothing lands between the write
    and its save. Fails if any setter's save moves back after its dispatch.
    """
    camera = session.scope.camera
    saves: list[tuple[str, object]] = []
    store = session.scope.store_settings
    calling: list[str] = []

    def spying(updates):
        saves.append((calling[-1], getattr(sequential_io_executor._lane_worker, 'executor', None)))
        store(updates)

    monkeypatch.setattr(session.scope, 'store_settings', spying)
    for name, call in (
        ('set_image_mode', lambda: camera.set_image_mode('12bit_scientific')),
        ('set_binning', lambda: camera.set_binning(2)),
        ('set_frame_size', lambda: camera.set_frame_size(800, 600)),
        ('set_frame_size at the frame held', lambda: camera.set_frame_size(800, 600)),
        ('set_high_conversion_gain', lambda: camera.set_high_conversion_gain(False)),
        ('set_line_noise_reduction', lambda: camera.set_line_noise_reduction(False)),
    ):
        calling.append(name)
        call()

    unsaved = [name for name in calling if name not in {saver for saver, _lane in saves}]
    off_lane = [saver for saver, lane in saves if lane is not session.camera_executor]
    assert unsaved == [], f'{unsaved} saved nothing'
    assert off_lane == [], f'{off_lane} saved off the camera lane'


def test_the_frame_already_held_is_refused_while_an_activity_holds_the_scope(session):
    """No storing setter answers past a hold, so a run's restore has nothing to put back."""
    camera = session.scope.camera
    held_frame = camera.frame_size.to_dict()
    before = _saved_geometry(session)
    claim = session.activity_claim.try_claim('diagnostic')
    try:
        with pytest.raises(HardwareCommandRefusedError) as refused:
            camera.set_frame_size(held_frame['width'], held_frame['height'])
    finally:
        claim.release()

    assert refused.value.reason == 'exclusive_activity_running'
    assert _saved_geometry(session) == before


def test_a_restore_that_changed_nothing_saves_nothing(session, monkeypatch):
    """Every run and autofocus ends with a restore; one that changed nothing wakes no save."""
    camera = session.scope.camera
    snapshot = camera.save_camera_state('test')
    saved = []
    monkeypatch.setattr(session.scope, 'store_settings', saved.append)

    camera.restore_camera_state(snapshot)

    assert saved == []


def test_a_saved_camera_setting_written_through_update_settings_names_the_cameras_member(session):
    with pytest.raises(SettingRefusedError) as refused:
        session.update_settings('binning.size', '2x2')

    assert refused.value.reason == 'has_member'
    assert refused.value.member == 'scope.camera.set_binning'
    assert 'scope.camera.set_binning' in str(refused.value)


def test_a_scope_no_session_composed_refuses_to_save(diagnostic_scope):
    """The diagnostic scope has no settings: a camera setting it saves is refused.

    Reached through the image mode: a scope with no camera has no formats,
    so the mode is saved without a camera write, and the binning and frame
    setters refuse ``not_connected`` before they save.
    """
    with pytest.raises(ConfigurationDefectError) as refused:
        diagnostic_scope.camera.set_image_mode('8bit')

    assert refused.value.reason == 'scope_not_composed'


def _bring_up(prepare=None, **settings):
    """A simulated scope bound to ``settings`` and brought up, and the settings it saved into."""
    scope = build_scope(simulate=True, sim_model='LS850')
    saved = bind_settings_like_a_session(scope, objective_id='4x Oly', **settings)
    if prepare is not None:
        prepare(scope)
    scope.initialize(
        ScopeInitConfig(
            turreted=False,
            binning_size=int(saved['binning']['size'][0]),
            frame_width=saved['frame']['width'],
            frame_height=saved['frame']['height'],
            acceleration_pct=100,
            image_mode=saved['image_mode'],
            high_conversion_gain=saved['camera']['high_conversion_gain'],
            line_noise_reduction=saved['camera']['line_noise_reduction'],
        )
    )
    return scope, saved


def test_a_format_the_camera_refuses_at_bring_up_leaves_the_saved_mode_and_is_told_once(
    centre_posts,
):
    """A mode is a save policy every camera honours, so nothing is substituted for it."""
    scope, saved = _bring_up(
        prepare=lambda s: s.simulation.refuse_camera_write('pixel_format', 'false'),
        image_mode='12bit_scientific',
    )

    assert saved['image_mode'] == '12bit_scientific'
    rejected = [p for p in centre_posts if p.reason == 'camera_rejected']
    assert len(rejected) == 1
    asked = image_mode.select_capture_pixel_format(12, scope.capabilities.camera_pixel_formats)
    assert scope.camera.pixel_format != asked


def test_a_feature_saved_on_for_a_camera_without_it_is_saved_off_and_told_once(centre_posts):
    """The simulated camera has neither feature, so it is at the standard setting, which is known."""
    scope, saved = _bring_up(camera={'high_conversion_gain': True, 'line_noise_reduction': True})

    assert saved['camera']['high_conversion_gain'] is False
    assert saved['camera']['line_noise_reduction'] is False
    told = [p for p in centre_posts if p.reason == 'camera_feature_absent']
    assert len(told) == 2
    substituted = {s.setting for s in scope.bring_up_record().substitutions}
    assert {'camera.high_conversion_gain', 'camera.line_noise_reduction'} <= substituted


def test_a_frame_bring_up_refits_is_saved_with_a_region_that_gives_it():
    """A frame above the scope's maximum runs at the maximum, and that is what is saved."""
    scope, saved = _bring_up(frame={'width': 9000, 'height': 9000})

    delivered = scope.camera.frame_size.to_dict()
    assert {key: saved['frame'][key] for key in ('width', 'height')} == delivered
    factor = int(saved['binning']['size'][0])
    assert saved['frame']['native_width'] == delivered['width'] * factor
    assert saved['frame']['native_height'] == delivered['height'] * factor


def test_an_older_settings_file_without_the_region_takes_the_shipped_one(tmp_path):
    """Settings preparation merges the template's region into a file saved before it existed."""
    import json
    import logging
    import shutil

    from modules import settings_init
    from tests.ast_seams import REPO_ROOT

    template = REPO_ROOT / 'data' / 'settings.json'

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    shutil.copy(template, data_dir / 'settings.json')
    older = json.loads(template.read_text())
    del older['frame']['native_width']
    del older['frame']['native_height']
    (data_dir / 'current.json').write_text(json.dumps(older))

    prepared, _rejected = settings_init.prepare_settings(
        logging.getLogger('test'), str(tmp_path), fall_back_to_template=False
    )

    shipped = json.loads(template.read_text())['frame']
    assert prepared['frame']['native_width'] == shipped['native_width']
    assert prepared['frame']['native_height'] == shipped['native_height']


def test_over_rest_the_cameras_routes_are_under_scope_camera_and_nowhere_else():
    """One wire path, ``/api/v1/scope/camera/...``; ``session.scope.camera`` is an in-process alias."""
    from modules import wire_encoding
    from modules.api_surface import mark_of
    from modules.scope_session import ScopeSession
    from rest import routes as rest_routes

    handed_out = wire_encoding.handed_out(
        ScopeSession, wire_encoding.project_classes(), wire_encoding.project_aliases()
    )
    camera_routes = {
        route.name
        for route in rest_routes.routes(ScopeSession, handed_out=handed_out)
        if route.owner is CameraAPI
    }
    marked = {name for name, member in vars(CameraAPI).items() if mark_of(member) is not None}

    assert 'scope/camera/set_frame_size' in camera_routes
    assert {path.rsplit('/', 1)[0] for path in camera_routes} == {'scope/camera'}
    root_names = {
        route.member.name
        for route in rest_routes.routes(ScopeSession, handed_out=handed_out)
        if route.owner is not CameraAPI
    }
    assert root_names & marked == set()


def _the_camera_holds_what_is_saved(session):
    """The saved binning and frame are the camera's, and the region gives the frame."""
    camera = session.scope.camera
    saved = session.settings
    factor = camera.read_binning()
    assert saved['binning']['size'] == f'{factor}x{factor}'
    held = camera.frame_size.to_dict()
    assert {key: saved['frame'][key] for key in ('width', 'height')} == held
    region = {'width': saved['frame']['native_width'], 'height': saved['frame']['native_height']}
    assert camera._target_frame(region, factor) == held


@pytest.mark.parametrize('node', ['binning', 'frame_size'])
def test_a_restore_the_camera_refuses_part_way_saves_what_the_camera_holds(session, node):
    """The restore stops at the refusal; what the camera took before it is saved, not left unsaved."""
    camera = session.scope.camera
    snapshot = camera.save_camera_state('test')
    camera.set_binning(2)
    camera.set_frame_size(400, 300)
    session.scope.simulation.refuse_camera_write(node, 'false')

    with pytest.raises(CameraSettingRejected):
        camera.restore_camera_state(snapshot)

    _the_camera_holds_what_is_saved(session)


def test_a_restore_whose_compare_read_fails_saves_the_binning_the_camera_took(session, monkeypatch):
    """A read that fails after the binning write landed still saves the binning the camera holds.

    The frame the camera holds after it is not known while its reads fail,
    so it is held as None; the binning is known, from the write the camera
    took, and is saved alone.
    """
    camera = session.scope.camera
    snapshot = camera.save_camera_state('test')
    camera.set_binning(2)
    session.scope.simulation.refuse_camera_read('frame_size', 'failed', count=5)

    with pytest.raises(CameraReadFailedError):
        camera.restore_camera_state(snapshot)

    assert camera.frame_size is None
    factor = camera.read_binning()
    assert factor == 1
    assert session.settings['binning']['size'] == f'{factor}x{factor}'


def test_an_older_settings_file_ends_bring_up_with_a_region_that_gives_its_frame():
    """A file saved before the region existed takes the shipped one, which need not give its frame.

    Saved at 2x and 800x600, it merges the template's 1900x1900 region; bring-up
    saves the region the frame it delivers implies, 1600x1200, so the next
    binning change delivers that region and not the template's.
    """
    _scope, saved = _bring_up(binning={'size': '2x2'}, frame={'width': 800, 'height': 600})

    assert (saved['frame']['native_width'], saved['frame']['native_height']) == (1600, 1200)
