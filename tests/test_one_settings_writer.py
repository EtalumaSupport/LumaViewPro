# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``update_settings(path, value)`` is the one write to the live settings.

It writes one setting, named by its dotted path, under the lock -- or it
refuses, naming why, and writes nothing: the path is not a setting, names a
block, belongs to a Session member, or the value is the wrong kind or out of
the setting's range. A REST caller, a script and the GUI all reach the same
checks, because they all reach the same member.
"""

import threading

import numpy as np
import pytest

from modules.exceptions import SettingRefusedError
from modules.protocol import ProtocolScheduleRefusedError
from modules.scope_session import ScopeSession
from tests.installation_fixtures import copy_installation_files
from tests.settings_fixtures import complete_settings


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path_factory.mktemp('live'))), simulate=True
    )
    yield s
    s.shutdown()


def test_a_nested_setting_is_written(session):
    session.update_settings('video.max_fps', 25)
    assert session.settings['video']['max_fps'] == 25
    session.update_settings('BF.sum', 3)
    assert session.settings['BF']['sum'] == 3


def test_a_leaf_with_no_shipped_value_takes_a_number(session):
    # The template ships these as null ("unset"); any single value is theirs.
    session.update_settings('stage.plate_bottom_z_estimate', 1200.0)
    assert session.settings['stage']['plate_bottom_z_estimate'] == 1200.0


@pytest.mark.parametrize(
    ('path', 'value', 'reason'),
    [
        ('no_such_setting', 1, 'not_a_setting'),
        ('video.no_such_limit', 1, 'not_a_setting'),
        ('video', {'max_fps': 1}, 'block'),
        ('video.max_fps', '30', 'wrong_kind'),
        ('show_tooltips', 1, 'wrong_kind'),
        ('video.max_fps', True, 'wrong_kind'),
        ('video.max_fps', np.float64(30.0), 'wrong_kind'),
        ('stage.plate_bottom_z_estimate', [1], 'wrong_kind'),
        ('video.max_fps', 201, 'out_of_range'),
        ('video.max_duration_seconds', 0, 'out_of_range'),
        ('tiling_overlap_percent', 75.0, 'out_of_range'),
        ('image_output_format.live', 'BMP', 'out_of_range'),
        ('live_folder', '/no/such\x00place', 'out_of_range'),
    ],
)
def test_a_refused_write_names_why_and_writes_nothing(session, path, value, reason):
    before = session.get_settings_snapshot()
    with pytest.raises(SettingRefusedError) as refused:
        session.update_settings(path, value)
    assert refused.value.reason == reason
    assert refused.value.path == path
    assert session.get_settings_snapshot() == before


@pytest.mark.parametrize(
    ('path', 'member'),
    [
        ('microscope', 'select_model'),
        ('objective_id', 'select_objective'),
        ('turret_objectives.1', 'assign_turret_objective'),
        ('protocol.labware', 'select_labware'),
        ('image_mode', 'set_image_mode'),
        ('binning.size', 'set_binning_size'),
        ('frame.width', 'set_frame_size'),
        ('Blue.acquire', 'set_layer_acquire'),
        ('Blue.auto_gain', 'set_layer_auto_gain'),
        ('Blue.focus', 'save_focus'),
    ],
)
def test_a_setting_with_a_member_is_refused_naming_it(session, path, member):
    with pytest.raises(SettingRefusedError, match=f'ScopeSession.{member}') as refused:
        session.update_settings(path, None)
    assert refused.value.reason == 'has_member'
    assert refused.value.member == member
    # The name it gives is a member a caller can call.
    assert callable(getattr(ScopeSession, member))


def test_a_protocol_schedule_no_protocol_can_run_is_refused(session):
    before = session.settings['protocol']['period']
    with pytest.raises(ProtocolScheduleRefusedError):
        session.update_settings('protocol.period', 0.001)
    assert session.settings['protocol']['period'] == before


def test_the_sessions_own_model_write_is_not_refused(session):
    # select_model writes the setting the writer refuses to everyone else.
    model = session.scope.layer_identity.model
    session.select_model(model)
    assert session.settings['microscope'] == model


def test_a_snapshot_never_sees_a_write_half_done(session):
    stop = threading.Event()
    torn = []

    def write():
        while not stop.is_set():
            session.update_settings('video.max_fps', 10)
            session.update_settings('video.max_fps', 20)

    writer = threading.Thread(target=write)
    writer.start()
    try:
        for _ in range(200):
            value = session.get_settings_snapshot()['video']['max_fps']
            if value not in (10, 20, 25, 0):
                torn.append(value)
    finally:
        stop.set()
        writer.join()
    assert torn == []


def test_a_relative_live_folder_is_stored_absolute_and_created(tmp_path):
    # The load rule, wherever the value enters: relative means the installation's.
    data = tmp_path / 'data'
    data.mkdir()
    copy_installation_files(data)
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live')),
        source_path=str(tmp_path),
        simulate=True,
    )
    try:
        s.update_settings('live_folder', 'captures/run7')
        stored = s.settings['live_folder']
        assert stored == str((tmp_path / 'captures' / 'run7').resolve())
        assert (tmp_path / 'captures' / 'run7').is_dir()
    finally:
        s.shutdown()


@pytest.mark.parametrize(
    ('member', 'imaging_setter', 'path'),
    [
        ('set_high_conversion_gain', 'set_conversion_gain_mode', 'high_conversion_gain'),
        ('set_line_noise_reduction', 'set_line_noise_reduction', 'line_noise_reduction'),
    ],
)
def test_a_camera_mode_is_stored_only_once_the_camera_took_it(
    session, monkeypatch, member, imaging_setter, path
):
    imaging = session.scope.imaging
    monkeypatch.setattr(imaging, imaging_setter, lambda value: True)
    assert getattr(session, member)(True) is True
    assert session.settings['camera'][path] is True

    monkeypatch.setattr(imaging, imaging_setter, lambda value: False)
    assert getattr(session, member)(False) is False
    assert session.settings['camera'][path] is True, 'a mode the camera refused is not stored'


def test_the_scale_bar_overlay_and_its_setting_change_together(session):
    session.set_scale_bar(True)
    assert session.scope.imaging.scale_bar_config['enabled'] is True
    assert session.settings['scale_bar']['enabled'] is True
    session.set_scale_bar(False)
    assert session.scope.imaging.scale_bar_config['enabled'] is False
    assert session.settings['scale_bar']['enabled'] is False


def test_an_acceleration_the_motors_refuse_is_not_stored(session, monkeypatch):
    session.set_acceleration_limit(60)
    assert session.settings['motion']['acceleration_max_pct'] == 60

    def refuse(val_pct):
        raise ValueError(f'{val_pct} is not a percentage')

    monkeypatch.setattr(session.scope.motion, 'set_acceleration_limit', refuse)
    with pytest.raises(ValueError):
        session.set_acceleration_limit(500)
    assert session.settings['motion']['acceleration_max_pct'] == 60
