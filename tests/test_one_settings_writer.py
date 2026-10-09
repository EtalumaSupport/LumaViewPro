# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``update_settings(path, value)`` is the one write to the live settings.

It writes one setting, named by its dotted path, under the lock -- or it
refuses, naming why, and writes nothing: the path is not a setting, names a
block, belongs to a Session member, is set only by the installation, or the
value is the wrong kind or out of the setting's range. A REST caller, a script and the GUI all reach the same
checks, because they all reach the same member.
"""

import math
import threading

import numpy as np
import pytest

from modules.exceptions import (
    AccelerationLimitRefusedError,
    CameraSettingRejected,
    HardwareCommandRefusedError,
    Refusal,
    SettingRefusedError,
)
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
        # NaN and an infinity are not numbers to the writer, ranged or not:
        # every range compares, and a comparison with NaN passes.
        ('zstack.range', math.nan, 'wrong_kind'),
        ('zstack.range', math.inf, 'wrong_kind'),
        ('stage.plate_bottom_z_estimate', math.nan, 'wrong_kind'),
        ('video.max_fps', 201, 'out_of_range'),
        ('video.max_duration_seconds', 0, 'out_of_range'),
        ('tiling_overlap_percent', 75.0, 'out_of_range'),
        ('image_output_format.live', 'BMP', 'out_of_range'),
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
        ('live_folder', 'set_live_folder'),
        ('protocol.filepath', 'set_protocol_filepath'),
        ('objective_id', 'select_objective'),
        ('turret_objectives.1', 'assign_turret_objective'),
        ('protocol.labware', 'select_labware'),
        ('image_mode', 'set_image_mode'),
        ('binning.size', 'set_binning_size'),
        ('frame.width', 'set_frame_size'),
        ('Blue.acquire', 'set_layer_acquire'),
        ('Blue.auto_gain', 'set_layer_auto_gain'),
        ('Blue.focus', 'save_layer_focus'),
    ],
)
def test_a_setting_with_a_member_is_refused_naming_it(session, path, member):
    with pytest.raises(SettingRefusedError, match=f'ScopeSession.{member}') as refused:
        session.update_settings(path, None)
    assert refused.value.reason == 'has_member'
    assert refused.value.member == member
    # The name it gives is a member a caller can call.
    assert callable(getattr(ScopeSession, member))


@pytest.mark.parametrize(
    ('path', 'value'),
    [
        ('rest_api.enabled', True),
        ('rest_api.host', '0.0.0.0'),
        ('rest_api.port', 9000),
        ('rest_api.api_key', 'k'),
        ('rest_api.cors_origins', ['*']),
        ('mode', 'engineering'),
        ('lvp_lock_port', 1),
        ('profile_trace_output_dir', '/elsewhere'),
        ('debug_mode', True),
        ('cprofile_enabled', True),
        ('profile_trace_enabled', True),
        ('tracemalloc_enabled', True),
        ('memory_profile_enabled', True),
        ('memory_profile_interval_s', 1),
        ('fx2_debug_wire_enabled', True),
    ],
)
def test_a_setting_only_the_installation_sets_is_refused(session, path, value):
    before = session.get_settings_snapshot()
    with pytest.raises(SettingRefusedError, match="installation's settings file") as refused:
        session.update_settings(path, value)
    assert refused.value.reason == 'installation_only'
    assert refused.value.path == path
    assert session.get_settings_snapshot() == before


def test_a_remembered_protocol_is_stored_and_forgotten(session):
    session.set_protocol_filepath('/data/plate.tsv')
    assert session.settings['protocol']['filepath'] == '/data/plate.tsv'
    session.set_protocol_filepath('')
    assert session.settings['protocol']['filepath'] == ''


def test_a_live_folder_no_file_system_can_name_is_refused(session):
    before = session.get_settings_snapshot()
    with pytest.raises(SettingRefusedError) as refused:
        session.set_live_folder('/no/such\x00place')
    assert refused.value.reason == 'out_of_range'
    assert refused.value.path == 'live_folder'
    assert session.get_settings_snapshot() == before


def test_a_protocol_schedule_no_protocol_can_run_is_refused(session):
    # The protocol's own range, refused in the writer's one vocabulary.
    before = session.settings['protocol']['period']
    with pytest.raises(SettingRefusedError) as refused:
        session.update_settings('protocol.period', 0.001)
    assert (refused.value.reason, refused.value.path) == ('out_of_range', 'protocol.period')
    assert session.settings['protocol']['period'] == before


def test_a_layer_exposure_or_gain_above_the_camera_is_stored_as_the_intent(session):
    # The camera's limit is the apply's to hold (applied_exposure_ms_for),
    # with the stored value kept for a camera that can reach it.
    assert session.scope.imaging.max_exposure_ms_cached == 1000.0
    assert session.scope.imaging.max_gain_db_cached == 48.0
    session.update_settings('Blue.exposure_ms', 1500.0)
    session.update_settings('Blue.gain_db', 60.0)
    assert session.settings['Blue']['exposure_ms'] == 1500.0
    assert session.settings['Blue']['gain_db'] == 60.0


@pytest.mark.parametrize('path', ['Blue.illumination_ma', 'Blue.stim_config.illumination_ma'])
def test_a_layer_current_above_the_led_board_is_refused_naming_the_board(session, path):
    # The board refuses to drive it and over-driving an LED is a damage
    # mode, so the writer refuses it where the camera's limit is applied.
    board_max = session.scope.capabilities.led_max_ma
    assert board_max == 1000
    before = session.get_settings_snapshot()
    with pytest.raises(SettingRefusedError, match="LED board's maximum") as refused:
        session.update_settings(path, board_max + 1)
    assert (refused.value.reason, refused.value.path) == ('out_of_range', path)
    assert session.get_settings_snapshot() == before
    session.update_settings(path, board_max)
    assert session.get_settings_snapshot() != before


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
        s.set_live_folder('captures/run7')
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
    monkeypatch.setattr(imaging, imaging_setter, lambda value: None)
    getattr(session, member)(True)
    assert session.settings['camera'][path] is True

    def refused(value):
        raise CameraSettingRejected(path, value, title='Not applied', message='refused')

    monkeypatch.setattr(imaging, imaging_setter, refused)
    with pytest.raises(CameraSettingRejected):
        getattr(session, member)(False)
    assert session.settings['camera'][path] is True, 'a mode the camera refused is not stored'


def test_the_scale_bar_overlay_and_its_setting_change_together(session):
    session.set_scale_bar(True)
    assert session.scope.imaging.scale_bar_config['enabled'] is True
    assert session.settings['scale_bar']['enabled'] is True
    session.set_scale_bar(False)
    assert session.scope.imaging.scale_bar_config['enabled'] is False
    assert session.settings['scale_bar']['enabled'] is False


def test_an_acceleration_out_of_range_is_refused_and_not_stored(session):
    session.set_acceleration_limit(60)
    assert session.settings['motion']['acceleration_max_pct'] == 60
    with pytest.raises(ValueError):
        session.set_acceleration_limit(500)
    assert session.settings['motion']['acceleration_max_pct'] == 60


def test_an_acceleration_out_of_range_is_a_refusal_not_a_fault(session):
    """Reported as a fault it read "Operation failed ... Check the main log"."""
    with pytest.raises(AccelerationLimitRefusedError) as refused:
        session.set_acceleration_limit(250)
    assert isinstance(refused.value, Refusal) and isinstance(refused.value, ValueError)
    assert '250' in str(refused.value) and '1 to 100' in str(refused.value)


def test_an_acceleration_with_no_motor_controller_is_refused_and_not_stored(session, monkeypatch):
    session.set_acceleration_limit(37)
    monkeypatch.setattr(type(session.scope), 'motor_connected', property(lambda self: False))
    for val_pct in (60, 500):
        with pytest.raises(HardwareCommandRefusedError) as refused:
            session.set_acceleration_limit(val_pct)
        assert refused.value.reason == 'not_connected'
    assert session.settings['motion']['acceleration_max_pct'] == 37


def test_a_bookmark_is_the_live_position_in_the_go_to_frame(session):
    session.scope.motion.home('ALL')
    saved = session.save_bookmark(('X', 'Y', 'Z'))
    here = session.get_current_plate_position()
    assert saved == {'x': here['x'], 'y': here['y'], 'z': here['z']}
    assert session.settings['bookmark'] == {**session.settings['bookmark'], **saved}


def test_set_all_bookmarks_stamps_the_focus_of_the_scopes_own_layers(session):
    session.scope.motion.home('ALL')
    z = session.save_all_bookmarks()
    on_scope = {record.key_name for record in session.scope.layer_identity.layers}
    assert session.settings['bookmark']['z'] == z
    for layer in on_scope:
        assert session.settings[layer]['focus'] == z
