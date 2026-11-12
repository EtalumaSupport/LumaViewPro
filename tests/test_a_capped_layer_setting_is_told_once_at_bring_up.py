# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A saved layer gain or exposure the camera cannot reach is kept, run at the limit, and told once.

The API capped a stored layer value to the attached camera's range at
every apply and logged it there, and nobody was told: a configuration
carried from a larger camera ran every channel at this body's limit with
only an api-log line to say so. Bring-up now reports every capped layer
value in one notice, the shape the load's replacements take, and the
stored value stays what the person wrote.
"""

from _lumascope.modules.exceptions import ChannelSettingCappedNotice
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.stock_channels import LED0


def _notices(heard):
    return [n for n in heard if n.reason == ChannelSettingCappedNotice.reason]


def _session(tmp_path, heard, **layers):
    settings = complete_settings(live_folder=str(tmp_path), **layers)
    return ScopeSession.create(settings, simulate=True, outcome_listener=heard.append)


def test_every_capped_layer_value_is_one_notice_and_the_saved_values_are_kept(tmp_path):
    heard = []
    session = _session(
        tmp_path,
        heard,
        BF={'gain_db': 1000.0, 'exposure_ms': 100000.0},
        **{LED0: {'exposure_ms': 100000.0}},
    )
    try:
        camera = session.scope.camera
        limits = session.scope.camera.setting_limits
        max_gain = limits.gain_max_db
        max_exposure = limits.exposure_max_ms
        assert max_gain < 1000.0 and max_exposure < 100000.0, 'the test needs a smaller camera'
        (notice,) = _notices(heard)
        assert notice.message == str(
            ChannelSettingCappedNotice(
                [
                    ('BF', 'gain_db', 1000.0, max_gain),
                    ('BF', 'exposure_ms', 100000.0, max_exposure),
                    (LED0, 'exposure_ms', 100000.0, max_exposure),
                ]
            )
        )
        assert session.settings['BF']['gain_db'] == 1000.0, 'the saved value is the intent'
        assert session.settings['BF']['exposure_ms'] == 100000.0
        assert session.settings[LED0]['exposure_ms'] == 100000.0
        # BF is the layer bring-up applies: the camera runs at its limit.
        assert camera.read_gain_db() == max_gain
        assert camera.read_exposure_ms() == max_exposure
    finally:
        session.shutdown()


def test_a_saved_exposure_under_the_floor_runs_at_the_floor_and_is_told(tmp_path):
    heard = []
    session = _session(tmp_path, heard, BF={'gain_db': 2.0, 'exposure_ms': 0.005})
    try:
        floor_ms = session.scope.camera.setting_limits.exposure_min_ms
        assert floor_ms is not None and floor_ms > 0.005, 'the test needs a camera with a floor'
        (notice,) = _notices(heard)
        assert notice.message == str(
            ChannelSettingCappedNotice([('BF', 'exposure_ms', 0.005, floor_ms)])
        )
        assert session.settings['BF']['exposure_ms'] == 0.005, 'the saved value is the intent'
        assert session.scope.camera.read_exposure_ms() == floor_ms
    finally:
        session.shutdown()


def test_a_still_under_a_floored_exposure_states_the_floor_in_its_file(tmp_path):
    from _lumascope.modules.image_utils import read_file_record
    from tests.test_manual_capture_member import _files, _open_session, _settings

    settings = _settings(tmp_path)
    settings['BF']['exposure_ms'] = 0.005
    with _open_session(settings) as session:
        floor_ms = session.scope.camera.setting_limits.exposure_min_ms
        (path,) = _files(session)
    assert read_file_record(path)['exposure_time_ms'] == floor_ms


def test_a_camera_that_reaches_every_saved_value_is_told_nothing(tmp_path):
    heard = []
    session = _session(tmp_path, heard, BF={'gain_db': 2.0, 'exposure_ms': 20.0})
    try:
        assert _notices(heard) == []
    finally:
        session.shutdown()
