# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A saved layer gain or exposure the camera cannot reach is kept, run at the limit, and told once.

The API capped a stored layer value to the attached camera's range at
every apply and logged it there, and nobody was told: a configuration
carried from a larger camera ran every channel at this body's limit with
only an api-log line to say so. Bring-up now reports every capped layer
value in one notice, the shape the load's replacements take, and the
stored value stays what the person wrote.
"""

from modules.exceptions import LayerSettingCappedNotice
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


def _notices(heard):
    return [n for n in heard if n.reason == LayerSettingCappedNotice.reason]


def _session(tmp_path, heard, **layers):
    settings = complete_settings(live_folder=str(tmp_path), **layers)
    return ScopeSession.create(settings, simulate=True, outcome_listener=heard.append)


def test_every_capped_layer_value_is_one_notice_and_the_saved_values_are_kept(tmp_path):
    heard = []
    session = _session(
        tmp_path,
        heard,
        BF={'gain_db': 1000.0, 'exposure_ms': 100000.0},
        Blue={'exposure_ms': 100000.0},
    )
    try:
        imaging = session.scope.imaging
        max_gain = imaging.max_gain_db_cached
        max_exposure = imaging.max_exposure_ms_cached
        assert max_gain < 1000.0 and max_exposure < 100000.0, 'the test needs a smaller camera'
        (notice,) = _notices(heard)
        assert notice.message == str(
            LayerSettingCappedNotice(
                [
                    ('BF', 'gain_db', 1000.0, max_gain),
                    ('BF', 'exposure_ms', 100000.0, max_exposure),
                    ('Blue', 'exposure_ms', 100000.0, max_exposure),
                ]
            )
        )
        assert session.settings['BF']['gain_db'] == 1000.0, 'the saved value is the intent'
        assert session.settings['BF']['exposure_ms'] == 100000.0
        assert session.settings['Blue']['exposure_ms'] == 100000.0
        # BF is the layer bring-up applies: the camera runs at its limit.
        assert imaging.get_gain_db() == max_gain
        assert imaging.get_exposure_ms() == max_exposure
    finally:
        session.shutdown()


def test_a_camera_that_reaches_every_saved_value_is_told_nothing(tmp_path):
    heard = []
    session = _session(tmp_path, heard, BF={'gain_db': 2.0, 'exposure_ms': 20.0})
    try:
        assert _notices(heard) == []
    finally:
        session.shutdown()
