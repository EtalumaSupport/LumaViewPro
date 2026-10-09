# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every range the writer holds is refused at the write and replaced at the load.

``settings_paths._RANGES`` is the one place a setting's range lives. Its two
halves are a write through ``ScopeSession.update_settings``, which refuses a
value outside the range with ``SettingRefusedError`` (``out_of_range``) and
writes nothing, and the load, which replaces a stored value outside it with
the shipped value for that key alone and tells it once. The load tests and the
writer tests each pin the cases they were written for; this guard is the
completeness check: the table below holds one out-of-range sample per range,
and its first assertion is that the table and ``_RANGES`` name the same paths,
so a range added to the writer without its sample here fails the build.
"""

import json
import logging
import shutil

import pytest

from modules import settings_init, settings_paths
from modules.exceptions import SettingRefusedError
from modules.scope_session import ScopeSession
from tests.ast_seams import REPO_ROOT
from tests.settings_fixtures import complete_settings

TEMPLATE = REPO_ROOT / 'data' / 'settings.json'

# One value outside each range. The keys are asserted equal to ``_RANGES``'s:
# the table mirrors the owner on purpose, and that assertion is what keeps
# the mirror from drifting. ``*`` is any layer; the tests walk it as Blue, the
# layer the template gives every block (a stimulation config among them).
_OUT_OF_RANGE = {
    'binning.size': '2x4',
    'simulator_tier': 'warp',
    'zstack.position': 'Current Position at Middle',
    'image_mode': '16bit',
    'logging.default.level': 'loud',
    'motion.acceleration_max_pct': 500,
    'protocol.period': 0.001,
    'protocol.duration': -1,
    'tiling_overlap_percent': 75.0,
    'image_output_format.live': 'BMP',
    'image_output_format.sequenced': 'BMP',
    'video.max_fps': settings_paths.VIDEO_MAX_FPS_LIMIT + 1,
    'video.max_duration_seconds': 0,
    'jpg_quality': settings_paths.JPG_QUALITY_RANGE[1] + 1,
    'live_view_fps': -3,
    '*.acquire': 'stack',
    '*.exposure_ms': 0.0,
    '*.gain_db': -3.0,
    '*.illumination_ma': -1.0,
    '*.sum': 0,
    '*.video_config.fps': 0,
    '*.video_config.duration': settings_paths.VIDEO_STEP_DURATION_S_MAX + 1,
    '*.composite_brightness_threshold': 101.0,
    '*.stim_config.frequency': settings_paths.STIM_FREQUENCY_HZ_RANGE[1] + 1,
    '*.stim_config.pulse_width': settings_paths.STIM_PULSE_WIDTH_MS_RANGE[0] - 1,
    '*.stim_config.pulse_count': settings_paths.STIM_PULSE_COUNT_RANGE[1] + 1,
    '*.stim_config.illumination_ma': -1,
}

_CASES = sorted(
    (pattern.replace('*', 'Blue', 1), value) for pattern, value in _OUT_OF_RANGE.items()
)


def test_the_table_names_every_range_the_writer_holds():
    assert set(_OUT_OF_RANGE) == set(settings_paths._RANGES)


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path_factory.mktemp('live'))), simulate=True
    )
    yield s
    s.shutdown()


@pytest.mark.parametrize(('path', 'value'), _CASES)
def test_a_write_outside_the_range_is_refused_and_nothing_is_written(session, path, value):
    before = session.get_settings_snapshot()
    if settings_paths.member_for(path) is not None:
        # A ranged setting with its own member is refused for the member
        # first; its range is the load's half only.
        with pytest.raises(SettingRefusedError) as refused:
            session.update_settings(path, value)
        assert refused.value.reason == 'has_member'
    else:
        with pytest.raises(SettingRefusedError) as refused:
            session.update_settings(path, value)
        assert refused.value.reason == 'out_of_range'
        assert refused.value.path == path
    assert session.get_settings_snapshot() == before


@pytest.mark.parametrize(('path', 'value'), _CASES)
def test_a_stored_value_outside_the_range_is_replaced_at_load_and_told_once(tmp_path, path, value):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    shutil.copy(TEMPLATE, data_dir / 'settings.json')
    shipped = json.loads(TEMPLATE.read_text())
    current = json.loads(TEMPLATE.read_text())
    *blocks, leaf = path.split('.')
    holder, shipped_holder = current, shipped
    for block in blocks:
        holder, shipped_holder = holder[block], shipped_holder[block]
    holder[leaf] = value
    (data_dir / 'current.json').write_text(json.dumps(current))

    settings, _rejected = settings_init.prepare_settings(
        logging.getLogger(__name__), str(tmp_path), fall_back_to_template=False
    )
    told = settings_init.take_stored_replacements()

    loaded = settings
    for block in blocks:
        loaded = loaded[block]
    assert loaded[leaf] == shipped_holder[leaf]
    (notice,) = told
    assert notice.replacements == [(path, value, shipped_holder[leaf])]
    assert notice.reason == 'stored_setting_replaced'
