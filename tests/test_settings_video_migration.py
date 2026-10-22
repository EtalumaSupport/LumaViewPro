# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The manual_video -> video settings-section fold, and the video keys' renames.

The fold must run before the settings.json default-merge: the merge only
adds missing keys, so without the fold an install carrying a configured
manual_video.max_fps would get the shipped video.max_fps = 0 merged in
and silently lose its cap. These tests pin the carry, the no-clobber
rule, the no-op paths, and a stored duration cap reaching its new name
through the whole load.
"""

import json
import logging
import pathlib
import shutil

import pytest

from modules import settings_init
from modules.settings_init import migrate_video_settings_dict

TEMPLATE = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'settings.json'


def test_configured_values_carry_to_video_section():
    settings = {'manual_video': {'max_fps': 10, 'max_duration_seconds': 120}}
    assert migrate_video_settings_dict(settings) is True
    assert 'manual_video' not in settings
    assert settings['video'] == {'max_fps': 10, 'max_duration_seconds': 120}


def test_existing_video_keys_are_not_clobbered():
    # A dict carrying BOTH sections keeps the video values: the new
    # section is the authority once it exists.
    settings = {
        'manual_video': {'max_fps': 10, 'max_duration_seconds': 120},
        'video': {'max_fps': 25},
    }
    assert migrate_video_settings_dict(settings) is True
    assert settings['video']['max_fps'] == 25
    assert settings['video']['max_duration_seconds'] == 120


def test_no_manual_video_section_is_a_noop():
    settings = {'video': {'max_fps': 0}}
    assert migrate_video_settings_dict(settings) is False
    assert settings == {'video': {'max_fps': 0}}


@pytest.mark.parametrize(
    'edit',
    [
        lambda current: current['video'].update(max_duration_seconds=120),
        # An older file still: the section folds first, then its key is renamed.
        lambda current: current.update(manual_video={'max_duration_seconds': 120}),
    ],
    ids=['video section', 'manual_video section'],
)
def test_a_stored_duration_cap_loads_under_its_unit_suffixed_name(tmp_path, edit):
    # Renamed before the default merge: the shipped 300 s is not merged in
    # beside the stored value, and the old name is not kept.
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    shutil.copy(TEMPLATE, data_dir / 'settings.json')
    current = json.loads(TEMPLATE.read_text())
    current['video'].pop('max_duration_s', None)
    edit(current)
    (data_dir / 'current.json').write_text(json.dumps(current))

    settings, _rejected = settings_init.prepare_settings(
        logging.getLogger(__name__), str(tmp_path), fall_back_to_template=False
    )

    assert settings['video']['max_duration_s'] == 120
    assert 'max_duration_seconds' not in settings['video']
