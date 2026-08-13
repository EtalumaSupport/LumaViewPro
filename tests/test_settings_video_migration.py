# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Unit tests for the manual_video -> video settings-section fold.

The fold must run before the settings.json default-merge: the merge only
adds missing keys, so without the fold an install carrying a configured
manual_video.max_fps would get the shipped video.max_fps = 0 merged in
and silently lose its cap. These tests pin the carry, the no-clobber
rule, and the no-op paths.
"""

from modules.settings_init import migrate_video_settings_dict


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


def test_empty_dict_is_a_noop():
    settings = {}
    assert migrate_video_settings_dict(settings) is False
    assert settings == {}
