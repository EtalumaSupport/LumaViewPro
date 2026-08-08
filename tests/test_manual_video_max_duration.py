# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Unit tests for the canonical max-duration accessor (video.max_duration_seconds).

The default used to be duplicated at every read site; these tests pin
the single accessor that now owns it
(config_helpers.get_manual_video_max_duration) at the ruled 300 s
manual default.
"""

from modules.config_helpers import get_manual_video_max_duration


def test_present_key_returns_stored_value():
    settings = {'video': {'max_duration_seconds': 120}}
    assert get_manual_video_max_duration(settings) == 120


def test_absent_key_returns_default_300():
    settings = {'video': {'max_fps': 0}}
    assert get_manual_video_max_duration(settings) == 300


def test_absent_video_block_returns_default_300():
    assert get_manual_video_max_duration({}) == 300


def test_stored_float_value_is_preserved():
    settings = {'video': {'max_duration_seconds': 1.0}}
    assert get_manual_video_max_duration(settings) == 1.0
