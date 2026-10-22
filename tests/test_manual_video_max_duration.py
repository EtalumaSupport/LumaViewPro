# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Unit tests for the max-duration accessor (video.max_duration_s).

The store is prepared, so the accessor reads the key: these tests pin
that config_helpers.get_manual_video_max_duration answers the stored
value as stored.
"""

from modules.config_helpers import get_manual_video_max_duration


def test_present_key_returns_stored_value():
    settings = {'video': {'max_duration_s': 120}}
    assert get_manual_video_max_duration(settings) == 120


def test_stored_float_value_is_preserved():
    settings = {'video': {'max_duration_s': 1.0}}
    assert get_manual_video_max_duration(settings) == 1.0
