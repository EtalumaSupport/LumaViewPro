# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The camera profile read reports a failure instead of answering "no camera".

``get_camera_profile_info`` answers None when no camera is connected. A read
that fails on a connected camera is a different fact, and a caller that sizes
an exposure sweep from the answer must not mistake it for an absent camera.
"""

from types import SimpleNamespace

import pytest

from modules.lumascope_api.diagnostics import DiagnosticsAPI


def _scope_with(driver):
    return SimpleNamespace(
        _camera_driver=driver,
        imaging=SimpleNamespace(max_exposure_ms_cached=178.0),
    )


def _profile():
    return SimpleNamespace(
        model_name='MT9P031',
        sensor='MT9P031',
        pixel_size_um=2.2,
        shutter='rolling',
        native_resolution={'width': 1900, 'height': 1900},
        gain=SimpleNamespace(total_min_db=0.0, total_max_db=42.1),
        binning_sizes=[1],
    )


def test_no_camera_answers_none():
    api = DiagnosticsAPI(_scope_with(None))
    assert api.get_camera_profile_info() is None


def test_a_connected_camera_answers_its_range():
    driver = SimpleNamespace(active=True, profile=_profile(), get_min_exposure=lambda: 0.1124)
    info = DiagnosticsAPI(_scope_with(driver)).get_camera_profile_info()
    assert info['exposure_min_ms'] == 0.1124
    assert info['max_exposure_ms'] == 178.0


def test_a_failed_read_reaches_the_caller():
    def broken():
        raise RuntimeError('exposure node unreadable')

    driver = SimpleNamespace(active=True, profile=_profile(), get_min_exposure=broken)
    with pytest.raises(RuntimeError, match='exposure node unreadable'):
        DiagnosticsAPI(_scope_with(driver)).get_camera_profile_info()
