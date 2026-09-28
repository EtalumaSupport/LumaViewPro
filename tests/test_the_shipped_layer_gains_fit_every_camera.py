# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The shipped template's layer gains are within reach of every camera.

A run refuses a step whose gain is outside the camera's range, and a new step
copies its layer's gain. The template once shipped Lumi at 48 dB, Red at 30
and Green at 20, so a fresh install on a camera with a smaller ceiling had its
first Lumi run refused. Every layer's default must fit the smallest ceiling a
camera profile declares.
"""

import json
import pathlib

import pytest

from drivers.camera_profiles import _PROFILES

SETTINGS = pathlib.Path(__file__).resolve().parent.parent / 'data' / 'settings.json'
LAYERS = ('BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi')


def _smallest_declared_ceiling() -> float:
    ceilings = [
        profile.gain.total_max_db
        for _key, profile in _PROFILES
        if profile.gain is not None and profile.gain.total_max_db is not None
    ]
    assert ceilings, 'no profile declares a gain ceiling; the check has nothing to compare to'
    return min(ceilings)


@pytest.mark.parametrize('layer', LAYERS)
def test_the_layer_gain_fits_the_smallest_camera(layer):
    gain = json.loads(SETTINGS.read_text())[layer]['gain_db']
    assert 0.0 <= gain <= _smallest_declared_ceiling(), (
        f'{layer} ships at {gain} dB, outside the smallest declared camera range'
    )
