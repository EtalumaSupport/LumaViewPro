# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated camera takes and refuses what the daA3840-45um does.

The simulated camera's limits described no shipped camera: gain to 20 dB
(the dart advertises 48), exposure to 10 s with no floor (the dart: about
14 us to 1 s), a 48 x 4 window grid copied from an IDS body (the dart
acquires a 1900-wide frame as asked), Mono10 (no real unit uses it) and a
global shutter. A script or REST client met in the simulator limits it would
not meet on the bench. The simulated profile is now built from the dart's,
so the fields an API caller reads are the dart's by construction.
"""

from __future__ import annotations

import pytest

from drivers import camera_profiles
from drivers.camera_profiles import lookup_profile, simulated_profile
from modules.exceptions import CameraSettingOutOfRangeError
from tests.scope_fakes import build_scope


@pytest.fixture
def scope():
    scope = build_scope(simulate=True)
    yield scope
    scope.disconnect()


@pytest.mark.parametrize('gain_db', [30.0, 48.0])
def test_a_gain_the_dart_takes_is_taken(scope, gain_db):
    assert scope.imaging.set_gain_db(gain_db) == pytest.approx(gain_db)


def test_a_gain_above_the_darts_is_refused(scope):
    with pytest.raises(CameraSettingOutOfRangeError):
        scope.imaging.set_gain_db(48.5)


def test_an_exposure_the_dart_takes_is_taken(scope):
    assert scope.imaging.set_exposure_ms(1000.0) == pytest.approx(1000.0)


@pytest.mark.parametrize('exposure_ms', [1500.0, 0.010])
def test_an_exposure_outside_the_darts_range_is_refused(scope, exposure_ms):
    with pytest.raises(CameraSettingOutOfRangeError):
        scope.imaging.set_exposure_ms(exposure_ms)


def test_a_1900_frame_is_acquired_at_1900(scope):
    scope.imaging.set_frame_size(1900, 1900)
    camera = scope._camera_driver
    assert (camera._width, camera._height) == (1900, 1900)


def test_the_smallest_frame_is_the_darts_4_by_4(scope):
    assert scope._camera_driver.get_min_frame_size() == {'width': 4, 'height': 4}


def test_mono10_is_not_offered(scope):
    camera = scope._camera_driver
    assert camera.get_supported_pixel_formats() == ('Mono8', 'Mono12')
    assert camera.set_pixel_format('Mono10') is False


def test_what_an_api_caller_reads_is_the_darts():
    simulated, dart = simulated_profile(), lookup_profile('daA3840-45um')
    for field in (
        'native_resolution',
        'pixel_size_um',
        'shutter',
        'binning_sizes',
        'has_auto_gain',
        'has_auto_exposure',
    ):
        assert getattr(simulated, field) == getattr(dart, field), field
    assert simulated.gain.has_digital == dart.gain.has_digital


def test_the_simulated_profile_shares_nothing_with_the_darts():
    simulated, dart = camera_profiles._simulated, camera_profiles._daA3840_45um
    assert simulated.gain is not dart.gain
    assert simulated.native_resolution is not dart.native_resolution
    assert simulated.binning_sizes is not dart.binning_sizes
    assert dart.gain.total_max_db is None, 'the dart fills its range at connect'
