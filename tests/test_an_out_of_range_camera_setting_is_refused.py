# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A gain or exposure outside the camera's range is refused, the same on
every camera, and the refusal names the range.

Before, what happened depended on the body: pylon refused an over-maximum
gain, IDS and FX2 silently clamped it and answered "applied", and the API
recorded the request. A caller that does not read the return -- a sweep that
sets 40, 45 and 50 dB on a 42 dB camera -- would then measure one gain three
times and record three. Now the public setters refuse before anything reaches
the camera, with the requested value and the camera's limits as fields, so a
REST or SDK caller reads the range without parsing words. A limit the camera
does not declare is not checked.
"""

import threading

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import CameraSettingOutOfRangeError, CameraSettingRejected, Refusal
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI


@pytest.fixture
def sim_imaging():
    cam = SimulatedCamera()
    cam.active = True
    cam.open_and_start()
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope._camera_executor = None
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    imaging._populate_camera_cache()
    return imaging, cam


def test_a_gain_above_the_maximum_is_refused_naming_the_range(sim_imaging, monkeypatch):
    imaging, cam = sim_imaging
    imaging.set_gain_db(3.0)
    maximum = imaging.max_gain_db_cached
    minimum = imaging.min_gain_db_cached
    written = []
    monkeypatch.setattr(cam, 'gain', lambda v: written.append(v) or v)

    with pytest.raises(CameraSettingOutOfRangeError) as excinfo:
        imaging.set_gain_db(maximum + 5.0)

    refusal = excinfo.value
    assert isinstance(refusal, Refusal) and isinstance(refusal, ValueError)
    assert not isinstance(refusal, CameraSettingRejected), 'nothing broke: a refusal, not a fault'
    assert refusal.reason == 'gain_db_out_of_range'
    assert (refusal.requested, refusal.minimum, refusal.maximum) == (
        maximum + 5.0,
        minimum,
        maximum,
    )
    assert f'{minimum:g} to {maximum:g} dB' in str(refusal)
    assert written == [], 'nothing reaches the camera'
    assert imaging.gain_db_cached == 3.0
    assert imaging.frame_validity.target('gain') == 3.0


def test_a_gain_below_a_declared_floor_is_refused(sim_imaging):
    imaging, _cam = sim_imaging
    imaging._commit_camera_writes({'min_gain_db': 2.0})

    with pytest.raises(CameraSettingOutOfRangeError) as excinfo:
        imaging.set_gain_db(1.0)

    assert excinfo.value.minimum == 2.0


def test_an_exposure_above_the_maximum_is_refused(sim_imaging):
    imaging, _cam = sim_imaging
    maximum = imaging.max_exposure_ms_cached

    with pytest.raises(CameraSettingOutOfRangeError) as excinfo:
        imaging.set_exposure_ms(maximum + 1.0)

    assert excinfo.value.reason == 'exposure_ms_out_of_range'
    assert excinfo.value.minimum is None, 'the simulated camera declares no floor'
    assert f'at most {maximum:g} ms' in str(excinfo.value)


def test_an_undeclared_floor_is_not_checked(sim_imaging):
    imaging, _cam = sim_imaging
    assert imaging.min_exposure_ms_cached is None

    imaging.set_exposure_ms(0.001)


def test_a_value_in_range_is_applied(sim_imaging):
    imaging, _cam = sim_imaging
    assert imaging.set_gain_db(imaging.max_gain_db_cached) == pytest.approx(
        imaging.max_gain_db_cached
    )
