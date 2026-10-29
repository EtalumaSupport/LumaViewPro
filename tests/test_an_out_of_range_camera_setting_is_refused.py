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

import pytest

from modules.exceptions import CameraSettingOutOfRangeError, CameraSettingRejected, Refusal
from tests.scope_fakes import set_camera_limits


def test_a_gain_above_the_maximum_is_refused_naming_the_range(sim_camera, monkeypatch):
    camera, cam = sim_camera
    camera.set_gain_db(3.0)
    limits = camera._scope.capabilities.camera_limits
    maximum = limits.gain_max_db
    minimum = limits.gain_min_db
    written = []
    monkeypatch.setattr(cam, 'gain', lambda v: written.append(v) or v)

    with pytest.raises(CameraSettingOutOfRangeError) as excinfo:
        camera.set_gain_db(maximum + 5.0)

    refusal = excinfo.value
    assert isinstance(refusal, Refusal) and isinstance(refusal, ValueError)
    assert not isinstance(refusal, CameraSettingRejected), 'nothing broke: a refusal, not a fault'
    assert refusal.reason == 'gain_db_out_of_range'
    assert (refusal.requested, refusal.limits) == (maximum + 5.0, (minimum, maximum))
    assert f'{minimum:g} to {maximum:g} dB' in str(refusal)
    assert written == [], 'nothing reaches the camera'
    assert camera.gain_db == 3.0
    assert camera.frame_validity.target('gain') == 3.0


def test_a_gain_below_a_declared_floor_is_refused(sim_camera, monkeypatch):
    camera, _cam = sim_camera
    set_camera_limits(monkeypatch, camera._scope, gain_min_db=2.0)

    with pytest.raises(CameraSettingOutOfRangeError) as excinfo:
        camera.set_gain_db(1.0)

    assert excinfo.value.limits[0] == 2.0


def test_an_exposure_above_the_maximum_is_refused(sim_camera):
    camera, _cam = sim_camera
    limits = camera._scope.capabilities.camera_limits
    maximum = limits.exposure_max_ms

    with pytest.raises(CameraSettingOutOfRangeError) as excinfo:
        camera.set_exposure_ms(maximum + 1.0)

    assert excinfo.value.reason == 'exposure_ms_out_of_range'
    minimum = limits.exposure_min_ms
    assert excinfo.value.limits[0] == minimum
    assert f'{minimum:g} to {maximum:g} ms' in str(excinfo.value)


def test_an_undeclared_floor_is_not_checked(sim_camera, monkeypatch):
    camera, _cam = sim_camera
    set_camera_limits(monkeypatch, camera._scope, exposure_min_ms=None)

    camera.set_exposure_ms(0.001)


def test_a_value_in_range_is_applied(sim_camera):
    camera, _cam = sim_camera
    maximum = camera._scope.capabilities.camera_limits.gain_max_db
    assert camera.set_gain_db(maximum) == pytest.approx(maximum)


def test_the_layer_apply_refuses_what_the_single_setters_refuse(sim_scope, monkeypatch):
    """``apply`` is a public door to the same gain and exposure, so it refuses
    the same values before anything reaches the camera."""
    camera = sim_scope.camera
    limits = sim_scope.capabilities.camera_limits
    written = []
    monkeypatch.setattr(sim_scope._camera_driver, 'gain', lambda v: written.append(v) or v)
    monkeypatch.setattr(sim_scope._camera_driver, 'exposure_t', lambda v: written.append(v) or v)

    for gain_db, exposure_ms, reason in (
        (limits.gain_max_db + 5.0, 20.0, 'gain_db_out_of_range'),
        (5.0, limits.exposure_max_ms + 1.0, 'exposure_ms_out_of_range'),
        (5.0, limits.exposure_min_ms / 2, 'exposure_ms_out_of_range'),
    ):
        with pytest.raises(CameraSettingOutOfRangeError) as refused:
            camera.apply(gain_db, exposure_ms, channel='BF')
        assert refused.value.reason == reason, (gain_db, exposure_ms)

    assert written == [], 'a refused apply writes neither value'
