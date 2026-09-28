# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera that refuses an auto-gain or auto-exposure change says so.

The auto-mode setters dropped the driver's answer: a refused arm was still
recorded as standing, so the next capture's lock wrote back values from a
loop that never ran, and an SDK or REST caller heard success. Now the
public members raise ``CameraSettingRejected`` and record nothing, the layer
apply names a refused auto-gain among the settings it could not apply, and
the lock's disarm reports a refusal it would otherwise drop.
"""

import threading

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import CameraSettingRejected
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI

AG_SETTINGS = {'target_brightness': 0.3, 'min_gain_db': 0.0, 'max_gain_db': 20.0}


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
    return imaging, cam


@pytest.fixture
def reported(monkeypatch):
    calls = []
    monkeypatch.setattr(
        'modules.notification_center.notifications.report_outcome',
        lambda exc, **kw: calls.append((exc, kw)),
    )
    return calls


def test_a_refused_auto_gain_arm_raises_and_is_not_recorded(sim_imaging, monkeypatch):
    imaging, cam = sim_imaging
    monkeypatch.setattr(cam, 'auto_gain', lambda *a, **kw: False)

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_auto_gain(True, AG_SETTINGS)

    assert excinfo.value.setting == 'auto_gain'
    assert imaging._auto_gain_arm is None, 'an arm the camera refused is not standing'


def test_a_refused_auto_exposure_raises(sim_imaging, monkeypatch):
    imaging, cam = sim_imaging
    monkeypatch.setattr(cam, 'auto_exposure_t', lambda state=True: False)

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.set_auto_exposure_time(True)

    assert excinfo.value.setting == 'auto_exposure'


def test_a_refused_target_brightness_raises(sim_imaging, monkeypatch):
    imaging, cam = sim_imaging
    monkeypatch.setattr(cam, 'update_auto_gain_target_brightness', lambda v: False)

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.update_auto_gain_target_brightness(0.4)

    assert excinfo.value.setting == 'auto_gain_target_brightness'


def test_the_layer_apply_names_a_refused_auto_gain(sim_imaging, monkeypatch):
    imaging, cam = sim_imaging
    monkeypatch.setattr(cam, 'auto_gain', lambda *a, **kw: False)

    with pytest.raises(CameraSettingRejected) as excinfo:
        imaging.apply_layer_camera_settings(
            gain_db=5.0,
            exposure_ms=40.0,
            auto_gain=True,
            auto_gain_settings=AG_SETTINGS,
            layer='BF',
        )

    assert excinfo.value.setting == 'auto_gain'
    assert imaging.gain_db_cached == pytest.approx(5.0), 'the other writes still ran'


def test_the_locks_refused_disarm_is_reported(sim_imaging, reported, monkeypatch):
    imaging, cam = sim_imaging
    imaging.set_auto_gain(True, AG_SETTINGS)
    monkeypatch.setattr(cam, 'auto_gain', lambda *a, **kw: False)

    imaging.lock_auto_gain()

    assert [exc.setting for exc, _kw in reported] == ['auto_gain']
    assert reported[0][1]['solicited'] is False


def _refusal(setting='auto_gain'):
    return CameraSettingRejected(setting, False, title='Camera Setting Not Applied', message='x')


def test_the_runs_arm_take_reports_a_refused_disarm(reported):
    from types import SimpleNamespace

    from modules.sequenced_capture_runner import SequencedCaptureRunMode
    from tests.protocol_drives import bare_capture_runner

    runner = bare_capture_runner()
    runner._saved_camera_state = {'auto_gain_arm': SimpleNamespace(settings={})}
    runner._run_mode = SequencedCaptureRunMode.FULL_PROTOCOL
    refusal = _refusal()
    runner._scope.imaging.set_auto_gain.side_effect = refusal

    runner._take_auto_gain_arm_for_run()

    assert [exc for exc, _kw in reported] == [refusal]


def test_the_runs_camera_take_reports_a_refused_target_brightness(reported):
    from tests.protocol_drives import bare_capture_runner

    runner = bare_capture_runner()
    runner._is_run_live = lambda: True
    runner._saved_camera_state = {}
    runner._autogain_settings = {'target_brightness': 0.4}
    runner._scope.imaging.save_camera_state.return_value = {}
    refusal = _refusal('auto_gain_target_brightness')
    runner._scope.imaging.update_auto_gain_target_brightness.side_effect = refusal

    assert runner._take_camera() is None, 'the run still takes the camera'
    assert [exc for exc, _kw in reported] == [refusal]


def test_the_step_end_disarm_reports_a_refusal_and_goes_on(reported):
    from tests.protocol_drives import protocol_step, scan_ready_runner

    runner = scan_ready_runner(protocol_step(Auto_Gain=True), _auto_gain_armed_step=0)
    refusal = _refusal()
    runner._scope.imaging.set_auto_gain.side_effect = refusal

    runner._step_executor.scan_iterate()

    assert [exc for exc, _kw in reported] == [refusal]
