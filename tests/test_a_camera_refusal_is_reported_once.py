# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A gain or exposure the camera takes is recorded as what it took; one it
refuses is reported once, by whoever ends the refusal's flight.

The setters answered with the driver's raw result -- ``True`` for gain,
microseconds for exposure -- and recorded the request as the camera's value,
so a driver that clamped, snapped or quantized left the cache, the listeners
and the display naming a value the sensor was not at. And a refusal was
reported three times: the impl logged it and posted a popup, the public setter
raised it, and the run's catcher logged it again.

Now the setters return the value in effect in the caller's unit, the impls
post nothing, the public setters raise, and the in-run catchers -- the image
writer, the autofocus sweep and the step's auto-gain arm -- report through the
one reporter and carry on at the value the camera holds.
"""

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import CameraSettingRejected
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI
from tests.scope_fakes import spec_scope


def _rejected(setting='gain_db', requested=7.0):
    return CameraSettingRejected(
        setting, requested, title='Camera Setting Not Applied', message='x'
    )


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
def posted(monkeypatch):
    captured = []
    for level in ('error', 'warning', 'info'):
        monkeypatch.setattr(
            f'modules.lumascope_api.imaging.notifications.{level}',
            lambda *a, _level=level, **kw: captured.append((_level, a)),
        )
    return captured


@pytest.fixture
def reported(monkeypatch):
    calls = []
    monkeypatch.setattr(
        'modules.notification_center.notifications.report_outcome',
        lambda exc, **kw: calls.append((exc, kw)),
    )
    return calls


class TestTheSetterAnswersWithTheValueInEffect:
    def test_a_gain_the_driver_snapped_is_recorded_as_snapped(self, sim_imaging, monkeypatch):
        imaging, cam = sim_imaging
        heard = []
        imaging.add_camera_listener(lambda name, value: heard.append((name, value)))
        monkeypatch.setattr(cam, 'gain', lambda v: 12.4)

        assert imaging.set_gain_db(12.5) == pytest.approx(12.4)
        assert imaging.gain_db_cached == pytest.approx(12.4)
        assert imaging.frame_validity.target('gain') == pytest.approx(12.4)
        assert ('gain', pytest.approx(12.4)) in heard

    def test_an_exposure_answers_in_milliseconds(self, sim_imaging):
        imaging, _cam = sim_imaging
        assert imaging.set_exposure_ms(33.0) == pytest.approx(33.0)

    def test_an_exposure_the_driver_clamped_is_recorded_as_clamped(self, sim_imaging, monkeypatch):
        """A sub-minimum request the body raises to its floor answers with the
        floor, and the cache says so -- not the request."""
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'exposure_t', lambda v: 50.0)

        assert imaging.set_exposure_ms(0.01) == pytest.approx(0.05)
        assert imaging.exposure_ms_cached == pytest.approx(0.05)


class TestTheApiPostsNothing:
    def test_a_refused_gain_raises_and_posts_nothing(self, sim_imaging, posted, monkeypatch):
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        with pytest.raises(CameraSettingRejected):
            imaging.set_gain_db(7.0)

        assert posted == []

    def test_a_refused_exposure_raises_and_posts_nothing(self, sim_imaging, posted, monkeypatch):
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        with pytest.raises(CameraSettingRejected):
            imaging.set_exposure_ms(25.0)

        assert posted == []


class TestTheLayerApply:
    def test_a_refused_gain_still_writes_the_exposure_then_raises(
        self, sim_imaging, posted, monkeypatch
    ):
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        with pytest.raises(CameraSettingRejected) as excinfo:
            imaging.apply_layer_camera_settings(gain_db=5.0, exposure_ms=40.0, layer='BF')

        assert excinfo.value.setting == 'gain_db'
        assert imaging.exposure_ms_cached == pytest.approx(40.0)
        assert posted == []


class TestTheRunReportsAndCarriesOn:
    def test_the_autofocus_sweep_reports_a_refused_target_once(self, reported):
        from modules.autofocus_runner import AutofocusRunner
        from modules.lumascope_api.imaging import AutoGainLock

        refusal = _rejected()
        runner = AutofocusRunner.__new__(AutofocusRunner)
        runner._scope = spec_scope()
        runner._scope.imaging.set_gain_db.side_effect = refusal
        runner._camera_gain = 7.0
        runner._camera_exposure = 20.0

        runner._apply_sweep_camera_targets(AutoGainLock(state=None))

        assert [exc for exc, _kw in reported] == [refusal]
        assert reported[0][1]['solicited'] is False
        runner._scope.imaging.set_exposure_ms.assert_called_once_with(20.0)

    def test_the_image_writer_reports_a_refused_step_value_once(self, reported):
        from modules.image_mode import ImageCaptureConfig
        from modules.protocol_callbacks import ProtocolCallbacks
        from modules.protocol_image_writer import ProtocolImageWriter
        from modules.run_outcome import EndingLatch
        from tests.protocol_drives import lent_run_claim

        writer = ProtocolImageWriter(
            scope=spec_scope(),
            callbacks=ProtocolCallbacks(),
            aborted=threading.Event(),
            file_io_executor=MagicMock(),
            abort_fn=lambda: None,
            fatal_abort_event=threading.Event(),
            ending=EndingLatch(),
            execution_record=None,
            leds_off_fn=lambda: None,
            is_run_in_progress_fn=lambda: True,
            image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
            timestamp_overlay=True,
            video_max_fps=0,
            engineering_mode=False,
            run_claim=lent_run_claim(),
        )
        scope = writer._scope
        scope.runtime_state.resolve_current_objective.return_value = ('4x Oly', {})
        scope.capabilities.has_turret = False
        scope.led_connected = False
        scope.imaging.capture_and_wait.return_value = np.zeros((4, 4), dtype=np.uint8)
        refusal = _rejected()
        scope.imaging.set_gain_db.side_effect = refusal
        protocol = MagicMock()
        protocol.capture_root.return_value = ''

        writer.capture(
            save_folder='/tmp',
            step={
                'Name': 'stepA',
                'Label': '',
                'Acquire': 'image',
                'Auto_Gain': False,
                'Color': 'BF',
                'Gain': 7.0,
                'Exposure': 10.0,
                'Objective': '4x',
                'Well': 'A1',
                'Z-Slice': 0,
                'Tile': '',
                'Illumination': 50.0,
                'False_Color': False,
            },
            output_format='TIFF',
            protocol=protocol,
            enable_image_saving=True,
        )

        assert [exc for exc, _kw in reported] == [refusal]
        assert reported[0][1]['solicited'] is False
        scope.imaging.set_exposure_ms.assert_called_once_with(10.0)
        assert scope.imaging.capture_and_wait.called, 'the step still captures'

    def test_the_step_auto_gain_arm_reports_a_refusal_and_goes_on(self, reported):
        from tests.protocol_drives import protocol_step, scan_ready_runner

        runner = scan_ready_runner(protocol_step(Auto_Gain=True))
        refusal = _rejected()
        runner._scope.imaging.apply_layer_camera_settings.side_effect = refusal

        runner._step_executor.scan_iterate()

        assert [exc for exc, _kw in reported] == [refusal]
        assert reported[0][1]['solicited'] is False
        assert runner._auto_gain_armed_step == 0, 'the arm is recorded, so the step goes on'
