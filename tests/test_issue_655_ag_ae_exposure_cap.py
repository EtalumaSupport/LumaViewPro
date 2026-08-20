# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#655 regression: AG/AE exposure is capped per channel class.

Bug
---
The earlier #655 fix opened the AutoExposureTime upper bound to the
sensor's native max. Combined with the MinimizeGain profile (#551,
exposure-first), continuous AG/AE drove exposure toward the sensor
maximum on any dim scene -- washing out brightfield and making the
live auto-exposure loop hunt (the "both go to 1000 ms" + flicker
reports, which postdated and refuted that fix).

Fix
---
AG/AE gets its own per-channel-class exposure ceiling, separate from
the manual exposure-slider limits:
  transmitted (BF/PC/DF) = 50 ms, fluorescence = 200 ms,
  luminescence = 1000 ms.
The ceiling is resolved by config_helpers.get_ag_ae_max_exposure_ms
(per-install override via settings['ag_ae_max_exposure_ms'][<class>],
else the documented default) and plumbed down to the driver, which
sets AutoExposureTimeUpperLimit to that cap (in microseconds, clamped
to the node's physical range) instead of the sensor max.

Test approach
-------------
Functional tests for the pure resolver. Behavioral tests for the pylon
driver (real _set_auto_exposure_time_bounds / auto_gain /
auto_gain_once on a bare PylonCamera via tests/camera_fakes.py), plus a
caller-cluster check that every AG-enable site forwards the cap.
Bench verification gates the actual stability claim (diag/issue-655).
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import modules.config_helpers as config_helpers

from tests.camera_fakes import bare_pylon_camera


REPO = pathlib.Path(__file__).resolve().parent.parent
PYLON_SRC = REPO / 'drivers' / 'pyloncamera.py'


# --------------------------------------------------------------------------
# Functional: the per-class resolver
# --------------------------------------------------------------------------


def test_default_caps_per_channel_class():
    """Transmitted 50 ms, fluorescence 200 ms, luminescence 1000 ms."""
    expected = {
        'BF': 50.0,
        'PC': 50.0,
        'DF': 50.0,
        'Blue': 200.0,
        'Green': 200.0,
        'Red': 200.0,
        'Lumi': 1000.0,
    }
    for layer, cap in expected.items():
        assert config_helpers.get_ag_ae_max_exposure_ms(layer) == cap, (
            f'{layer} AG/AE cap should default to {cap} ms'
        )


def test_settings_override_honored_per_class():
    settings = {'ag_ae_max_exposure_ms': {'fluorescence': 150}}
    assert config_helpers.get_ag_ae_max_exposure_ms('Red', settings) == 150.0
    # A class without an override key falls back to its default.
    assert config_helpers.get_ag_ae_max_exposure_ms('BF', settings) == 50.0


def test_unknown_layer_falls_back_to_fluorescence_cap():
    assert config_helpers.get_ag_ae_max_exposure_ms('Nonexistent') == 200.0


# --------------------------------------------------------------------------
# Behavioral: pylon driver applies the cap (not the sensor max)
# --------------------------------------------------------------------------


def _bounded_camera(node_min=30.0, node_max=1_000_000.0, sensor_min=20.0):
    cam = bare_pylon_camera()
    cam.active.AutoExposureTimeLowerLimit.Min = sensor_min
    cam.active.AutoExposureTimeUpperLimit.Min = node_min
    cam.active.AutoExposureTimeUpperLimit.Max = node_max
    return cam


def test_old_sensor_max_bound_helper_is_gone():
    """The uncapped helper that opened bounds to the sensor max must be
    replaced -- its presence would mean the regression path still exists."""
    src = PYLON_SRC.read_text()
    assert '_open_auto_exposure_time_bounds_to_camera_max' not in src, (
        'The sensor-max AutoExposureTime bound helper must be removed; '
        'AG/AE exposure is now capped per channel class. (#655)'
    )


def test_bound_helper_converts_ms_cap_to_microseconds():
    """A 50 ms class cap must land on the camera as 50_000 us (Pylon
    AutoExposureTime nodes are in us), with the lower bound opened to
    the sensor minimum so AG can still drop exposure."""
    cam = _bounded_camera()
    cam._set_auto_exposure_time_bounds(max_exposure_ms=50.0)
    cam.active.AutoExposureTimeUpperLimit.SetValue.assert_called_once_with(50_000.0)
    cam.active.AutoExposureTimeLowerLimit.SetValue.assert_called_once_with(20.0)


def test_bound_helper_clamps_cap_to_node_max():
    """A cap above the node's physical range must clamp to the node Max
    -- never exceed it (the SDK would raise)."""
    cam = _bounded_camera(node_max=1_000_000.0)
    cam._set_auto_exposure_time_bounds(max_exposure_ms=2000.0)
    cam.active.AutoExposureTimeUpperLimit.SetValue.assert_called_once_with(1_000_000.0)


def test_bound_helper_opens_to_node_max_when_uncapped():
    """max_exposure_ms=None keeps the legacy open-to-node-max behavior
    for callers that do not supply a class ceiling."""
    cam = _bounded_camera(node_max=1_000_000.0)
    cam._set_auto_exposure_time_bounds(max_exposure_ms=None)
    cam.active.AutoExposureTimeUpperLimit.SetValue.assert_called_once_with(1_000_000.0)


def test_auto_gain_forwards_cap_to_bound_helper():
    """Both AG-arm entry points must pass the per-class cap down to the
    bound helper before enabling the auto loop."""
    for method_name, expected_mode in (('auto_gain', 'Continuous'), ('auto_gain_once', 'Once')):
        cam = bare_pylon_camera()
        cam.update_auto_gain_target_brightness = MagicMock()
        cam.update_auto_gain_min_max = MagicMock()
        cam._set_auto_exposure_time_bounds = MagicMock()
        getattr(cam, method_name)(state=True, ae_max_exposure_ms=123.0)
        cam._set_auto_exposure_time_bounds.assert_called_once_with(max_exposure_ms=123.0)
        cam.active.GainAuto.SetValue.assert_called_once_with(expected_mode)
        cam.active.ExposureAuto.SetValue.assert_called_once_with(expected_mode)


# --------------------------------------------------------------------------
# Caller cluster: every AG-enable site forwards the per-class cap (Rule 16)
# --------------------------------------------------------------------------


def test_api_set_auto_gain_forwards_cap_from_settings_dict():
    from drivers.simulated_camera import SimulatedCamera
    from modules.lumascope_api import Lumascope
    from modules.lumascope_api.imaging import ImagingAPI

    cam = SimulatedCamera()
    cam.connect()
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    imaging = ImagingAPI(scope, cam)

    recorded = {}
    orig_auto_gain = cam.auto_gain

    def recording_auto_gain(state, **kwargs):
        recorded.update(kwargs)
        return orig_auto_gain(state, **kwargs)

    cam.auto_gain = recording_auto_gain
    imaging.set_auto_gain(
        True,
        {
            'target_brightness': 0.5,
            'min_gain_db': 0.0,
            'max_gain_db': 24.0,
            'max_exposure_ms': 800.0,
        },
    )
    assert recorded.get('ae_max_exposure_ms') == 800.0, (
        'imaging.set_auto_gain must forward settings["max_exposure_ms"] '
        f'to the driver as the AE cap (#655); driver saw {recorded}'
    )


def test_live_caller_injects_per_class_cap():
    src = (REPO / 'ui' / 'layer_control.py').read_text()
    assert 'get_ag_ae_max_exposure_ms' in src and 'max_exposure_ms' in src, (
        'layer_control.apply_settings must inject the per-class AG/AE cap '
        'into the auto-gain settings dict. (#655)'
    )


def test_protocol_caller_injects_per_class_cap(monkeypatch):
    """The AG arm tick must write the per-class cap (resolved via
    config_helpers) into the settings dict the apply carries. (#655)"""
    from tests.protocol_drives import protocol_step, scan_ready_runner

    monkeypatch.setattr(
        'modules.config_helpers.get_ag_ae_max_exposure_ms',
        lambda color, settings: 456.0,
    )
    runner = scan_ready_runner(protocol_step(Auto_Gain=True))
    runner._step_executor.scan_iterate()
    applies = [
        c.args[0]
        for c in runner._io_executor.protocol_put.call_args_list
        if c.args[0].action is runner._scope.imaging.apply_layer_camera_settings
    ]
    assert applies, 'the AG step must queue the apply on the io executor'
    assert applies[0].kwargs['auto_gain_settings']['max_exposure_ms'] == 456.0, (
        "the step's channel-class cap must reach the AG apply. (#655)"
    )


def _video_session_autogain_call(autogain_settings):
    """Drive a frame-less protocol video step and return the auto_gain_once
    kwargs the imaging API received at the first-frame re-arm."""
    import threading

    import modules.protocol_recording as protocol_recording
    from modules.protocol_recording import ProtocolVideoStep

    scope = MagicMock()
    scope.imaging.frames_until_valid.return_value = 0
    scope.imaging.active_cached = False  # wait loop exits on its first tick
    scope.imaging.camera_identity = {
        'model': 'sim',
        'serial': '0',
        'timestamp_tick_frequency_hz': None,
    }
    scope.imaging.frame_size_cached = {'width': 8, 'height': 8}
    step = {
        'Auto_Gain': True,
        'Exposure': 10.0,
        'Video Config': {'fps': 5, 'duration': 1},
        'Color': 'BF',
        'False_Color': False,
    }
    import tempfile

    recorder = ProtocolVideoStep(
        scope=scope,
        step=step,
        save_folder=pathlib.Path(tempfile.mkdtemp()),
        name='clip',
        video_as_frames=True,
        capture_config=MagicMock(capture_depth=8, save_encoding='8bit'),
        timestamp_overlay=True,
        global_max_fps=0,
        autogain_settings=autogain_settings,
        callbacks={},
        aborted_event=threading.Event(),
        is_run_in_progress=lambda: True,
        abort_run_fatal=MagicMock(),
        abort_run_on_writer_death=MagicMock(),
        record_step_row=MagicMock(),
        record_dropped_capture=MagicMock(),
    )
    from unittest.mock import patch

    with patch.object(protocol_recording, 'check_disk_space_ok', lambda *a, **k: (True, 999999)):
        outcome = recorder.run_blocking()
    assert outcome == protocol_recording.NO_FRAMES
    assert scope.imaging.auto_gain_once.called, 'the first-frame AG re-arm must fire'
    return scope.imaging.auto_gain_once.call_args.kwargs


def test_video_capture_rearm_forwards_cap():
    kwargs = _video_session_autogain_call(
        {
            'target_brightness': 0.5,
            'min_gain_db': 0.0,
            'max_gain_db': 24.0,
            'max_exposure_ms': 777.0,
        }
    )
    assert kwargs.get('ae_max_exposure_ms') == 777.0, (
        f'video first-frame AG re-arm must forward the cap (#655); got {kwargs}'
    )


def test_video_capture_rearm_tolerates_missing_cap():
    kwargs = _video_session_autogain_call(
        {'target_brightness': 0.5, 'min_gain_db': 0.0, 'max_gain_db': 24.0}
    )
    assert kwargs.get('ae_max_exposure_ms') is None, (
        'an install without a per-class cap must re-arm uncapped, not raise'
    )
