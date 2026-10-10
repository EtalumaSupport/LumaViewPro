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
The ceilings ship in settings['ag_ae_max_exposure_ms'][<class>], set per
install there; config_helpers.get_ag_ae_max_exposure_ms resolves a
layer's from that map, and the ceiling is plumbed down to the driver, which
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

import json
import pathlib
import threading
from unittest.mock import MagicMock

from tests.protocol_drives import lent_run_claim
import modules.config_helpers as config_helpers

from tests.camera_fakes import bare_pylon_camera
from tests.scope_fakes import give_camera_capabilities


REPO = pathlib.Path(__file__).resolve().parent.parent
PYLON_SRC = REPO / 'drivers' / 'pyloncamera.py'


# --------------------------------------------------------------------------
# Functional: the per-class resolver
# --------------------------------------------------------------------------


def test_the_shipped_ceilings_per_channel_class():
    """Transmitted 50 ms, fluorescence 200 ms, luminescence 1000 ms, from
    the shipped settings, the one store of the ceilings."""
    shipped = json.loads((REPO / 'data' / 'settings.json').read_text())['ag_ae_max_exposure_ms']
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
        assert config_helpers.get_ag_ae_max_exposure_ms(layer, shipped) == cap, (
            f'{layer} AG/AE cap should ship at {cap} ms'
        )


def test_each_layer_takes_its_class_from_the_map():
    """The resolver is handed the per-class map itself, not the settings
    dict that holds it, and answers each layer with its class's entry."""
    ceilings = {'transmitted': 11.0, 'fluorescence': 123.0, 'luminescence': 456.0}
    assert config_helpers.get_ag_ae_max_exposure_ms('BF', ceilings) == 11.0
    assert config_helpers.get_ag_ae_max_exposure_ms('Red', ceilings) == 123.0
    assert config_helpers.get_ag_ae_max_exposure_ms('Lumi', ceilings) == 456.0


def test_unknown_layer_takes_the_fluorescence_cap():
    ceilings = {'transmitted': 11.0, 'fluorescence': 123.0, 'luminescence': 456.0}
    assert config_helpers.get_ag_ae_max_exposure_ms('Nonexistent', ceilings) == 123.0


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
    give_camera_capabilities(scope, cam)
    imaging = ImagingAPI(scope, cam)

    recorded = {}
    orig_auto_gain = cam.auto_gain

    def recording_auto_gain(state, **kwargs):
        recorded.update(kwargs)
        return orig_auto_gain(state, **kwargs)

    cam.auto_gain = recording_auto_gain
    # The impl seam: this harness builds a bare ImagingAPI with no scope
    # composition, and the forwarding contract under test lives in the
    # body, not the dispatcher.
    imaging._set_auto_gain_impl(
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


def test_protocol_caller_injects_per_class_cap(monkeypatch):
    """The AG arm tick must write the per-class cap (resolved via
    config_helpers) into the settings dict the apply carries. (#655)"""
    from tests.protocol_drives import protocol_step, scan_ready_runner

    monkeypatch.setattr(
        'modules.config_helpers.get_ag_ae_max_exposure_ms',
        lambda color, overrides: 456.0,
    )
    runner = scan_ready_runner(protocol_step(Auto_Gain=True))
    runner._step_executor.scan_iterate()
    applies = runner._scope.imaging.apply_layer_camera_settings.call_args_list
    assert applies, 'the AG step must make the apply'
    assert applies[0].kwargs['auto_gain_settings']['max_exposure_ms'] == 456.0, (
        "the step's channel-class cap must reach the AG apply. (#655)"
    )


def test_a_run_arms_the_camera_at_the_ceiling_the_settings_hold(tmp_path, monkeypatch):
    """A run reads the ceilings from its session's settings at prepare,
    and a fluorescence step with auto-gain arms the camera at that class's
    ceiling: the whole chain, settings to driver, on the simulated scope.
    No caller hands the map in, so none can leave it out."""
    from modules.run_events import RunEvents
    from modules.scope_session import ScopeSession
    from tests.protocol_drives import wait_until_not_running
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings
    from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT
    from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

    step = dict(_make_single_step_protocol(color='Blue').step(0))
    step['Auto_Gain'] = True
    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path),
            objective_id='10x Oly',
            ag_ae_max_exposure_ms={
                'transmitted': 50.0,
                'fluorescence': 123.0,
                'luminescence': 1000.0,
            },
        ),
        simulate=True,
    )
    try:
        home_sim_scope(session.scope)
        camera = session.scope.imaging._driver
        armed = []
        real_auto_gain = camera.auto_gain

        # a stand-in by design: the simulated camera takes the AE ceiling
        # for interface parity and keeps nothing of it (a Goal 5 gap), so
        # the arm is recorded at the driver's door and passed through.
        def record_auto_gain(state, **kwargs):
            if state:
                armed.append(kwargs.get('ae_max_exposure_ms'))
            return real_auto_gain(state, **kwargs)

        monkeypatch.setattr(camera, 'auto_gain', record_auto_gain)
        done = threading.Event()
        started = session.create_protocol_runner().run_single_scan(
            protocol=_build_real_protocol([dict(step)]),
            sequence_name='ag_ae_ceiling',
            parent_dir=str(tmp_path),
            events=RunEvents(run_ended=lambda *_ended: done.set()),
        )
        assert done.wait(timeout=COMPLETION_TIMEOUT), 'the run did not end'
        settled = started.wait(timeout_s=COMPLETION_TIMEOUT)
        assert (settled.status, settled.reason) == ('completed', 'completed')
        assert wait_until_not_running(session)
        assert armed and set(armed) == {123.0}, f'the camera was armed at {armed}'
    finally:
        session.shutdown()


def test_protocol_arm_resolves_the_cap_from_the_run():
    """The AG arm tick must resolve the ceiling from the map the run
    carries -- no monkeypatched resolver here, so this is the whole chain:
    a fluorescence step on a run whose override map says 123 ms arms at
    123 ms, not at the 200 ms fluorescence default. (#655)"""
    from tests.protocol_drives import protocol_step, scan_ready_runner

    runner = scan_ready_runner(
        protocol_step(Auto_Gain=True, Color='Blue'),
        _ag_ae_max_exposure_ms={'fluorescence': 123.0},
    )
    runner._step_executor.scan_iterate()
    applies = runner._scope.imaging.apply_layer_camera_settings.call_args_list
    assert applies, 'the AG step must make the apply'
    assert applies[0].kwargs['auto_gain_settings']['max_exposure_ms'] == 123.0, (
        "the run's per-install fluorescence ceiling must reach the AG apply. (#655)"
    )


def _video_session_autogain_call(autogain_settings):
    """Drive a frame-less protocol video step and return the auto_gain_once
    kwargs the imaging API received at the first-frame re-arm."""
    import threading

    import modules.protocol_recording as protocol_recording
    from modules.protocol_recording import ProtocolVideoStep
    from modules.run_events import RunEvents

    scope = MagicMock()
    scope.imaging.frames_until_valid.return_value = 0
    scope.imaging.active_cached = False  # wait loop exits on its first tick
    scope.runtime_state.resolve_current_objective.return_value = ('4x Oly', {'focal_length': 45.0})
    scope.capabilities.camera_model = 'sim'
    scope.capabilities.camera_serial_number = '0'
    scope.capabilities.camera_timestamp_tick_hz = None
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
        events=RunEvents(),
        aborted_event=threading.Event(),
        is_run_in_progress=lambda: True,
        abort_run_fatal=MagicMock(),
        record_step_row=MagicMock(),
        record_dropped_capture=MagicMock(),
        run_claim=lent_run_claim(),
        to_plate=None,
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
