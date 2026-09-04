# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture under a continuous auto-gain arm locks the achieved values and
proves them through the evidence gate.

Bug shape: arming continuous auto-gain records nothing for the gate, so the
manual exposure target set just before the arm stays recorded while the
camera's auto-exposure moves the real exposure. Every gated capture then
compares the frame's chunk exposure against the stale target and rejects
it; in a protocol three rejections abort the run. The fix locks the arm
inside the capture: disarm, take the achieved exposure and gain from the
last frame's chunks, write them back through the ordinary setters (which
re-target the gate), capture through the UNCHANGED gate, classify the
outcome against the layer class's usable range, and re-arm afterwards when
the arm was a live-view arm.

Only the camera driver is a double (a SimulatedCamera with per-frame chunks
and a modelled auto-exposure). ImagingAPI, FrameValidity, the drain loop
and the chunk gate are the production objects.
"""

from __future__ import annotations

import ast
import threading
from unittest.mock import patch

import lvp_logger
import modules.config_helpers as config_helpers
from drivers.simulated_camera import SimulatedCamera
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI
from tests import ast_seams

AG_SETTINGS_FLUORESCENCE = {
    'target_brightness': 0.5,
    'min_gain_db': 0.0,
    'max_gain_db': 20.0,
    'max_exposure_ms': 200.0,
    'min_exposure_ms': 1.0,
}
AG_SETTINGS_TRANSMITTED = {
    'target_brightness': 0.5,
    'min_gain_db': 0.0,
    'max_gain_db': 20.0,
    'max_exposure_ms': 50.0,
    'min_exposure_ms': 0.1,
}


class _ChunkHandler:
    """The shape ImagingAPI._get_latest_chunks reaches: a handler whose
    get_last_chunks() reports the exposure (us) and gain (dB) the last
    stored frame was taken with."""

    def __init__(self, cam):
        self._cam = cam

    def get_last_chunks(self):
        if self._cam.chunks_absent:
            return {}
        return {'ExposureTime': self._cam._exposure_us, 'Gain': self._cam._gain}


class _ChunkAeSim(SimulatedCamera):
    """SimulatedCamera + per-frame chunks + a modelled continuous auto-exposure.

    On auto_gain(True, ..., ae_max_exposure_ms=c) the auto loop lands the
    exposure on clamp(ae_lands_on_ms, camera_floor_ms, c): the value the
    camera picks is deliberately NOT the value the caller last requested.
    """

    def __init__(self, ae_lands_on_ms: float, camera_floor_ms: float = 0.03):
        super().__init__()
        self.ae_lands_on_ms = ae_lands_on_ms
        self.camera_floor_ms = camera_floor_ms
        self.chunks_absent = False
        self.fail_readback = False
        self.cam_image_handler = _ChunkHandler(self)
        self.profile.has_auto_gain = True

    def auto_gain(
        self,
        state=True,
        target_brightness: float = 0.5,
        min_gain_db=None,
        max_gain_db=None,
        ae_max_exposure_ms=None,
    ):
        with self._lock:
            self._auto_gain_enabled = state
            if state:
                ceiling = ae_max_exposure_ms if ae_max_exposure_ms is not None else 1e9
                landed = min(max(self.ae_lands_on_ms, self.camera_floor_ms), ceiling)
                self._exposure_us = landed * 1000.0
                self._gain = ((min_gain_db or 0.0) + (max_gain_db or 20.0)) / 2.0
        return True

    def get_exposure_t(self) -> float:
        if self.fail_readback:
            return -1.0
        return super().get_exposure_t()

    def get_gain(self) -> float:
        if self.fail_readback:
            return -1.0
        return super().get_gain()


class _StubRuntimeState:
    def get_current_objective(self):
        return None


class _StubIllumination:
    def get_led_states(self):
        return {}

    def state_color2ch(self, color):
        return None


def _build(ae_lands_on_ms: float) -> tuple[ImagingAPI, _ChunkAeSim]:
    cam = _ChunkAeSim(ae_lands_on_ms)
    cam.active = True
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope._camera_executor = None
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    scope.illumination = _StubIllumination()
    scope.runtime_state = _StubRuntimeState()
    imaging.start_streaming()
    return imaging, cam


def _logged(level: str) -> list[str]:
    """Messages the production code sent to lvp_logger.logger.<level>.

    The test conftest installs a MagicMock in place of lvp_logger, so the
    module-level ``logger`` every production module imports is that mock
    and its calls are the record.
    """
    return [str(call.args[0]) for call in getattr(lvp_logger.logger, level).call_args_list]


def _arm(imaging, settings, *, resume_after_capture):
    imaging._apply_layer_camera_settings_impl(
        gain_db=1.0,
        exposure_ms=100.0,
        auto_gain=True,
        auto_gain_settings=settings,
        resume_after_capture=resume_after_capture,
    )


def test_auto_gain_capture_locks_and_passes_the_gate():
    """A per-capture arm at a 100 ms step exposure whose auto-exposure lands
    at 62.003 ms: the capture returns a frame proven at 62.003 ms through
    the gate, reports CONVERGED, and leaves the camera locked (auto off)."""
    imaging, cam = _build(ae_lands_on_ms=62.003)
    _arm(imaging, AG_SETTINGS_FLUORESCENCE, resume_after_capture=False)
    image = imaging._capture_and_wait_impl(timeout_s=1.0)
    assert image is not None, 'the capture was rejected against the stale exposure target'
    info = imaging.last_capture_info
    assert info['auto_gain'] == 'CONVERGED'
    assert info['auto_gain_exposure_ms'] == 62.003
    assert imaging.frame_validity.target('exposure') == 62003.0
    assert info['chunk_exposure_us'] == 62003.0
    assert cam._auto_gain_enabled is False


def test_auto_gain_capture_reports_maxed_and_saves():
    """Auto-exposure pegged at the 200 ms class ceiling is MAXED: the frame
    is still returned and the lock line is logged at INFO."""
    imaging, _cam = _build(ae_lands_on_ms=500.0)
    lvp_logger.logger.reset_mock()
    _arm(imaging, AG_SETTINGS_FLUORESCENCE, resume_after_capture=False)
    image = imaging._capture_and_wait_impl(timeout_s=1.0)
    assert image is not None
    info = imaging.last_capture_info
    assert info['auto_gain'] == 'MAXED'
    assert info['auto_gain_exposure_ms'] == 200.0
    lock_lines = [m for m in _logged('info') if '[AG CONVERGE] locked: state=MAXED' in m]
    assert len(lock_lines) == 1


def test_auto_gain_capture_reports_at_minimum_below_class_floor():
    """A fluorescence auto-exposure landing at 0.4 ms -- above the camera's
    own floor but below the class's 1.0 ms usable floor -- is AT_MINIMUM,
    and the frame is still captured at 0.4 ms."""
    imaging, _cam = _build(ae_lands_on_ms=0.4)
    _arm(imaging, AG_SETTINGS_FLUORESCENCE, resume_after_capture=False)
    image = imaging._capture_and_wait_impl(timeout_s=1.0)
    assert image is not None
    info = imaging.last_capture_info
    assert info['auto_gain'] == 'AT_MINIMUM'
    assert info['auto_gain_exposure_ms'] == 0.4
    assert imaging.frame_validity.target('exposure') == 400.0


def test_lock_result_carries_the_value_to_store():
    """The API decides what a caller stores as the manual setting after a
    lock: the achieved exposure floored to the class's usable floor, so a
    GUI and a REST caller store the same value."""
    imaging, _cam = _build(ae_lands_on_ms=0.4)
    _arm(imaging, AG_SETTINGS_FLUORESCENCE, resume_after_capture=False)
    lock = imaging._lock_auto_gain_impl()
    assert lock.exposure_ms == 0.4
    assert lock.stored_exposure_ms == 1.0

    imaging, _cam = _build(ae_lands_on_ms=62.003)
    _arm(imaging, AG_SETTINGS_FLUORESCENCE, resume_after_capture=False)
    assert imaging._lock_auto_gain_impl().stored_exposure_ms == 62.003


def test_auto_gain_capture_failed_readback_still_captures():
    """No chunks and a failed hardware readback leave nothing to lock: the
    state is FAILED, no exposure/gain target is recorded, the capture still
    returns a frame, and the lock line is logged."""
    imaging, cam = _build(ae_lands_on_ms=62.0)
    lvp_logger.logger.reset_mock()
    _arm(imaging, AG_SETTINGS_FLUORESCENCE, resume_after_capture=False)
    cam.chunks_absent = True
    cam.fail_readback = True
    image = imaging._capture_and_wait_impl(timeout_s=1.0)
    assert image is not None
    info = imaging.last_capture_info
    assert info['auto_gain'] == 'FAILED'
    assert info['auto_gain_exposure_ms'] is None
    assert imaging.frame_validity.target('exposure') is None
    assert imaging.frame_validity.target('gain') is None
    assert any('[AG CONVERGE] locked: state=FAILED' in m for m in _logged('warning'))


def test_live_view_arm_resumes_after_capture():
    """A live-view arm (the default) is re-armed after the capture so the
    view keeps adjusting; a protocol step's arm stays locked off."""
    imaging, cam = _build(ae_lands_on_ms=62.0)
    _arm(imaging, AG_SETTINGS_TRANSMITTED, resume_after_capture=True)
    assert imaging._capture_and_wait_impl(timeout_s=1.0) is not None
    assert cam._auto_gain_enabled is True
    assert imaging.frame_validity.frames_until_valid() == 20

    imaging, cam = _build(ae_lands_on_ms=62.0)
    _arm(imaging, AG_SETTINGS_TRANSMITTED, resume_after_capture=False)
    assert imaging._capture_and_wait_impl(timeout_s=1.0) is not None
    assert cam._auto_gain_enabled is False


def test_live_view_lock_tells_the_user_and_a_protocol_lock_does_not():
    """A limit state reaches an attended user through the notification
    center from the API itself, so the GUI renders nothing of its own and
    a headless caller gets the same notice; a protocol step's arm is
    unattended and gets the log line only."""
    imaging, _cam = _build(ae_lands_on_ms=0.4)
    _arm(imaging, AG_SETTINGS_FLUORESCENCE, resume_after_capture=True)
    with patch('modules.lumascope_api.imaging.notifications') as notifications:
        lock = imaging._lock_auto_gain_impl()
    assert lock.state.value == 'AT_MINIMUM'
    assert notifications.info.call_count == 1
    assert '0.4' in notifications.info.call_args.args[2]

    imaging, _cam = _build(ae_lands_on_ms=0.4)
    _arm(imaging, AG_SETTINGS_FLUORESCENCE, resume_after_capture=False)
    with patch('modules.lumascope_api.imaging.notifications') as notifications:
        assert imaging._lock_auto_gain_impl().state.value == 'AT_MINIMUM'
    assert notifications.info.call_count == 0
    assert notifications.error.call_count == 0


def test_failed_lock_under_a_live_view_arm_is_an_error_to_the_user():
    imaging, cam = _build(ae_lands_on_ms=62.0)
    _arm(imaging, AG_SETTINGS_TRANSMITTED, resume_after_capture=True)
    cam.chunks_absent = True
    cam.fail_readback = True
    with patch('modules.lumascope_api.imaging.notifications') as notifications:
        assert imaging._capture_and_wait_impl(timeout_s=1.0) is not None
    assert imaging.last_capture_info['auto_gain'] == 'FAILED'
    assert notifications.error.call_count == 1


def test_public_lock_runs_the_impl_without_an_executor():
    """A bare scope with no executor runs the body on the calling thread;
    the toggle-off path uses this member, never the impl."""
    imaging, cam = _build(ae_lands_on_ms=8.0)
    _arm(imaging, AG_SETTINGS_TRANSMITTED, resume_after_capture=True)
    lock = imaging.lock_auto_gain()
    assert lock.state.value == 'CONVERGED'
    assert lock.stored_exposure_ms == 8.0
    assert cam._auto_gain_enabled is False
    assert imaging.lock_auto_gain().state is None


def test_class_floor_lives_in_config_helpers():
    """The usable exposure floor per layer class has one home beside the
    AG/AE ceiling; the layer control reads it and no longer defines it."""
    assert config_helpers.get_ag_ae_min_exposure_ms('BF') == 0.1
    assert config_helpers.get_ag_ae_min_exposure_ms('PC') == 0.1
    assert config_helpers.get_ag_ae_min_exposure_ms('Blue') == 1.0
    assert config_helpers.get_ag_ae_min_exposure_ms('Lumi') == 1.0
    tree = ast_seams.parse_module('ui/layer_control.py')
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert 'TRANSMITTED_MIN_EXPOSURE_MS' not in names
    assert 'FLUORESCENCE_MIN_EXPOSURE_MS' not in names
    assert 'get_ag_ae_min_exposure_ms' in names


def test_converged_lock_leaves_an_info_line():
    """Every lock state leaves one line in a customer bundle, the normal
    one included: a converged lock logged only at debug left no trace of
    what the camera settled at when debug was off."""
    imaging, _cam = _build(ae_lands_on_ms=62.003)
    lvp_logger.logger.reset_mock()
    _arm(imaging, AG_SETTINGS_FLUORESCENCE, resume_after_capture=False)
    assert imaging._capture_and_wait_impl(timeout_s=1.0) is not None
    lock_lines = [m for m in _logged('info') if '[AG CONVERGE] locked: state=CONVERGED' in m]
    assert len(lock_lines) == 1, _logged('info')
