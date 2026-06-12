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
import re
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
        'BF': 50.0, 'PC': 50.0, 'DF': 50.0,
        'Blue': 200.0, 'Green': 200.0, 'Red': 200.0,
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
    src = (REPO / 'modules' / 'lumascope_api' / 'imaging.py').read_text()
    assert re.search(
        r"ae_max_exposure_ms\s*=\s*settings\.get\(\s*['\"]max_exposure_ms['\"]", src
    ), 'imaging.set_auto_gain must forward settings["max_exposure_ms"] as the cap. (#655)'


def test_live_caller_injects_per_class_cap():
    src = (REPO / 'ui' / 'layer_control.py').read_text()
    assert 'get_ag_ae_max_exposure_ms' in src and 'max_exposure_ms' in src, (
        'layer_control.apply_settings must inject the per-class AG/AE cap '
        'into the auto-gain settings dict. (#655)'
    )


def test_protocol_caller_injects_per_class_cap():
    src = (REPO / 'modules' / 'protocol_step_runner.py').read_text()
    assert 'get_ag_ae_max_exposure_ms' in src, (
        'protocol_step_runner must set the per-class AG/AE cap for the '
        "step's channel class before arming. (#655)"
    )


def test_video_capture_rearm_forwards_cap():
    src = (REPO / 'modules' / 'video_capture.py').read_text()
    assert re.search(
        r"ae_max_exposure_ms\s*=\s*self\._autogain_settings\.get\(\s*['\"]max_exposure_ms['\"]",
        src,
    ), 'video_capture first-frame AG re-arm must forward the cap. (#655)'
