# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for ScopeInitConfig and the partial-hardware notification filter.

Background: pre-fix, `Lumascope.__init__` warned "Partial Hardware
Detected" whenever any of LED / motor / camera failed to construct.
For an LS620 (which legitimately has no motor — `Focus=false,
XYStage=false, Turret=false` in scopes.json) every startup popped the
warning twice (once for the initial connect attempt and once on the
auto-reconnect). The fix moves the notification into
`Lumascope.initialize(config)` and filters `missing` against the
scope's expected hardware as captured on `ScopeInitConfig`.
"""

# Heavy deps are mocked by tests/conftest.py at module-import time.

import pytest

from modules.lumascope_api import Lumascope
from modules.notification_center import NotificationCenter, Severity
from modules.scope_init_config import ScopeInitConfig
from drivers.null_motorboard import NullMotionBoard
from drivers.null_ledboard import NullLEDBoard


# ---------- ScopeInitConfig.from_settings ----------

_BASE_SETTINGS = {
    'binning': {'size': '1x1'},
    'frame': {'width': 1900, 'height': 1900},
    'objective_id': '4x',
    'turret_objectives': None,
    'motion': {'acceleration_max_pct': 100},
    'stage_offset': {'x': 0, 'y': 0},
    'scale_bar': {'enabled': False},
    'microscope': 'LS820',
}

_LS620_CONFIG = {
    'Focus': False, 'XYStage': False, 'Turret': False,
    'Layers': {'Lumi': False, 'Fluorescence': True, 'Darkfield': False,
               'Brightfield': True, 'PhaseContrast': True},
}
_LS820_CONFIG = {
    'Focus': True, 'XYStage': False, 'Turret': False,
    'Layers': {'Lumi': False, 'Fluorescence': True, 'Darkfield': True,
               'Brightfield': True, 'PhaseContrast': True},
}
_LS850T_CONFIG = {
    'Focus': True, 'XYStage': True, 'Turret': True,
    'Layers': {'Lumi': False, 'Fluorescence': True, 'Darkfield': True,
               'Brightfield': True, 'PhaseContrast': True},
}


class TestFromSettings:

    def test_default_no_scope_config_preserves_pre_filter_behavior(self):
        config = ScopeInitConfig.from_settings(_BASE_SETTINGS, labware=None)
        assert config.expects_motion is True
        assert config.expects_led is True

    def test_ls620_no_motor_expected(self):
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS, labware=None, scope_config=_LS620_CONFIG,
        )
        assert config.expects_motion is False
        assert config.expects_led is True

    def test_ls820_motor_expected_via_focus(self):
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS, labware=None, scope_config=_LS820_CONFIG,
        )
        assert config.expects_motion is True
        assert config.expects_led is True

    def test_ls850t_motor_expected_via_xystage_and_turret(self):
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS, labware=None, scope_config=_LS850T_CONFIG,
        )
        assert config.expects_motion is True
        assert config.expects_led is True

    def test_all_layers_false_means_no_led_expected(self):
        scope_config = {
            'Focus': True, 'XYStage': False, 'Turret': False,
            'Layers': {'Lumi': False, 'Fluorescence': False,
                       'Darkfield': False, 'Brightfield': False,
                       'PhaseContrast': False},
        }
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS, labware=None, scope_config=scope_config,
        )
        assert config.expects_led is False


# ---------- _notify_partial_hardware filter ----------

def _make_scope_with_no_hardware():
    """Sim scope, then strip drivers to Null* / no camera, then flip
    `_simulated` off so the early-return doesn't fire."""
    scope = Lumascope(simulate=True)
    scope.led = NullLEDBoard()
    scope.motion = NullMotionBoard()
    if hasattr(scope, 'camera'):
        delattr(scope, 'camera')
    scope._simulated = False
    return scope


@pytest.fixture
def captured_warnings(monkeypatch):
    """Swap a fresh NotificationCenter (no dedup) into lumascope_api so
    each test sees only its own notifications."""
    fresh_nc = NotificationCenter(dedup_window_s=0)
    received = []
    fresh_nc.add_listener(lambda n: received.append(n),
                          min_severity=Severity.WARNING)
    monkeypatch.setattr('modules.lumascope_api.notifications', fresh_nc)
    return received


class TestNotifyPartialHardware:

    def test_simulator_never_warns(self, captured_warnings):
        scope = Lumascope(simulate=True)
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS, labware=None, scope_config=_LS620_CONFIG,
        )
        scope._notify_partial_hardware(config)
        assert captured_warnings == []

    def test_ls620_no_motor_no_warning(self, captured_warnings):
        scope = _make_scope_with_no_hardware()
        # LS620 has Layers — pretend the LED board did connect by
        # swapping Null out for a real-ish object.
        scope.led = object()  # truthy non-Null sentinel
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS, labware=None, scope_config=_LS620_CONFIG,
        )
        scope._notify_partial_hardware(config)
        # No motor expected, LED present, no camera attached → only
        # camera should be reported as missing.
        assert len(captured_warnings) == 1
        assert "Camera" in captured_warnings[0].message
        assert "Motor Controller" not in captured_warnings[0].message

    def test_ls820_motor_failed_warns(self, captured_warnings):
        scope = _make_scope_with_no_hardware()
        scope.led = object()
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS, labware=None, scope_config=_LS820_CONFIG,
        )
        scope._notify_partial_hardware(config)
        assert len(captured_warnings) == 1
        assert "Motor Controller" in captured_warnings[0].message

    def test_no_scope_config_warns_for_missing_motor(self, captured_warnings):
        """Backward-compat: callers that don't supply scope_config get
        the pre-filter behavior (any Null driver → warning)."""
        scope = _make_scope_with_no_hardware()
        scope.led = object()
        config = ScopeInitConfig.from_settings(_BASE_SETTINGS, labware=None)
        scope._notify_partial_hardware(config)
        assert len(captured_warnings) == 1
        assert "Motor Controller" in captured_warnings[0].message
