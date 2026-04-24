# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stage 2 tests: Lumascope fan API + UI-gating policy + listener.

Stage 2 layers a thin Lumascope API over MotorBoard's Stage 1 fan
methods. The headline addition is `fan_ui_kind()` — one probe that
encodes the UI-gating policy (decided 2026-04-24 bench):

  - Legacy firmware: hide fan UI entirely (return None).
  - FW4.0 + PWM fan: 'PWM' — slider + RPM readout.
  - FW4.0 + HiLo fan: 'HILO' — HI/LO/OFF radio.

Stage 4 (Kivy) code does one `fan_ui_kind()` call and branches. No
combinatoric reasoning about protocol × hardware in UI code.
"""
import threading
from unittest.mock import MagicMock

import pytest


def _make_scope(with_motor=True, fan_status=None, use_v4=True,
                motor_has_fan_api=True):
    """Build a minimal Lumascope sufficient for fan-API tests."""
    from modules.lumascope_api import Lumascope
    s = Lumascope.__new__(Lumascope)
    if with_motor:
        m = MagicMock()
        m._use_v4 = MagicMock(return_value=use_v4)
        if motor_has_fan_api:
            m.get_fan_status = MagicMock(return_value=fan_status)
            m.fan_supports_pwm = MagicMock(
                return_value=(fan_status or {}).get('mode') == 'PWM')
            m.set_fan_hilo = MagicMock(return_value=True)
            m.set_fan_pwm = MagicMock(return_value=True)
        else:
            # Strip the fan attrs to simulate an older MotorBoard or
            # an alternative motor driver without the Stage 1 methods.
            for attr in ('get_fan_status', 'fan_supports_pwm',
                         'set_fan_hilo', 'set_fan_pwm'):
                if hasattr(m, attr):
                    delattr(m, attr)
            # MagicMock auto-attrs everything, so explicit delattr is
            # insufficient. Use spec=[] to lock down the API surface.
            m = MagicMock(spec=['_use_v4'])
            m._use_v4 = MagicMock(return_value=use_v4)
        s.motion = m
    else:
        s.motion = None
    s._fan_listeners_lock = threading.Lock()
    s._fan_listeners = []
    return s


# ---------------------------------------------------------------------------
# fan_ui_kind — the one policy probe
# ---------------------------------------------------------------------------

class TestFanUiKind:
    def test_pwm_when_fw40_and_pwm_fan(self):
        s = _make_scope(fan_status={'mode': 'PWM', 'state': None,
                                    'fan_pct': 50, 'tach_rpm': 2000})
        assert s.fan_ui_kind() == 'PWM'

    def test_hilo_when_fw40_and_hilo_fan(self):
        s = _make_scope(fan_status={'mode': 'HILO', 'state': 'HI',
                                    'fan_pct': None, 'tach_rpm': None})
        assert s.fan_ui_kind() == 'HILO'

    def test_none_when_legacy_firmware(self):
        # Even if hardware has a fan, legacy firmware hides UI per policy.
        s = _make_scope(use_v4=False,
                        fan_status={'mode': 'HILO', 'state': 'HI',
                                    'fan_pct': None, 'tach_rpm': None})
        assert s.fan_ui_kind() is None

    def test_none_when_no_fan_hardware(self):
        s = _make_scope(fan_status={'mode': 'NONE', 'state': None,
                                    'fan_pct': None, 'tach_rpm': None})
        assert s.fan_ui_kind() is None

    def test_none_when_no_motor(self):
        s = _make_scope(with_motor=False)
        assert s.fan_ui_kind() is None

    def test_none_when_status_probe_fails(self):
        s = _make_scope(fan_status=None)  # driver returned None (error)
        assert s.fan_ui_kind() is None

    def test_none_when_motor_missing_fan_api(self):
        # Alternative motor drivers (e.g. FX2, future silicon) may not
        # implement Stage 1 — fan_ui_kind must not crash.
        s = _make_scope(motor_has_fan_api=False)
        assert s.fan_ui_kind() is None


# ---------------------------------------------------------------------------
# get_fan_status / set_fan_hilo / set_fan_pwm / fan_supports_pwm
# ---------------------------------------------------------------------------

class TestFanApiPassthrough:
    def test_get_fan_status_delegates(self):
        expected = {'mode': 'HILO', 'state': 'LO', 'fan_pct': None,
                    'tach_rpm': None}
        s = _make_scope(fan_status=expected)
        assert s.get_fan_status() == expected
        s.motion.get_fan_status.assert_called_once()

    def test_get_fan_status_none_when_no_motor(self):
        s = _make_scope(with_motor=False)
        assert s.get_fan_status() is None

    def test_set_fan_hilo_delegates(self):
        s = _make_scope(fan_status={'mode': 'HILO', 'state': 'OFF'})
        assert s.set_fan_hilo('HI') is True
        s.motion.set_fan_hilo.assert_called_once_with('HI')

    def test_set_fan_hilo_false_when_no_motor(self):
        s = _make_scope(with_motor=False)
        assert s.set_fan_hilo('HI') is False

    def test_set_fan_pwm_delegates(self):
        s = _make_scope(fan_status={'mode': 'PWM', 'fan_pct': 0,
                                    'tach_rpm': 0})
        assert s.set_fan_pwm(75) is True
        s.motion.set_fan_pwm.assert_called_once_with(75)

    def test_set_fan_pwm_false_when_no_motor(self):
        s = _make_scope(with_motor=False)
        assert s.set_fan_pwm(50) is False

    def test_fan_supports_pwm_delegates(self):
        s = _make_scope(fan_status={'mode': 'PWM'})
        assert s.fan_supports_pwm() is True
        s.motion.fan_supports_pwm.assert_called_once()

    def test_fan_supports_pwm_false_when_no_motor(self):
        s = _make_scope(with_motor=False)
        assert s.fan_supports_pwm() is False


# ---------------------------------------------------------------------------
# Listeners — add / remove / fire
# ---------------------------------------------------------------------------

class TestFanListeners:
    def test_set_fan_hilo_fires_listeners_on_success(self):
        status = {'mode': 'HILO', 'state': 'HI', 'fan_pct': None,
                  'tach_rpm': None}
        s = _make_scope(fan_status=status)
        captured = []
        s.add_fan_listener(captured.append)
        s.set_fan_hilo('HI')
        assert captured == [status]

    def test_set_fan_pwm_fires_listeners_on_success(self):
        status = {'mode': 'PWM', 'state': None, 'fan_pct': 80,
                  'tach_rpm': 2500}
        s = _make_scope(fan_status=status)
        captured = []
        s.add_fan_listener(captured.append)
        s.set_fan_pwm(80)
        assert captured == [status]

    def test_listener_exception_does_not_break_other_listeners(self):
        s = _make_scope(fan_status={'mode': 'HILO', 'state': 'HI'})
        good_captured = []
        def bad(_status):
            raise RuntimeError('listener broken')
        s.add_fan_listener(bad)
        s.add_fan_listener(good_captured.append)
        s.set_fan_hilo('HI')
        # bad listener raised, but the good one still fired
        assert len(good_captured) == 1

    def test_remove_listener(self):
        s = _make_scope(fan_status={'mode': 'HILO', 'state': 'HI'})
        captured = []
        s.add_fan_listener(captured.append)
        s.remove_fan_listener(captured.append)
        s.set_fan_hilo('HI')
        # listener was removed before the set call
        assert captured == []

    def test_remove_missing_listener_silent(self):
        # Remove a listener that was never added — must not raise.
        s = _make_scope(fan_status={'mode': 'HILO', 'state': 'HI'})
        s.remove_fan_listener(lambda _s: None)

    def test_add_listener_noop_in_diagnostic_mode(self):
        # Lumascope.create_diagnostic builds a minimal scope without
        # _fan_listeners_lock. add_fan_listener must not crash.
        from modules.lumascope_api import Lumascope
        s = Lumascope.__new__(Lumascope)
        s.motion = MagicMock()
        # Intentionally do NOT set _fan_listeners_lock.
        s.add_fan_listener(lambda _s: None)  # must not raise

    def test_fire_noop_when_status_none(self):
        # If get_fan_status returns None, listeners don't fire — no
        # point delivering an empty event.
        s = _make_scope(fan_status=None)
        captured = []
        s.add_fan_listener(captured.append)
        s._fire_fan_listeners()
        assert captured == []
