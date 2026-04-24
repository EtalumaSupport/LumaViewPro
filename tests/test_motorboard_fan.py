# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stage 1 tests: MotorBoard fan control (driver layer).

Fan firmware was already complete on both v3.0.x (FAN:HI/LO/OFF,
FANPWM:0-100, FANSPEED:HILO/PWM) and FW4.0 (@fw.command('FAN') with
PWM, HiLo, and tach readback). These tests cover the new driver
wrappers — get_fan_status, set_fan_hilo, set_fan_pwm, fan_supports_pwm
— for both protocol versions.

UI-level gating ("only show fan UI on FW4.0; PWM widget iff PWM fan")
lives at Stage 2/4 — the driver exposes the raw capability and the
API/UI compose the policy.
"""
from unittest.mock import MagicMock

import pytest

from drivers.motorboard import MotorBoard


def _make_motor(v4=True):
    m = MotorBoard.__new__(MotorBoard)
    m._use_v4 = MagicMock(return_value=v4)
    # _state_lock / _fullinfo are consulted by LEGACY path.
    import threading
    m._state_lock = threading.Lock()
    m._fullinfo = None
    return m


# ---------------------------------------------------------------------------
# get_fan_status — V4 path
# ---------------------------------------------------------------------------

class TestGetFanStatusV4:
    def test_pwm_fan_full_response(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={
            'ok': True, 'fan': 'PWM', 'fan_pct': 50, 'tach_rpm': 2345,
        })
        s = m.get_fan_status()
        assert s['mode'] == 'PWM'
        assert s['state'] is None
        assert s['fan_pct'] == 50
        assert s['tach_rpm'] == 2345

    def test_hilo_fan_state_hi(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={'ok': True, 'fan': 'HI'})
        s = m.get_fan_status()
        assert s['mode'] == 'HILO'
        assert s['state'] == 'HI'
        assert s['fan_pct'] is None
        assert s['tach_rpm'] is None

    def test_hilo_fan_state_off(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={'ok': True, 'fan': 'OFF'})
        s = m.get_fan_status()
        assert s['mode'] == 'HILO'
        assert s['state'] == 'OFF'

    def test_hilo_with_pwm_tach_piggyback(self):
        # Firmware _fan_state_response returns HiLo state AND tach_rpm
        # if a PWM fan is present alongside a HiLo controller.
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={
            'ok': True, 'fan': 'LO', 'tach_rpm': 1200,
        })
        s = m.get_fan_status()
        assert s['mode'] == 'HILO'
        assert s['state'] == 'LO'
        assert s['tach_rpm'] == 1200

    def test_no_fan_hardware(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={'ok': True, 'fan': 'NONE'})
        s = m.get_fan_status()
        assert s['mode'] == 'NONE'
        assert s['state'] is None

    def test_firmware_error_returns_none(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={'ok': False, 'err': 'X'})
        assert m.get_fan_status() is None

    def test_no_response_returns_none(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value=None)
        assert m.get_fan_status() is None


# ---------------------------------------------------------------------------
# get_fan_status — LEGACY (v3.0.x) path parses FULLINFO
# ---------------------------------------------------------------------------

class TestGetFanStatusLegacy:
    def test_hilo_from_fullinfo(self):
        m = _make_motor(v4=False)
        m.exchange_command = MagicMock(
            return_value='Model: LS720 Serial: 42 FanCntl: HI/LO   Speed:HI Other: stuff'
        )
        s = m.get_fan_status()
        assert s['mode'] == 'HILO'
        assert s['state'] == 'HI'
        assert s['fan_pct'] is None
        assert s['tach_rpm'] is None

    def test_pwm_from_fullinfo(self):
        m = _make_motor(v4=False)
        m.exchange_command = MagicMock(
            return_value='Model: LS850 Serial: 100 FanCntl: PWM  Speed: 75% Tach: 3100 RPM'
        )
        s = m.get_fan_status()
        assert s['mode'] == 'PWM'
        assert s['fan_pct'] == 75
        assert s['tach_rpm'] == 3100

    def test_no_fan_info_in_fullinfo(self):
        m = _make_motor(v4=False)
        m.exchange_command = MagicMock(
            return_value='Model: LS720 Serial: 42'  # no FanCntl line
        )
        s = m.get_fan_status()
        assert s['mode'] == 'NONE'
        assert s['state'] is None

    def test_uses_cached_fullinfo_when_available(self):
        m = _make_motor(v4=False)
        m._fullinfo = {'_raw': 'FanCntl: HI/LO   Speed:LO'}
        m.exchange_command = MagicMock()
        s = m.get_fan_status()
        assert s['state'] == 'LO'
        # Cached path — no FULLINFO re-fetch.
        m.exchange_command.assert_not_called()

    def test_empty_response_returns_none_mode(self):
        m = _make_motor(v4=False)
        m.exchange_command = MagicMock(return_value=None)
        s = m.get_fan_status()
        # None raw means NONE mode with no state
        assert s['mode'] == 'NONE'


# ---------------------------------------------------------------------------
# fan_supports_pwm capability probe
# ---------------------------------------------------------------------------

class TestFanSupportsPwm:
    def test_true_when_mode_pwm(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={
            'ok': True, 'fan': 'PWM', 'fan_pct': 50, 'tach_rpm': 2345,
        })
        assert m.fan_supports_pwm() is True

    def test_false_when_mode_hilo(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={'ok': True, 'fan': 'HI'})
        assert m.fan_supports_pwm() is False

    def test_false_when_no_fan(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={'ok': True, 'fan': 'NONE'})
        assert m.fan_supports_pwm() is False

    def test_false_when_status_fails(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value=None)
        assert m.fan_supports_pwm() is False


# ---------------------------------------------------------------------------
# set_fan_hilo
# ---------------------------------------------------------------------------

class TestSetFanHilo:
    def test_v4_hi_ok(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={'ok': True, 'fan': 'HI'})
        assert m.set_fan_hilo('HI') is True
        m.exchange_json.assert_called_once_with({'cmd': 'FAN', 'mode': 'HI'})

    def test_v4_case_insensitive(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={'ok': True, 'fan': 'LO'})
        assert m.set_fan_hilo('lo') is True
        m.exchange_json.assert_called_once_with({'cmd': 'FAN', 'mode': 'LO'})

    def test_v4_firmware_rejects(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={
            'ok': False, 'err': 'NOT_PRESENT', 'msg': 'discrete fan not present'
        })
        assert m.set_fan_hilo('HI') is False

    def test_legacy_fan_hi(self):
        m = _make_motor(v4=False)
        m.exchange_command = MagicMock(return_value='Fan set to high')
        assert m.set_fan_hilo('HI') is True
        m.exchange_command.assert_called_once_with('FAN:HI', timeout=5)

    def test_legacy_error(self):
        m = _make_motor(v4=False)
        m.exchange_command = MagicMock(return_value='ERROR: Fan not present.')
        assert m.set_fan_hilo('HI') is False

    def test_invalid_state_raises(self):
        m = _make_motor(v4=True)
        with pytest.raises(ValueError):
            m.set_fan_hilo('SUPER_HIGH')


# ---------------------------------------------------------------------------
# set_fan_pwm
# ---------------------------------------------------------------------------

class TestSetFanPwm:
    def test_v4_ok(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={
            'ok': True, 'fan': 'PWM', 'fan_pct': 60,
        })
        assert m.set_fan_pwm(60) is True
        m.exchange_json.assert_called_once_with({'cmd': 'FAN', 'mode': 60})

    def test_v4_boundary_zero(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={
            'ok': True, 'fan': 'PWM', 'fan_pct': 0,
        })
        assert m.set_fan_pwm(0) is True

    def test_v4_boundary_hundred(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={
            'ok': True, 'fan': 'PWM', 'fan_pct': 100,
        })
        assert m.set_fan_pwm(100) is True

    def test_v4_firmware_rejects(self):
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={
            'ok': False, 'err': 'NOT_PRESENT', 'msg': 'PWM fan not present'
        })
        assert m.set_fan_pwm(50) is False

    def test_legacy_ok(self):
        m = _make_motor(v4=False)
        m.exchange_command = MagicMock(return_value='Fan set to 75%')
        assert m.set_fan_pwm(75) is True
        m.exchange_command.assert_called_once_with('FANPWM:75', timeout=5)

    def test_legacy_error(self):
        m = _make_motor(v4=False)
        m.exchange_command = MagicMock(
            return_value='ERROR: PWM fan not present.')
        assert m.set_fan_pwm(50) is False

    def test_out_of_range_raises(self):
        m = _make_motor(v4=True)
        with pytest.raises(ValueError):
            m.set_fan_pwm(-1)
        with pytest.raises(ValueError):
            m.set_fan_pwm(101)

    def test_non_int_raises(self):
        m = _make_motor(v4=True)
        with pytest.raises(ValueError):
            m.set_fan_pwm('high')

    def test_float_coerces(self):
        # int() coerces — 50.7 -> 50. Acceptable for UI slider drag.
        m = _make_motor(v4=True)
        m.exchange_json = MagicMock(return_value={
            'ok': True, 'fan': 'PWM', 'fan_pct': 50,
        })
        assert m.set_fan_pwm(50.7) is True
