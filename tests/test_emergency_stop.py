# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for MotorBoard.stop / LEDBoard.stop / Lumascope.emergency_stop.

Closes the audit-surfaced gap (2026-04-24 session 37 audit): firmware
exposes STOP on both boards but LVP had no driver or API caller on any
firmware version. The three entry points here are:

  - MotorBoard.stop() — V4 JSON STOP, LEGACY text STOP, normalized dict
  - LEDBoard.stop()   — V4 JSON STOP, LEGACY degrades to leds_off
  - Lumascope.emergency_stop() — one-shot "stop everything safe"
"""
from unittest.mock import MagicMock, patch

import pytest

from drivers.motorboard import MotorBoard
from drivers.ledboard import LEDBoard


# ---------------------------------------------------------------------------
# MotorBoard.stop
# ---------------------------------------------------------------------------

class TestMotorStop:
    def _make_motor(self):
        m = MotorBoard.__new__(MotorBoard)
        return m

    def test_v4_returns_normalized_dict_with_positions(self):
        m = self._make_motor()
        m._use_v4 = MagicMock(return_value=True)
        m.exchange_json = MagicMock(return_value={
            'ok': True, 'stopped': True,
            'X': 12345, 'Y': 67890, 'Z': 111, 'T': 222,
        })
        result = m.stop()
        assert result == {
            'ok': True, 'stopped': True,
            'positions': {'X': 12345, 'Y': 67890, 'Z': 111, 'T': 222},
            'response': None,
        }
        m.exchange_json.assert_called_once_with({'cmd': 'STOP'}, timeout=5)

    def test_v4_partial_axes_only_included(self):
        # Two-axis board (Z, T only) — firmware omits X/Y.
        m = self._make_motor()
        m._use_v4 = MagicMock(return_value=True)
        m.exchange_json = MagicMock(return_value={
            'ok': True, 'stopped': True, 'Z': 500, 'T': 1000,
        })
        result = m.stop()
        assert result['positions'] == {'Z': 500, 'T': 1000}

    def test_v4_none_response_returns_none(self):
        m = self._make_motor()
        m._use_v4 = MagicMock(return_value=True)
        m.exchange_json = MagicMock(return_value=None)
        assert m.stop() is None

    def test_legacy_returns_normalized_dict(self):
        m = self._make_motor()
        m._use_v4 = MagicMock(return_value=False)
        m.exchange_command = MagicMock(return_value='STOPPED')
        result = m.stop()
        assert result == {
            'ok': True, 'stopped': True,
            'positions': None, 'response': 'STOPPED',
        }
        m.exchange_command.assert_called_once_with('STOP', timeout=5)

    def test_legacy_none_response_returns_none(self):
        m = self._make_motor()
        m._use_v4 = MagicMock(return_value=False)
        m.exchange_command = MagicMock(return_value=None)
        assert m.stop() is None

    def test_legacy_non_stopped_response_flagged(self):
        # If firmware returns something unexpected, we record ok=True
        # (the command went through) but stopped=False (the content
        # didn't confirm). Defensive — callers can decide.
        m = self._make_motor()
        m._use_v4 = MagicMock(return_value=False)
        m.exchange_command = MagicMock(return_value='ERROR: something went wrong')
        result = m.stop()
        assert result['ok'] is True
        assert result['stopped'] is False
        assert 'ERROR' in result['response']


# ---------------------------------------------------------------------------
# LEDBoard.stop
# ---------------------------------------------------------------------------

class TestLedStop:
    def _make_led(self):
        l = LEDBoard.__new__(LEDBoard)
        return l

    def test_v4_returns_normalized_dict(self):
        l = self._make_led()
        l._use_v4 = MagicMock(return_value=True)
        l.exchange_json = MagicMock(return_value={'ok': True, 'stopped': True})
        result = l.stop()
        assert result == {
            'ok': True, 'stopped': True,
            'response': None, 'note': None,
        }
        l.exchange_json.assert_called_once_with({'cmd': 'STOP'}, timeout=5)

    def test_v4_none_returns_none(self):
        l = self._make_led()
        l._use_v4 = MagicMock(return_value=True)
        l.exchange_json = MagicMock(return_value=None)
        assert l.stop() is None

    def test_legacy_degrades_to_leds_off(self):
        l = self._make_led()
        l._use_v4 = MagicMock(return_value=False)
        l.leds_off = MagicMock()
        result = l.stop()
        l.leds_off.assert_called_once()
        assert result['ok'] is True
        assert result['stopped'] is True
        assert 'LEGACY' in result['note']
        assert 'leds_off' in result['note']


# ---------------------------------------------------------------------------
# Lumascope.emergency_stop
# ---------------------------------------------------------------------------

class TestLumascopeEmergencyStop:
    def _make_scope(self, with_motor=True, with_led=True):
        """Minimal Lumascope sufficient to exercise emergency_stop.

        Bypasses __init__ so we don't need real hardware or Kivy setup.
        """
        import threading
        from modules.lumascope_api import Lumascope
        s = Lumascope.__new__(Lumascope)
        s.motion = MagicMock() if with_motor else None
        s.led = MagicMock() if with_led else None
        s._led_owner_lock = threading.Lock()
        s._led_listeners_lock = threading.Lock()  # hasattr guard uses this
        s._led_owners = {}
        s._led_state = {}
        s.frame_validity = MagicMock()
        s._fire_led_listeners = MagicMock()
        if with_led:
            s.led.available_colors = MagicMock(return_value=['Blue', 'Green', 'Red'])
        return s

    def test_calls_both_stops(self):
        s = self._make_scope()
        s.motion.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                                'positions': {'X': 0}, 'response': None})
        s.led.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                             'response': None, 'note': None})
        result = s.emergency_stop()
        s.motion.stop.assert_called_once()
        s.led.stop.assert_called_once()
        assert result['motion']['stopped'] is True
        assert result['led']['stopped'] is True

    def test_clears_led_owners_and_state(self):
        s = self._make_scope()
        s._led_owners = {'Blue': 'protocol_1'}
        s._led_state = {'Blue': {'enabled': True, 'illumination': 50.0, 'owner': 'protocol_1'}}
        s.motion.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                                'positions': None, 'response': None})
        s.led.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                             'response': None, 'note': None})
        s.emergency_stop()
        assert s._led_owners == {}
        assert s._led_state == {}

    def test_fires_led_listeners_for_all_colors(self):
        s = self._make_scope()
        s.motion.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                                'positions': None, 'response': None})
        s.led.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                             'response': None, 'note': None})
        s.emergency_stop()
        # One call per available color, all with enabled=False, mA=0.0.
        assert s._fire_led_listeners.call_count == 3
        for call in s._fire_led_listeners.call_args_list:
            _color, enabled, ma, _owner = call.args
            assert enabled is False
            assert ma == 0.0

    def test_motion_exception_still_runs_led_stop(self):
        """Safety promise: one side exploding doesn't skip the other."""
        s = self._make_scope()
        s.motion.stop = MagicMock(side_effect=RuntimeError('motor wedged'))
        s.led.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                             'response': None, 'note': None})
        result = s.emergency_stop()
        assert result['motion'] == 'error'
        assert result['led']['stopped'] is True
        s.led.stop.assert_called_once()

    def test_led_exception_still_after_motion(self):
        s = self._make_scope()
        s.motion.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                                'positions': None, 'response': None})
        s.led.stop = MagicMock(side_effect=RuntimeError('led fried'))
        result = s.emergency_stop()
        assert result['motion']['stopped'] is True
        assert result['led'] == 'error'
        s.motion.stop.assert_called_once()

    def test_motor_returns_none_treated_as_error(self):
        s = self._make_scope()
        s.motion.stop = MagicMock(return_value=None)
        s.led.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                             'response': None, 'note': None})
        result = s.emergency_stop()
        assert result['motion'] == 'error'
        assert result['led']['stopped'] is True

    def test_absent_motion_marked_absent(self):
        s = self._make_scope(with_motor=False)
        s.led.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                             'response': None, 'note': None})
        result = s.emergency_stop()
        assert result['motion'] == 'absent'
        assert result['led']['stopped'] is True

    def test_absent_led_marked_absent(self):
        s = self._make_scope(with_led=False)
        s.motion.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                                'positions': None, 'response': None})
        result = s.emergency_stop()
        assert result['motion']['stopped'] is True
        assert result['led'] == 'absent'

    def test_both_absent_no_crash(self):
        s = self._make_scope(with_motor=False, with_led=False)
        result = s.emergency_stop()
        assert result == {'motion': 'absent', 'led': 'absent'}

    def test_diagnostic_mode_missing_ui_infra_does_not_contaminate_status(self):
        """Lumascope.create_diagnostic builds a minimal scope without
        frame_validity / _led_listeners_lock / _led_owner_lock. The
        emergency_stop safety promise: firmware-side stops succeed
        cleanly even when those API-side UI attrs are missing. Caught
        on bench 2026-04-24 — diagnostic scope flipped LED to 'error'
        because a listener-fire raised AttributeError after led.stop
        actually succeeded.
        """
        import threading
        from modules.lumascope_api import Lumascope
        s = Lumascope.__new__(Lumascope)
        s.motion = MagicMock()
        s.led = MagicMock()
        # INTENTIONALLY omit _led_owner_lock, _led_listeners_lock,
        # frame_validity, _led_state, _led_owners — mirrors
        # create_diagnostic's minimal init.
        s.motion.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                                'positions': None, 'response': None})
        s.led.stop = MagicMock(return_value={'ok': True, 'stopped': True,
                                             'response': None, 'note': None})
        result = s.emergency_stop()
        # Status reflects actual firmware-side success, not the missing
        # UI plumbing.
        assert result['motion']['stopped'] is True
        assert result['led']['stopped'] is True
        assert result['led'] != 'error'
