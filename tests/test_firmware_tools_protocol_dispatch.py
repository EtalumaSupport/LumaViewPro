"""Regression tests — firmware_tools subcommands route through driver
methods that handle protocol dispatch (Architecture Rule 22).

Covers the cluster fix where:
  - `cmd_info` was sending raw `FULLINFO`, broken on FW4.0 motor.
  - `homing-test` helpers (_wait_for_stop, _move_to_step, _home_single,
    _home_all) were sending raw v3.0.x commands.
  - `bench --raw-commands` exposed a raw-command escape hatch.

After the fix, every external action uses production driver methods
that branch by `_use_v4()` / `_use_v35()` per protocol.
"""
import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

from drivers.motorboard import MotorBoard
from tools.firmware_tools import (
    _home_all,
    _home_single,
    _move_to_step,
    _wait_for_stop,
    cmd_info,
)


# ---------------------------------------------------------------------------
# cmd_info — motor + LED, both protocols
# ---------------------------------------------------------------------------

def _make_motor_mock(use_v4, fullinfo_dict):
    motor = MagicMock()
    motor._use_v4.return_value = use_v4
    motor.fullinfo.return_value = fullinfo_dict
    return motor


def _make_led_mock(use_v35, info_dict):
    led = MagicMock()
    led._use_v35.return_value = use_v35
    led.get_info.return_value = info_dict
    return led


class TestCmdInfo:
    def test_motor_v3_0_x_branch(self):
        motor = _make_motor_mock(
            use_v4=False,
            fullinfo_dict={
                'model': 'LS820T',
                'serial_number': '7162-19',
                '_raw': 'Model: LS820T Serial: 7162-19 X present: True ...',
            },
        )
        led = _make_led_mock(use_v35=False, info_dict={
            'version': '3.0.7', 'date': '2024-01-15', 'cal_status': 'calibrated',
            'raw': 'multi-line legacy INFO',
        })
        buf = io.StringIO()
        with patch('tools.firmware_tools._connect_motor_board', return_value=motor), \
             patch('tools.firmware_tools.LEDBoard', return_value=led), \
             redirect_stdout(buf):
            cmd_info(MagicMock())
        out = buf.getvalue()
        assert '=== Motor board ===' in out
        assert 'protocol: v3.0.x' in out
        assert 'LS820T' in out
        assert '7162-19' in out
        # Driver dispatcher was used, not raw exchange_command.
        motor.fullinfo.assert_called_once()
        motor.exchange_command.assert_not_called()
        # Both boards disconnected.
        motor.disconnect.assert_called_once()
        led.disconnect.assert_called_once()

    def test_motor_fw40_branch(self):
        motor = _make_motor_mock(
            use_v4=True,
            fullinfo_dict={
                'model': 'LS820T',
                'serial_number': '7162-19',
                '_raw': {'cmd': 'INFO', 'ok': True, 'model': 'LS820T'},
                '_info': {'cmd': 'INFO', 'ok': True, 'model': 'LS820T'},
            },
        )
        led = _make_led_mock(use_v35=True, info_dict={
            'version': '3.5.0', 'date': '2026-04-26',
            'cal_status': 'calibrated',
            'features': ['led', 'stim', 'boot_log'],
            'raw': 'INFO ver=3.5.0 ...',
        })
        buf = io.StringIO()
        with patch('tools.firmware_tools._connect_motor_board', return_value=motor), \
             patch('tools.firmware_tools.LEDBoard', return_value=led), \
             redirect_stdout(buf):
            cmd_info(MagicMock())
        out = buf.getvalue()
        assert 'protocol: FW4.0' in out
        assert 'protocol: v3.5' in out
        assert '3.5.0' in out
        assert 'led,stim,boot_log' in out
        # Raw FW4.0 dict surfaces directly (per "raw should be raw").
        assert "'cmd': 'INFO'" in out
        # No raw exchange_command anywhere.
        motor.exchange_command.assert_not_called()

    def test_disconnect_runs_on_motor_failure(self):
        motor = _make_motor_mock(use_v4=False, fullinfo_dict=None)
        motor.fullinfo.side_effect = RuntimeError('boom')
        led = _make_led_mock(use_v35=False, info_dict={})
        with patch('tools.firmware_tools._connect_motor_board', return_value=motor), \
             patch('tools.firmware_tools.LEDBoard', return_value=led), \
             pytest.raises(RuntimeError):
            cmd_info(MagicMock())
        motor.disconnect.assert_called_once()
        # LED is opened only after motor block exits cleanly.
        led.disconnect.assert_not_called()


# ---------------------------------------------------------------------------
# homing-test helpers — driver-method dispatch
# ---------------------------------------------------------------------------

class TestHomingTestHelpers:
    def test_wait_for_stop_uses_target_status(self):
        board = MagicMock()
        # First two polls: not at target. Third: at target.
        board.target_status.side_effect = [False, False, True]
        ok = _wait_for_stop(board, 'Z', timeout=5)
        assert ok is True
        assert board.target_status.call_count == 3
        assert all(c.args == ('Z',) for c in board.target_status.call_args_list)
        board.exchange_command.assert_not_called()

    def test_wait_for_stop_handles_driver_exception(self):
        board = MagicMock()
        board.target_status.side_effect = [
            RuntimeError('disconnect'), RuntimeError('disconnect'), True,
        ]
        ok = _wait_for_stop(board, 'X', timeout=5)
        assert ok is True

    def test_wait_for_stop_timeout(self):
        board = MagicMock()
        board.target_status.return_value = False
        ok = _wait_for_stop(board, 'Z', timeout=0.3)
        assert ok is False

    def test_move_to_step_uses_move_and_wait(self):
        board = MagicMock()
        board.target_status.return_value = True
        ok = _move_to_step(board, 'X', 12345)
        assert ok is True
        board.move.assert_called_once_with('X', 12345)
        board.exchange_command.assert_not_called()

    def test_home_single_routes_through_home_axis(self):
        board = MagicMock()
        board.home_axis.return_value = True
        ok, msg, dt = _home_single(board, 'Z')
        assert ok is True
        assert 'home_axis(Z)' in msg
        assert dt >= 0
        board.home_axis.assert_called_once_with('Z')
        board.exchange_command.assert_not_called()

    def test_home_single_propagates_failure(self):
        board = MagicMock()
        board.home_axis.return_value = False
        ok, msg, _ = _home_single(board, 'X')
        assert ok is False
        assert 'home_axis(X)' in msg

    def test_home_single_catches_exception(self):
        board = MagicMock()
        board.home_axis.side_effect = ValueError('axis Q invalid')
        ok, msg, _ = _home_single(board, 'Q')
        assert ok is False
        assert 'raised' in msg

    def test_home_all_uses_home(self):
        board = MagicMock()
        board.home.return_value = True
        ok, msg, _ = _home_all(board)
        assert ok is True
        assert 'home() returned True' in msg
        board.home.assert_called_once()
        board.exchange_command.assert_not_called()


# ---------------------------------------------------------------------------
# MotorBoard.home_axis — per-axis dispatcher
# ---------------------------------------------------------------------------

class TestMotorBoardHomeAxis:
    def _make_board(self, use_v4=False):
        # Bypass __init__'s auto-connect and registry plumbing so we can
        # exercise home_axis() in isolation.
        board = MotorBoard.__new__(MotorBoard)
        board._use_v4 = MagicMock(return_value=use_v4)
        board.zhome = MagicMock(return_value=True)
        board.thome = MagicMock(return_value=True)
        board.home = MagicMock(return_value=True)
        board._v4_home_wait = MagicMock(return_value=(True, {}))
        board.exchange_command = MagicMock()
        board.exchange_json = MagicMock()
        return board

    def test_z_routes_to_zhome(self):
        b = self._make_board()
        assert b.home_axis('Z') is True
        b.zhome.assert_called_once()
        b.thome.assert_not_called()
        b.home.assert_not_called()

    def test_t_routes_to_thome(self):
        b = self._make_board()
        assert b.home_axis('T') is True
        b.thome.assert_called_once()

    def test_x_v4_uses_xy_group(self):
        b = self._make_board(use_v4=True)
        assert b.home_axis('X') is True
        b._v4_home_wait.assert_called_once()
        payload, _kw = b._v4_home_wait.call_args.args[0], b._v4_home_wait.call_args.kwargs
        assert payload == {'cmd': 'HOME', 'axis': 'XY'}

    def test_y_v4_uses_xy_group(self):
        b = self._make_board(use_v4=True)
        assert b.home_axis('Y') is True
        b._v4_home_wait.assert_called_once()

    def test_x_legacy_falls_through_to_full_home(self):
        b = self._make_board(use_v4=False)
        assert b.home_axis('X') is True
        b.home.assert_called_once()
        b._v4_home_wait.assert_not_called()

    def test_y_legacy_falls_through_to_full_home(self):
        b = self._make_board(use_v4=False)
        assert b.home_axis('Y') is True
        b.home.assert_called_once()

    def test_invalid_axis_raises(self):
        b = self._make_board()
        with pytest.raises(ValueError):
            b.home_axis('Q')
