# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for NullMotionBoard — verifies interface compliance and no-op behavior."""

import threading
from unittest.mock import MagicMock

# Heavy deps are mocked by tests/conftest.py at module-import time.

import pytest
from drivers.null_motorboard import NullMotionBoard


class TestNullMotionBoardInterface:
    """Verify NullMotionBoard has the same interface as real MotorBoard."""

    @pytest.fixture
    def board(self):
        return NullMotionBoard()

    def test_has_driver_attribute(self, board):
        assert board.driver  # truthy

    def test_has_thread_lock(self, board):
        assert hasattr(board.thread_lock, 'acquire')
        assert hasattr(board.thread_lock, 'release')

    def test_has_overshoot(self, board):
        assert board.overshoot is False

    def test_has_axes_config(self, board):
        assert 'X' in board.axes_config
        assert 'Y' in board.axes_config
        assert 'Z' in board.axes_config
        assert 'T' in board.axes_config


class TestNullMotionBoardMovement:
    """Movement methods return immediately without error."""

    @pytest.fixture
    def board(self):
        return NullMotionBoard()

    def test_move_noop(self, board):
        board.move('Z', 1000)  # should not raise

    def test_move_abs_pos_noop(self, board):
        board.move_abs_pos('Z', 5000.0)

    def test_move_rel_pos_noop(self, board):
        board.move_rel_pos('X', 100.0)


class TestNullMotionBoardPosition:
    """Position queries return 0."""

    @pytest.fixture
    def board(self):
        return NullMotionBoard()

    def test_target_pos_zero(self, board):
        assert board.target_pos('Z') == 0.0
        assert board.target_pos('X') == 0.0

    def test_current_pos_zero(self, board):
        assert board.current_pos('Z') == 0.0

    def test_target_pos_steps_zero(self, board):
        assert board.target_pos_steps('Z') == 0

    def test_current_pos_steps_zero(self, board):
        assert board.current_pos_steps('Z') == 0


class TestNullMotionBoardStatus:
    """Status queries return safe defaults."""

    @pytest.fixture
    def board(self):
        return NullMotionBoard()

    def test_target_status_arrived(self, board):
        assert board.target_status('Z') is True

    def test_has_turret_false(self, board):
        assert board.has_turret() is False

    def test_has_homed_true(self, board):
        assert board.has_homed() is True

    def test_has_thomed_true(self, board):
        assert board.has_thomed() is True

    def test_homing_complete(self, board):
        assert board.initial_homing_complete is True
        assert board.initial_t_homing_complete is True


class TestNullMotionBoardHoming:
    """Homing returns immediately."""

    @pytest.fixture
    def board(self):
        return NullMotionBoard()

    def test_zhome(self, board):
        assert board.zhome() is True

    def test_home(self, board):
        assert board.home() is True

    def test_thome(self, board):
        assert board.thome() is True


class TestNullMotionBoardInfo:
    """Info methods return safe defaults."""

    @pytest.fixture
    def board(self):
        return NullMotionBoard()

    def test_fullinfo(self, board):
        info = board.fullinfo()
        assert isinstance(info, dict)
        assert info['model'] is None

    def test_get_microscope_model(self, board):
        assert board.get_microscope_model() is None

    def test_get_axis_limits(self, board):
        limits = board.get_axis_limits('Z')
        assert limits is not None
        assert 'min' in limits
        assert 'max' in limits

    def test_is_connected_false(self, board):
        assert board.is_connected() is False


class TestNullMotionBoardCoordinateTransforms:
    """Coordinate transforms work (use defaults)."""

    @pytest.fixture
    def board(self):
        return NullMotionBoard()

    def test_z_roundtrip(self, board):
        um = 5000.0
        steps = board.z_um2ustep(um)
        back = board.z_ustep2um(steps)
        assert abs(back - um) < 1.0  # within 1um

    def test_xy_roundtrip(self, board):
        um = 10000.0
        steps = board.xy_um2ustep(um)
        back = board.xy_ustep2um(steps)
        assert abs(back - um) < 1.0


class TestNullMotionBoardNoOps:
    """Verify all no-op methods don't raise."""

    @pytest.fixture
    def board(self):
        return NullMotionBoard()

    def test_connect(self, board):
        board.connect()

    def test_disconnect(self, board):
        board.disconnect()

    def test_set_precision_mode(self, board):
        board.set_precision_mode('Z', True)

    def test_set_acceleration(self, board):
        board.set_acceleration_limit('Z', 'acceleration', 50)

    def test_spi_read(self, board):
        result = board.spi_read('Z', 0x4B)
        assert result == '0'

    def test_exchange_command(self, board):
        result = board.exchange_command('STATUS')
        assert result is None

    def test_enter_raw_repl(self, board):
        assert board.enter_raw_repl() is False

    def test_repl_list_files(self, board):
        assert board.repl_list_files() == []
