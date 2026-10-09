# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A command the board did not answer is reported as no reply.

pyserial's readline() answers a read timeout with an empty line, and the
exchange used to hand that on as the reply ''. Neither board ever answers
a command with an empty line, so '' always meant silence, and every caller
that checked for None took silence for an answer: a move nobody
acknowledged was a move, a fan write was a success, a stop probe cached
STOP as supported.

Each test runs the production driver over the firmware-backed simulator
(the field firmware) and loses one reply on the wire.
"""

import sys

import pytest

from drivers.exceptions import HardwareError
from drivers.firmware_updater import BOARD_CONFIGS, BoardType, _run_post_update_test
from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.sim_wire.backend import LedBoardSpec, MotorBoardSpec, SimWireBackend
from modules.lumascope_api.diagnostics import NO_REPLY, DiagnosticsAPI
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS
from tools import firmware_tools

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )


@pytest.fixture
def motor():
    """(the production motor driver, the simulated board)."""
    backend = SimWireBackend(MotorBoardSpec('LS850T', frozenset('XYZT')))
    board = MotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=backend)
    try:
        yield board, backend.motor_board
    finally:
        board.disconnect()


@pytest.fixture
def led():
    """(the production LED driver, the simulated board)."""
    backend = SimWireBackend(None, led=LedBoardSpec('LS850T'))
    board = LEDBoard(backend=backend)
    try:
        yield board, backend.led_board
    finally:
        board.disconnect()


def test_a_lost_reply_is_none(motor):
    board, sim = motor
    sim.drop_next_reply()
    assert board.exchange_command('ACTUAL_RZ', timeout=0.5) is None


@pytest.mark.slow
def test_a_move_the_board_did_not_acknowledge_raises(motor):
    board, sim = motor
    sim.drop_next_reply()
    with pytest.raises(HardwareError, match='did not happen'):
        board.move('Z', 1000)


@pytest.mark.slow
def test_a_status_read_with_no_reply_is_a_hardware_error(motor):
    board, sim = motor
    sim.drop_next_reply()
    with pytest.raises(HardwareError, match='no response'):
        board.target_status('Z')


@pytest.mark.slow
def test_an_spi_write_with_no_reply_raises(motor):
    board, sim = motor
    sim.drop_next_reply()
    with pytest.raises(HardwareError, match='no response'):
        board.spi_write('Z', 0x4B, 100)


@pytest.mark.slow
def test_a_fan_write_with_no_reply_raises(motor):
    board, sim = motor
    sim.drop_next_reply()
    with pytest.raises(HardwareError, match='no response'):
        board.set_fan_duty(50)


@pytest.mark.slow
def test_a_stop_with_no_reply_raises_and_caches_nothing(motor):
    """The board may have taken a STOP it did not answer: that is neither
    a stop nor firmware without STOP, so it raises."""
    board, sim = motor
    sim.drop_next_reply()
    with pytest.raises(HardwareError):
        board.motor_stop()
    assert getattr(board, '_supports_stop_cached', None) is None
    # The next stop reaches the board, which answers that it has no STOP.
    assert board.motor_stop() is False
    assert board._supports_stop_cached is False


@pytest.mark.slow
def test_an_acceleration_read_with_no_reply_is_not_cached(motor):
    board, sim = motor
    sim.drop_next_reply()
    board.acceleration_limit('X', 'acceleration')
    assert 'X_acceleration' not in (board._accel_cache or {})


@pytest.mark.slow
def test_a_diagnostic_command_with_no_reply_is_no_reply(motor):
    board, sim = motor
    sim.drop_next_reply()
    assert DiagnosticsAPI._exchange_command_impl(board, 'motor', 'ACTUAL_RZ') == NO_REPLY


@pytest.mark.slow
def test_the_tools_move_reports_a_move_with_no_reply(motor):
    board, sim = motor
    sim.drop_next_reply()
    assert 'did not happen' in firmware_tools._move_to_step(board, 'Z', 1000, timeout=2)


def test_the_tools_homing_cycle_moves_away_and_homes_through_the_driver(monkeypatch, capsys, motor):
    board, _ = motor
    monkeypatch.setattr(firmware_tools, '_connect_motor_board', lambda: board)
    args = firmware_tools.argparse.Namespace(axes=['Z'], cycles=1, move_between=True)
    firmware_tools.cmd_homing_test(args)
    assert '[  1/1] OK' in capsys.readouterr().out


def test_the_tool_refuses_a_lone_x_or_y_home(monkeypatch, capsys, motor):
    # Neither firmware has XHOME or YHOME; the tool used to send one anyway.
    board, _ = motor
    monkeypatch.setattr(firmware_tools, '_connect_motor_board', lambda: board)
    args = firmware_tools.argparse.Namespace(axes=['X'], cycles=1, move_between=False)
    with pytest.raises(SystemExit):
        firmware_tools.cmd_homing_test(args)
    assert 'X and Y home only together' in capsys.readouterr().out


def test_leaving_engineering_mode_answers_whether_it_happened(led):
    board, _ = led
    assert board.enter_engineering_mode(timeout=5)
    assert board.exit_engineering_mode() is True


def test_the_simulated_board_answers_its_exit_as_the_real_one_does():
    assert SimulatedLEDBoard().exit_engineering_mode() is True


@pytest.mark.slow
def test_the_post_update_led_check_fails_on_no_reply(monkeypatch, led):
    board, sim = led
    exchange = board.exchange_command

    def hold_the_enable(command, *args, **kwargs):
        # Held past the check's 2 s read, so LEDS_ENT gets no line at all.
        if command == 'LEDS_ENT':
            sim.delay_next_reply(2.5)
        return exchange(command, *args, **kwargs)

    monkeypatch.setattr(board, 'exchange_command', hold_the_enable)
    passed, detail = _run_post_update_test(board, BOARD_CONFIGS[BoardType.LED])
    assert passed is False
    assert 'no response' in detail
