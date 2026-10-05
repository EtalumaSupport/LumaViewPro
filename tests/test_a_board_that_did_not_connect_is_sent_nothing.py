# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A serial board whose connect did not succeed is sent nothing at construction.

A launch with no motor or LED board on USB is a supported start, and the
registry already records the board `not_detected` and bring-up reports it
once. The drivers used to connect to a board their own port search had not
found, then send it their first command (the LED's safety LEDS_OFF, the
motor's CONFIG read), and log each failure: four ERRORs and a WARNING on
every launch without hardware. A board on USB whose port another program
holds is a real failure and logs its connect failure, once.

Every board here is the real firmware behind the simulated USB.
"""

import logging

import pytest
from serial.serialutil import SerialException

import drivers.ledboard as ledboard_mod
import drivers.motorboard as motorboard_mod
import drivers.null_ledboard  # registers the LED fallback
import drivers.null_motorboard  # noqa: F401 -- registers the motor fallback
import drivers.serialboard as serialboard_mod
from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from drivers.registry import led_registry, motor_registry
from drivers.sim_wire.backend import LedBoardSpec, MotorBoardSpec, SimWireBackend
from tests.log_capture import capture_module_log
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS


@pytest.fixture
def driver_log(monkeypatch):
    """Every record the two board drivers and their base class log."""
    return [
        capture_module_log(monkeypatch, module)
        for module in (serialboard_mod, ledboard_mod, motorboard_mod)
    ]


def _loud(driver_log):
    return [
        f'{r.levelname} {r.getMessage()}'
        for records in driver_log
        for r in records
        if r.levelno >= logging.WARNING
    ]


def _present():
    return SimWireBackend(MotorBoardSpec('LS850T', frozenset('XYZT')), led=LedBoardSpec('LS850T'))


def _held(backend):
    def refuse(**_kwargs):
        raise SerialException('port held by another program')

    backend.open = refuse
    return backend


def _build(kind, backend):
    if kind == 'motor':
        return MotorBoard(backend=backend, motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
    return LEDBoard(backend=backend)


@pytest.mark.parametrize('kind', ['motor', 'led'])
def test_an_absent_board_logs_nothing_and_is_not_found(kind, driver_log):
    board = _build(kind, SimWireBackend(None, None))
    assert board.found is False
    assert _loud(driver_log) == []


@pytest.mark.parametrize('kind', ['motor', 'led'])
def test_a_held_port_logs_its_connect_failure_once(kind, driver_log):
    board = _build(kind, _held(_present()))
    assert board.found is True
    assert not board.is_connected()
    loud = _loud(driver_log)
    assert len(loud) == 1, loud
    assert loud[0].startswith('ERROR') and 'connect() failed' in loud[0], loud


def test_a_present_led_board_connects_and_confirms_its_safety_off(driver_log):
    board = LEDBoard(backend=_present())
    try:
        assert board.is_connected()
        assert board.last_safety_off_error is None
        assert _loud(driver_log) == []
    finally:
        board.disconnect()


def test_a_present_motor_board_connects_and_reads_its_config(driver_log):
    board = MotorBoard(backend=_present(), motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
    try:
        assert board.is_connected()
        assert board.motorconfig.board_config_read_ok
        assert _loud(driver_log) == []
    finally:
        board.disconnect()


def test_the_registry_falls_back_on_absent_boards_with_no_driver_line(driver_log):
    backend = SimWireBackend(None, None)
    _led, led_fallback = led_registry.create_with_fallback('auto', backend=backend)
    _motor, motor_fallback = motor_registry.create_with_fallback(
        'auto', backend=backend, motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS
    )
    assert led_fallback is not None and led_fallback.cause == 'not_detected'
    assert motor_fallback is not None and motor_fallback.cause == 'not_detected'
    assert _loud(driver_log) == []
