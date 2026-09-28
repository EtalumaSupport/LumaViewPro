# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated LED board at the wire: the shipped LED firmware behind its
own port, beside the motor board's.

What is pinned here is that a simulated scope has each of its boards as a
board of its own: discovered under the vendor and product IDs the real
board enumerates with, opened by its own device name, and holding its own
hardware, faults included; and that the production LEDBoard connects to
it, the firmware reaching its DAC through the chip select the board wires
to it.
"""

import pathlib
import sys

import pytest
import serial

from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from drivers.sim_wire.backend import (
    LED_DEVICE,
    LED_PID,
    LED_VID,
    MOTOR_DEVICE,
    MOTOR_PID,
    MOTOR_VID,
    LedBoardSpec,
    MotorBoardSpec,
    SimWireBackend,
)
from drivers.sim_wire.mp import tmc5072

sys.path.insert(0, str(pathlib.Path('drivers/sim_wire/mp').resolve()))
import dac80508

firmware_only = pytest.mark.skipif(
    not (sys.platform == 'darwin' or sys.platform.startswith('linux')),
    reason='the firmware-backed simulator runs on macOS and Linux only',
)

MOTOR = MotorBoardSpec('LS850T', frozenset('XYZT'))
LED = LedBoardSpec('LS850T')


def _open(backend, device):
    return backend.open(port=device, baudrate=115200, timeout=0.5, write_timeout=1)


class TestDiscovery:
    def test_a_scope_with_both_boards_lists_each_under_its_own_ids(self):
        ports = {p.device: (p.vid, p.pid) for p in SimWireBackend(MOTOR, led=LED).comports()}
        assert ports == {
            MOTOR_DEVICE: (MOTOR_VID, MOTOR_PID),
            LED_DEVICE: (LED_VID, LED_PID),
        }

    def test_the_led_board_is_found_under_the_real_boards_ids(self):
        # The IDs the production LEDBoard discovers U50 by.
        assert (LED_VID, LED_PID) == (0x0424, 0x704C)

    def test_a_scope_with_no_motor_board_lists_the_led_board_alone(self):
        backend = SimWireBackend(None, led=LED)
        assert backend.motor_board is None
        assert [p.device for p in backend.comports()] == [LED_DEVICE]

    def test_a_scope_with_no_led_board_lists_the_motor_board_alone(self):
        backend = SimWireBackend(MOTOR)
        assert backend.led_board is None
        assert [p.device for p in backend.comports()] == [MOTOR_DEVICE]

    def test_an_unplugged_led_board_is_not_listed_and_the_motor_board_still_is(self):
        backend = SimWireBackend(MOTOR, led=LED)
        backend.led_board.unplug()
        assert [p.device for p in backend.comports()] == [MOTOR_DEVICE]


class TestOpen:
    def test_a_device_the_scope_does_not_have_is_refused(self):
        backend = SimWireBackend(MOTOR)
        with pytest.raises(serial.SerialException, match='no simulated board'):
            _open(backend, LED_DEVICE)

    @firmware_only
    def test_each_device_opens_its_own_board(self):
        backend = SimWireBackend(MOTOR, led=LED)
        led_port = _open(backend, LED_DEVICE)
        motor_port = _open(backend, MOTOR_DEVICE)
        try:
            # A second open finds the board it names already held, and says
            # which board that is.
            with pytest.raises(serial.SerialException, match=r'\[sim led .*already open'):
                _open(backend, LED_DEVICE)
            with pytest.raises(serial.SerialException, match=r'\[sim motor .*already open'):
                _open(backend, MOTOR_DEVICE)
        finally:
            led_port.close()
            motor_port.close()


class TestFaults:
    def test_the_led_board_has_no_hardware_fault_to_inject(self):
        # The field LED firmware detects no DAC fault, so the simulated DAC
        # offers none: a fault on it would be one no firmware reaction shows.
        backend = SimWireBackend(None, led=LED)
        with pytest.raises(ValueError, match='unknown fault target'):
            backend.led_board.inject('X', tmc5072.STALL)

    def test_the_motor_board_beside_it_keeps_its_own_faults(self):
        backend = SimWireBackend(MOTOR, led=LED)
        backend.motor_board.inject('X', tmc5072.STALL)
        backend.motor_board.clear('X', tmc5072.STALL)


class TestSpec:
    def test_an_unknown_timing_is_refused(self):
        with pytest.raises(ValueError, match='timing mode'):
            LedBoardSpec('LS850T', timing='warp')

    def test_the_image_runs_the_shipped_led_firmware_on_its_runtime(self):
        image = LED.image()
        assert image.firmware_mpy.endswith('led-field.mpy')
        assert image.runtime == MOTOR.image().runtime


class TestTheDac:
    def test_a_frame_is_three_bytes_and_miso_answers_zeros(self):
        # The firmware only writes; nothing drives MISO back.
        assert dac80508.Board().datagram('DAC', bytes((0x05, 0x00, 0x0A))) == bytes(3)

    def test_a_transfer_that_is_not_a_dac_frame_is_refused(self):
        with pytest.raises(ValueError, match='3-byte'):
            dac80508.Board().datagram('DAC', bytes(5))


@firmware_only
class TestTheFirmwareBoots:
    def test_the_production_led_board_connects_to_the_shipped_firmware(self):
        board = LEDBoard(backend=SimWireBackend(None, led=LED))
        try:
            # The banner's date, and no version: the field firmware carries none.
            assert board.firmware_date == '2024-06-05'
            assert board.firmware_version is None
            assert not board.firmware_silent
            # The connect's safety LEDS_OFF was acknowledged.
            assert board.last_safety_off_error is None
        finally:
            board.disconnect()

    def test_the_motor_and_led_boards_each_reach_their_own_chip_at_gp1(self):
        # GP1 is the XY TMC5072's chip select on the motor board and the
        # DAC's on the LED board; each board's firmware must meet its own.
        backend = SimWireBackend(MOTOR, led=LED)
        motor = MotorBoard(backend=backend)
        led = LEDBoard(backend=backend)
        try:
            assert led.firmware_date == '2024-06-05'
            assert motor.home()
            motor.move_abs_pos('X', 20000.0, overshoot_enabled=False)
            assert motor.wait_for_position('X', timeout=2.0)
            assert motor.current_pos('X') == pytest.approx(20000.0, abs=0.1)
        finally:
            led.disconnect()
            motor.disconnect()
