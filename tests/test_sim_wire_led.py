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

import os
import pathlib
import subprocess
import sys
import time

import pytest
import serial

from drivers.exceptions import HardwareError
from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from drivers.sim_wire import backend as sim_backend
from drivers.sim_wire.backend import (
    LED_DEVICE,
    LED_ENABLE_PINS,
    LED_PID,
    LED_VID,
    MOTOR_DEVICE,
    MOTOR_PID,
    MOTOR_VID,
    LedBoardSpec,
    MotorBoardSpec,
    SimWireBackend,
)
from drivers.sim_wire.port import RegisterWrite
from drivers.sim_wire.mp import tmc5072
from tests.sim_wire_bench import FRESH, led_on_args

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


# The field firmware's DAC frames (K78): soft reset, CONFIG with DAC6 and
# DAC7 powered down, gain 1x, and channel n's code at address n + 8.
SOFT_RESET = bytes((0x05, 0x00, 0x0A))
CONFIG_6_7_DOWN = bytes((0x03, 0x04, 0xC0))
GAIN_1X = bytes((0x04, 0x00, 0x00))


def _code(channel, code):
    return bytes((channel + 8, code >> 8, code & 0xFF))


def _dac_booted():
    dac = dac80508.Board(LED_ENABLE_PINS)
    for frame in (SOFT_RESET, CONFIG_6_7_DOWN, GAIN_1X):
        dac.datagram('DAC', frame)
    return dac


def _pins(high=()):
    return lambda pin: int(pin in high)


class TestTheDac:
    def test_after_the_boot_frames_channels_0_to_5_are_powered_and_6_and_7_are_not(self):
        channels = _dac_booted().state(_pins())['DAC']
        assert [powered for _enabled, powered, _code in channels] == [True] * 6 + [False] * 2

    def test_a_channel_frame_sets_that_channels_code(self):
        dac = _dac_booted()
        dac.datagram('DAC', _code(3, 6983))
        assert [code for *_, code in dac.state(_pins())['DAC']] == [0, 0, 0, 6983, 0, 0, 0, 0]

    def test_a_soft_reset_clears_the_channel_codes(self):
        dac = _dac_booted()
        dac.datagram('DAC', _code(3, 6983))
        dac.datagram('DAC', SOFT_RESET)
        assert all(code == 0 for *_, code in dac.state(_pins())['DAC'])

    def test_a_channel_is_enabled_by_its_own_enable_pin(self):
        state = _dac_booted().state(_pins(high={LED_ENABLE_PINS[2]}))['DAC']
        assert [enabled for enabled, *_ in state] == [False, False, True] + [False] * 5

    def test_the_state_changes_with_the_enable_pins_it_watches(self):
        assert _dac_booted().state_pins == LED_ENABLE_PINS

    def test_a_frame_is_reported_as_the_register_write_it_makes(self):
        assert _dac_booted().written('DAC', _code(3, 6983)) == (None, 0x0B, 6983)

    def test_a_read_frame_is_refused(self):
        # The firmware never reads the DAC, so the model answers no read.
        with pytest.raises(ValueError, match='read'):
            _dac_booted().datagram('DAC', bytes((0x88, 0x00, 0x00)))

    def test_a_frame_is_three_bytes_and_miso_answers_zeros(self):
        # The firmware only writes; nothing drives MISO back.
        assert dac80508.Board(LED_ENABLE_PINS).datagram('DAC', bytes((0x05, 0x00, 0x0A))) == bytes(
            3
        )

    def test_a_transfer_that_is_not_a_dac_frame_is_refused(self):
        with pytest.raises(ValueError, match='3-byte'):
            dac80508.Board(LED_ENABLE_PINS).datagram('DAC', bytes(5))


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


@pytest.fixture
def lit():
    """(LEDBoard, the simulated LED board behind it), the oracle on."""
    backend = SimWireBackend(None, led=LedBoardSpec('LS850T', oracle=True))
    board = LEDBoard(backend=backend)
    try:
        yield board, backend.led_board
    finally:
        board.disconnect()


@firmware_only
def test_the_first_command_after_connect_gets_its_own_reply():
    # Connect turns the LEDs off. A reply to that left unread is taken by
    # the next command as its own: INFO would read 'LED 0 has been turned off'.
    board = LEDBoard(backend=SimWireBackend(None, led=LED))
    try:
        assert board.last_safety_off_error is None
        assert 'Version' in board.exchange_command('INFO', timeout=2)
    finally:
        board.disconnect()


@firmware_only
def test_a_multiline_exchange_ends_when_the_reply_ends():
    # The support report reads INFO up to its CALIBRATION line, and FACTORY
    # up to its Y/N prompt. Once the reply is over, waiting the call's
    # per-line window for each trailing line took 25 s and 10 s.
    board = LEDBoard(backend=SimWireBackend(None, led=LED))
    try:
        t0 = time.monotonic()
        info = board.exchange_multiline(
            'INFO',
            timeout=5,
            end_markers=['RESET CAUSE', 'POWER-ON', 'HARD', 'WDT', 'CALIBRATION'],
        )
        assert 'Calibration' in info
        assert time.monotonic() - t0 < 1.0
        t0 = time.monotonic()
        assert board.enter_engineering_mode(timeout=5)
        assert time.monotonic() - t0 < 1.5
        board.exit_engineering_mode()
    finally:
        board.disconnect()


@firmware_only
class TestWhatTheDacDrives:
    OFF = (True, True, 0)

    def test_after_connect_every_channel_is_enabled_powered_and_dark(self, lit):
        _, sim = lit
        assert sim.state('DAC') == (self.OFF,) * 6 + ((False, False, 0),) * 2

    def test_led_on_drives_the_channels_code_for_its_current(self, lit):
        board, sim = lit
        board.led_on(3, 100)
        # The firmware's mA_to_dac: int(60.936 * 100 + 890).
        assert sim.state('DAC')[3] == (True, True, 6983)
        board.led_off(3)
        assert sim.state('DAC')[3] == self.OFF

    def test_the_write_behind_led_on_is_in_the_register_writes(self, lit):
        board, sim = lit
        sim.take_writes()
        board.led_on(3, 100)
        assert sim.take_writes() == [RegisterWrite('DAC', None, 0x0B, 6983)]

    def test_disabling_the_leds_opens_every_enable_switch_and_leaves_the_codes(self, lit):
        board, sim = lit
        board.led_on(1, 50)
        board.leds_disable()
        state = sim.state('DAC')
        assert [enabled for enabled, *_ in state[:6]] == [False] * 6
        assert state[1][2] == 3936  # int(60.936 * 50 + 890)
        board.leds_enable()
        assert [enabled for enabled, *_ in sim.state('DAC')[:6]] == [True] * 6

    def test_one_channels_enable_command_opens_that_channels_switch_alone(self, lit):
        board, sim = lit
        board.exchange_command('LED1_ENF')
        assert [enabled for enabled, *_ in sim.state('DAC')[:6]] == [
            True,
            False,
            True,
            True,
            True,
            True,
        ]

    def test_every_led_call_the_bench_made_drives_what_it_asked(self, lit):
        board, sim = lit
        for record in FRESH:
            kind = record['kind']
            if kind == 'led_on':
                channel, ma = led_on_args(record)
                board.led_on(channel, ma)
                expected = [self.OFF] * 6
                # The firmware's mA_to_dac.
                expected[channel] = (True, True, int(60.936 * ma + 890))
            elif kind == 'led_off':
                board.led_off(record['channel'])
                expected = [self.OFF] * 6
            elif kind == 'leds_off':
                board.leds_off()
                expected = [self.OFF] * 6
            else:
                continue
            assert list(sim.state('DAC')[:6]) == expected, record['command']

    def test_the_state_needs_the_oracle(self):
        backend = SimWireBackend(None, led=LED)
        with pytest.raises(serial.SerialException, match='oracle is off'):
            backend.led_board.state('DAC')


def _read_for(port, seconds):
    """Every byte the board sends within `seconds`."""
    deadline = time.monotonic() + seconds
    out = b''
    while time.monotonic() < deadline:
        out += port.read(4096)
    return out


def _booted_led_port():
    backend = SimWireBackend(None, led=LED)
    port = backend.open(port=LED_DEVICE, baudrate=115200, timeout=0.1, write_timeout=1)
    deadline = time.monotonic() + 5
    while b'Calibration' not in _read_for(port, 0.2):
        if time.monotonic() > deadline:
            raise AssertionError('the LED board never printed its INFO banner at boot')
    return backend, port


@firmware_only
class TestTheConsole:
    """The field LED firmware's factory() asks with input(), which on the
    board's MicroPython 1.19 is readline.c: a line ends on a carriage return
    only; a newline is not a line end and is dropped; a printable byte is
    echoed as it is typed."""

    def test_a_newline_does_not_answer_the_factory_prompt_and_a_return_does(self):
        _backend, port = _booted_led_port()
        try:
            port.write(b'FACTORY\n')
            assert b'Y/N' in _read_for(port, 1.0)
            port.write(b'Y\n')
            # The Y is echoed; the newline ends nothing, so factory() still waits.
            assert _read_for(port, 1.0) == b'Y'
            port.write(b'\r')
            answer = _read_for(port, 1.0)
            assert answer.startswith(b'\r\n')
            assert b'Engineering Mode' in answer
        finally:
            port.close()

    def test_the_driver_enters_and_leaves_factory_without_a_recovery(self, monkeypatch):
        # SN 12075's tech support run sent Y ended by a newline: factory()
        # never saw a line end, every command after it went unanswered, and
        # exit_engineering_mode had to soft-reset the board.
        board = LEDBoard(backend=SimWireBackend(None, led=LED))
        recovery = []
        safe_write = board._safe_write

        def _recorded(data, context):
            recovery.append(data)
            return safe_write(data, context=context)

        monkeypatch.setattr(board, '_safe_write', _recorded)
        try:
            assert board.enter_engineering_mode(timeout=1.0)
            assert 'Version' in board.exchange_command('INFO', timeout=1)
            board.exit_engineering_mode()
            assert recovery == []
        finally:
            board.disconnect()

    def test_the_factory_exchange_answers_as_sn_12075_did(self, monkeypatch):
        # The support report's entry and exit on SN 12075 once Y was sent
        # with a return (its serial.log, 2026-09-28 16:09): each reply the
        # driver took, in order.
        bench = [
            (
                'FACTORY',
                '------------------------\n'
                'ETALUMA FACTORY MODE\n'
                '------------------------\n'
                'Changing these values is not recommended and may void product warranty.\n'
                'Do you accept these terms?  Y/N',
            ),
            (
                'Y',
                '-' * 90 + '\n'
                'Engineering Mode: Press q or Q to exit\n' + '-' * 90 + '\n'
                "board info: 'INFO' case insensitive\n"
                "LED enable:   'LED' channel '_ENT' where channel is 0 through 5, "
                'or S (plural/all)\n'
                "LED disable:  'LED' channel '_ENF' where channel is 0 through 5, "
                'or S (plural/all)\n'
                "LED on:       'LED' channel '_MA' where channel is 0 through 5, "
                'or S (plural/all)',
            ),
            ('Q', '------------------------'),
            ('INFO', 'Version:      EL-0925 Gen3 LED Controller'),
        ]
        board = LEDBoard(backend=SimWireBackend(None, led=LED))
        replies = []
        for name in ('exchange_command', 'exchange_multiline'):
            exchange = getattr(board, name)

            def _recorded(command, *args, _exchange=exchange, **kwargs):
                reply = _exchange(command, *args, **kwargs)
                replies.append((command, reply))
                return reply

            monkeypatch.setattr(board, name, _recorded)
        try:
            assert board.enter_engineering_mode(timeout=5)
            board.exit_engineering_mode()
        finally:
            board.disconnect()
        assert replies == bench

    def test_a_y_the_board_does_not_take_is_refused_at_entry(self, monkeypatch):
        # Answered with the main loop's newline, the prompt never returns and
        # no banner comes: entry fails there, not later at exit, and leaves
        # the board answering, because its caller will not call exit.
        board = LEDBoard(backend=SimWireBackend(None, led=LED))
        exchange = board.exchange_multiline

        def _newline_framed(command, **kwargs):
            kwargs['line_end'] = b'\n'
            return exchange(command, **kwargs)

        monkeypatch.setattr(board, 'exchange_multiline', _newline_framed)
        try:
            with pytest.raises(HardwareError, match='did not enter engineering mode'):
                board.enter_engineering_mode(timeout=1.0)
            assert 'Version' in board.exchange_command('INFO', timeout=2)
        finally:
            board.disconnect()


def _console(dialect, typed):
    """What `input('P> ')` prints and answers on a dialect's runtime with the
    board's console in place, given the bytes typed at it."""
    mp = pathlib.Path('drivers/sim_wire/mp').resolve()
    result = subprocess.run(
        [
            str(sim_backend.runtime_path(dialect)),
            '-c',
            "import console\ntry:\n    print(repr(input('P> ')))\n"
            'except BaseException as e:\n    print(type(e).__name__)',
        ],
        input=typed,
        capture_output=True,
        timeout=10,
        env=dict(os.environ, MICROPYPATH=str(mp)),
    )
    return result.stdout


@firmware_only
class TestTheConsoleByteByByte:
    """readline.c 1.19's answer to each byte the model takes, run on the
    board's runtime with nothing between the bytes and input()."""

    def test_a_return_ends_the_line_and_a_newline_does_not(self):
        assert _console('field', b'Y\nES\r') == b"P> YES\r\n'YES'\n"

    def test_ctrl_c_is_a_keyboard_interrupt(self):
        assert _console('field', b'Y\x03') == b'P> YKeyboardInterrupt\n'

    def test_ctrl_d_on_an_empty_line_is_end_of_file(self):
        assert _console('field', b'\x04') == b'P> EOFError\n'

    def test_ctrl_b_on_an_empty_line_is_an_empty_answer(self):
        assert _console('field', b'\x02') == b"P> ''\n"

    def test_ctrl_d_and_ctrl_e_in_a_line_leave_it_as_typed(self):
        assert _console('field', b'Y\x04\x05\r') == b"P> Y\r\n'Y'\n"

    def test_a_byte_that_would_reach_line_editing_is_refused(self):
        assert _console('field', b'Y\x1b') == b'P> YValueError\n'

    def test_the_host_closing_the_input_is_end_of_file(self):
        assert _console('field', b'Y') == b'P> YEOFError\n'

    def test_a_later_runtime_keeps_its_own_input(self):
        # 1.28's readline ends a line on a newline too; the model is 1.19's.
        assert b"'Y'" in _console('3.0', b'Y\n')
