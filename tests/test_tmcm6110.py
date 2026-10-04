# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The LS720's stage driver, run against the simulated TMCM-6110 at the wire.

Every test drives the production ``Tmcm6110Board`` through a
``SimulatedTmcm6110Backend``: the datagrams are encoded, sent, answered and
decoded as on the bench, and the simulated board keeps time, so a move is
in flight until it arrives.
"""

import logging
import time

import pytest

from drivers.exceptions import HardwareError, MotionInterlockError
from drivers.registry import DriverRegistry
from drivers.simulated_tmcm6110 import (
    SimulatedTmcm6110,
    SimulatedTmcm6110Backend,
    SimulatedTmcm6110Port,
)
from drivers.tmcm6110 import (
    AP_ACTUAL_VELOCITY,
    AP_TARGET_POSITION,
    FIRMWARE_VERSION,
    GAP,
    GIO,
    LID_INPUT,
    MST,
    MVP,
    SAP,
    STATUS_OK,
    USB_IDS,
    Tmcm6110Board,
    decode_command,
    decode_reply,
    encode_command,
    encode_reply,
    encode_version_reply,
)
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS

# How much faster than the wall clock the simulated board runs in these
# tests, so a millimetre's move takes milliseconds.
FAST = 50


def _fast_clock(factor=FAST):
    start = time.monotonic()
    return lambda: (time.monotonic() - start) * factor


@pytest.fixture
def sim():
    return SimulatedTmcm6110(clock=_fast_clock())


@pytest.fixture
def board(sim):
    driver = Tmcm6110Board(
        motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=SimulatedTmcm6110Backend(sim)
    )
    yield driver
    driver.disconnect()


def _wait_arrival(board, axis, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not board.target_status(axis):
        assert time.monotonic() < deadline, f'{axis} did not arrive'
        time.sleep(0.005)


def _sent(sim):
    commands = list(sim.commands)
    sim.commands.clear()
    return commands


class RecordingBackend(SimulatedTmcm6110Backend):
    """Records every port it opens, so a test can see which were closed."""

    def __init__(self, sim):
        super().__init__(sim)
        self.opened = []

    def open(self, **kwargs):
        port = super().open(**kwargs)
        self.opened.append(port)
        return port


# --- The datagram ---------------------------------------------------------


def test_a_command_is_laid_out_as_tmcl_sends_it():
    """[address, command, type, motor, value big-endian, checksum of the eight]."""
    assert encode_command(MVP, 0, 1, -10000) == bytes(
        [1, 4, 0, 1, 0xFF, 0xFF, 0xD8, 0xF0, (1 + 4 + 0 + 1 + 0xFF + 0xFF + 0xD8 + 0xF0) & 0xFF]
    )
    assert decode_command(encode_command(MVP, 0, 1, -10000)) == (1, MVP, 0, 1, -10000)


def test_a_reply_logged_from_classic_decodes():
    """A GAP reply LumaView Classic logged: reply address 2, module 1, status 100, value 1."""
    reply = decode_reply(bytes([0x2, 0x1, 0x64, 0x6, 0x0, 0x0, 0x0, 0x1, 0x6E]))
    assert reply == (2, 1, STATUS_OK, GAP, 1)
    assert encode_reply(STATUS_OK, GAP, 1) == bytes([0x2, 0x1, 0x64, 0x6, 0x0, 0x0, 0x0, 0x1, 0x6E])


def test_a_reply_whose_checksum_fails_is_refused():
    bad = bytearray(encode_reply(STATUS_OK, GAP, 1))
    bad[8] ^= 0xFF
    with pytest.raises(ValueError, match='checksum'):
        decode_reply(bytes(bad))


def test_the_firmware_reply_is_ascii_with_no_checksum(board):
    """The board answers the firmware query with its version across bytes
    1-8; the last byte is a character, not a checksum, and is read as one."""
    reply = encode_version_reply('6110V135')
    assert reply == b'\x02' + b'6110V135'
    assert board.firmware_version == '6110V135'


# --- Finding the board ----------------------------------------------------


@pytest.mark.parametrize('usb_id', USB_IDS)
def test_either_usb_identity_is_found(usb_id):
    sim = SimulatedTmcm6110(usb_id=usb_id)
    board = Tmcm6110Board(
        motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=SimulatedTmcm6110Backend(sim)
    )
    assert board.found
    assert board.is_connected()


def test_a_port_that_is_not_a_6110_is_closed_and_not_found():
    sim = SimulatedTmcm6110()
    sim.version = '3110V100'
    backend = RecordingBackend(sim)
    board = Tmcm6110Board(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=backend)
    assert not board.found
    assert not board.is_connected()
    assert len(backend.opened) == 1
    assert not backend.opened[0].is_open


def test_a_port_with_another_usb_identity_is_never_opened():
    sim = SimulatedTmcm6110(usb_id=(0x2E8A, 0x0005))
    backend = RecordingBackend(sim)
    board = Tmcm6110Board(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=backend)
    assert not board.found
    assert backend.opened == []


def test_on_a_host_with_no_6110_the_fallback_is_not_detected():
    """Tried by the registry on a host without one, the constructor raises
    nothing and the registry falls back naming nothing found."""
    sim = SimulatedTmcm6110()
    sim.plugged = False
    registry = DriverRegistry('motor')
    registry.register('tmcm6110', priority=80)(Tmcm6110Board)

    @registry.register('null', priority=0)
    class Null:
        def __init__(self, **kwargs):
            self.found = False

    instance, fallback = registry.create_with_fallback(
        'auto', motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=SimulatedTmcm6110Backend(sim)
    )
    assert isinstance(instance, Null)
    assert fallback.cause == 'not_detected'


def test_the_config_is_built_only_once_a_6110_is_found():
    """A host without a 6110 never reads the section, so a broken one
    cannot fail bring-up there; on an LS720 it is refused and the port closed."""
    sim = SimulatedTmcm6110()
    sim.plugged = False
    assert not Tmcm6110Board(motorconfig_defaults={}, backend=SimulatedTmcm6110Backend(sim)).found

    sim = SimulatedTmcm6110()
    backend = RecordingBackend(sim)
    with pytest.raises(ValueError, match='TMCM-6110'):
        Tmcm6110Board(motorconfig_defaults={}, backend=backend)
    assert not backend.opened[0].is_open


def test_unmeasured_travel_limits_are_a_warning_at_bring_up(caplog, sim):
    with caplog.at_level(logging.WARNING, logger='LVP.drivers.tmcm6110'):
        Tmcm6110Board(
            motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=SimulatedTmcm6110Backend(sim)
        )
    assert any('not measured' in r.getMessage() for r in caplog.records)


# --- The surface every motor driver provides ------------------------------


def test_the_members_the_api_reads(board):
    assert board.found is True
    assert board.firmware_responding is True
    assert board.firmware_date is None
    assert board.overshoot is False
    assert board.detect_present_axes() == ['X', 'Y', 'Z']
    assert board.detect_homed_axes() == []
    assert board.get_microscope_model() == 'LS720'
    assert not hasattr(board, 'exchange_multiline')
    assert board.exchange_command('INFO') is None
    assert board.motorconfig.ramp_params('X')
    assert board.get_axis_limits('Y') == {'min': 0.0, 'max': 80_000.0}


# --- Moves, arrival, the lid ----------------------------------------------


def test_target_status_is_false_straight_after_mvp(board):
    board.move_abs_pos('X', 2000)
    assert board.target_status('X') is False
    _wait_arrival(board, 'X')
    assert board.current_pos('X') == pytest.approx(2000)
    assert board.target_pos('X') == pytest.approx(2000)


def test_positions_are_positive_away_from_the_reference_on_the_board_negative(board, sim):
    """The API's micrometres are 0 at the reference and positive away; the
    board counts the other way (direction -1 on every axis)."""
    board.move_abs_pos('Y', 1000)
    _wait_arrival(board, 'Y')
    assert sim.position('Y') == -6400


def test_a_relative_move_is_from_the_board_target(board):
    board.move_abs_pos('Z', 100)
    board.move_rel_pos('Z', 50)
    _wait_arrival(board, 'Z')
    assert board.current_pos('Z') == pytest.approx(150, abs=0.1)


@pytest.mark.parametrize('axis', ['X', 'Y'])
def test_the_lid_is_read_before_every_command_that_starts_x_or_y(board, sim, axis):
    _sent(sim)
    board.move_abs_pos(axis, 100)
    board.move_rel_pos(axis, 100)
    board.move(axis, 3200)
    sent = _sent(sim)
    starts = [i for i, c in enumerate(sent) if c.command == MVP]
    assert len(starts) == 3
    for i in starts:
        assert (sent[i - 1].command, sent[i - 1].type, sent[i - 1].motor) == (GIO, *LID_INPUT)


def test_an_open_lid_refuses_x_and_stops_all_three(board, sim):
    board.move_abs_pos('Z', 5000)
    sim.lid_open = True
    _sent(sim)
    with pytest.raises(MotionInterlockError) as refused:
        board.move_abs_pos('X', 1000)
    assert refused.value.reason == 'lid_open'
    assert refused.value.moved is False
    assert refused.value.stopped is True
    sent = _sent(sim)
    assert not any(c.command == MVP for c in sent)
    assert sorted(c.motor for c in sent if c.command == MST) == [0, 1, 2]
    # The stopped Z stands where it stopped, its target there with it.
    assert board.target_status('Z')
    assert board.current_pos('X') == 0


def test_z_moves_with_the_lid_open(board, sim):
    sim.lid_open = True
    board.move_abs_pos('Z', 200)
    _wait_arrival(board, 'Z')
    assert board.current_pos('Z') == pytest.approx(200, abs=0.1)


def test_interlocks_reads_the_lid_and_the_power(board, sim):
    assert board.interlocks() == frozenset()
    sim.lid_open = True
    assert board.interlocks() == {'lid_open'}
    sim.powered = False
    assert board.interlocks() == {'lid_open', 'stage_unpowered'}


# --- The one stop ---------------------------------------------------------


def test_a_stop_leaves_each_target_at_its_actual_written_with_sap_0(board, sim):
    board.move_abs_pos('X', 60_000)
    board.move_abs_pos('Y', 40_000)
    time.sleep(0.02)
    _sent(sim)
    assert board.motor_stop() is True
    sent = _sent(sim)
    assert sorted(c.motor for c in sent if (c.command, c.type) == (SAP, AP_TARGET_POSITION)) == [
        0,
        1,
        2,
    ]
    # Each target written only once its axis read velocity 0.
    for motor in (0, 1, 2):
        write = next(
            i
            for i, c in enumerate(sent)
            if (c.command, c.type, c.motor) == (SAP, AP_TARGET_POSITION, motor)
        )
        polls = [
            i
            for i, c in enumerate(sent[:write])
            if (c.command, c.type, c.motor) == (GAP, AP_ACTUAL_VELOCITY, motor)
        ]
        assert polls
    assert not any(c.command == MVP for c in sent)
    for axis in 'XYZ':
        assert board.target_status(axis)
    assert 0 < board.current_pos('X') < 60_000


def test_a_stop_works_with_the_lid_open(board, sim):
    board.move_abs_pos('X', 60_000)
    sim.lid_open = True
    assert board.motor_stop() is True
    assert board.target_status('X')


def test_an_axis_that_keeps_moving_past_the_bound_makes_the_stop_raise():
    sim = SimulatedTmcm6110()
    board = Tmcm6110Board(
        motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=SimulatedTmcm6110Backend(sim)
    )
    board.move_abs_pos('X', 110_000)
    time.sleep(0.05)
    sim.ignore_stop('X')
    _sent(sim)
    with pytest.raises(HardwareError, match='X still moving'):
        board.motor_stop()
    # Every axis that did stop still has its target written.
    written = [c.motor for c in _sent(sim) if (c.command, c.type) == (SAP, AP_TARGET_POSITION)]
    assert sorted(written) == [1, 2]


def test_a_relative_move_with_an_unreadable_target_does_not_move(board, sim):
    sim.silent = True
    with pytest.raises(HardwareError, match='cannot read the current target'):
        board.move_rel_pos('Z', 50)
    sim.silent = False
    assert not any(c.command == MVP for c in _sent(sim))


def test_the_limit_switches_read_left_then_right(board, sim):
    assert board.limit_switch_status('Z') == (0, 0)
    sim.axes['Z'].p = float(sim.axes['Z'].layout.right_switch_at)
    assert board.limit_switch_status('Z') == (0, 1)
    sim.silent = True
    assert board.limit_switch_status('Z') == (-1, -1)


# --- Acceleration ---------------------------------------------------------


def test_an_acceleration_change_writes_the_speed_first_then_the_scaled_acceleration(board, sim):
    _sent(sim)
    board.set_acceleration_limits(40)
    sent = [(c.command, c.type, c.motor, c.value) for c in _sent(sim)]
    assert sent == [
        (SAP, 4, 0, 1000),
        (SAP, 5, 0, 200),
        (SAP, 4, 1, 1000),
        (SAP, 5, 1, 200),
    ]
    assert board.acceleration_limits() == {
        'X': {'acceleration': 500, 'deceleration': 500},
        'Y': {'acceleration': 500, 'deceleration': 500},
    }


def test_z_has_no_acceleration_limit(board):
    with pytest.raises(NotImplementedError):
        board.set_acceleration_limit('Z', 'acceleration', 50)


# --- A board that goes away -----------------------------------------------


class _Garbling(SimulatedTmcm6110):
    """Spoils the reply to the first GAP after ``spoil`` is set, one way."""

    spoil = None

    def answer(self, datagram):
        reply = super().answer(datagram)
        if self.spoil and datagram[1] == GAP:
            how, self.spoil = self.spoil, None
            if how == 'checksum':
                reply = reply[:8] + bytes([reply[8] ^ 0xFF])
            elif how == 'command':
                reply = encode_reply(STATUS_OK, MVP, 0)
        return reply


@pytest.mark.parametrize(
    ('how', 'words'), [('checksum', 'checksum mismatch'), ('command', 'answers command 4')]
)
def test_a_reply_that_does_not_line_up_raises_and_closes_the_port(how, words):
    sim = _Garbling(clock=_fast_clock())
    board = Tmcm6110Board(
        motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=SimulatedTmcm6110Backend(sim)
    )
    sim.spoil = how
    with pytest.raises(HardwareError, match=words):
        board.target_status('X')
    assert not board.is_connected()


def test_a_refused_command_raises_naming_it_and_its_status(sim):
    class Refusing(SimulatedTmcm6110):
        def _execute(self, cmd):
            if cmd.command == MVP:
                return 6, 0
            return super()._execute(cmd)

    refusing = Refusing(clock=_fast_clock())
    board = Tmcm6110Board(
        motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=SimulatedTmcm6110Backend(refusing)
    )
    with pytest.raises(HardwareError, match=r'MVP 0, motor 2.*status 6 \(command not available\)'):
        board.move_abs_pos('Z', 100)
    assert board.is_connected()


def test_a_silent_board_raises_naming_the_command_and_closes_the_port(board, sim):
    sim.silent = True
    with pytest.raises(HardwareError, match=r'GAP 3, motor 0.*no reply'):
        board.target_status('X')
    assert not board.is_connected()
    assert board.current_pos('X') is None

    sim.silent = False
    assert board.current_pos('X') == 0
    assert board.is_connected()


def test_an_unplugged_board_is_disconnected(board, sim):
    sim.plugged = False
    with pytest.raises(HardwareError, match='port failed'):
        board.target_status('X')
    assert not board.is_connected()
    with pytest.raises(HardwareError, match='not connected'):
        board.target_status('X')


# --- The simulated board itself -------------------------------------------


def test_the_simulated_board_answers_the_firmware_query_as_the_bench_unit(sim):
    port = SimulatedTmcm6110Port(sim, port='sim:tmcm6110', timeout=0)
    port.write(encode_command(FIRMWARE_VERSION, 0, 0))
    assert port.read(9) == b'\x026110V135'


def test_a_relative_mvp_answers_the_absolute_target():
    """As the 2016-2018 wire logs show: MVP REL's reply carries the
    resulting absolute target; MVP ABS echoes. The board's clock stands
    still, so the axis has not moved between the two."""
    sim = SimulatedTmcm6110(clock=lambda: 0.0)
    port = SimulatedTmcm6110Port(sim, port='sim:tmcm6110', timeout=0)
    port.write(encode_command(SAP, 1, 0, 2000))
    port.read(9)
    port.write(encode_command(MVP, 1, 0, -500))
    assert decode_reply(port.read(9)).value == 1500
    port.write(encode_command(MVP, 0, 0, -700))
    assert decode_reply(port.read(9)).value == -700


def test_a_right_switch_search_stops_on_the_switch_and_zeroes_there(sim):
    port = SimulatedTmcm6110Port(sim, port='sim:tmcm6110', timeout=0)

    def send(command, type_, motor, value=0):
        port.write(encode_command(command, type_, motor, value))
        return decode_reply(port.read(9)).value

    send(SAP, 193, 2, 65)
    send(SAP, 194, 2, 2047)
    send(13, 0, 2)
    deadline = time.monotonic() + 5
    while send(13, 2, 2):
        assert time.monotonic() < deadline
        time.sleep(0.005)
    assert send(GAP, 1, 2) == 0
    assert send(GAP, 10, 2) == 1
    assert send(GAP, 3, 2) == 0


def test_an_axis_driven_into_its_switch_stops_there(sim):
    port = SimulatedTmcm6110Port(sim, port='sim:tmcm6110', timeout=0)

    def send(command, type_, motor, value=0):
        port.write(encode_command(command, type_, motor, value))
        return decode_reply(port.read(9)).value

    send(MVP, 0, 2, 10_000_000)
    deadline = time.monotonic() + 5
    while not send(GAP, 10, 2):
        assert time.monotonic() < deadline
        time.sleep(0.005)
    stopped_at = send(GAP, 1, 2)
    time.sleep(0.01)
    assert send(GAP, 3, 2) == 0
    assert send(GAP, 1, 2) == stopped_at
    assert stopped_at < 10_000_000


# --- The simulated LS720 --------------------------------------------------


@pytest.mark.parametrize('tier', ['fast', 'firmware'])
def test_a_simulated_row_naming_the_6110_gets_the_driver_on_the_simulated_board(tier):
    from modules.lumascope_api._lumascope import Lumascope

    board = Lumascope._build_simulated_motor_board(
        'LS720', frozenset('XYZ'), 'TMCM-6110', tier, SHIPPED_MOTOR_DEFAULTS
    )
    assert isinstance(board, Tmcm6110Board)
    assert board.found
    board.move_abs_pos('Z', 10)
    assert board.target_pos('Z') == pytest.approx(10, abs=0.1)
