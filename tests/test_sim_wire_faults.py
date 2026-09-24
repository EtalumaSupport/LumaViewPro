# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Hardware faults and the register-write oracle of the firmware-backed
simulator.

A test switches a fault on in the simulated hardware and the real firmware
meets it, so what comes back through the production driver is the
firmware's own failure. What is pinned: each fault is what the hardware
does, a fault set before a command is in effect from that command's first
SPI transfer, faults outlive a soft reset as hardware does, and the oracle
reports every register write in order without a byte of it reaching the
driver.
"""

import sys
import time

import pytest
from serial.serialutil import SerialException

from drivers.motorboard import MotorBoard
from drivers.sim_wire import port as sim_port
from drivers.sim_wire.backend import DIALECTS, MotorBoardSpec, SimWireBackend
from drivers.sim_wire.mp import channel, tmc5072
from drivers.sim_wire.port import BoardImage, EmulatedPort, RegisterWrite

firmware_only = pytest.mark.skipif(
    not (sys.platform == 'darwin' or sys.platform.startswith('linux')),
    reason='the firmware-backed simulator runs on macOS and Linux only',
)


def _motor(**kwargs) -> tmc5072.Motor:
    args = {
        'present': True,
        'usteps_per_mm': 1000.0,
        'travel_mm': 100.0,
        'flag_active_low': False,
        'start_usteps': 5000.0,
    }
    args.update(kwargs)
    motor = tmc5072.Motor(**args)
    motor.write(tmc5072.VMAX, 100_000)
    motor.write(tmc5072.AMAX, 10_000)
    motor.write(tmc5072.SW_MODE, tmc5072.STOP_L_ENABLE)
    return motor


def _go_to(motor: tmc5072.Motor, target: int) -> None:
    motor.write(tmc5072.RAMPMODE, tmc5072.MODE_POSITION)
    motor.write(tmc5072.XTARGET, target)
    motor.advance(0, lambda a, b: a - b, instant=True)
    motor.advance(1, lambda a, b: a - b, instant=True)


class TestTheChipModelsFaults:
    def test_a_move_into_the_flag_stops_at_the_switch(self):
        motor = _motor()
        start = motor.actual()
        _go_to(motor, start - 10_000)
        assert motor.actual() == start - 5000
        assert motor.read(tmc5072.RAMP_STAT) & tmc5072.EVENT_STOP_L

    def test_a_switch_that_never_trips_lets_the_move_run_through_the_flag(self):
        motor = _motor()
        motor.set_fault(tmc5072.SWITCH_NEVER_TRIPS, True)
        start = motor.actual()
        _go_to(motor, start - 10_000)
        assert motor.actual() == start - 10_000
        assert not motor.read(tmc5072.RAMP_STAT) & (tmc5072.STATUS_STOP_L | tmc5072.EVENT_STOP_L)

    @pytest.mark.parametrize('fault', [tmc5072.STALL, tmc5072.ABSENT])
    def test_a_motor_the_stage_does_not_follow_reaches_its_target_while_the_stage_stays(
        self, fault
    ):
        # Open loop: the chip counts the steps it sends, whether or not the
        # stage moves, so XACTUAL arrives and the switch never sees the flag.
        motor = _motor()
        motor.set_fault(fault, True)
        start = motor.actual()
        _go_to(motor, start - 10_000)
        assert motor.actual() == start - 10_000
        assert motor.read(tmc5072.RAMP_STAT) & tmc5072.POSITION_REACHED
        assert motor.stop_status() == (0, 0)

    @pytest.mark.parametrize('fault', tmc5072.FAULTS)
    def test_a_faulted_motor_has_no_flag_edge_ahead(self, fault):
        # A stage that is not following, or a switch that never changes, can
        # meet no edge; offering one would walk a move to its goal an edge's
        # distance at a time, since the stalled stage never reaches it.
        motor = _motor(start_usteps=1.0)
        assert motor.next_edge(-1) == 0.0
        motor.set_fault(fault, True)
        assert motor.next_edge(-1) is None

    def test_once_a_stall_clears_the_stage_moves_from_where_it_stalled(self):
        motor = _motor()
        motor.set_fault(tmc5072.STALL, True)
        start = motor.actual()
        _go_to(motor, start + 3000)
        motor.set_fault(tmc5072.STALL, False)
        # The stage is still 5000 from the flag, not the 8000 XACTUAL implies.
        _go_to(motor, start - 20_000)
        assert motor.actual() == start + 3000 - 5000

    def test_an_absent_motor_reads_open_load_on_both_coils(self):
        motor = _motor()
        assert not motor.drv_status() & (tmc5072.DRV_OLA | tmc5072.DRV_OLB)
        motor.set_fault(tmc5072.ABSENT, True)
        assert motor.drv_status() & tmc5072.DRV_OLA
        assert motor.drv_status() & tmc5072.DRV_OLB

    def test_an_unknown_fault_is_refused(self):
        with pytest.raises(ValueError, match='unknown fault'):
            _motor().set_fault('melted', True)


class TestTheChannel:
    def test_a_fault_line_round_trips(self):
        for on in (True, False):
            assert channel.parse_fault_line(channel.fault_line(on, 'Z', tmc5072.STALL)) == (
                on,
                'Z',
                tmc5072.STALL,
            )

    def test_a_register_write_frame_round_trips(self):
        frame = channel.write_frame('ZT', 'Z', 0x0D, 0xFFB1E012).encode('latin-1')
        assert frame[0] == channel.ORACLE_START and frame[-1] == channel.ORACLE_END
        assert channel.parse_write(frame[1:-1]) == ('ZT', 'Z', 0x0D, 0xFFB1E012)
        frame = channel.write_frame('XY', None, 0x00, 1).encode('latin-1')
        assert channel.parse_write(frame[1:-1]) == ('XY', None, 0x00, 1)


def _unopened_port(oracle: bool) -> EmulatedPort:
    image = BoardImage(
        runtime='-', firmware_mpy='-', files={}, module_path=(), label='[test]', oracle=oracle
    )
    return EmulatedPort(image)


class TestTheDemux:
    def test_a_frame_split_across_reads_is_taken_out_whole(self):
        port = _unopened_port(oracle=True)
        frame = channel.write_frame('XY', 'X', 0x0D, 1234).encode('latin-1')
        stream = b'Z home ' + frame + b'successful\r\n'
        to_driver = b''.join(port._demux(stream[i : i + 5]) for i in range(0, len(stream), 5))
        assert to_driver == b'Z home successful\r\n'
        assert port._writes == [RegisterWrite('XY', 'X', 0x0D, 1234)]

    def test_a_frame_with_the_oracle_off_fails_the_port(self):
        port = _unopened_port(oracle=False)
        port._demux(channel.write_frame('XY', 'X', 0x0D, 1).encode('latin-1'))
        assert 'oracle off' in port._failure

    def test_untaken_writes_past_the_limit_fail_the_port(self, monkeypatch):
        monkeypatch.setattr(sim_port, 'WRITES_LIMIT', 3)
        port = _unopened_port(oracle=True)
        port._demux(channel.write_frame('XY', 'X', 0x0D, 1).encode('latin-1') * 4)
        assert 'nothing is taking them' in port._failure
        assert len(port._writes) == 3

    def test_a_fault_needs_a_known_axis_and_name(self):
        port = _unopened_port(oracle=False)
        with pytest.raises(ValueError, match='unknown axis'):
            port.inject('Q', tmc5072.STALL)
        with pytest.raises(ValueError, match='unknown fault'):
            port.inject('Z', 'melted')


@pytest.fixture
def scope(request):
    """(MotorBoard, EmulatedPort) for an LS850T on the requested dialect."""
    dialect, oracle = getattr(request, 'param', ('3.0', False))
    backend = SimWireBackend(
        MotorBoardSpec('LS850T', frozenset('XYZT'), dialect=dialect, oracle=oracle)
    )
    board = MotorBoard(backend=backend)
    try:
        yield board, backend.motor_port
    finally:
        board.disconnect()


@firmware_only
class TestTheFirmwareMeetsTheFault:
    @pytest.mark.parametrize('scope', [(d, False) for d in DIALECTS], indirect=True)
    def test_a_z_switch_that_never_trips_is_the_firmwares_home_timeout(self, scope):
        board, port = scope
        assert board.exchange_command('ZHOME') == 'Z home successful'
        port.inject('Z', tmc5072.SWITCH_NEVER_TRIPS)
        assert board.exchange_command('ZHOME') == 'ERROR: Z home timeout'
        port.clear('Z', tmc5072.SWITCH_NEVER_TRIPS)
        assert board.exchange_command('ZHOME') == 'Z home successful'

    @pytest.mark.parametrize('scope', [(d, False) for d in DIALECTS], indirect=True)
    def test_a_stalled_x_is_the_firmwares_xy_home_timeout(self, scope):
        board, port = scope
        port.inject('X', tmc5072.STALL)
        assert board.exchange_command('HOME') == 'ERROR: XY home timeout'

    def test_a_fault_set_before_a_command_is_in_effect_from_its_first_transfer(self, scope):
        # DRVSTAT_Z is one register read; a fault picked up any later than
        # the command's first SPI transfer would miss it entirely.
        board, port = scope
        for _ in range(50):
            port.inject('Z', tmc5072.ABSENT)
            assert 'OPEN_A OPEN_B' in board.exchange_command('DRVSTAT_Z')
            port.clear('Z', tmc5072.ABSENT)
            assert 'OPEN_A' not in board.exchange_command('DRVSTAT_Z')

    def test_an_absent_motor_is_not_detected_by_the_firmware(self, scope):
        board, port = scope
        port.inject('X', tmc5072.ABSENT)
        # X is the reply's first line.
        assert board.exchange_command('MOTORDETECT') == 'X: detected=False  configured=True'

    def test_a_fault_outlives_a_soft_reset(self, scope):
        board, port = scope
        port.inject('Z', tmc5072.ABSENT)
        port.write(b'\x03\x04')
        deadline = time.monotonic() + 5
        while not board.exchange_command('INFO', timeout=0.5):
            assert time.monotonic() < deadline, 'the board did not come back from the soft reset'
        assert 'OPEN_A OPEN_B' in board.exchange_command('DRVSTAT_Z')


@firmware_only
class TestTheOracle:
    @pytest.mark.parametrize('scope', [('3.0', True)], indirect=True)
    def test_every_z_home_makes_the_same_writes_before_its_reply(self, scope):
        board, port = scope
        port.take_writes()
        seen = []
        for _ in range(20):
            assert board.exchange_command('ZHOME') == 'Z home successful'
            seen.append(tuple(port.take_writes()))
        assert seen[0] and all(writes == seen[0] for writes in seen[1:])
        # The firmware's own approach target: 30 mm toward the flag.
        assert RegisterWrite('ZT', 'Z', tmc5072.XTARGET, 0xFFB1E012) in seen[0]

    def test_with_the_oracle_off_there_are_no_writes_to_take(self, scope):
        _board, port = scope
        with pytest.raises(SerialException, match='oracle is off'):
            port.take_writes()
