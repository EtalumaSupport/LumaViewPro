# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated TMC5072 pair, driven the way the firmware drives it.

The model runs inside MicroPython behind the SPI stub; these tests import
it under CPython with a fake clock and send it the firmware's own datagram
sequences, so what is pinned is the chip's answers: pipelined reads, the
switch logic under every SW_MODE the firmware writes, hard stops, the
position and velocity bits, the open-load bits, and the ramp's duration
against the Stage 0 bench record.
"""

import json
import pathlib
import struct
import sys

import pytest

sys.path.insert(0, str(pathlib.Path('drivers/sim_wire/mp').resolve()))
import tmc5072 as chip

BASE_CONFIG = json.loads(
    pathlib.Path('drivers/sim_wire/firmware/motorconfig-base.json').read_text()
)
FCLK = 16_000_000
START = {'X': 600_000, 'Y': 400_000, 'Z': 600_000, 'T': 150_000}

# The firmware's register addresses and literals (main.py).
XY, ZT = 'XY', 'ZT'
M1, M2 = 0x20, 0x40
X_5MM_RIGHT = 0x000189B1
X_130MM_LEFT = 0xFFD80404
X_5MM_LEFT = 0xFFFB1E00


class Clock:
    def __init__(self):
        self.us = 0

    def ticks_us(self):
        return self.us

    @staticmethod
    def ticks_diff(a, b):
        return a - b


def board(timing='instant', present=None, start=START):
    config = json.loads(json.dumps(BASE_CONFIG))
    if present is not None:
        config['Axis Present'] = {a: int(a in present) for a in 'XYZT'}
    clock = Clock()
    sim = {'timing': timing, 'fclk_hz': FCLK, 'start_usteps': start}
    return chip.Board(config, sim, clock.ticks_us, clock.ticks_diff), clock


def write(b, cs, addr, value):
    b.datagram(cs, struct.pack('>BI', 0x80 | addr, value & 0xFFFFFFFF))


def read(b, cs, addr, signed=False):
    b.datagram(cs, struct.pack('>BI', addr, 0))
    reply = b.datagram(cs, bytes(5))
    value = int.from_bytes(reply[1:5], 'big')
    return value - (1 << 32) if signed and value >= (1 << 31) else value


def configure(b, cs, base, amax, vmax, dmax, sw_mode):
    for reg, value in (
        (chip.AMAX, amax),
        (chip.VMAX, vmax),
        (chip.DMAX, dmax),
        (chip.SW_MODE, sw_mode),
        (chip.RAMPMODE, chip.MODE_POSITION),
    ):
        write(b, cs, base + reg, value)


def configure_x(b):
    configure(b, XY, M1, 50000, 800000, 50000, 0x0F)


def settle(clock, ms=1):
    clock.us += ms * 1000


class TestReadsArePipelined:
    def test_a_read_answers_the_previous_datagrams_register(self):
        b, _ = board()
        write(b, XY, M1 + chip.VMAX, 12345)
        b.datagram(XY, struct.pack('>BI', M1 + chip.VMAX, 0))
        first = b.datagram(XY, struct.pack('>BI', M1 + chip.AMAX, 0))
        second = b.datagram(XY, bytes(5))
        assert int.from_bytes(first[1:5], 'big') == 12345
        assert int.from_bytes(second[1:5], 'big') == 0  # AMAX, unwritten

    def test_each_chip_has_its_own_shift_register(self):
        b, _ = board()
        write(b, XY, M1 + chip.VMAX, 111)
        write(b, ZT, M1 + chip.VMAX, 222)
        b.datagram(XY, struct.pack('>BI', M1 + chip.VMAX, 0))
        b.datagram(ZT, struct.pack('>BI', M1 + chip.VMAX, 0))
        assert int.from_bytes(b.datagram(XY, bytes(5))[1:5], 'big') == 111
        assert int.from_bytes(b.datagram(ZT, bytes(5))[1:5], 'big') == 222


class TestTheHomingSequence:
    """The firmware's xyzhome, datagram for datagram, on X."""

    def test_the_fast_approach_stops_on_the_flag_short_of_its_target(self):
        b, clock = board()
        configure_x(b)
        write(b, XY, M1 + chip.XTARGET, X_130MM_LEFT)
        settle(clock)
        status = read(b, XY, M1 + chip.RAMP_STAT)
        assert status & (chip.STATUS_STOP_L | chip.VZERO) == chip.STATUS_STOP_L | chip.VZERO
        assert not status & chip.POSITION_REACHED
        assert b.motors['X'].phys == 0  # the flag edge, which is on the flag

    def test_writing_xactual_in_hold_mode_moves_the_frame_not_the_stage(self):
        b, clock = board()
        configure_x(b)
        write(b, XY, M1 + chip.XTARGET, X_130MM_LEFT)
        settle(clock)
        write(b, XY, M1 + chip.RAMPMODE, chip.MODE_HOLD)
        write(b, XY, M1 + chip.XACTUAL, 0x8000)
        write(b, XY, M1 + chip.RAMPMODE, chip.MODE_POSITION)
        settle(clock)
        assert read(b, XY, M1 + chip.XACTUAL, signed=True) == 0x8000
        assert b.motors['X'].phys == 0

    def test_flipped_polarity_stops_where_the_flag_clears_and_normal_stops_where_it_sets(self):
        b, clock = board()
        configure_x(b)
        write(b, XY, M1 + chip.XTARGET, X_130MM_LEFT)
        settle(clock)
        write(b, XY, M1 + chip.RAMPMODE, chip.MODE_HOLD)
        write(b, XY, M1 + chip.XACTUAL, 0x8000)
        write(b, XY, M1 + chip.RAMPMODE, chip.MODE_POSITION)
        # Flip: stop_l | stop_r | swap_lr, slow, right 5 mm.
        write(b, XY, M1 + chip.SW_MODE, 0x13)
        write(b, XY, M1 + chip.VMAX, 4000)
        write(b, XY, M1 + chip.XTARGET, X_5MM_RIGHT)
        settle(clock)
        status = read(b, XY, M1 + chip.RAMP_STAT)
        assert status & (chip.STATUS_STOP_L | chip.VZERO) == chip.STATUS_STOP_L | chip.VZERO
        clear = read(b, XY, M1 + chip.XACTUAL, signed=True)
        # Normal polarity, left 5 mm: back onto the flag.
        write(b, XY, M1 + chip.SW_MODE, 0x0F)
        write(b, XY, M1 + chip.XTARGET, X_5MM_LEFT)
        settle(clock)
        status = read(b, XY, M1 + chip.RAMP_STAT)
        assert status & (chip.STATUS_STOP_L | chip.VZERO) == chip.STATUS_STOP_L | chip.VZERO
        set_ = read(b, XY, M1 + chip.XACTUAL, signed=True)
        assert clear > set_
        assert clear - set_ <= 4  # the model's hysteresis is a microstep each way

    def test_the_move_to_zero_reaches_its_target(self):
        b, clock = board()
        configure_x(b)
        write(b, XY, M1 + chip.XTARGET, X_130MM_LEFT)
        settle(clock)
        write(b, XY, M1 + chip.RAMPMODE, chip.MODE_HOLD)
        write(b, XY, M1 + chip.XACTUAL, 0)
        write(b, XY, M1 + chip.RAMPMODE, chip.MODE_POSITION)
        write(b, XY, M1 + chip.XTARGET, 10078)
        settle(clock)
        status = read(b, XY, M1 + chip.RAMP_STAT)
        assert status & (chip.POSITION_REACHED | chip.VZERO) == chip.POSITION_REACHED | chip.VZERO
        assert read(b, XY, M1 + chip.XACTUAL, signed=True) == 10078


class TestTheSwitchLogicOfEachAxis:
    def test_z_flag_is_active_high_and_its_right_input_rests_low(self):
        b, clock = board(start={'X': 1, 'Y': 1, 'Z': 100_000, 'T': 150_000})
        configure(b, ZT, M2, 25000, 400000, 25000, 0x03)
        assert read(b, ZT, M2 + chip.RAMP_STAT) & 0x3 == 0
        write(b, ZT, M2 + chip.XTARGET, (-30 * 170666) & 0xFFFFFFFF)
        settle(clock)
        status = read(b, ZT, M2 + chip.RAMP_STAT)
        assert status & (chip.STATUS_STOP_L | chip.VZERO) == chip.STATUS_STOP_L | chip.VZERO
        # Flipped (0x1F): the left bit reads set from the right input, the
        # right bit reports the flag clearing.
        write(b, ZT, M2 + chip.SW_MODE, 0x1F)
        assert read(b, ZT, M2 + chip.RAMP_STAT) & chip.STATUS_STOP_L

    def test_the_turret_flag_is_a_slot_found_within_one_turn(self):
        b, clock = board()
        configure(b, ZT, M1, 5000, 0x1F400, 5000, 0x01)
        write(b, ZT, M1 + chip.XTARGET, (-340000) & 0xFFFFFFFF)
        settle(clock)
        status = read(b, ZT, M1 + chip.RAMP_STAT)
        assert status & (chip.STATUS_STOP_L | chip.VZERO) == chip.STATUS_STOP_L | chip.VZERO
        assert b.motors['T'].phys == chip.TURRET_FLAG_WIDTH - 1
        # With the switches disabled the turret turns freely through the flag.
        write(b, ZT, M1 + chip.SW_MODE, 0)
        write(b, ZT, M1 + chip.XTARGET, (-200000) & 0xFFFFFFFF)
        settle(clock)
        assert read(b, ZT, M1 + chip.RAMP_STAT) & chip.POSITION_REACHED

    def test_a_disabled_switch_does_not_stop_the_motor(self):
        b, clock = board()
        configure(b, XY, M1, 50000, 800000, 50000, 0x00)
        write(b, XY, M1 + chip.XTARGET, X_130MM_LEFT)
        settle(clock)
        assert read(b, XY, M1 + chip.RAMP_STAT) & chip.POSITION_REACHED


class TestDrvStatus:
    def test_an_absent_motor_reads_open_load_on_both_coils(self):
        b, _ = board(present='Z')
        assert read(b, XY, chip.DRV_STATUS[0]) & (chip.DRV_OLA | chip.DRV_OLB)
        assert not read(b, ZT, chip.DRV_STATUS[1]) & (chip.DRV_OLA | chip.DRV_OLB)

    def test_a_motor_at_rest_reads_standstill_and_its_hold_current(self):
        b, _ = board()
        write(b, XY, M1 + chip.IHOLD_IRUN, 0x00071103)
        status = read(b, XY, chip.DRV_STATUS[0])
        assert status & chip.DRV_STST
        assert (status >> 16) & 0x1F == 3


class TestTheRealisticRamp:
    def _time_move(self, distance_usteps, amax, vmax, dmax, usteps_per_mm=20157):
        b, clock = board(timing='realistic')
        configure(b, XY, M1, amax, vmax, dmax, 0x0F)
        settle(clock)
        write(b, XY, M1 + chip.XTARGET, distance_usteps)  # XACTUAL starts at 0
        ms = 0
        while not read(b, XY, M1 + chip.RAMP_STAT) & chip.POSITION_REACHED:
            settle(clock)
            ms += 1
            assert ms < 20000, 'the move never finished'
        return ms

    def test_a_96_mm_x_move_takes_what_the_bench_measured(self):
        # Stage 0, LS850T field firmware, X 96 mm by API wait: 2947 ms median,
        # of which the API's poll is the 220 ms plateau its 1 um moves show.
        # The bench unit runs the field XY INI: AMAX 30000, VMAX 800000.
        ms = self._time_move(96 * 20157, 30000, 800000, 30000)
        assert abs(ms - (2947 - 220)) <= 40, ms

    def test_a_short_move_is_a_triangle_not_a_trapezoid(self):
        # 1 mm at 5.8 M usteps/s^2 never reaches 763 k usteps/s.
        ms = self._time_move(20157, 50000, 800000, 50000)
        assert 100 <= ms <= 125, ms

    def test_a_target_rewrite_mid_move_stops_the_motor_short(self):
        b, clock = board(timing='realistic')
        configure_x(b)
        settle(clock)
        write(b, XY, M1 + chip.XTARGET, 60 * 20157)
        settle(clock, 500)
        read(b, XY, M1 + chip.RAMP_STAT)
        actual = read(b, XY, M1 + chip.XACTUAL, signed=True)
        assert 0 < actual < 60 * 20157
        assert not read(b, XY, M1 + chip.RAMP_STAT) & chip.VZERO
        # The firmware's STOP: actual written as the new target.
        write(b, XY, M1 + chip.XTARGET, actual)
        settle(clock, 300)
        status = read(b, XY, M1 + chip.RAMP_STAT)
        assert status & chip.VZERO
        stopped = read(b, XY, M1 + chip.XACTUAL, signed=True)
        assert abs(stopped - actual) < 200_000  # decelerated past the rewrite, then came back
        assert stopped < 60 * 20157

    def test_velocity_is_reported_in_register_units(self):
        b, clock = board(timing='realistic')
        configure_x(b)
        settle(clock)
        write(b, XY, M1 + chip.XTARGET, 60 * 20157)
        settle(clock, 500)
        vactual = read(b, XY, M1 + chip.VACTUAL)
        assert vactual == pytest.approx(800000, abs=2)
