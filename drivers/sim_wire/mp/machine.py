# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The RP2040 `machine` module as the EL-0940 firmware sees it, for the
# firmware running inside MicroPython on the host. This file runs INSIDE the
# MicroPython child process, not in LumaViewPro's interpreter: it shadows the
# runtime's built-in `machine` because it is first on MICROPYPATH.
#
# Pure MicroPython, nothing host-only, so the same file can be carried onto a
# physical RP2350 later.
#
# SPI is the two TMC5072 drivers in `tmc5072.py`, built on the first
# transfer from the unit config the firmware reads (`motorconfig.json`) and
# the simulator's own settings (`sim_chip.json`), both in the working
# directory the port gives the child. Chip select is read from the pins the
# firmware drives, so the model needs no wiring of its own.
#
# The control channel. The firmware blocks in stdin's readline() while idle,
# so this code runs only inside the firmware's own SPI transfers:
# - Faults come in on fd 3, a pipe the port writes, one line per change.
#   It is read at every transfer: a pipe write the port has finished is
#   readable at once, so a fault written before a command is in effect from
#   its first transfer.
# - With the oracle on, every register write goes out on stdout as a frame,
#   in order with the firmware's own output; the port takes the frames out
#   before the driver reads.
# The messages themselves are `channel.py`'s.

import json
import select
import sys
import time

import channel

_PINS = {}


class Pin:
    IN = 0
    OUT = 1
    OPEN_DRAIN = 2
    PULL_UP = 1
    PULL_DOWN = 2
    IRQ_RISING = 8
    IRQ_FALLING = 4

    def __init__(
        self, id: int, mode: int | None = None, pull: int | None = None, value: int | None = None
    ) -> None:
        self.id = id
        self._v = 1 if pull == Pin.PULL_UP else 0
        if value is not None:
            self._v = 1 if value else 0
        _PINS[id] = self

    def init(self, *a: object, **k: object) -> None:
        pass

    def value(self, v: int | None = None) -> int | None:
        if v is None:
            return self._v
        self._v = 1 if v else 0

    def on(self) -> None:
        self._v = 1

    def off(self) -> None:
        self._v = 0

    high = on
    low = off

    def toggle(self) -> None:
        self._v ^= 1

    def irq(self, *a: object, **k: object) -> None:
        pass


class PWM:
    def __init__(self, pin: 'Pin', **k: object) -> None:
        self._f = 0
        self._d = 0

    def freq(self, f: int | None = None) -> int | None:
        if f is None:
            return self._f
        self._f = f

    def duty_u16(self, d: int | None = None) -> int | None:
        if d is None:
            return self._d
        self._d = d

    def deinit(self) -> None:
        pass


class Timer:
    # Timer callbacks never fire: the firmware uses them only for LEDs and the
    # fan tachometer, which nothing reads back over the port.
    PERIODIC = 1
    ONE_SHOT = 0

    def __init__(self, *a: object, **k: object) -> None:
        pass

    def init(self, *a: object, **k: object) -> None:
        pass

    def deinit(self) -> None:
        pass


# The firmware's chip-select pins: XY_chip = Pin(1), ZT_chip = Pin(5), low
# while a transfer is in flight.
_CHIP_SELECT = ((1, 'XY'), (5, 'ZT'))

_board = None
_oracle = False

# Open for the life of the process: the port holds the other end.
_faults_in = open('/dev/fd/3', 'rb')  # noqa: SIM115
_faults_poll = select.poll()
_faults_poll.register(_faults_in, select.POLLIN)


def _read_faults(board) -> None:
    while _faults_poll.poll(0):
        line = _faults_in.readline()
        if not line:
            # The port closed its end; no fault can change any more.
            _faults_poll.unregister(_faults_in)
            return
        on, axis, name = channel.parse_fault_line(line)
        board.set_fault(axis, name, on)


def _selected_chip():
    for pin_id, name in _CHIP_SELECT:
        pin = _PINS.get(pin_id)
        if pin is not None and pin._v == 0:
            return name
    return None


def _the_board():
    global _board, _oracle
    if _board is None:
        import tmc5072

        with open('motorconfig.json') as f:
            motorconfig = json.load(f)
        with open('sim_chip.json') as f:
            sim = json.load(f)
        _board = tmc5072.Board(motorconfig, sim, time.ticks_us, time.ticks_diff)
        _oracle = bool(sim.get('oracle', False))
    return _board


class SPI:
    MSB = 0
    LSB = 1

    def __init__(self, *a: object, **k: object) -> None:
        pass

    def _datagram(self, buf: bytes) -> bytes:
        chip = _selected_chip()
        if chip is None:
            # No chip selected: nothing drives MISO.
            return bytes(len(buf))
        frame = bytes(buf[:5]) + bytes(max(0, 5 - len(buf)))
        board = _the_board()
        _read_faults(board)
        out = board.datagram(chip, frame)
        if _oracle and frame[0] & 0x80:
            axis, reg = board.register_name(chip, frame[0] & 0x7F)
            value = (frame[1] << 24) | (frame[2] << 16) | (frame[3] << 8) | frame[4]
            sys.stdout.write(channel.write_frame(chip, axis, reg, value))
        return out[: len(buf)]

    def write(self, buf: bytes) -> None:
        self._datagram(buf)

    def read(self, n: int, write: int = 0) -> bytes:
        return self._datagram(bytes((write,) * n))

    def readinto(self, buf: bytearray, write: int = 0) -> None:
        out = self._datagram(bytes((write,) * len(buf)))
        for i in range(len(buf)):
            buf[i] = out[i]

    def write_readinto(self, wb: bytes, rb: bytearray) -> None:
        out = self._datagram(wb)
        for i in range(len(rb)):
            rb[i] = out[i] if i < len(out) else 0


class _Mem:
    def __init__(self) -> None:
        self._d = {}

    def __getitem__(self, a: int) -> int:
        return self._d.get(a, 0)

    def __setitem__(self, a: int, v: int) -> None:
        self._d[a] = v


mem32 = _Mem()


def bootloader() -> None:
    raise SystemExit('bootloader')


def reset() -> None:
    raise SystemExit('reset')


def freq(f: int | None = None) -> int:
    return 125_000_000


def unique_id() -> bytes:
    return b'\x00' * 8
