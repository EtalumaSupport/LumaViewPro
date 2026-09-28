# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The RP2040 `machine` module as the EL-0940 boards' firmware sees it, for
# the firmware running inside MicroPython on the host. This file runs INSIDE
# the MicroPython child process, not in LumaViewPro's interpreter: it
# shadows the runtime's built-in `machine` because it is first on
# MICROPYPATH.
#
# Pure MicroPython, nothing host-only, so the same file can be carried onto a
# physical RP2350 later.
#
# SPI is the board's chips: the model module and the chip-select wiring the
# board's settings (`sim_board.json`, in the working directory the port
# gives the child) name. The model is built on the first transfer, and each
# transfer goes, as sent, to the chip whose select pin the firmware holds
# low; the chip frames it. So one stub serves every board: GP1 is the XY
# TMC5072 on the motor board and the DAC on the LED board.
#
# The control channel. The firmware blocks in stdin's readline() while idle,
# so this code runs only inside the firmware's own SPI transfers:
# - Faults come in on a pipe the port writes, one line per change, at the
#   fd `channel.FAULTS_FD_ENV` names.
#   It is read at every transfer: a pipe write the port has finished is
#   readable at once, so a fault written before a command is in effect from
#   its first transfer.
# - With the oracle on, every register write the model names goes out on
#   stdout as a frame, in order with the firmware's own output; the port
#   takes the frames out before the driver reads.
# The messages themselves are `channel.py`'s.

import json
import os
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


_board = None
_oracle = False
# (pin, chip name) for each chip on the board, from its settings; a chip's
# select pin is low while a transfer to it is in flight.
_chip_select = ()

# Open for the life of the process: the port holds the other end.
_faults_in = open('/dev/fd/' + os.getenv(channel.FAULTS_FD_ENV), 'rb')  # noqa: SIM115
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
    for pin_id, name in _chip_select:
        pin = _PINS.get(pin_id)
        if pin is not None and pin._v == 0:
            return name
    return None


def _the_board():
    global _board, _oracle, _chip_select
    if _board is None:
        with open('sim_board.json') as f:
            sim = json.load(f)
        model = __import__(sim['model'])
        _board = model.build(sim, time.ticks_us, time.ticks_diff)
        _oracle = bool(sim['oracle'])
        _chip_select = tuple((pin, name) for pin, name in sim['chip_select'])
    return _board


class SPI:
    MSB = 0
    LSB = 1

    def __init__(self, *a: object, **k: object) -> None:
        pass

    def _datagram(self, buf: bytes) -> bytes:
        board = _the_board()
        chip = _selected_chip()
        if chip is None:
            # The firmware selects a chip around every transfer it makes, so
            # a transfer with none selected is this board's wiring naming the
            # wrong pin; answered, it would pass as a chip that took it.
            raise ValueError(
                f'an SPI transfer with no chip selected; the board wires {_chip_select}'
            )
        _read_faults(board)
        out = board.datagram(chip, bytes(buf))
        if _oracle:
            written = board.written(chip, bytes(buf))
            if written is not None:
                sys.stdout.write(channel.write_frame(chip, *written))
        return out

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
