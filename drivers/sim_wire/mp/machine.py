# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The RP2040 `machine` module as the EL-0940 firmware sees it, for the
# firmware running inside MicroPython on the host. This file runs INSIDE the
# MicroPython child process, not in LumaViewPro's interpreter: it shadows the
# runtime's built-in `machine` because it is first on MICROPYPATH.
#
# Pure MicroPython, nothing host-only, so the same file can be carried onto a
# physical RP2350 later.
#
# SPI answers zeros here: no motor-driver chips are modelled yet, so the
# firmware boots and answers, and anything that waits on a chip (homing)
# runs to the firmware's own timeout.


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


class SPI:
    MSB = 0
    LSB = 1

    def __init__(self, *a: object, **k: object) -> None:
        pass

    def write(self, buf: bytes) -> None:
        pass

    def read(self, n: int, write: int = 0) -> bytes:
        return bytes(n)

    def readinto(self, buf: bytearray, write: int = 0) -> None:
        for i in range(len(buf)):
            buf[i] = 0

    def write_readinto(self, wb: bytes, rb: bytearray) -> None:
        for i in range(len(rb)):
            rb[i] = 0


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
