# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The clock for the realistic timing mode. Runs INSIDE the MicroPython child
# and shadows the built-in `time`.
#
# The board's power-on boot is in the past when a host connects (`power_on`),
# so while it runs the clock is virtual: a sleep advances it and returns at
# once. From the boot's end the clock is the runtime's, continuing from where
# the virtual one stopped. It must continue, not jump: the chip model stamps
# every motor at each transfer, the boot's included, and a jump to the
# runtime's raw ticks would have it step through decades of travel before it
# answers. A process with no power-on boot (a soft reset, a reboot) is never
# booting, and runs on the runtime's clock from its start.
#
# The switch is the flag read inside each function, never a rebinding: the
# firmware binds `from time import sleep` at import, and the chip model holds
# ticks_us from its first transfer.

import utime as _runtime

import power_on

_virtual_us = 0


def ticks_us() -> int:
    if power_on.booting:
        return _virtual_us
    return _virtual_us + _runtime.ticks_diff(_runtime.ticks_us(), power_on.ended_us)


def ticks_ms() -> int:
    return ticks_us() // 1000


def ticks_cpu() -> int:
    return ticks_us()


def ticks_diff(a: int, b: int) -> int:
    return a - b


def ticks_add(a: int, b: int) -> int:
    return a + b


def sleep_us(us: int) -> None:
    global _virtual_us
    if power_on.booting:
        _virtual_us += int(us)
    else:
        _runtime.sleep_us(us)


def sleep_ms(ms: int) -> None:
    global _virtual_us
    if power_on.booting:
        _virtual_us += int(ms * 1000)
    else:
        _runtime.sleep_ms(ms)


def sleep(s: float) -> None:
    global _virtual_us
    if power_on.booting:
        _virtual_us += int(s * 1_000_000)
    else:
        _runtime.sleep(s)


def time() -> int:
    return ticks_us() // 1_000_000


def time_ns() -> int:
    return ticks_us() * 1000


def localtime(*a: int) -> tuple:
    return _runtime.localtime(*a)
