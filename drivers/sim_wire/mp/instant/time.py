# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The virtual clock for the instant timing mode. Runs INSIDE the MicroPython
# child and shadows the built-in `time`: a sleep advances the clock and
# returns at once, so a board boots in milliseconds and a firmware timeout
# expires as soon as the firmware has slept through it. Anything else that
# must move with the firmware's sense of time (a chip model's ramps) reads
# this clock too.
_now_us = 0


def advance_us(us: int) -> None:
    global _now_us
    _now_us += int(us)


def ticks_us() -> int:
    return _now_us


def ticks_ms() -> int:
    return _now_us // 1000


def ticks_cpu() -> int:
    return _now_us


def ticks_diff(a: int, b: int) -> int:
    return a - b


def ticks_add(a: int, b: int) -> int:
    return a + b


def sleep_us(us: int) -> None:
    advance_us(us)


def sleep_ms(ms: int) -> None:
    advance_us(ms * 1000)


def sleep(s: float) -> None:
    advance_us(s * 1_000_000)


def time() -> int:
    return _now_us // 1_000_000


def time_ns() -> int:
    return _now_us * 1000


def localtime(*a: int) -> tuple:
    return (2026, 1, 1, 0, 0, 0, 3, 1)
