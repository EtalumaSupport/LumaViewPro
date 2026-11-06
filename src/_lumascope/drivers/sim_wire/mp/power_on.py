# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The board's power-on boot, which the firmware's first command read ends.
# Runs INSIDE the MicroPython child, imported by the launch after the console
# and before the firmware.
#
# A board is powered with its scope, so its power-on boot is over before a
# host opens its port. A process the port starts as that power-on
# (channel.POWER_ON_ENV set) is booting until the firmware first reads a
# command: the realistic clock (`realistic/time.py`) runs the boot's sleeps on
# a virtual clock, and the port holds a host's open until a BOOTED frame on
# stdout says the boot is over. A soft reset or a reboot starts a process with
# no power-on: it boots on the real clock while a host waits, as a board does.
#
# The first read is the firmware's first sys.stdin.readline(), the main loop of
# both motor dialects and of the LED firmware. sys.stdin cannot be reassigned
# on either runtime, so `sys` itself is replaced in sys.modules before the
# firmware imports it, by a stand-in whose stdin wraps the real one. The
# wrapper is an io.IOBase with ioctl: the 3.0 firmware registers stdin with
# select.poll at import, and poll refuses a plain object.

import io
import os
import select
import sys

import utime

import channel

booting = os.getenv(channel.POWER_ON_ENV) == '1'
# The runtime's ticks_us when the power-on boot ended; for a process with no
# power-on boot, when it started.
ended_us = None if booting else utime.ticks_us()

# MicroPython's stream ioctl request for a poll.
_MP_STREAM_POLL = 3


def _boot_over() -> None:
    global booting, ended_us
    ended_us = utime.ticks_us()
    booting = False
    sys.stdout.write(channel.booted_frame())


class _Stdin(io.IOBase):
    def __init__(self, real: object) -> None:
        self._real = real
        self._poll = select.poll()
        self._poll.register(real, select.POLLIN)

    def readline(self) -> str:
        if booting:
            _boot_over()
        return self._real.readline()

    def read(self, *a: int) -> str:
        return self._real.read(*a)

    def ioctl(self, request: int, arg: int) -> int:
        if request == _MP_STREAM_POLL:
            return select.POLLIN if self._poll.poll(0) else 0
        return 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


class _Sys:
    def __init__(self, real: object) -> None:
        self._real = real
        self.stdin = _Stdin(real.stdin)

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


if booting:
    sys.modules['sys'] = _Sys(sys)
