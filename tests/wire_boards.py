# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The production board drivers over the firmware-backed simulator.

The one place a test builds a `MotorBoard` or `LEDBoard` on the wire
emulation: the shipped firmware behind an emulated port, reached through
the driver's own ``backend=`` seam, so what the board answers is the
firmware's reply rather than a hand-written one. Each builder returns the
driver and the simulated board behind it, through which a test reaches
the board's faults (``drop_next_reply``, ``delay_next_reply``,
``unplug``), its chips (``state``, ``take_writes`` with the oracle on)
and its link. The driver's open port is a `WatchedPort`, which a test
that holds a reply back, or writes without reading, waits on rather than
on the clock.

A connect costs about 1.2 s: the driver resets the firmware and drains
its boot output, as it does on a real board. The test's teardown
disconnects what it built; a module-scoped fixture's board is built
before the test begins and is that fixture's to disconnect.

The tests that use these carry the ``firmware_tier`` mark: the firmware
runs on macOS and Linux only.
"""

from __future__ import annotations

import threading

from serial.serialutil import SerialException

from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from drivers.sim_wire.backend import (
    LED_DEVICE,
    MOTOR_DEVICE,
    LedBoardSpec,
    MotorBoardSpec,
    SimWireBackend,
)
from drivers.sim_wire.port import EmulatedPort
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS
from tests.scope_fakes import _TEARDOWNS


class WatchedPort(EmulatedPort):
    """The emulated port, telling a test when the driver has written and how
    many lines of the board's output have landed in it.

    ``wrote`` is set by every write. A held reply lands all at once when its
    delay ends, and a write-only command's reply lands with no read waiting
    for it; `wait_for_lines` is how a test knows it has.
    """

    def __init__(self, *args, **kwargs):
        self.wrote = threading.Event()
        self._landed = threading.Condition()
        self._lines = 0
        super().__init__(*args, **kwargs)

    def write(self, data):
        written = super().write(data)
        self.wrote.set()
        return written

    def _deliver(self, data: bytes) -> None:
        super()._deliver(data)
        with self._landed:
            self._lines += data.count(b'\n')
            self._landed.notify_all()

    def lines_landed(self) -> int:
        with self._landed:
            return self._lines

    def wait_for_lines(self, count: int, timeout: float = 5.0) -> bool:
        """Whether ``count`` lines have landed since the port opened, waiting
        up to ``timeout`` seconds for them."""
        with self._landed:
            return self._landed.wait_for(lambda: self._lines >= count, timeout)


class _WatchedBackend(SimWireBackend):
    def open(self, **kwargs) -> WatchedPort:
        port = kwargs.get('port')
        board = {MOTOR_DEVICE: self.motor_board, LED_DEVICE: self.led_board}.get(port)
        if board is None:
            raise SerialException(f'no simulated board at {port!r}')
        return WatchedPort(board, **kwargs)


def motor_on_the_wire(model: str = 'LS850T', axes: str = 'XYZT', **spec):
    """(MotorBoard, the simulated motor board) on *model*'s firmware.

    *spec* goes to `MotorBoardSpec`: ``dialect`` ('field', the default
    firmware, or '3.0'), ``timing``, ``oracle``.
    """
    backend = _WatchedBackend(MotorBoardSpec(model, frozenset(axes), **spec))
    board = MotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=backend)
    _TEARDOWNS.append(board.disconnect)
    return board, backend.motor_board


def led_on_the_wire(model: str = 'LS850T', **spec):
    """(LEDBoard, the simulated LED board) on the field LED firmware.

    *spec* goes to `LedBoardSpec`: ``timing``, ``oracle``.
    """
    backend = _WatchedBackend(None, led=LedBoardSpec(model, **spec))
    board = LEDBoard(backend=backend)
    _TEARDOWNS.append(board.disconnect)
    return board, backend.led_board
