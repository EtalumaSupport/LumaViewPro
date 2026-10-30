# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The production board drivers over the firmware-backed simulator.

The one place a test builds a `MotorBoard` or `LEDBoard` on the wire
emulation: the shipped firmware behind an emulated port, reached through
the driver's own ``backend=`` seam, so what the board answers is the
firmware's reply rather than a hand-written one. Each builder returns the
driver and the simulated board behind it, through which a test reaches
the board's faults (``drop_next_reply``, ``delay_next_reply``,
``unplug``), its chips (``state``, ``take_writes`` with the oracle on)
and its link.

A connect costs about 1.2 s: the driver resets the firmware and drains
its boot output, as it does on a real board. The test's teardown
disconnects what it built; a module-scoped fixture's board is built
before the test begins and is that fixture's to disconnect.

The tests that use these carry the ``firmware_tier`` mark: the firmware
runs on macOS and Linux only.
"""

from __future__ import annotations

from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from drivers.sim_wire.backend import LedBoardSpec, MotorBoardSpec, SimWireBackend
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS
from tests.scope_fakes import _TEARDOWNS


def motor_on_the_wire(model: str = 'LS850T', axes: str = 'XYZT', **spec):
    """(MotorBoard, the simulated motor board) on *model*'s firmware.

    *spec* goes to `MotorBoardSpec`: ``dialect`` ('field', the default
    firmware, or '3.0'), ``timing``, ``oracle``.
    """
    backend = SimWireBackend(MotorBoardSpec(model, frozenset(axes), **spec))
    board = MotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=backend)
    _TEARDOWNS.append(board.disconnect)
    return board, backend.motor_board


def led_on_the_wire(model: str = 'LS850T', **spec):
    """(LEDBoard, the simulated LED board) on the field LED firmware.

    *spec* goes to `LedBoardSpec`: ``timing``, ``oracle``.
    """
    backend = SimWireBackend(None, led=LedBoardSpec(model, **spec))
    board = LEDBoard(backend=backend)
    _TEARDOWNS.append(board.disconnect)
    return board, backend.led_board
