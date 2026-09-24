# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A home returns when the stage has stopped, on the firmware that ships.

The field firmware answers HOME and then drives X, Y and Z to the centre
of travel, and answers THOME and then drives Z back to where it was,
without waiting for either move. What the driver returns after a home is
therefore where the stage IS only if the driver waits: these run the real
field firmware in realistic timing and read the stage the moment the home
returns.
"""

import sys

import pytest

from drivers.motorboard import MotorBoard
from drivers.sim_wire.backend import MotorBoardSpec, SimWireBackend

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )


@pytest.fixture
def board():
    spec = MotorBoardSpec('LS850T', frozenset('XYZT'), dialect='field', timing='realistic')
    b = MotorBoard(backend=SimWireBackend(spec))
    try:
        yield b
    finally:
        b.disconnect()


def _raw(board, query, axis):
    return int(board.exchange_command(f'{query}_R{axis}'))


def test_after_home_every_axis_is_at_its_target(board):
    assert board.home()
    for axis in 'XYZT':
        assert _raw(board, 'ACTUAL', axis) == _raw(board, 'TARGET', axis), axis


def test_after_a_turret_home_z_is_back_at_its_target(board):
    assert board.home()
    board.move_abs_pos('Z', 3000.0, overshoot_enabled=False)
    assert board.wait_for_position('Z', timeout=10.0)
    assert board.thome()
    for axis in 'ZT':
        assert _raw(board, 'ACTUAL', axis) == _raw(board, 'TARGET', axis), axis
