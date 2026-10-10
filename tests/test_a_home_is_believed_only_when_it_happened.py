# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A home is believed only when every axis it homes is homed.

The field firmware answers HOME with 'XYZ home complete' without reading
whether its Z and T homes succeeded, answers THOME with 'T home successful'
after a failed Z re-home, and answers HOME on a board with no XY with
'X not present' whatever Z did. Its own per-axis homed flags, in FULLINFO,
are set only by a home that succeeded. Each test is the first home since
the board booted, with a Z switch that never trips, on the field firmware.
"""

import sys

import pytest

from drivers.exceptions import HardwareError
from drivers.motorboard import MotorBoard
from drivers.sim_wire.backend import MotorBoardSpec, SimWireBackend
from drivers.sim_wire.mp import tmc5072
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )


def _board(model, axes):
    backend = SimWireBackend(MotorBoardSpec(model, frozenset(axes)))
    board = MotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=backend)
    backend.motor_board.inject('Z', tmc5072.SWITCH_NEVER_TRIPS)
    return board


@pytest.mark.parametrize(
    ('model', 'axes', 'home'),
    [
        ('LS850T', 'XYZT', 'home'),
        ('LS850T', 'XYZT', 'thome'),
        ('LS820', 'Z', 'home'),
    ],
)
def test_a_home_whose_z_did_not_home_raises_naming_z(model, axes, home):
    board = _board(model, axes)
    try:
        with pytest.raises(HardwareError, match='Z did not home'):
            getattr(board, home)()
        assert not board.has_thomed()
    finally:
        board.disconnect()
