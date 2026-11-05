# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A home is believed only when every axis it homes is homed.

The field firmware answers HOME with 'XYZ home complete' without reading
whether its Z and T homes succeeded, and answers THOME with 'T home
successful' after a failed Z re-home. Its own per-axis homed flags, in
FULLINFO, are set only by a home that succeeded. A board with no XY is
homed by ZHOME, which answers its own timeout. Each test is the first home
since the board booted, with a Z switch that never trips, on the field
firmware.
"""

import pytest

from _lumascope.drivers.exceptions import HardwareError
from _lumascope.drivers.motorboard import MotorBoard
from _lumascope.drivers.sim_wire.backend import MotorBoardSpec, SimWireBackend
from _lumascope.drivers.sim_wire.mp import tmc5072
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS

pytestmark = pytest.mark.firmware_tier


def _board(model, axes):
    backend = SimWireBackend(MotorBoardSpec(model, frozenset(axes)))
    board = MotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=backend)
    backend.motor_board.inject('Z', tmc5072.SWITCH_NEVER_TRIPS)
    return board


@pytest.mark.parametrize(
    ('model', 'axes', 'home', 'says'),
    [
        ('LS850T', 'XYZT', 'home', 'Z did not home'),
        ('LS850T', 'XYZT', 'thome', 'Z did not home'),
        ('LS820', 'Z', 'home', 'Z home timeout'),
    ],
)
def test_a_home_whose_z_did_not_home_raises_naming_z(model, axes, home, says):
    board = _board(model, axes)
    try:
        with pytest.raises(HardwareError, match=says):
            getattr(board, home)()
        assert not board.has_thomed()
    finally:
        board.disconnect()
