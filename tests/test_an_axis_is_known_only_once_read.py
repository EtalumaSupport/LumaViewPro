# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An axis starts known only when the board reports it homed and its position was read.

IDLE says the cache holds where the axis is. Bring-up set an axis IDLE from
the board's homed answer alone and read its position in a separate later
call, which the support report's diagnostic scope never made and which an
``except Exception: pass`` abandoned on any defect: every homed axis then
stood IDLE on the cache's 0.0 seed, a number nothing read, with moves
permitted from it. ``MotionAPI._init_axes`` now reads each axis as it
decides its state.
"""

import pytest

import drivers.simulated_motorboard as simulated_motorboard
from modules.exceptions import AxisStateUnknownError
from modules.lumascope_api.motion import AxisState
from tests.scope_fakes import build_scope


def test_a_homed_axis_whose_read_fails_starts_unknown():
    """``create_diagnostic`` (the support report's scope, hardware only) builds
    its state with ``_init_axes`` and made no read, so the read is
    ``_init_axes``'s own and is pinned there: a simulated scope's store
    rebuilt, every axis reported homed, over a board whose Z read fails."""
    scope = build_scope(simulate=True)
    motion = scope.motion
    scope.simulation.drop_replies('motor', count=None, command='TARGET_RZ')

    motion._init_axes(['X', 'Y', 'Z', 'T'], ['X', 'Y', 'Z', 'T'])

    scope.simulation.clear()
    states = {axis: at.state for axis, at in motion.axis_positions().items()}
    assert states == {
        'X': AxisState.IDLE,
        'Y': AxisState.IDLE,
        'Z': AxisState.UNKNOWN,
        'T': AxisState.IDLE,
    }
    assert motion.axis_positions()['Z'].position is None
    with pytest.raises(AxisStateUnknownError) as exc:
        motion.move_z(1000.0)
    assert exc.value.reason == 'position_unknown'
    assert list(exc.value.axes) == ['Z']


def test_a_defect_in_the_bring_up_read_fails_construction(monkeypatch):
    """The read was wrapped in ``except Exception: pass``: a defect left every
    homed axis IDLE at 0.0, T at a slot that does not exist."""
    read = simulated_motorboard.SimulatedMotorBoard.target_pos

    def defective(self, axis):
        # A stand-in by design: a defect in the read, which no simulated
        # fault produces, since every fault the board can have is a
        # HardwareError that the read already turns into UNKNOWN.
        if axis == 'Z':
            raise KeyError('a defect in the position read')
        return read(self, axis)

    monkeypatch.setattr(simulated_motorboard.SimulatedMotorBoard, 'target_pos', defective)
    with pytest.raises(KeyError, match='a defect in the position read'):
        build_scope(simulate=True)
