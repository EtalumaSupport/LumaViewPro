"""A motor board's positions, in user units and in microsteps, converted once.

X, Y and Z are microns, each axis converted by its own row of the board's
microsteps per mm and rounded half-up to a microstep; T is the turret's
1-based slot, at the microsteps the board's slot table gives it. Every
motor board and the simulator convert through these two functions, so the
simulator moves by the numbers the scope uses. A board whose hardware lacks
an axis refuses it before it calls here.

The config is the board's: anything with ``usteps_per_mm(axis)`` serves the
linear axes, and a turret's config adds ``turret_position_usteps(slot)``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from _lumascope.drivers.exceptions import PartAbsentError

if TYPE_CHECKING:
    from _lumascope.drivers.motorconfig import MotorConfig
    from _lumascope.drivers.tmcm6110_config import Tmcm6110Config

LINEAR_AXES = ('X', 'Y', 'Z')


def position_to_steps(config: MotorConfig | Tmcm6110Config, axis: str, position: float) -> int:
    """The microsteps of ``position``: microns for X/Y/Z, the slot for T.

    Raises:
        HardwareError: ``axis`` is no motor board's axis, or the slot table
            lists no such slot.
    """
    if axis in LINEAR_AXES:
        return math.floor(config.usteps_per_mm(axis) * position / 1000 + 0.5)
    if axis == 'T':
        return config.turret_position_usteps(position)
    raise PartAbsentError(f'Unsupported axis ({axis})')


def steps_to_position(config: MotorConfig | Tmcm6110Config, axis: str, steps: int) -> float | int:
    """The position of ``steps`` microsteps: microns for X/Y/Z, the
    1-based slot for T.

    Raises:
        HardwareError: ``axis`` is no motor board's axis.
    """
    if axis in LINEAR_AXES:
        return steps * 1000 / config.usteps_per_mm(axis)
    if axis == 'T':
        # The T row of microsteps per mm is one slot's, 90 degrees.
        degrees = 90.0 / config.usteps_per_mm('T') * steps
        return int(degrees / 90) + 1
    raise PartAbsentError(f'Unsupported axis ({axis})')
