# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Shared constants for the lumascope_api package.

Module-level home for values that both `_lumascope.Lumascope` and the
sub-API modules (motion.py, etc.) need to read. Lifting them here
breaks the import cycle that would otherwise force lazy imports in
the sub-API files.

Other modules in the package alias these onto their classes for
back-compat (e.g. `Lumascope._VALID_AXIS_NAMES = _VALID_AXIS_NAMES`)
so existing callers (`scope._VALID_AXIS_NAMES`, tests reading the
class attribute) keep working.
"""

import numbers
from typing import NamedTuple

from modules.exceptions import AccelerationLimitRefusedError
from modules.api_surface import api_fields

# Structural axis-name vocabulary used only for input sanity checks
# ("did the caller pass a real axis letter?"). NOT a capability query --
# use `scope.capabilities.axes` for "what does this scope have?".
_VALID_AXIS_NAMES = ('X', 'Y', 'Z', 'T')

# The simulated boards a simulated scope can be built on: a Python
# stand-in with no timing; the production driver against the real firmware
# in a MicroPython process, its motors moving at once; and the same
# firmware with its motors moving at the ramp fitted to bench moves, so a
# move or a home takes seconds rather than none. A home starts with the
# axes mid-travel, so it is shorter than a stage's (an LS850's 5 s against
# under 20 s). The session validates the setting against this tuple and
# the constructor dispatches on it.
SIMULATOR_TIERS = ('fast', 'firmware', 'realistic')

# Absolute position bounds in um -- generous outer limits. Per-axis
# travel limits are enforced by the motor board itself.
MOTOR_POSITION_LIMIT = 1_000_000  # 1 meter in um

# The turret's four positions. Unlike X/Y/Z this is not travel and not um:
# the T axis publishes no limits, so the um-based range check cannot refuse
# anything for it and the motor's answer to a nonsense slot is to drive
# there -- 99 is 24.5 revolutions. Both refusal sites in motion.py read this
# rather than spelling the range twice: they guard the same illegal state at
# two depths of one call chain, and drifting apart would leave one door open.
TURRET_SLOT_MIN = 1
TURRET_SLOT_MAX = 4

# The acceleration limit a caller may ask for, as a percentage of the
# firmware's own maximum. The API refuses outside it before any board is
# commanded: held in the real motor driver alone, the simulated and absent
# boards took any number and the Session stored it.
ACCELERATION_PCT_MIN = 1
ACCELERATION_PCT_MAX = 100


def refuse_acceleration_pct(val_pct: object) -> None:
    """Refuse an acceleration limit no board may be given.

    The one check of the range, for every place a limit enters: the motion
    API's setter, a stored value at load and a settings dict handed to a
    session.

    Raises:
        AccelerationLimitRefusedError: ``val_pct`` is not a number, or is
            outside ``ACCELERATION_PCT_MIN`` to ``ACCELERATION_PCT_MAX``. A
            ValueError, so a caller catching a bad argument keeps working.
    """
    # bool is excluded by name: True is an int, and a number to Real.
    if (
        isinstance(val_pct, bool)
        or not isinstance(val_pct, numbers.Real)
        or not ACCELERATION_PCT_MIN <= val_pct <= ACCELERATION_PCT_MAX
    ):
        raise AccelerationLimitRefusedError(val_pct, ACCELERATION_PCT_MIN, ACCELERATION_PCT_MAX)


def is_turret_slot(position: object) -> bool:
    """Whether ``position`` names a turret slot.

    A bool is excluded explicitly: it is an int in Python, so ``True``
    would otherwise pass as slot 1.
    """
    return (
        isinstance(position, int)
        and not isinstance(position, bool)
        and TURRET_SLOT_MIN <= position <= TURRET_SLOT_MAX
    )


class AxisState:
    """Possible states for a motion axis."""

    UNKNOWN = 'unknown'  # Not homed / state not known
    IDLE = 'idle'  # At known position, not moving
    MOVING = 'moving'  # Move commanded, not yet arrived
    HOMING = 'homing'  # Homing sequence in progress


@api_fields('position', 'state')
class AxisPosition(NamedTuple):
    """One axis's state and its position, read together.

    ``position`` is None unless the axis is IDLE or MOVING: an axis whose
    reference is lost or still being established keeps answering the last
    number it reported, and a caller writing a position into a file must
    not be handed it.
    """

    state: str
    position: float | int | None
