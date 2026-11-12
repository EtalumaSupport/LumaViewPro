# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The axis names and states: the closed sets a motion member, a position and a capability name an axis by.

A leaf module, importing nothing of the project's, so a protocol, a
recording and the capabilities name an axis without importing the scope's
API package, which imports them.
"""

import enum
from collections.abc import Mapping


class StageAxis(enum.StrEnum):
    """A stage axis, X, Y or Z: an axis with a position in um.

    The axes a position, a jog, a bookmark and a position event name. A
    name's spelling is exact: ``'x'`` is not ``'X'``, so each axis has one
    spelling in every log and on every wire. Whether this scope has the
    axis is ``ScopeCapabilities.axes``.
    """

    X = 'X'
    Y = 'Y'
    Z = 'Z'


class Axis(enum.StrEnum):
    """An axis a motion member names: a stage axis, or the turret, T.

    The turret has a slot, not a position (``TurretStatus``), so a member
    that answers a position takes a ``StageAxis``.
    """

    X = StageAxis.X.value
    Y = StageAxis.Y.value
    Z = StageAxis.Z.value
    T = 'T'


class AxisState(enum.StrEnum):
    """Possible states for a motion axis."""

    UNKNOWN = 'unknown'  # Not homed / state not known
    IDLE = 'idle'  # At known position, not moving
    MOVING = 'moving'  # Move commanded, not yet arrived
    HOMING = 'homing'  # Homing sequence in progress


def axis_states(states: Mapping[str, str] | None) -> dict[Axis, AxisState]:
    """Each axis named in *states* as its ``Axis``, with its state as its ``AxisState``.

    The one build of an axis-to-state map a refusal or an ending carries,
    so each names the axes as ``MotionAPI.axes_without_position`` answers
    them, in the order given. None is no axes.

    Raises:
        ValueError: a name that is no ``Axis``, or a state that is no
            ``AxisState``.
    """
    return {Axis(axis): AxisState(state) for axis, state in (states or {}).items()}
