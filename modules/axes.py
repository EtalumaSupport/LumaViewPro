# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The axis names: the closed sets a motion member, a position and a capability name an axis by.

A leaf module, importing nothing of the project's, so a protocol, a
recording and the capabilities name an axis without importing the scope's
API package, which imports them.
"""

import enum


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
