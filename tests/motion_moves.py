# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""One axis's move through the member that moves it, for a test that walks the axes.

A caller asks for the move it wants -- X and Y as one request, Z on its
own -- so there is no member that takes an axis name. A test whose subject
holds for each axis (a travel bound, an arrival, a judged move) names the
axis and reaches it here: X or Y alone through ``start_move_xy`` /
``start_move_xy_by``, Z through ``start_move_z`` / ``start_move_z_by``.
``overshoot_enabled`` reaches Z only; X and Y have no approach to choose.
"""

from __future__ import annotations

from modules.operation import Operation


def start_axis_move(motion, axis: str, position: float, **kwargs) -> Operation:
    """Start ``axis`` moving to ``position``; the started move."""
    if axis == 'Z':
        return motion.start_move_z(position, **kwargs)
    kwargs.pop('overshoot_enabled', None)
    return motion.start_move_xy(**{f'{axis.lower()}_um': position}, **kwargs)


def axis_move(motion, axis: str, position: float, **kwargs) -> None:
    """Move ``axis`` to ``position``, and return once it has arrived."""
    if axis == 'Z':
        motion.move_z(position, **kwargs)
        return
    kwargs.pop('overshoot_enabled', None)
    motion.move_xy(**{f'{axis.lower()}_um': position}, **kwargs)


def start_axis_move_by(motion, axis: str, distance: float, **kwargs) -> Operation:
    """Start ``axis`` moving by ``distance``; the started move."""
    if axis == 'Z':
        return motion.start_move_z_by(distance, **kwargs)
    kwargs.pop('overshoot_enabled', None)
    return motion.start_move_xy_by(**{f'd{axis.lower()}_um': distance})


def axis_move_by(motion, axis: str, distance: float, **kwargs) -> None:
    """Move ``axis`` by ``distance``, and return once it has arrived."""
    if axis == 'Z':
        motion.move_z_by(distance, **kwargs)
        return
    kwargs.pop('overshoot_enabled', None)
    motion.move_xy_by(**{f'd{axis.lower()}_um': distance})
