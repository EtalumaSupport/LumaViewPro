# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The one test of "a number" every owner of a numeric argument asks before its range.

A leaf: it imports nothing of the project but the refusal it raises, so
the modules low in the import graph that own a number (``labware``, whose
importers ``common_utils`` reaches while it is still loading) can ask it.
"""

import math
import numbers
import sys

from modules.exceptions import ArgumentRefusedError


def is_finite_number(value: object) -> bool:
    """True when ``value`` is a real number that is finite as a float, and not a ``bool``.

    A comparison with NaN is False, so ``value < low`` passes it, and a
    ``bool`` is an ``int`` to Python. Numpy's numbers are real numbers;
    ``np.bool_`` is not one. A Python ``int`` past the largest float is not
    one either: every owner uses the number as a float, and ``math.isfinite``
    raises ``OverflowError`` on it rather than answer. Numpy's fixed-width
    integers always fit.
    """
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= sys.float_info.max
    return math.isfinite(value)


def refuse_unless_finite_number(value: object, argument: str) -> None:
    """Refuse ``value`` unless ``is_finite_number`` holds for it.

    Raises:
        ArgumentRefusedError: ``'not_a_number'``, naming ``argument``.
    """
    if not is_finite_number(value):
        raise ArgumentRefusedError('not_a_number', argument=argument, value=value)


def refuse_outside(value: object, argument: str, limits: tuple[float, float]) -> None:
    """Refuse ``value`` unless it is a finite number within ``limits``, ends included.

    Raises:
        ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``.
    """
    refuse_unless_finite_number(value, argument)
    low, high = limits
    if not low <= value <= high:
        raise ArgumentRefusedError('out_of_range', argument=argument, value=value, limits=limits)


# The longest wait a caller may ask for, in seconds: 30 days, room for a
# multi-day protocol's run, and short of the smallest wait the host can
# hold. Python's waits raise past threading.TIMEOUT_MAX, which is about
# 292 years on macOS and Linux but 49.7 days on Windows, so a bound read
# from the host would accept a wait on one platform it refuses on another.
WAIT_S = (0.0, 30 * 24 * 3600.0)


def refuse_unless_wait_s(value: object, argument: str) -> None:
    """Refuse ``value`` unless it is a wait every host can hold: 0 to 30 days.

    Every member that takes a wait from its caller asks this where the wait
    enters, before anything starts, so a wait too long for the host is a
    refusal rather than an ``OverflowError`` out of the wait itself.

    Raises:
        ArgumentRefusedError: ``'not_a_number'`` or ``'out_of_range'``.
    """
    refuse_outside(value, argument, WAIT_S)
