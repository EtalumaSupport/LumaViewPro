# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The one test of "a number" every owner of a numeric argument asks before its range.

A leaf: it imports nothing of the project but the refusal it raises, so
the modules low in the import graph that own a number (``labware``, whose
importers ``common_utils`` reaches while it is still loading) can ask it.
"""

import math
import numbers

from modules.exceptions import ArgumentRefusedError


def is_finite_number(value: object) -> bool:
    """True when ``value`` is a real number that is finite, and not a ``bool``.

    A comparison with NaN is False, so ``value < low`` passes it, and a
    ``bool`` is an ``int`` to Python. Numpy's numbers are real numbers;
    ``np.bool_`` is not one.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def refuse_unless_finite_number(value: object, argument: str) -> None:
    """Refuse ``value`` unless ``is_finite_number`` holds for it.

    Raises:
        ArgumentRefusedError: ``'not_a_number'``, naming ``argument``.
    """
    if not is_finite_number(value):
        raise ArgumentRefusedError('not_a_number', argument=argument, value=value)
