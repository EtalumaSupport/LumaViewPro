# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""What counts as a number, once for each kind a check asks about.

Every numeric check in the program asks one of these; a check then applies
its own bound to the answer. A ``bool`` is never a number: Python counts it
as an ``int``, so ``True`` would otherwise pass as 1. Numpy registers its
scalars with ``numbers``, so a numpy integer is an integer here and a numpy
float a real number, and ``np.bool_`` is neither; this module needs no
numpy of its own.

Here, in the layer every other layer may import, because drivers judge the
numbers in their config files and readings and may not import ``modules``.
"""

import math
import numbers
import sys


def is_real_number(value: object) -> bool:
    """True when ``value`` is a real number, NaN and the infinities included, and not a ``bool``.

    For a value converted or printed rather than judged: the owner that
    judges it refuses NaN in its own words.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def is_finite_number(value: object) -> bool:
    """True when ``value`` is a real number that is finite as a float, and not a ``bool``.

    A comparison with NaN is False, so ``value < low`` passes it, and a
    ``bool`` is an ``int`` to Python. Numpy's numbers are real numbers;
    ``np.bool_`` is not one. A Python ``int`` past the largest float is not
    one either: every owner uses the number as a float, and ``math.isfinite``
    raises ``OverflowError`` on it rather than answer. Numpy's fixed-width
    integers always fit.
    """
    if not is_real_number(value):
        return False
    if isinstance(value, int):
        return abs(value) <= sys.float_info.max
    return math.isfinite(value)


def is_integer_number(value: object) -> bool:
    """True when ``value`` is of an integer type, Python's or numpy's, and not a ``bool``.

    By type: ``2.0`` is not one. For an argument or a field whose kind is
    an integer; a count written as ``2.0`` is ``is_count``'s.
    """
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def is_count(value: object) -> bool:
    """True when ``value`` is a finite number whose value is whole, ``2`` and ``2.0`` alike.

    By value, for a whole number that arrives as JSON or text, where a
    client may write ``2.0``: the settings store's counts, a protocol's
    whole cells, a request's whole fields. Its owner converts it to an
    ``int`` where it takes it.
    """
    return is_finite_number(value) and (
        isinstance(value, numbers.Integral) or float(value).is_integer()
    )


def whole_number_in(text: str) -> int | None:
    """The whole number ``text`` spells in ASCII decimal digits, or None for any other text.

    For a whole number written as text: a header, a query argument, a
    file name's index, a board's or the OS's reply. ``str.isdigit`` is
    true for digits ``int`` will not read (a superscript two, U+00B2), and ``int`` refuses a
    decimal string longer than its digit cap (``sys.get_int_max_str_digits``),
    so a check written as ``isdigit`` then ``int`` raises on text it
    passed. Empty, signed or spaced text is None: each reader strips what
    its format allows before it asks.
    """
    if not (text.isascii() and text.isdigit()):
        return None
    cap = sys.get_int_max_str_digits()
    if cap and len(text) > cap:
        return None
    return int(text)
