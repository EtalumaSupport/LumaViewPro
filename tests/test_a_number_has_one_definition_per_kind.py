# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What counts as a number, for each of the four kinds a check asks about.

An integer is by type (``2.0`` is not one), a count by value (``2.0`` is
one), a finite number refuses NaN, the infinities and an integer past
every float, and a real number takes NaN for the code that converts or
prints a value rather than judging it. None of them takes a ``bool``,
Python's or numpy's; numpy's numbers are numbers of their kind.
"""

import math
from decimal import Decimal
from fractions import Fraction

import numpy as np

from _lumascope.lib.real_numbers import (
    is_count,
    is_finite_number,
    is_integer_number,
    is_real_number,
)

# value: (integer, count, finite, real)
_TABLE = [
    (True, (False, False, False, False)),
    (np.bool_(True), (False, False, False, False)),
    (1, (True, True, True, True)),
    (np.int64(1), (True, True, True, True)),
    (2.0, (False, True, True, True)),
    (2.5, (False, False, True, True)),
    (np.float32(1.5), (False, False, True, True)),
    (np.float64(3.0), (False, True, True, True)),
    (math.nan, (False, False, False, True)),
    (math.inf, (False, False, False, True)),
    (-math.inf, (False, False, False, True)),
    (10**400, (True, False, False, True)),
    (Fraction(1, 2), (False, False, True, True)),
    (Fraction(4, 2), (False, True, True, True)),
    (Decimal('1'), (False, False, False, False)),
    ('1', (False, False, False, False)),
    (None, (False, False, False, False)),
]


def test_each_value_is_a_number_of_the_kinds_its_row_names():
    predicates = (is_integer_number, is_count, is_finite_number, is_real_number)
    wrong = [
        (value, predicate.__name__, predicate(value), expected)
        for value, row in _TABLE
        for predicate, expected in zip(predicates, row, strict=True)
        if predicate(value) is not expected
    ]
    assert wrong == []
