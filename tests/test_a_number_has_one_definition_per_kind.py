# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What counts as a number, for each of the four kinds a check asks about, and in text.

An integer is by type (``2.0`` is not one), a count by value (``2.0`` is
one), a finite number refuses NaN, the infinities and an integer past
every float, and a real number takes NaN for the code that converts or
prints a value rather than judging it. None of them takes a ``bool``,
Python's or numpy's; numpy's numbers are numbers of their kind. A whole
number in text is ASCII decimal digits, no more than ``int`` converts.
"""

import math
import sys
from decimal import Decimal
from fractions import Fraction

import numpy as np

from _lumascope.lib.real_numbers import (
    is_count,
    is_finite_number,
    is_integer_number,
    is_real_number,
    whole_number_in,
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


def test_a_whole_number_in_text_is_ascii_digits_int_converts_and_nothing_else():
    cap = sys.get_int_max_str_digits()
    read = {'0': 0, '12': 12, '9' * cap: int('9' * cap)}
    # A superscript two, and twelve in Arabic-Indic digits: digits to str.isdigit.
    not_whole = ['', '-1', '+1', ' 1', '1 ', '1.0', '\u00b2', '\u0661\u0662', '9' * (cap + 1)]

    assert {text: whole_number_in(text) for text in read} == read
    assert [text for text in not_whole if whole_number_in(text) is not None] == []
