# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The case walk names every case its check fails, however the case is shaped.

The walk is what a collapsed parametrize reports through: a case it passed
over silently would be a green test with the behaviour gone.
"""

import pytest

from tests.case_walk import failing_cases


def test_every_failing_case_is_named_with_its_own_message():
    def check(value):
        assert value % 2 == 0, f'{value} is odd'

    failed = failing_cases([1, 2, 3], check)

    assert list(failed) == ['1', '3']
    assert failed['1'].startswith('1 is odd') and failed['3'].startswith('3 is odd')


def test_a_one_argument_check_is_handed_a_tuple_case_whole():
    seen = []

    failing_cases([(1, 2)], seen.append)

    assert seen == [(1, 2)]


def test_a_check_of_several_arguments_is_handed_the_case_unpacked():
    def check(low, high):
        assert low < high

    assert list(failing_cases([(1, 2), (3, 2)], check)) == ['(3, 2)']


def test_a_pytest_param_hands_over_its_values_and_is_named_by_its_id():
    def check(low, high):
        assert low < high

    cases = [pytest.param(1, 2, id='rising'), pytest.param(3, 2, id='falling')]

    assert list(failing_cases(cases, check)) == ['falling']


def test_a_raise_that_never_came_is_a_failing_case():
    def check(value):
        with pytest.raises(ValueError):
            int(value)

    assert list(failing_cases(['x', '7'], check)) == ["'7'"]
