# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""One test over a set: every case is checked and every one that fails is named.

The form a parametrize over a computed set takes when it is one assertion
over the set (``tests/guards/test_no_parametrize_over_a_discovered_set.py``):
the case's body stays as it was, in a check that takes the case, and the
test asserts that no case failed, its failure naming each one with its own
message, so one red run lists every offender.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable

import pytest

# What a failed check raises: an assert, or pytest.raises meeting no raise.
_FAILED = (AssertionError, pytest.fail.Exception)

# A ``pytest.param(...)`` case: its values are the arguments, its id the name.
_PARAM = type(pytest.param(None))


def failing_cases(cases: Iterable, check: Callable, *, label: Callable = repr) -> dict:
    """``{label(case): message}`` for each case ``check`` fails on.

    ``check`` takes one argument per parametrized name: a one-argument check
    is handed the case whole, a tuple included; a check of several is handed
    the case unpacked. A ``pytest.param`` case hands over its values and is
    named by its id.
    """
    takes_one = len(inspect.signature(check).parameters) == 1
    failed = {}
    for case in cases:
        if isinstance(case, _PARAM):
            args, name = case.values, case.id or repr(case.values)
        else:
            args, name = ((case,) if takes_one else tuple(case)), label(case)
        try:
            check(*args)
        except _FAILED as ex:
            failed[name] = str(ex) or 'assertion failed'
    return failed
