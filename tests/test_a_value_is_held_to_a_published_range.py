# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A value past a range the hardware publishes is held to it, by one rule.

``cap_stored_value`` is the one resolution of a value against a hardware
limit, for a stored value applied at the limit and for a value written now.
"""

from _lumascope.modules.finite_number import cap_stored_value


def test_a_value_past_either_end_applies_as_that_end_and_is_returned_beside_it():
    assert cap_stored_value(3000.0, 0.014, 1000.0) == (3000.0, 1000.0)
    assert cap_stored_value(0.001, 0.014, 1000.0) == (0.001, 0.014)
    assert cap_stored_value(5.0, 0.014, 1000.0) == (5.0, 5.0)


def test_an_unknown_bound_narrows_nothing():
    # No camera, or a driver that publishes no limit: a missing bound is
    # not a bound of zero, and inventing one would apply a limit no
    # hardware asked for.
    assert cap_stored_value(5.0, None, None) == (5.0, 5.0)


def test_a_whole_number_held_to_a_whole_number_ceiling_stays_one():
    # A count held to its ceiling is still a count: a sum of 150 capped at
    # 100 is the int 100, which a count's own check takes, where 100.0 is
    # refused as the wrong kind.
    value, applied = cap_stored_value(150, 1, 100)
    assert (value, applied) == (150, 100)
    assert type(applied) is int
