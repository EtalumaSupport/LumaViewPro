# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every setting the template holds says what holds it, beyond its kind.

``settings_paths`` refuses a write by the leaf's kind and, where it has one,
its range. A leaf added without a rule took any finite number, and one
shipped null any single value, so a stored ``BF.acquire`` of 5 or a
``BF.focus`` of ``'abc'`` reached the code that reads it. Each leaf of the
template as the application reads it (``settings_init.load_template``) now
has a member, an installation-only or derived mark, a range, a switch's kind
or an owner that bounds it at use (``settings_paths.is_classified``), and
each leaf shipped null the kinds it holds when set; the failure names every
leaf that has none.
"""

from modules import settings_paths
from tests.settings_fixtures import shipped_template


def test_every_leaf_says_what_holds_it_and_every_null_leaf_its_kinds():
    leaves = list(settings_paths._leaf_paths(shipped_template()))
    unclassified = [path for path, _ in leaves if not settings_paths.is_classified(path)]
    null_without_kinds = [
        path
        for path, shipped in leaves
        if shipped is None and settings_paths.nullable_kinds_for(path) is None
    ]
    assert (unclassified, null_without_kinds) == ([], [])


def test_a_switch_is_a_leaf_shipped_true_or_false():
    # The kind is the whole range only for a bool; a number marked a switch
    # would take any value of its kind with no range.
    not_switches = [
        path
        for path, shipped in settings_paths._leaf_paths(shipped_template())
        if settings_paths._pattern_for(settings_paths.KIND_IS_THE_RANGE, path)
        and type(shipped) is not bool
    ]
    assert not_switches == []
