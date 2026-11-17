# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every setting's range is a row: one of the module's refusers with its numbers.

``settings_paths._RANGES`` is the one table of the writer's ranges. A rule
written for one setting, whose body called a refuser with that setting's
numbers, was copied 25 times before the rows became ``functools.partial``
over the refusers; a new range is a new row. This guard refuses a row that
is a function written for its setting, a lambda, or a ``_judged`` row whose
judge is this module's own (the judge is another owner's; a rule of this
module's is a refuser). A rule that holds logic no refuser holds is named in
``_LOGIC_RULES``, so adding one is a reviewed change to this file.
"""

from __future__ import annotations

import functools

from _lumascope.modules import settings_paths

_REFUSERS = frozenset(
    {
        settings_paths._refuse_outside,
        settings_paths._refuse_below,
        settings_paths._refuse_non_count,
        settings_paths._refuse_unless_member,
        settings_paths._refuse_unknown_format,
        settings_paths._judged,
        settings_paths._or_none,
    }
)

# A rule with logic of its own: a level name in any case, a colour or no override.
_LOGIC_RULES = frozenset({settings_paths._log_level, settings_paths._display_color})


def _why_not_a_row(rule: object) -> str | None:
    """Why ``rule`` is not a row of a refuser, or None when it is."""
    if rule in _LOGIC_RULES:
        return None
    if not isinstance(rule, functools.partial):
        return f'{getattr(rule, "__name__", rule)!r} is not a partial of a refuser'
    if rule.func not in _REFUSERS:
        return f'a partial of {rule.func.__name__!r}, which is not a refuser'
    if rule.func is settings_paths._or_none:
        return _why_not_a_row(rule.args[0])
    if rule.func is settings_paths._judged:
        judge = rule.args[0]
        owner = getattr(
            judge.func if isinstance(judge, functools.partial) else judge, '__module__', None
        )
        if owner == settings_paths.__name__:
            return f"its judge {judge!r} is this module's own, not another owner's"
    return None


def test_every_range_is_a_row_of_a_refuser():
    offenders = {
        path: why for path, rule in settings_paths._RANGES.items() if (why := _why_not_a_row(rule))
    }
    assert not offenders, f'A range must be a row of a refuser in _RANGES: {offenders}'
