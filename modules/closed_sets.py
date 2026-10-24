# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""A closed set's one judge: whether a value is one of its members, and the one refusal when not.

A closed set is a ``StrEnum``. A caller may give a member or its plain
string, and whoever takes it is handed the member. Every place a value of
one enters judges it here -- a member's parameter (``api_arguments``), a
field of a record as it is built (``api_surface.api_fields``), a value
inside a dict, a protocol file or the settings, by its owner -- so a set
is defined once, by its class, and a value outside it has one refusal:
``ArgumentRefusedError`` ``'not_offered'``, offering the set's values.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence

from modules.exceptions import ArgumentRefusedError


def member_of[E: enum.StrEnum](cls: type[E], value: object) -> E | None:
    """The member of ``cls`` that ``value`` is or spells; None for anything else.

    Exact, as the values are: ``'off'`` is not ``'Off'``.
    """
    if isinstance(value, cls):
        return value
    if isinstance(value, str):
        try:
            return cls(value)
        except ValueError:
            return None
    return None


def refuse_unless_offered[E: enum.StrEnum](
    cls: type[E], value: object, argument: str, *, within: Sequence[E] | None = None
) -> E:
    """The member of ``cls`` that ``value`` is or spells, refused unless it is one the taker offers.

    Args:
        cls: The set.
        value: What was given.
        argument: The argument's name for the refusal: a parameter, a
            record's ``Record.field``, a key's dotted path.
        within: The part of the set the taker offers, in the order a
            person reads it; None for the whole set.

    Raises:
        ArgumentRefusedError: ``'not_offered'``, offering the values the
            taker offers.
    """
    member = member_of(cls, value)
    offered = tuple(cls) if within is None else tuple(within)
    if member is None or member not in offered:
        raise ArgumentRefusedError(
            'not_offered', argument=argument, value=value, offered=tuple(m.value for m in offered)
        )
    return member
