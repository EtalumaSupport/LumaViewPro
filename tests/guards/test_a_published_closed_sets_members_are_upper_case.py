# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every closed set the API publishes spells its members in UPPER_CASE.

A Python caller names a member (``RunStatus.COMPLETED``,
``ZProjectMethod.MAX``); the names freeze with the API, so one spelling
holds across every set. ``ZProjectMethod`` once spelled its members as its
values (``ZProjectMethod.Max``), the one published set that did. A member's
value is the wire's word and is not judged here: ``ZProjectMethod.MAX`` is
``'Max'``.

The published sets are the enums the public face exports
(``lumascope.__all__``), which is every name the reference has a section
for.
"""

import enum

import lumascope


def _published_closed_sets():
    for name in lumascope.__all__:
        value = getattr(lumascope, name)
        if isinstance(value, type) and issubclass(value, enum.Enum):
            yield name, value


def test_every_published_closed_sets_members_are_upper_case():
    sets = list(_published_closed_sets())
    assert sets, 'the public face exports no enum; the walk found nothing to judge'
    offenders = {
        name: [member for member in cls.__members__ if not member.isupper()] for name, cls in sets
    }
    offenders = {name: members for name, members in offenders.items() if members}
    assert not offenders, (
        'a published closed set spells a member other than UPPER_CASE '
        f'(rename the member; its value, the wire word, stays): {offenders}'
    )
