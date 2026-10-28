# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every ``@api`` member that takes an argument has a door that builds.

The door resolves a member's annotations at its first call and turns each
into a check. An annotation that names something no module defines, or a
form no check is written for, would fail that first call -- in a client's
hands. Building every door here makes it a build failure instead. The
members are the declared API's (``tests.marked_set``): a module's marked
function among them, whose door takes every parameter as an argument.
"""

from modules import api_arguments, api_surface
from tests.marked_set import marked_set


def _members_with_a_door():
    found = []
    for label, marked in sorted(marked_set().items()):
        member = marked.value
        if marked.field or isinstance(member, property):
            continue
        original = getattr(api_surface._function_of(member), '__wrapped__', None)
        if original is None:
            continue
        bound = not marked.at_top_level and not isinstance(member, staticmethod)
        found.append((label, original, bound))
    return found


MEMBERS = _members_with_a_door()


def test_the_members_with_an_argument_are_found():
    assert len(MEMBERS) > 150
    assert 'rest.app.build_app' in {label for label, _, _ in MEMBERS}, 'the walk lost the functions'


def test_every_door_builds():
    failed = {}
    for label, function, bound in MEMBERS:
        try:
            api_arguments.door_for(function, bound=bound)
        except Exception as ex:  # the defect under test: any door that does not build
            failed[label] = f'{type(ex).__name__}: {ex}'
    assert not failed, f'a door does not build: {failed}'
