# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every ``@api`` member that takes an argument has a door that builds.

The door resolves a member's annotations at its first call and turns each
into a check. An annotation that names something no module defines, or a
form no check is written for, would fail that first call -- in a client's
hands. Building every door here makes it a build failure instead.
"""

import pytest

from modules import api_arguments, api_surface, wire_encoding


def _members_with_a_door():
    found = []
    for cls_name, cls in sorted(wire_encoding.project_classes().items()):
        for name, member in vars(cls).items():
            if api_surface.mark_of(member) is None or isinstance(member, property):
                continue
            function = api_surface._function_of(member)
            original = getattr(function, '__wrapped__', None)
            if original is None:
                continue
            found.append(
                pytest.param(
                    original, not isinstance(member, staticmethod), id=f'{cls_name}.{name}'
                )
            )
    return found


MEMBERS = _members_with_a_door()


def test_the_members_with_an_argument_are_found():
    assert len(MEMBERS) > 150


@pytest.mark.parametrize('function, bound', MEMBERS)
def test_the_door_builds(function, bound):
    api_arguments.door_for(function, bound=bound)
