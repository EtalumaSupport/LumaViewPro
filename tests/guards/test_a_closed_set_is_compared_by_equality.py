# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A closed set's value is compared with ``==`` or ``in``, never ``is``.

A closed set is a ``StrEnum``, and a caller may hand in a member or its
plain string: an unmarked internal member takes ``'Z'`` as readily as
``Axis.Z``, and a record built before its judge ran, or a dict read back
from a file, holds the string. ``'lit' == LedState.LIT`` holds and
``'lit' is LedState.LIT`` does not, so an identity comparison against a
member answers False for the very value it names, silently. A snapshot
built with the plain word once restored no LED for exactly that reason.

The walk is over ``modules/``, ``ui/`` and ``rest/``: every comparison by
``is`` or ``is not`` whose side is a member of a ``StrEnum`` defined there,
by class or by the functional form.
"""

from __future__ import annotations

import ast

from tests.ast_seams import Packages, iter_package_modules

_PACKAGES = Packages({'modules', 'ui', 'rest'})


def _is_str_enum(base: ast.expr) -> bool:
    return (isinstance(base, ast.Name) and base.id == 'StrEnum') or (
        isinstance(base, ast.Attribute) and base.attr == 'StrEnum'
    )


def _str_enum_names(trees) -> set[str]:
    names = set()
    for _, tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(_is_str_enum(b) for b in node.bases):
                names.add(node.name)
            elif (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and _is_str_enum(node.value.func)
            ):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _identity_comparisons(trees, names: set[str]) -> list[str]:
    def member(side: ast.expr) -> bool:
        if not isinstance(side, ast.Attribute):
            return False
        owner = side.value
        return (isinstance(owner, ast.Name) and owner.id in names) or (
            isinstance(owner, ast.Attribute) and owner.attr in names
        )

    found = []
    for path, tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            sides = [node.left, *node.comparators]
            for index, op in enumerate(node.ops):
                if isinstance(op, ast.Is | ast.IsNot) and (
                    member(sides[index]) or member(sides[index + 1])
                ):
                    found.append(f'{path}:{node.lineno}: {ast.unparse(node)}')
    return sorted(found)


def test_no_closed_set_member_is_compared_by_identity():
    trees = list(iter_package_modules(sorted(_PACKAGES)))
    names = _str_enum_names(trees)
    assert 'LedState' in names, 'the walk no longer finds the string enums'
    found = _identity_comparisons(trees, names)
    assert found == [], 'compare a closed set with == or !=, not is:\n' + '\n'.join(found)
