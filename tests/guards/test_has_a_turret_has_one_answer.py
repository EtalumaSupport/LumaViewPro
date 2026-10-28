# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Whether the scope has a turret has one answer: ``capabilities.has_turret``.

It is decided once, at the scope's construction (``scope_capabilities``):
the board's axes when a board came up, else the declared model's catalogue
entry. A second answer -- a model's entry read elsewhere, a value recorded at
bring-up -- differs from it exactly when the motor board did not come up,
where a turret scope then answers with its stored objective instead of the
one in the light path. This walk over every production module, drivers,
tools and the REST host included, refuses a ``has_turret`` read outside the
capabilities' own module that is not read off the capabilities.
"""

from __future__ import annotations

import ast

from tests.ast_seams import Words, iter_package_modules, production_modules

_OWNER_MODULE = 'modules/scope_capabilities.py'
# The names the capabilities are read through: the scope's attribute and its local alias.
_CAPABILITIES = Words({'capabilities', 'caps'})


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(('drivers', 'tools', 'rest'))


def _off_the_capabilities(node: ast.Attribute) -> bool:
    receiver = node.value
    name = receiver.attr if isinstance(receiver, ast.Attribute) else getattr(receiver, 'id', None)
    return name in _CAPABILITIES


def test_no_module_but_the_capabilities_answers_has_a_turret():
    found = [
        f'{rel_path}:{node.lineno}: {ast.unparse(node)}'
        for rel_path, tree in _all_production_modules()
        if rel_path != _OWNER_MODULE
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == 'has_turret'
        and not _off_the_capabilities(node)
    ]
    assert not found, (
        'has_turret read off something other than the scope capabilities; ask '
        '`capabilities.has_turret`:\n' + '\n'.join(found)
    )
