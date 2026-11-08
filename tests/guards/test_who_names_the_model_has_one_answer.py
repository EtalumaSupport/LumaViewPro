# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Whether the board named the scope's model has one answer: ``ChannelIdentity.model_named_by_board``.

It is decided once, by the identity resolver (``channel_record``), where the
board's name is weighed against the operator's selection. A second answer --
the field set anywhere else, or read off something other than the identity --
can disagree with the model the scope runs as, and then the model line and
REST's ``model_at_next_start`` name a start that does not happen. This walk
over every production module, drivers, tools and the REST host included,
refuses the field set outside the resolver's module and read off anything
but the identity.
"""

from __future__ import annotations

import ast

from tests.ast_seams import Words, iter_package_modules, production_modules

_OWNER_MODULE = 'src/_lumascope/modules/channel_record.py'
_FIELD = 'model_named_by_board'
# The names the identity is read through: the scope's attribute and its local alias.
_IDENTITY = Words({'channel_identity', 'identity'})


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(('src/_lumascope/drivers', 'tools', 'src/_lumascope/rest'))


def _off_the_identity(node: ast.Attribute) -> bool:
    receiver = node.value
    name = receiver.attr if isinstance(receiver, ast.Attribute) else getattr(receiver, 'id', None)
    return name in _IDENTITY


def _second_answers(rel_path: str, tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == _FIELD:
            yield f'{rel_path}:{node.value.lineno}: sets {_FIELD}'
        elif (
            isinstance(node, ast.Attribute) and node.attr == _FIELD and not _off_the_identity(node)
        ):
            yield f'{rel_path}:{node.lineno}: {ast.unparse(node)}'


def test_no_module_but_the_resolver_answers_who_named_the_model():
    found = [
        line
        for rel_path, tree in _all_production_modules()
        if rel_path != _OWNER_MODULE
        for line in _second_answers(rel_path, tree)
    ]
    assert not found, (
        f'{_FIELD} set or read outside the channel identity; ask '
        f'`channel_identity.{_FIELD}`:\n' + '\n'.join(found)
    )
