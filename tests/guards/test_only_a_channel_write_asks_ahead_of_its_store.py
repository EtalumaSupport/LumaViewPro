# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Only the Session's channel writes ask the camera for values ahead of the store.

``CameraAPI.hold_channel_writing`` puts a channel on the camera at the values
a write is about to store, and records the camera as holding that channel. A
caller that did not then store them would leave the record naming a channel
at values it never had: a still of that channel would be taken, and named,
at settings nobody stored. So its only callers are the two writes that store
what they asked for once it returns, ``ScopeSession._write_channels`` and
``ScopeSession.update_settings``. Every reference counts, a call, an alias, a
lambda or a ``functools.partial`` argument, under the outermost member it sits
in. Not seen: a ``getattr`` by string, and the engineering plugin, which runs
in process from the Firmware repo.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, tracked_files

_MEMBER = 'hold_channel_writing'
_CALLERS = frozenset({'ScopeSession._write_channels', 'ScopeSession.update_settings'})
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _references(tree: ast.AST) -> list[tuple[int, str]]:
    """(line, outermost member's qualname) of every attribute reference to the member."""
    found = []

    def walk(node, prefix, member):
        if isinstance(node, ast.ClassDef) and member is None:
            prefix = f'{prefix}{node.name}.'
        elif isinstance(node, _DEFS) and member is None:
            member = f'{prefix}{node.name}'
        if isinstance(node, ast.Attribute) and node.attr == _MEMBER:
            found.append((node.lineno, member or '<module>'))
        for child in ast.iter_child_nodes(node):
            walk(child, prefix, member)

    walk(tree, '', None)
    return found


def test_the_walk_sees_a_call_an_alias_a_lambda_and_a_partial():
    """The instrument reports each form it must see, before its silence counts."""
    tree = ast.parse(
        'import functools\n'
        'class Elsewhere:\n'
        '    def call(self):\n'
        "        self.scope.camera.hold_channel_writing('BF', {})\n"
        '    def alias(self):\n'
        '        ask = self.scope.camera.hold_channel_writing\n'
        '    def deferred(self):\n'
        "        return lambda: camera.hold_channel_writing('BF', {})\n"
        '    def partial(self):\n'
        '        return functools.partial(camera.hold_channel_writing, "BF")\n'
    )
    assert _references(tree) == [
        (4, 'Elsewhere.call'),
        (6, 'Elsewhere.alias'),
        (8, 'Elsewhere.deferred'),
        (10, 'Elsewhere.partial'),
    ]


def test_only_the_channel_writes_ask_ahead_of_their_store():
    files = [REPO_ROOT / 'lumaviewpro.py']
    files += [REPO_ROOT / rel for rel in tracked_files(('src', 'ui', 'tools'))]
    references = {
        f'{path.relative_to(REPO_ROOT)}:{line} {member}': member
        for path in files
        for line, member in _references(ast.parse(path.read_text(encoding='utf-8')))
    }
    assert set(references.values()) == _CALLERS, (
        f'only {sorted(_CALLERS)} ask the camera ahead of their store; found {sorted(references)}'
    )
