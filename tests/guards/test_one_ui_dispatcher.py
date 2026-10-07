# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The process has one UI dispatcher, held by kivy_utils, with one door.

Each lane, the listener bridge and kivy_utils once held a dispatcher of
their own, each given by a different caller. A session built over a caller's
scope gave its file, post-processing, worker-pool and diagnostics lanes none
at all, so on the GUI host their callbacks ran on the worker while the
scope's lanes delivered on the UI thread. One process has one UI thread, so
one store: ``kivy_utils``' ``_ui_dispatcher``, written only through
``ScopeSession.set_ui_dispatcher``. This walk, over production and tests
alike, refuses a second holder and a second writer.
"""

from __future__ import annotations

import ast
import re

from tests.ast_seams import iter_package_modules, production_modules, walk_defs

_STORE_MODULE = 'modules/kivy_utils.py'
_DOOR_MODULE = 'modules/scope_session.py'
_DOOR_DEF = 'ScopeSession.set_ui_dispatcher'
_WRITER = '_set_ui_dispatcher'
# The door's own name is how every caller reaches the store; the writer's
# name is policed by the second test.
_ALLOWED = frozenset({'set_ui_dispatcher', _WRITER})
# A name that holds one, not one that merely contains the letters (``gui_dispatcher``).
_HOLDER = re.compile(r'(?<![a-z])ui_dispatch')


def _every_module():
    yield from production_modules()
    yield from iter_package_modules(('drivers', 'tools', 'tests'))


def _names(node: ast.AST) -> list[str]:
    """The identifiers *node* binds or reads: a parameter, keyword, attribute, name or import."""
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.keyword):
        return [node.arg] if node.arg else []
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.alias):
        return [node.name.rsplit('.', 1)[-1], node.asname or '']
    return []


def test_no_module_but_kivy_utils_holds_a_dispatcher():
    found = []
    for rel_path, tree in _every_module():
        if rel_path == _STORE_MODULE:
            continue
        for node in ast.walk(tree):
            for name in _names(node):
                if _HOLDER.search(name) and name not in _ALLOWED:
                    found.append(f'{rel_path}:{getattr(node, "lineno", "?")} {name}')
    assert found == [], (
        'a UI dispatcher is held or passed outside kivy_utils; the process has one, '
        'set through ScopeSession.set_ui_dispatcher and read by kivy_utils.schedule_ui:\n'
        + '\n'.join(found)
    )


def test_only_the_session_door_writes_the_store():
    found = []
    for rel_path, tree in _every_module():
        inside = set()
        for qualname, node in walk_defs(tree.body):
            for inner in ast.walk(node):
                if id(inner) in inside or _WRITER not in _names(inner):
                    continue
                inside.add(id(inner))
                if (rel_path, qualname) != (_DOOR_MODULE, _DOOR_DEF):
                    found.append(f'{rel_path}:{inner.lineno} {qualname}')
        for inner in ast.walk(tree):
            if id(inner) not in inside and _WRITER in _names(inner):
                found.append(f'{rel_path}:{getattr(inner, "lineno", "?")} <module>')
    assert found == [], (
        f'kivy_utils.{_WRITER} is reached outside {_DOOR_DEF}; set the dispatcher '
        'through the door:\n' + '\n'.join(found)
    )
