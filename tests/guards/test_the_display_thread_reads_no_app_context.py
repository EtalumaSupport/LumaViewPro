# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The display thread renders through what it was started with, never the app context.

It used to look its widget and the scope up through a context provider the
GUI handed the session factory, so a GUI-agnostic module reached into the
GUI's app context on every frame and the factory carried a host-only
parameter for it. ``ScopeDisplay.start()`` now hands the widget to the
thread's ``start``; a name for the app context in the thread's module would
be the provider coming back.
"""

from __future__ import annotations

import ast

from tests.ast_seams import parse_module

_MODULE = 'modules/scope_display_thread.py'


def _names(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            yield node.lineno, node.arg
        elif isinstance(node, ast.Attribute):
            yield node.lineno, node.attr
        elif isinstance(node, ast.Name):
            yield node.lineno, node.id
        elif isinstance(node, ast.ImportFrom):
            yield node.lineno, node.module or ''
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name


def test_the_display_thread_names_no_app_context():
    found = [
        f'{_MODULE}:{lineno} {name}'
        for lineno, name in _names(parse_module(_MODULE))
        if 'app_context' in name or 'ctx' in name.lower().split('.')[-1].split('_')
    ]
    assert found == [], (
        'the display thread names the app context; it renders through the renderer '
        'its start() was given:\n' + '\n'.join(found)
    )
