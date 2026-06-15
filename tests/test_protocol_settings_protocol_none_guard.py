# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Protocol-settings handlers must guard self._protocol before using it.

Bug
---
ProtocolSettings sets self._protocol from a scheduled _init_ui. _init_ui has a
failure path that returns without setting self._protocol when ctx never comes
up. Several user-facing handlers then dereferenced self._protocol with no guard,
so a silent init failure turned the first interaction (typing a period/duration,
stepping prev/next) into an AttributeError crash. Sibling handlers in the same
file (update_capture_root, step_name_validation) already used the
hasattr/None-guard pattern, showing the gap was an omission.

Fix
---
Guard update_period, update_duration, prev_step, and next_step with the same
`hasattr(self, '_protocol') and self._protocol is not None` check before the
first self._protocol.<...> call, returning early otherwise.

Test approach
-------------
These handlers are Kivy-bound (touch self.ids, _app_ctx.ctx) and cannot be
instantiated headless, so -- as with the other ProtocolSettings regression
tests in this suite -- the contract is locked structurally: parse the method
AST and assert a _protocol None-guard appears before the first
self._protocol.<attr> dereference. Fails on pre-fix source (no guard present).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'

GUARDED_HANDLERS = ['update_period', 'update_duration', 'prev_step', 'next_step']


def _method_node(class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(PROTOCOL_SETTINGS_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in source')


def _guard_linenos(method: ast.FunctionDef) -> list[int]:
    """Line numbers of `if` tests that reference self._protocol and None."""
    out = []
    for node in ast.walk(method):
        if isinstance(node, ast.If):
            dumped = ast.dump(node.test)
            if ("'_protocol'" in dumped or "attr='_protocol'" in dumped) and 'None' in dumped:
                out.append(node.lineno)
    return out


def _protocol_deref_linenos(method: ast.FunctionDef) -> list[int]:
    """Line numbers of `self._protocol.<attr>` accesses (the crashing use)."""
    out = []
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == '_protocol'
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == 'self'
        ):
            out.append(node.lineno)
    return out


@pytest.mark.parametrize('handler', GUARDED_HANDLERS)
def test_handler_guards_protocol_before_use(handler):
    method = _method_node('ProtocolSettings', handler)
    guards = _guard_linenos(method)
    derefs = _protocol_deref_linenos(method)

    assert derefs, f'{handler} no longer dereferences self._protocol -- update this test'
    assert guards, (
        f'{handler} dereferences self._protocol with no None-guard; a silent '
        f'_init_ui failure will crash on first interaction'
    )
    assert min(guards) < min(derefs), (
        f'{handler} guard (line {min(guards)}) must precede the first '
        f'self._protocol use (line {min(derefs)})'
    )
