# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A test fakes the clock of the module it tests, never the process's.

``mock.patch('modules.protocol_run_loop.time.sleep', fake)`` reads like a
patch of one module, but ``modules.protocol_run_loop.time`` IS the time
module, so the patch replaces ``time.sleep`` for every thread in the pytest
worker. A thread left over from an earlier test then sleeps on the fake: it
spins hot, and when the fake advances a clock, it advances the clock under
test -- the #239 cadence test read a 100 s interval as 131 s with one such
thread sleeping 1 ms, and 5292 s under a loaded suite. A frozen
``time.monotonic`` stops every other thread's deadline the same way.

The form that fakes one module's clock replaces the module's own name:
``mock.patch.object(module, 'time', SimpleNamespace(sleep=..., ...))``.
This guard refuses every spelling that sets an attribute on the time
module instead: a patch target naming ``<...>.time.<member>`` or
``time.<member>``, and ``patch.object`` / ``setattr`` whose object is
``time`` or ``<...>.time``.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT

# Calls whose first argument is a dotted target string, and calls whose
# first argument is the object an attribute is set on.
_TARGET_PATCHERS = frozenset({'patch', 'setattr'})
_ATTR_SETTERS = frozenset({'object', 'setattr'})


def _names_the_time_module(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == 'time'
    return isinstance(node, ast.Attribute) and node.attr == 'time'


def _target_is_a_time_member(target: str) -> bool:
    parts = target.split('.')
    return len(parts) >= 2 and parts[-2] == 'time'


def _offences(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
        first = node.args[0]
        names_a_member = (
            name in _TARGET_PATCHERS
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and _target_is_a_time_member(first.value)
        )
        sets_on_the_module = name in _ATTR_SETTERS and _names_the_time_module(first)
        if names_a_member or sets_on_the_module:
            lines.append(node.lineno)
    return lines


def test_no_test_patches_a_member_of_the_time_module():
    offenders = []
    for path in sorted((REPO_ROOT / 'tests').rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for lineno in _offences(tree):
            offenders.append(f'{path.relative_to(REPO_ROOT)}:{lineno}')
    assert not offenders, (
        "these patch the process's time module, so every thread sees the fake; "
        "replace the module's own name instead, "
        "mock.patch.object(module, 'time', SimpleNamespace(...)):\n" + '\n'.join(offenders)
    )
