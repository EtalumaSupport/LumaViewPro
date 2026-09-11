# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The turret slot writers must push the config to the microscope.

Bug
---
The GUI's assign and reset handlers updated settings['turret_objectives']
but never called set_turret_config(), so the assignment/clear persisted to
settings while the microscope was never synced. Hardware dispatch happened
only on the objective-selection path, so the buttons silently did half
their job.

Fix
---
The slot writers now live on the Session (assign_turret_objective and
clear_turret_objective) and each pushes
scope.runtime_state.set_turret_config(settings['turret_objectives']) after
the settings write, for every host.

Test approach
-------------
The contract is locked structurally: assert each writer's AST contains a
set_turret_config call. Hardware behavior itself is bench-validated.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCOPE_SESSION_SRC = REPO / 'modules' / 'scope_session.py'

DISPATCH_HANDLERS = ['assign_turret_objective', 'clear_turret_objective']


def _method_node(class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(SCOPE_SESSION_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in source')


def _calls_attr(method: ast.FunctionDef, attr: str) -> bool:
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
        ):
            return True
    return False


@pytest.mark.parametrize('handler', DISPATCH_HANDLERS)
def test_handler_dispatches_turret_config(handler):
    method = _method_node('ScopeSession', handler)
    assert _calls_attr(method, 'set_turret_config'), (
        f'{handler} writes settings but never calls set_turret_config(); the '
        f'microscope never syncs the turret assignment'
    )
