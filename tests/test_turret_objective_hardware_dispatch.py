# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Turret assign/reset buttons must push the config to the microscope.

Bug
---
set_turret_objective and reset_turret_objective updated
settings['turret_objectives'] but never called set_turret_config(), so the
assignment/clear persisted to settings while the microscope was never synced.
Hardware dispatch happened only in select_objective() (the spinner path), not
on these buttons -- so the buttons silently did half their job.

Fix
---
After the settings write, both handlers now call
scope.runtime_state.set_turret_config(turret_config=settings['turret_objectives'])
guarded by motion.has_turret(), mirroring select_objective().

Test approach
-------------
The handlers are Kivy-bound (self.ids, _app_ctx.ctx) and cannot run headless,
so -- as with the other vertical/protocol UI regression tests -- the contract
is locked structurally: assert each handler's AST contains a set_turret_config
call. Fails on pre-fix source (no such call). Hardware behavior itself is
bench-validated.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
VERTICAL_CONTROL_SRC = REPO / 'ui' / 'vertical_control.py'

DISPATCH_HANDLERS = ['set_turret_objective', 'reset_turret_objective']


def _method_node(class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(VERTICAL_CONTROL_SRC.read_text())
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
    method = _method_node('VerticalControl', handler)
    assert _calls_attr(method, 'set_turret_config'), (
        f'{handler} writes settings but never calls set_turret_config(); the '
        f'microscope never syncs the turret assignment'
    )
