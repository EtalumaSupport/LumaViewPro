# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The scope holds no copy of the configuration the settings store, and the Session never calls it under the settings lock.

The labware, stage offset, turret map, selected objective and whether the
scale bar is drawn are read from the session's settings; a field or a
setter for one of them on ``RuntimeState`` or ``ImagingAPI`` is the
hand-kept mirror coming back. The scope reads the settings under
``settings_lock``, a plain non-reentrant lock, so a Session member that
called into the scope while holding it would deadlock. Reading a plain
attribute of the scope (its layer identity) takes no lock; a call may.
"""

import ast

from tests.ast_seams import parse_module

_HELD_NOWHERE = {
    '_labware',
    '_stage_offset',
    '_turret_config',
    '_objective',
    '_objective_id',
    '_scale_bar',
}
_NO_SETTER = {
    'set_labware',
    'set_stage_offset',
    'set_turret_config',
    'set_objective',
    'set_scale_bar',
}


def _class(path: str, name: str) -> ast.ClassDef:
    tree = parse_module(path)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)


def _self_attributes(cls: ast.ClassDef) -> set[str]:
    return {
        node.attr
        for node in ast.walk(cls)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
        and isinstance(node.ctx, ast.Store)
    }


def _methods(cls: ast.ClassDef) -> set[str]:
    return {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_neither_sub_api_holds_or_sets_one():
    for path, name in (
        ('modules/lumascope_api/runtime_state.py', 'RuntimeState'),
        ('modules/lumascope_api/imaging.py', 'ImagingAPI'),
    ):
        cls = _class(path, name)
        assert not _self_attributes(cls) & _HELD_NOWHERE, name
        assert not _methods(cls) & _NO_SETTER, name


def _holds_the_settings_lock(node: ast.With) -> bool:
    return any(
        isinstance(item.context_expr, ast.Attribute) and item.context_expr.attr == 'settings_lock'
        for item in node.items
    )


def _reaches_the_scope(func: ast.expr) -> bool:
    node = func
    while isinstance(node, ast.Attribute):
        if node.attr == 'scope' and isinstance(node.value, ast.Name) and node.value.id == 'self':
            return True
        node = node.value
    return False


def test_no_session_member_calls_the_scope_under_the_settings_lock():
    tree = parse_module('modules/scope_session.py')
    calls = [
        inner.lineno
        for with_node in ast.walk(tree)
        if isinstance(with_node, ast.With) and _holds_the_settings_lock(with_node)
        for inner in ast.walk(with_node)
        if isinstance(inner, ast.Call) and _reaches_the_scope(inner.func)
    ]
    assert calls == []
