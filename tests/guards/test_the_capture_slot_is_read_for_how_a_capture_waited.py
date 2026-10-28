# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``ImagingAPI.last_capture_info`` is read only for how a capture waited.

A frame's own facts are its ``FrameRecord``, handed back with it
(``capture_still``). The slot once carried the record too, and a reader
that took it from there with ``.get('frame_record')`` got None the day it
left and reported the frame as having no record -- a plausible wrong
answer, not a failure (Goal 1). So every key read from the slot, by
subscript or ``.get``, through the member or a name bound to it in the
same function, is one the slot holds.
"""

from __future__ import annotations

import ast

from tests import ast_seams

# The member's docstring, ``Returns``: what a capture leaves, frame or not.
_SLOT_KEYS = frozenset({'hold_ms', 'drained', 'rechecks', 'deadline_s', 'n_entry', 'active_s'})


def _is_the_slot(node: ast.AST) -> bool:
    """The member itself, or ``(... .last_capture_info or {})``."""
    if isinstance(node, ast.BoolOp):
        return _is_the_slot(node.values[0])
    return isinstance(node, ast.Attribute) and node.attr == 'last_capture_info'


def _key_read(node: ast.AST) -> tuple[ast.AST, ast.AST] | None:
    """(the expression read from, the key) when ``node`` reads a key."""
    if isinstance(node, ast.Subscript):
        return node.value, node.slice
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'get'
        and node.args
    ):
        return node.func.value, node.args[0]
    return None


def _from_the_slot(expr: ast.AST, slot_names: set[str]) -> bool:
    return _is_the_slot(expr) or (isinstance(expr, ast.Name) and expr.id in slot_names)


def _offenders(tree: ast.Module) -> list[tuple[int, str]]:
    found = []
    scopes = [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ] + [tree]
    for scope in scopes:
        slot_names = {
            target.id
            for node in ast.walk(scope)
            if isinstance(node, ast.Assign) and _is_the_slot(node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(scope):
            read = _key_read(node)
            if read is None or not _from_the_slot(read[0], slot_names):
                continue
            key = read[1]
            if isinstance(key, ast.Constant) and key.value not in _SLOT_KEYS:
                found.append((node.lineno, repr(key.value)))
    return found


def test_every_key_read_from_the_capture_slot_is_one_it_holds():
    offenders = sorted(
        f'{path}:{line} reads {key}'
        for path in ast_seams.tracked_python_files()
        if path != 'tests/guards/test_the_capture_slot_is_read_for_how_a_capture_waited.py'
        for line, key in set(_offenders(ast_seams.parse_module(path)))
    )
    assert not offenders, (
        "last_capture_info holds how a capture waited; a frame's facts are the record "
        'capture_still hands back with it:\n' + '\n'.join(offenders)
    )
