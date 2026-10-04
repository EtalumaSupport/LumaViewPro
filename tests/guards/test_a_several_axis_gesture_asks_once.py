# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A GUI gesture that moves several axes is one task on the IO lane.

Each ``move_absolute`` / ``move_relative`` submitted on its own is its own
lane task, refused on the lane when its axis does not know its position. A
gesture issuing two or three of them on an un-homed scope was refused once
per axis, after it had already moved on; and a check asked on the GUI thread
first could be overtaken by a home or a stop before the moves ran.
``ui_helpers.submit_gesture`` asks the motion API once and moves, in one
task; this keeps every several-axis gesture going through it, including the
next one.
"""

import ast

from tests.ast_seams import iter_package_modules

_MOVES = frozenset({'move_absolute', 'move_relative'})


def _called_name(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _own_calls(fn):
    """Calls in the function's own body, nested defs excluded."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _axis_of(call):
    """The axis literal a move names, positionally or as ``axis=``."""
    candidates = list(call.args[:1]) + [kw.value for kw in call.keywords if kw.arg == 'axis']
    for value in candidates:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _defs_with_parents(tree):
    """``(qualname, def, enclosing def or None)`` for every def, however nested."""
    found = []

    def visit(node, prefix, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f'{prefix}{child.name}.', enclosing)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f'{prefix}{child.name}'
                found.append((qualname, child, enclosing))
                visit(child, f'{qualname}.', child)
            else:
                visit(child, prefix, enclosing)

    visit(tree, '', None)
    return found


def _submits_as_gesture(enclosing, fn):
    """True when *enclosing* hands *fn* to ``submit_gesture`` as its moves."""
    if enclosing is None:
        return False
    return any(
        _called_name(call) == 'submit_gesture'
        and any(
            kw.arg == 'moves' and isinstance(kw.value, ast.Name) and kw.value.id == fn.name
            for kw in call.keywords
        )
        for call in _own_calls(enclosing)
    )


def _several_axis_movers():
    """``{'path::qualname': (enclosing def's qualname, submitted as a gesture)}``."""
    found = {}
    for rel, tree in iter_package_modules(['ui']):
        if rel == 'ui/ui_helpers.py':
            continue  # the move helpers themselves
        defs = _defs_with_parents(tree)
        names = {id(node): qualname for qualname, node, _ in defs}
        for qualname, fn, enclosing in defs:
            moves = [c for c in _own_calls(fn) if _called_name(c) in _MOVES]
            if len({_axis_of(c) for c in moves} - {None}) < 2:
                continue
            found[f'{rel}::{qualname}'] = (
                names.get(id(enclosing)),
                _submits_as_gesture(enclosing, fn),
            )
    return found


def test_the_gestures_are_the_ones_known():
    """A new several-axis gesture is caught by the rule below; this pins that the
    scan still sees the ones it was written against, so a refactor that hides a
    gesture from it (a variable axis, a new helper) is noticed."""
    gestures = {
        f'{key.split("::")[0]}::{enclosing}'
        for key, (enclosing, _) in _several_axis_movers().items()
    }
    assert gestures >= {
        'ui/scope_display.py::ScopeDisplay.touch',
        'ui/stage.py::Stage.on_touch_down',
    }


def test_every_several_axis_gesture_is_one_lane_task():
    offenders = [key for key, (_, submitted) in _several_axis_movers().items() if not submitted]

    assert offenders == [], (
        'these move several axes outside ui_helpers.submit_gesture, so each axis is '
        f'asked and refused on its own: {offenders}'
    )
