# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A GUI gesture that moves several axes asks, once, before it moves any.

Each ``move_absolute`` / ``move_relative`` is its own fire-and-forget task,
refused on the motion lane when its axis does not know its position. A
gesture issuing two or three of them on an un-homed scope was refused once
per axis, after it had already moved on, and the notification centre's dedup
showed one popup naming X. The motion API answers the whole gesture at once
(``refuse_unknown_positions``, through ``ui_helpers.unknown_position_refused``);
this keeps every several-axis gesture asking it, including the next one.
"""

import ast

from tests.ast_seams import iter_package_modules, walk_defs

_MOVES = frozenset({'move_absolute', 'move_relative'})


def _own_calls(fn):
    """Calls in the function's own body, nested defs excluded."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _axis_of(call):
    """The axis literal a move names, positionally or as ``axis=``."""
    candidates = list(call.args[:1]) + [kw.value for kw in call.keywords if kw.arg == 'axis']
    for value in candidates:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _several_axis_gestures():
    """``{'path::qualname': (first move line, [ask lines])}`` for every such gesture."""
    found = {}
    for rel, tree in iter_package_modules(['ui']):
        if rel == 'ui/ui_helpers.py':
            continue  # the move helpers themselves
        for qualname, fn in walk_defs(tree.body):
            calls = list(_own_calls(fn))
            moves = [c for c in calls if c.func.id in _MOVES]
            axes = {_axis_of(c) for c in moves} - {None}
            if len(axes) < 2:
                continue
            asks = [c.lineno for c in calls if c.func.id == 'unknown_position_refused']
            found[f'{rel}::{qualname}'] = (min(c.lineno for c in moves), asks)
    return found


def test_the_gestures_are_the_ones_known():
    """A new several-axis gesture is caught by the rule below; this pins that the
    scan still sees the ones it was written against, so a refactor that hides a
    gesture from it (a variable axis, a new helper) is noticed."""
    assert set(_several_axis_gestures()) >= {
        'ui/scope_display.py::ScopeDisplay.touch',
        'ui/stage.py::Stage.on_touch_down',
        'ui/step_navigation.py::go_to_step',
    }


def test_every_several_axis_gesture_asks_before_it_moves():
    offenders = [
        key
        for key, (first_move, asks) in _several_axis_gestures().items()
        if not asks or min(asks) > first_move
    ]

    assert offenders == [], (
        'these move several axes without first asking the motion API whether the '
        f'scope knows where they are (ui_helpers.unknown_position_refused): {offenders}'
    )
