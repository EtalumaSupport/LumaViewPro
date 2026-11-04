# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A GUI gesture that moves several axes is one task on the IO lane.

Each move submitted on its own is its own lane task, refused on the lane
when its axes do not know their position. A gesture issuing several of them
on an un-homed scope was refused once per move, after it had already moved
on; and a check asked on the GUI thread first could be overtaken by a home
or a stop before the moves ran. ``ui_helpers.submit_gesture`` asks the
motion API once and moves, in one task, under one command, so a capture
waits for the whole gesture; this keeps every several-axis gesture going
through it -- an X and Y request as much as several moves -- including the
next one.
"""

import ast

from tests.ast_seams import MemberNames, iter_package_modules

_XY_MOVES = MemberNames(
    'modules/lumascope_api/motion.py',
    {'move_xy', 'move_xy_by', 'start_move_xy', 'start_move_xy_by'},
)
_Z_MOVES = MemberNames(
    'modules/lumascope_api/motion.py', {'move_z', 'move_z_by', 'start_move_z', 'start_move_z_by'}
)


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


def _axes_of(call):
    """The axes a move asks for: Z for a Z move; for an X and Y one, each it is given."""
    name = _called_name(call)
    if name in _Z_MOVES:
        return {'Z'}
    if name not in _XY_MOVES:
        return set()
    axes = {'X', 'Y'}.intersection(('X', 'Y')[: len(call.args)])
    for kw in call.keywords:
        if kw.arg in ('x_um', 'dx_um'):
            axes.add('X')
        elif kw.arg in ('y_um', 'dy_um'):
            axes.add('Y')
    return axes


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
            if len(set().union(*(_axes_of(c) for c in _own_calls(fn)))) < 2:
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
    # The stage map's click is one call, ``MotionAPI.start_move_to_plate``,
    # which puts its axes on one lane task itself.
    assert gestures >= {'ui/scope_display.py::ScopeDisplay.touch'}


def test_every_several_axis_gesture_is_one_lane_task():
    offenders = [key for key, (_, submitted) in _several_axis_movers().items() if not submitted]

    assert offenders == [], (
        'these move several axes outside ui_helpers.submit_gesture, so each axis is '
        f'asked and refused on its own: {offenders}'
    )
