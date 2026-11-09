# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A GUI gesture that moves several axes is one API member, which opens its own command.

Each move submitted on its own is its own lane task, refused on the lane
when its axes do not know their position. A gesture issuing several of them
on an un-homed scope was refused once per move, after it had already moved
on; and a capture asked between two of them took the new X at the old Y.
A several-axis gesture is an API member that asks about its axes once and
opens ``MotionAPI.one_command`` itself (``start_move_to_plate``,
``start_center_on``, ``start_go_to_bookmark``), so REST and the SDK move
the same way. This holds the GUI to it: no function or lambda in ``ui/``
calls an X and Y move member, or more than one move member.
"""

import ast

from tests.ast_seams import MemberNames, iter_package_modules

_XY_MOVES = MemberNames(
    'src/_lumascope/modules/lumascope_api/motion.py',
    {'move_xy', 'move_xy_by', 'start_move_xy', 'start_move_xy_by'},
)
_Z_MOVES = MemberNames(
    'src/_lumascope/modules/lumascope_api/motion.py',
    {'move_z', 'move_z_by', 'start_move_z', 'start_move_z_by'},
)
_CALLABLES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _called_name(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _own_calls(fn):
    """Calls in the callable's own body, nested defs and lambdas excluded."""
    stack = [fn.body] if isinstance(fn, ast.Lambda) else list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (*_CALLABLES, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _offenders(tree, rel):
    """``'rel:line'`` for each callable that moves X and Y, or makes more than one move."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, _CALLABLES):
            continue
        moves = [
            name
            for call in _own_calls(node)
            if (name := _called_name(call)) in _XY_MOVES or name in _Z_MOVES
        ]
        if len(moves) > 1 or any(name in _XY_MOVES for name in moves):
            found.append(f'{rel}:{node.lineno} {"/".join(sorted(moves))}')
    return found


def test_the_scan_sees_a_lambda_and_a_def():
    """The known positives: an X and Y move in a lambda, two Z moves in a def."""
    tree = ast.parse(
        'def touch(motion):\n'
        '    submit(call=lambda: motion.start_move_xy_by(1.0, 2.0))\n'
        'def nudge(motion):\n'
        '    motion.start_move_z(1.0)\n'
        '    motion.start_move_z_by(2.0)\n'
        'def one(motion):\n'
        '    submit(call=lambda: motion.start_move_z(1.0))\n'
    )

    assert sorted(_offenders(tree, 'x.py')) == [
        'x.py:2 start_move_xy_by',
        'x.py:3 start_move_z/start_move_z_by',
    ]


def test_no_gui_callable_composes_a_several_axis_move():
    offenders = [hit for rel, tree in iter_package_modules(['ui']) for hit in _offenders(tree, rel)]

    assert offenders == [], (
        'these move X and Y, or several axes, from the GUI; a several-axis gesture is an '
        f'API member that opens one_command itself: {offenders}'
    )
