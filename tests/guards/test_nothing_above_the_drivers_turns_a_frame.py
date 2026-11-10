# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Nothing above the camera drivers turns a frame; the drivers turn one through ``copy_oriented`` alone.

Every frame a driver delivers has row 0 at the top of the picture
(``OwedReverse`` in ``_lumascope/drivers/camera.py``), so every client and
file agrees without knowing a rule. The rule used to be each consumer's:
the saver and the recorder each flipped every frame, and five clients that
did not got it upside down. A turn written above the drivers is that rule
coming back, and a second turn in a driver is a second way to orient one.

A turn is a flip (``np.flip``, ``np.flipud``, ``np.fliplr``, ``cv2.flip``),
a rotation (``np.rot90``, ``cv2.rotate``), or a subscript that steps back
(``::-1``) through an array's first or second axis. A step back through a
later axis, or after ``...``, reverses an image's colour channels, which
turns nothing. The simulated sensors (``simulated_*.py``) are hardware
stand-ins: their readout reverses as a sensor's does, and is not a driver's
copy.
"""

from __future__ import annotations

import ast

from tests.ast_seams import Words, parse_module, tracked_files

_DRIVERS = 'src/_lumascope/drivers/'
_TURNING_CALLS = Words({'flip', 'flipud', 'fliplr', 'rot90', 'rotate'})
# The one helper a driver turns a frame in, by file and function.
_THE_HELPER = ('src/_lumascope/drivers/camera.py', 'copy_oriented')


def _steps_back(index: ast.AST) -> bool:
    return (
        isinstance(index, ast.Slice)
        and isinstance(index.step, ast.UnaryOp)
        and isinstance(index.step.op, ast.USub)
        and isinstance(index.step.operand, ast.Constant)
        and index.step.operand.value == 1
    )


def _turns(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        return isinstance(func, ast.Attribute) and func.attr in _TURNING_CALLS
    if isinstance(node, ast.Subscript):
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        for position, index in enumerate(indices[:2]):
            if isinstance(index, ast.Constant) and index.value is Ellipsis:
                return False
            if _steps_back(index):
                return position < 2
    return False


def _turns_in(rel_path: str):
    """``(line, enclosing function)`` for each turn in the file."""
    tree = parse_module(rel_path)
    found = []

    def visit(node, function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if _turns(node):
            found.append((node.lineno, function))
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, None)
    return found


def test_no_module_above_the_drivers_and_no_driver_copy_turns_a_frame():
    offenders = []
    for rel_path in tracked_files():
        if rel_path.startswith('tests/'):
            continue
        in_drivers = rel_path.startswith(_DRIVERS)
        if in_drivers and rel_path.rsplit('/', 1)[-1].startswith('simulated_'):
            continue
        for line, function in _turns_in(rel_path):
            if (rel_path, function) == _THE_HELPER:
                continue
            offenders.append(f'{rel_path}:{line} ({function})')
    assert not offenders, (
        'a frame turned outside copy_oriented; every driver delivers row 0 at the top, '
        'and nothing above the drivers turns a frame:\n' + '\n'.join(offenders)
    )


def test_the_helper_is_where_the_drivers_turn_a_frame():
    """The exemption names a function that turns: renamed or moved, the guard says so."""
    path, function = _THE_HELPER
    assert function in {f for _line, f in _turns_in(path)}, _THE_HELPER
