# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI starts a move; it never waits on one.

A person's gesture -- a jog, a scroll-wheel tick, a click on the stage or
the image, a typed position -- returns at once, and a move that fails on
the way is the motion monitor's to report. ``scope.motion.move_absolute``
and ``move_relative`` wait for the axis to arrive, so a gesture that
called them would hold the IO lane for the whole travel, serialise X and
Y, and make each scroll tick wait for the last. The GUI calls the start
members (``start_move_absolute`` / ``start_move_relative``); a Session
member that waits, such as ``go_to_step``, is submitted off the lane.

The scan finds every call in ``ui/`` of a waited move member on an
expression that names the motion API (``motion.move_absolute(...)``,
``ctx.scope.motion.move_relative(...)``). The gesture wrappers in
``ui/ui_helpers.py`` share those names but are plain functions, called
by bare name, and are not matched.
"""

import ast

from tests.ast_seams import iter_package_modules

_WAITED = frozenset({'move_absolute', 'move_relative'})


def _waited_motion_calls():
    found = []
    for rel, tree in iter_package_modules(['ui']):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _WAITED
                and ast.unparse(node.func.value).split('.')[-1] == 'motion'
            ):
                found.append(f'{rel}:{node.lineno} {ast.unparse(node.func)}')
    return found


def test_the_gui_calls_no_waited_move_member():
    assert _waited_motion_calls() == []


def test_the_scan_sees_the_gesture_moves_it_was_written_for():
    """The start members are matched by the same shape, so a refactor that
    hid the gestures from the scan is noticed."""
    started = []
    for rel, tree in iter_package_modules(['ui']):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {'start_move_absolute', 'start_move_relative'}
                and ast.unparse(node.func.value).split('.')[-1] == 'motion'
            ):
                started.append(rel)
    assert {'ui/stage.py', 'ui/scope_display.py', 'ui/ui_helpers.py'} <= set(started)
