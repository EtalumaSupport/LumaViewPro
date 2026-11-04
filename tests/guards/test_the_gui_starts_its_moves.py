# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI starts a move; it never waits on one.

A person's gesture -- a jog, a scroll-wheel tick, a click on the stage or
the image, a typed position -- returns at once, and a move that fails on
the way is the motion monitor's to report. ``scope.motion.move_xy``,
``move_xy_by``, ``move_z``, ``move_z_by``, ``move_to_plate`` and
``move_to_well`` wait for the axes to arrive, so a gesture that
called them would hold the IO lane for the whole travel, serialise X and
Y, and make each scroll tick wait for the last. The GUI calls the start
members (``start_move_xy``, ``start_move_xy_by``, ``start_move_z``,
``start_move_z_by``, ``start_move_to_plate``); a Session
member that waits, such as ``go_to_step``, is submitted off the lane.

The scan finds every call in ``ui/`` of a waited move member on an
expression that names the motion API (``motion.move_xy(...)``,
``ctx.scope.motion.move_z_by(...)``). The gesture wrappers in
``ui/ui_helpers.py`` are plain functions, called by bare name, and are
not matched.
"""

import ast

from tests.ast_seams import MemberNames, iter_package_modules

_WAITED = MemberNames(
    'modules/lumascope_api/motion.py',
    {'move_xy', 'move_xy_by', 'move_z', 'move_z_by', 'move_to_plate', 'move_to_well'},
)


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
                and node.func.attr
                in {
                    'start_move_xy',
                    'start_move_xy_by',
                    'start_move_z',
                    'start_move_z_by',
                    'start_move_to_plate',
                }
                and ast.unparse(node.func.value).split('.')[-1] == 'motion'
            ):
                started.append(rel)
    assert {'ui/scope_display.py', 'ui/ui_helpers.py'} <= set(started)


def test_the_gui_goes_to_a_step_without_waiting_for_the_stage():
    """A click on a step is a gesture too: ``start_go_to_step``, not the
    waiting ``go_to_step``. Waiting made rapid Next clicks travel to every
    step in turn, and made the person's own scroll during the travel a
    'superseded' failure popup."""
    found = []
    for rel, tree in iter_package_modules(['ui']):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'go_to_step'
                and ast.unparse(node.func.value).split('.')[-1] == 'session'
            ):
                found.append(f'{rel}:{node.lineno}')
    assert found == []
