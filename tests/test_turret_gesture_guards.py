# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The double-press guard and the interaction record belong to the CLICK,
not to the turret move.

While both sat on ``turret_select``, they applied to every caller of it.
``@debounce`` keys on the instance plus the method name, so one window was
shared by the four turret buttons, the step-navigation path, the XY home
and the protocol lane: a click within half a second of a run's turret move
dropped the RUN's move, while X, Y and Z proceeded to the step's
coordinates and the capture was taken through whatever glass was mounted.
The record had the same fault mirrored -- a startup sequence wrote 14
``BUTTON TURRET_POS_N`` entries for presses nobody made, which is a support
bundle telling an investigator the user did something they did not do.

These are source seams rather than behavioural calls because the subject is
a Kivy widget method and the wiring under test is a ``.kv`` binding; what
must hold is WHICH function carries the guards and WHICH one the buttons
reach.
"""

import ast

import pytest

from tests.ast_seams import REPO_ROOT, direct_call_names, find_def, parse_module


VERTICAL_CONTROL = 'ui/vertical_control.py'


def _decorator_names(fn):
    names = []
    for dec in fn.decorator_list:
        node = dec.func if isinstance(dec, ast.Call) else dec
        names.append(node.attr if isinstance(node, ast.Attribute) else getattr(node, 'id', ''))
    return names


@pytest.fixture
def gesture():
    fn = find_def(VERTICAL_CONTROL, 'turret_gesture', class_name='VerticalControl')
    assert fn is not None, 'the gesture entry point is what the buttons bind to'
    return fn


@pytest.fixture
def select():
    fn = find_def(VERTICAL_CONTROL, 'turret_select', class_name='VerticalControl')
    assert fn is not None, 'every turret caller still routes through turret_select'
    return fn


def test_the_gesture_absorbs_a_double_press(gesture):
    assert 'debounce' in _decorator_names(gesture)


def test_the_gesture_records_the_press(gesture):
    assert 'button' in direct_call_names(gesture)


def test_the_gesture_does_the_turret_work_through_the_shared_path(gesture):
    """It is a thin entry point, not a second implementation."""
    assert 'turret_select' in direct_call_names(gesture)


def test_the_move_is_not_debounced(select):
    """A run's turret move must not be droppable by a user's click."""
    assert 'debounce' not in _decorator_names(select)


def test_the_move_writes_no_interaction_record(select):
    """Program-initiated motion is not a press, so the bundle must not
    report one. No ``protocol`` test guards it any more -- the record is
    simply not written here."""
    assert 'button' not in direct_call_names(select)


def test_every_turret_button_reaches_the_gesture():
    """The guards moving without the bindings moving would leave the
    buttons both unrecorded and unprotected."""
    kv = (REPO_ROOT / 'ui/lumaviewpro.kv').read_text()

    for slot in (1, 2, 3, 4):
        assert f'root.turret_gesture({slot})' in kv
    assert 'root.turret_select(' not in kv


def _names_used(rel_path):
    """Every attribute and plain name the module mentions."""
    used = set()
    for node in ast.walk(parse_module(rel_path)):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    return used


@pytest.mark.parametrize('rel_path', ['ui/ui_helpers.py', 'ui/motion_settings.py'])
def test_the_program_still_calls_the_undecorated_path(rel_path):
    """The callers that are not a person must not be routed at the
    gesture: that would hand them back the window and the false record."""
    used = _names_used(rel_path)

    assert 'turret_select' in used
    assert 'turret_gesture' not in used
