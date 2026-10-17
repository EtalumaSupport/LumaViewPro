# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A press the debounce drops is recorded, and a press with other arguments is not a double-press.

Every rejected or dropped user press is recorded (K17): the handlers write
their own record inside the method, so a drop in the wrapper must write one
itself. The window is one per set of arguments, so the turret's four slot
buttons, which share one method, do not swallow each other's presses.
"""

import logging

import pytest

from modules.debounce import debounce


class _Turret:
    def __init__(self):
        self.moves = []

    @debounce(60.0)
    def turret_gesture(self, slot):
        self.moves.append(slot)


@pytest.fixture
def records():
    """The interaction log's lines, read from its own logger: the record is the product."""
    lines = []

    class _Catch(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    log = logging.getLogger('LVP.gui_interactions')
    handler = _Catch()
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    yield lines
    log.removeHandler(handler)
    log.setLevel(old_level)


def test_a_press_inside_the_window_is_dropped_and_recorded(records):
    turret = _Turret()

    turret.turret_gesture(2)
    turret.turret_gesture(2)

    assert turret.moves == [2]
    assert records == ['BUTTON _Turret.turret_gesture dropped args=(2,)']


def test_a_press_with_other_arguments_runs(records):
    turret = _Turret()

    turret.turret_gesture(2)
    turret.turret_gesture(3)

    assert turret.moves == [2, 3]
    assert records == []
