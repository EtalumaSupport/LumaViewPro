# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An EL-0940's limits sit a margin inside its far ends, so no move they allow reaches one.

A probe drove an LS850T's X to the limit the API reported, the stated
travel, and the stage reached its far end with Z raised, into the
objective. At the bench the far switches stop X and Y short of the stated
travel (Firmware docs/reference/bench/el0940/2026-11-11_far_end_positions.md),
and a unit's calibration offset moves its switch nearer still. Each axis's
limit is now its stated travel less its `"Axis Travel Margin"`: a board's
own figure where it states a usable one, the shipped default otherwise.

On the production driver over the shipped motor firmware, whose simulated
far switches sit at the bench's span (`MotorBoardSpec.far_switch_mm`).
"""

import logging

import pytest

from _lumascope.drivers.sim_wire.backend import MotorBoardSpec
from tests.wire_boards import motor_on_the_wire

pytestmark = pytest.mark.firmware_tier

STATED_TRAVEL_UM = {'X': 120000.0, 'Y': 80000.0, 'Z': 14000.0}
# The shipped margin, 3 mm on X and Y and none on Z, written out so a
# defaults change fails here by name.
LIMIT_UM = {'X': 117000.0, 'Y': 77000.0, 'Z': 14000.0}


def _unit_config(**sections):
    """A stock LS850T's own motorconfig.json with ``sections`` merged over it."""
    config = MotorBoardSpec('LS850T', frozenset('XYZT')).motorconfig()
    for section, figures in sections.items():
        config.setdefault(section, {}).update(figures)
    return config


def test_each_axis_reaches_its_limit_with_its_far_switch_clear():
    board, _sim = motor_on_the_wire(dialect='3.0')
    board.home()

    for axis, limit in LIMIT_UM.items():
        assert board.axis_limits(axis) == {'min': 0.0, 'max': limit}
        board.move_abs_pos(axis, limit)
        assert board.wait_for_position(axis)
        assert board.current_pos(axis) == pytest.approx(limit, abs=0.1)
        assert board.limit_switch_status(axis) == (0, 0), axis


@pytest.mark.parametrize('axis', ['X', 'Y'])
def test_past_the_limit_the_far_switch_stops_the_stage_short_of_the_stated_travel(axis):
    """Driven to the stated travel with no limit asked (the driver checks
    none), the stage stops on its far switch, which lies past the limit."""
    board, _sim = motor_on_the_wire(dialect='3.0')
    board.home()

    board.move_abs_pos(axis, STATED_TRAVEL_UM[axis])
    board.wait_for_position(axis, timeout=1.0)

    assert board.limit_switch_status(axis) == (0, 1)
    assert LIMIT_UM[axis] < board.current_pos(axis) < STATED_TRAVEL_UM[axis]


def test_a_boards_own_travel_and_margin_set_its_limit():
    """An OEM stage shorter than the stock one keeps its own travel, and a
    board that states its own margin is held to it."""
    board, _sim = motor_on_the_wire(
        dialect='3.0',
        unit_config=_unit_config(
            **{'Axis Travel Limit': {'X': 100}, 'Axis Travel Margin': {'X': 1, 'Y': 2}}
        ),
    )

    assert board.axis_limits('X')['max'] == 99000.0
    assert board.axis_limits('Y')['max'] == 78000.0
    assert board.axis_limits('Z')['max'] == LIMIT_UM['Z']


def test_a_boards_figures_that_give_no_limit_are_dropped_for_the_defaults(caplog):
    """A board figure the limit cannot be computed from is dropped and the
    shipped figure kept, logged by name: alone (a negative travel, a word
    for a margin) or as the pair the merge would give (a margin as long as
    the travel: the board's margin goes first, then its travel)."""
    with caplog.at_level(logging.WARNING):
        board, _sim = motor_on_the_wire(
            dialect='3.0',
            unit_config=_unit_config(
                **{
                    'Axis Travel Limit': {'X': 2.0, 'Y': -5},
                    'Axis Travel Margin': {'X': 2.5, 'Y': 130, 'Z': 'banana'},
                }
            ),
        )

    assert {axis: board.axis_limits(axis)['max'] for axis in LIMIT_UM} == LIMIT_UM
    rejected = {
        r.getMessage().split('=')[0].removeprefix('[MotorConfig] Rejecting ')
        for r in caplog.records
        if 'Rejecting' in r.getMessage()
    }
    assert rejected == {
        'board value Axis Travel Limit.Y',
        'non-numeric board value Axis Travel Margin.Z',
        'board value Axis Travel Margin.Y',
        'board value Axis Travel Margin.X',
        'board value Axis Travel Limit.X',
    }


def test_a_z_only_board_gets_its_z_limit():
    board, _sim = motor_on_the_wire('LS820', 'Z', dialect='3.0')

    assert board.axis_limits('Z') == {'min': 0.0, 'max': LIMIT_UM['Z']}
