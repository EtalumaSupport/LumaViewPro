# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A move on the plate is its own member, in plate mm, and moves only the axes it is given.

``MotionAPI.start_move_to_plate`` takes a ``PlateLocation``: X and Y in mm
on the selected plate, an axis that is None left where it is, a move under
way on it included. A Z is refused: a plate position has no Z. A well is
gone to by its label (``start_move_to_well``), and the bookmark by the axes
asked (``ScopeSession.start_go_to_bookmark``). ``move_absolute`` takes stage
microns only, so a position's unit never depends on another argument.
"""

import pytest

from modules.coord_transformations import PlateLocation
from modules.exceptions import ArgumentRefusedError


@pytest.fixture(scope='module')
def session(sim_turreted_session):
    return sim_turreted_session


def _stage_um(session, x_mm, y_mm):
    return session.scope.runtime_state.plate_to_stage(x_mm, y_mm)


def test_a_plate_move_of_x_alone_leaves_a_y_move_under_way(session):
    scope = session.scope
    motion = scope.motion
    halted = scope.simulation.hold_axis('Y')
    y_move = motion.start_move_absolute('Y', 60_000.0)
    try:
        assert halted.wait(10.0), 'the held Y move never halted'

        motion.move_to_plate(PlateLocation(x_mm=40.0, y_mm=None, z_um=None))

        assert not y_move.done(), 'the X move ended the Y move under way'
        assert motion.get_target_position('Y') == pytest.approx(60_000.0)
        assert motion.get_target_position('X') == pytest.approx(_stage_um(session, 40.0, 40.0).x_um)
    finally:
        scope.simulation.clear()
    y_move.result()


def test_a_move_to_a_well_lands_on_its_centre_and_an_unknown_label_is_refused(session):
    motion = session.scope.motion
    x_mm, y_mm = session.scope.runtime_state.get_labware().get_well_position(11, 7)

    motion.move_to_well('H12')

    target = _stage_um(session, x_mm, y_mm)
    assert motion.get_target_position('X') == pytest.approx(target.x_um)
    assert motion.get_target_position('Y') == pytest.approx(target.y_um)
    with pytest.raises(ArgumentRefusedError) as refused:
        motion.move_to_well('Z99')
    assert (refused.value.reason, refused.value.value) == ('well_unknown', 'Z99')
    assert motion.get_target_position('X') == pytest.approx(target.x_um)


def test_each_go_to_bookmark_axis_lands_on_the_bookmark_and_moves_no_other(session):
    motion = session.scope.motion
    motion.move_to_plate(PlateLocation(x_mm=30.0, y_mm=20.0, z_um=None))
    motion.move_absolute('Z', 3000.0)
    session.save_bookmark(('X', 'Y', 'Z'))
    elsewhere = _stage_um(session, 60.0, 50.0)
    marked = _stage_um(session, 30.0, 20.0)
    wrong = {}
    for axis, at_mark in (('X', marked.x_um), ('Y', marked.y_um), ('Z', 3000.0)):
        motion.move_to_plate(PlateLocation(x_mm=60.0, y_mm=50.0, z_um=None))
        motion.move_absolute('Z', 1000.0)

        session.go_to_bookmark((axis,))

        landed = {other: motion.get_target_position(other) for other in ('X', 'Y', 'Z')}
        expected = {'X': elsewhere.x_um, 'Y': elsewhere.y_um, 'Z': 1000.0, axis: at_mark}
        if landed != pytest.approx(expected):
            wrong[axis] = landed
    assert not wrong, wrong


def test_move_absolute_takes_stage_microns_only(session):
    with pytest.raises(TypeError):
        session.scope.motion.move_absolute('X', 40.0, frame='plate')
