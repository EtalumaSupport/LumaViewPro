# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A position off the plate names no well.

The well lookup clamped the nearest well's index into the grid, so any
position, however far off the plate, was named after an edge well: a
still taken half a plate away was saved as "H12". Within the grid a
position still names its nearest well. No shipped plate states a well
size, so "inside a well" cannot be judged; a position more than half a
pitch beyond the outer well centres is off the grid and names none.
"""

import pytest

from modules.labware_loader import WellPlateLoader


@pytest.fixture
def plate():
    return WellPlateLoader().get_plate('96 well microplate')


def _centre(plate, i, j):
    return plate.get_well_position(i, j)


def test_a_position_off_the_plate_has_no_well_index(plate):
    assert plate.get_well_index(500, 500) is None


def test_a_position_off_the_plate_is_labelled_as_no_well(plate):
    assert plate.get_well_label(500, 500) == ''


def test_the_index_is_plain_ints(plate):
    i, j = plate.get_well_index(*_centre(plate, 3, 2))
    assert (type(i), type(j)) == (int, int)
    assert (i, j) == (3, 2)


@pytest.mark.parametrize('fraction', [0.0, 0.25, 0.49])
def test_within_half_a_pitch_of_an_edge_well_names_it(plate, fraction):
    columns = plate.config['columns']
    x, y = _centre(plate, columns - 1, 0)
    x += fraction * plate.config['spacing']['x']
    assert plate.get_well_index(x, y) == (columns - 1, 0)


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_more_than_half_a_pitch_beyond_the_outer_centres_names_none(plate, axis):
    x, y = _centre(plate, 0, 0)
    pitch = plate.config['spacing'][axis]
    if axis == 'x':
        x -= 0.51 * pitch
    else:
        y -= 0.51 * pitch
    assert plate.get_well_index(x, y) is None


def test_a_plate_with_no_wells_names_no_well():
    """The Blank plate has no grid, so no position on it names a well.

    It once clipped every position to index (-1, -1), saved as '@0' in
    filenames and drawn as a well ring at the plate's origin, and then
    raised for any position, so every caller had to ask has_wells() first.
    A plate with no wells answers as a position off the grid does.
    """
    blank = WellPlateLoader().get_plate('Blank')

    assert blank.get_well_index(10.0, 10.0) is None
    assert blank.get_well_label(x=10.0, y=10.0) == ''
    assert blank.get_positions_with_labels() == []
