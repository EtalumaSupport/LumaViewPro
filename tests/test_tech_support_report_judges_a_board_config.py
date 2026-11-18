# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A support report judges a board's motorconfig.json by the rules LVP reads it by.

The report's check was written for a schema no board has (``Model``,
``X Axis``, ``Steps Per mm``), so it warned on every real board's file and
judged none of its figures. It now judges each figure the file states by
the one table the defaults and the board merge are judged by
(``motorconfig.figure_problem``); a figure the file leaves out is not
judged, since a board's file states few.
"""

import ast
import json
import pathlib

import pytest

from _lumascope.modules.tech_support_report import validate_motorconfig

_CAPTURE = pathlib.Path(__file__).parent / 'data' / 'sim_wire_conformance_ls850t_field.json'


def _field_board_config() -> dict:
    """The CONFIG a field LS850T answered on the bench (a Python dict repr)."""
    capture = json.loads(_CAPTURE.read_text(encoding='utf-8'))
    reply = next(
        record['reply']
        for record in capture['runs'][0]['records']
        if record['board'] == 'motor' and record['command'] == 'CONFIG'
    )
    return ast.literal_eval(reply)


def test_a_real_boards_file_passes_with_nothing_said():
    judged = validate_motorconfig(json.dumps(_field_board_config()))

    assert (judged['valid'], judged['errors'], judged['warnings']) == (True, [], [])


@pytest.mark.parametrize(
    ('section', 'key', 'figure', 'says'),
    [
        (
            'Axis Microsteps per mm / Objective',
            'Z',
            0,
            'microsteps per mm 0 is not a whole number of 1 or more',
        ),
        ('Optics', 'PixelSize', -2.0, 'optics figure -2.0 is not a positive number'),
    ],
    ids=['zero-microsteps', 'negative-pixel-size'],
)
def test_a_figure_lvp_cannot_use_makes_the_file_invalid_naming_it(section, key, figure, says):
    board = _field_board_config()
    board[section][key] = figure

    judged = validate_motorconfig(json.dumps(board))

    assert (judged['valid'], judged['errors']) == (False, [f'{section}.{key}: {says}'])


def test_a_travel_pair_is_judged_only_when_the_file_states_both():
    board = _field_board_config()
    assert 'Axis Travel Margin' not in board, 'a field board states no margin'
    board['Axis Travel Margin'] = {'X': 120}

    judged = validate_motorconfig(json.dumps(board))

    assert judged['valid'] is False
    assert judged['errors'] == ['X travel limit: margin 120 mm leaves none of the travel 120 mm']
