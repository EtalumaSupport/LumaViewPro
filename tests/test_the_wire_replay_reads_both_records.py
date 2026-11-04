# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The wire replay reads a move's record in the words it was taken in.

A recorded bench session is what that bench did. Its move records were
taken when the plugin moved one axis by name (``move_absolute(X, 123.000)``);
a record taken since names the move it asked for (``move_xy(x_um=...)``,
``move_z(...)``). The replay reads both, so a recording is kept as recorded.
"""

import pytest

from tests.sim_wire_bench import _target


@pytest.mark.parametrize(
    ('command', 'move'),
    [
        ('move_absolute(X, 60000.000)', ('X', 60000.0)),
        ('move_absolute(Z, -12.500)', ('Z', -12.5)),
        ('move_xy(x_um=60000.000)', ('X', 60000.0)),
        ('move_xy(y_um=40000.000)', ('Y', 40000.0)),
        ('move_z(5000.000)', ('Z', 5000.0)),
    ],
)
def test_a_move_record_names_its_axis_and_target(command, move):
    assert _target(command) == move


def test_a_record_that_names_no_move_is_refused():
    with pytest.raises(AssertionError):
        _target('move_turret(2)')
