"""Micrometre-to-microstep conversion rounds to nearest, not toward zero.

Truncation biased every conversion downward by up to one microstep, so a
typed position could be commanded short and the read-back could disagree
with what the user entered. The conversions round half-up instead.

These tests DERIVE their probe values from the board's own
``usteps_per_mm``: the value that distinguishes half-up from Python's
banker's ``round`` differs per axis resolution (250.0 um at 170666
usteps/mm, 150.0 at 80630), so a hardcoded constant silently stops
discriminating the moment a board reports a different resolution.
"""

import math

import pytest

from drivers.simulated_motorboard import SimulatedMotorBoard


@pytest.fixture
def board():
    return SimulatedMotorBoard()


def _first_tie_um(usteps_per_mm, axis_scale=1000.0):
    """Smallest positive um at a one-decimal step landing exactly on X.5 usteps.

    An exact .5 is the only place half-up and banker's rounding can differ,
    so it is the only value that tests which one is in use.
    """
    for tenth in range(1, 4_000_000):
        um = tenth / 10.0
        usteps = usteps_per_mm * um / axis_scale
        on_exact_tie = abs(usteps - math.floor(usteps) - 0.5) < 1e-9
        if on_exact_tie and round(usteps) != math.floor(usteps + 0.5):
            return um
    raise AssertionError('no discriminating tie value for this resolution')


@pytest.mark.parametrize('axis,conv', [('Z', 'z_um2ustep'), ('X', 'xy_um2ustep')])
def test_exact_tie_rounds_up_not_to_even(board, axis, conv):
    """Python's round() is banker's; a tie must go up, not to the even step."""
    usteps_per_mm = board.motorconfig.usteps_per_mm(axis)
    um = _first_tie_um(usteps_per_mm)
    exact = usteps_per_mm * um / 1000

    got = getattr(board, conv)(um)

    assert got == math.floor(exact + 0.5)
    assert got != round(exact), (
        f"{conv}({um}) returned the banker's result {got}; a tie must round up"
    )


@pytest.mark.parametrize('axis,conv', [('Z', 'z_um2ustep'), ('X', 'xy_um2ustep')])
def test_conversion_is_never_more_than_half_a_microstep_off(board, axis, conv):
    """Truncation could be a full microstep short; nearest is half, worst case."""
    usteps_per_mm = board.motorconfig.usteps_per_mm(axis)
    um_per_ustep = 1000.0 / usteps_per_mm

    worst = 0.0
    for tenth in range(0, 100_000, 7):
        um = tenth / 10.0
        commanded_um = getattr(board, conv)(um) * um_per_ustep
        worst = max(worst, abs(commanded_um - um))

    assert worst <= um_per_ustep / 2 + 1e-9, (
        f'{conv} is off by {worst} um, more than half a microstep'
    )


@pytest.mark.parametrize('axis,conv', [('Z', 'z_um2ustep'), ('X', 'xy_um2ustep')])
def test_error_is_not_biased_downward(board, axis, conv):
    """Truncation always rounded down; nearest must not favour one direction."""
    usteps_per_mm = board.motorconfig.usteps_per_mm(axis)
    um_per_ustep = 1000.0 / usteps_per_mm

    signed = [
        getattr(board, conv)(t / 10.0) * um_per_ustep - t / 10.0 for t in range(0, 100_000, 7)
    ]
    mean_error = sum(signed) / len(signed)

    assert abs(mean_error) < um_per_ustep / 10, (
        f'mean conversion error {mean_error} um is biased, not centred'
    )


def test_turret_degrees_are_deliberately_left_truncating(board):
    """The turret is excluded on purpose, and only stays safe while its
    callers pass exact quarter turns -- pin that so a new caller shows up
    here rather than as a silently short turret move."""
    usteps_per_90 = board.motorconfig.usteps_per_mm('T')

    for position in range(1, 5):
        degrees = 90 * (position - 1)
        exact = degrees * usteps_per_90 / 90.0
        assert exact == int(exact), (
            f'turret position {position} no longer lands on a whole microstep; '
            f'it must be converted with rounding like the linear axes'
        )
        assert board.t_deg2ustep(degrees) == exact
