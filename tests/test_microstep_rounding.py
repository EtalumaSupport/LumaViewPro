"""A position in microns converts to microsteps half-up, by its own axis's
row of the board's config, the same on every motor board.

Truncation biased every conversion downward by up to one microstep, so a
typed position could be commanded short and the read-back could disagree
with what the user entered; Python's ``round`` sends an exact tie to the
even microstep. The conversion rounds half-up instead, on the real board,
the simulator and the TMCM-6110 alike, since all three convert through
``motion_units``.

The probe values are DERIVED from each board's own ``usteps_per_mm``: the
value on which half-up and banker's rounding differ depends on the axis's
resolution, so a hardcoded constant silently stops discriminating the
moment a board reports a different one.
"""

import copy

import pytest

from _lumascope.drivers.simulated_motorboard import SimulatedMotorBoard
from _lumascope.drivers.simulated_tmcm6110 import SimulatedTmcm6110, SimulatedTmcm6110Backend
from _lumascope.drivers.tmcm6110 import Tmcm6110Board
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS
from tests.wire_boards import motor_on_the_wire

LINEAR_AXES = ('X', 'Y', 'Z')


def _real_board():
    board, _sim = motor_on_the_wire()
    return board


def _simulated_board():
    return SimulatedMotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)


def _tmcm_board():
    sim = SimulatedTmcm6110(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS)
    return Tmcm6110Board(
        motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=SimulatedTmcm6110Backend(sim)
    )


# Each board is its own driver with its own connect, so each is its own case.
@pytest.fixture(
    params=[
        pytest.param(_real_board, id='real', marks=pytest.mark.firmware_tier),
        pytest.param(_simulated_board, id='simulated'),
        pytest.param(_tmcm_board, id='tmcm6110'),
    ]
)
def board(request):
    driver = request.param()
    yield driver
    driver.disconnect()


def _ties(usteps_per_mm, count=2000):
    """Microns that land exactly on a half microstep where banker's
    rounding goes down: (k + 1/2) microsteps for even k, kept only where
    the float product is exactly the half, each with its half-up answer."""
    ties = []
    for k in range(0, 2 * count, 2):
        um = (k + 0.5) * 1000 / usteps_per_mm
        if usteps_per_mm * um / 1000 == k + 0.5:
            ties.append((um, k + 1))
    return ties


def test_an_exact_tie_rounds_up_not_to_even(board):
    """A tie goes up a microstep, where ``round`` would send it to the even
    one below."""
    wrong = []
    for axis in LINEAR_AXES:
        ties = _ties(board.motorconfig.usteps_per_mm(axis))
        assert ties, f'{axis}: no exact tie at this resolution to test with'
        wrong += [
            (axis, um, got, want)
            for um, want in ties
            if (got := board.position_to_steps(axis, um)) != want
        ]
    assert not wrong, f'ties not rounded half-up (axis, um, got, want): {wrong[:5]}'


def test_a_conversion_is_never_more_than_half_a_microstep_off(board):
    """Truncation could be a full microstep short; nearest is half, worst case."""
    for axis in LINEAR_AXES:
        um_per_ustep = 1000.0 / board.motorconfig.usteps_per_mm(axis)
        worst = max(
            abs(board.position_to_steps(axis, t / 10.0) * um_per_ustep - t / 10.0)
            for t in range(0, 100_000, 7)
        )
        assert worst <= um_per_ustep / 2 + 1e-9, (
            f'{axis} is off by {worst} um, more than half a microstep'
        )


def test_the_error_is_not_biased_downward(board):
    """Truncation always rounded down; nearest must not favour one direction."""
    for axis in LINEAR_AXES:
        um_per_ustep = 1000.0 / board.motorconfig.usteps_per_mm(axis)
        signed = [
            board.position_to_steps(axis, t / 10.0) * um_per_ustep - t / 10.0
            for t in range(0, 100_000, 7)
        ]
        mean_error = sum(signed) / len(signed)
        assert abs(mean_error) < um_per_ustep / 10, (
            f'{axis}: mean conversion error {mean_error} um is biased, not centred'
        )


def test_each_axis_converts_by_its_own_row_of_the_config():
    """Y is converted by Y's microsteps per mm, not by X's: a board whose
    two rows differ moves and reads Y by its own."""
    defaults = copy.deepcopy(SHIPPED_MOTOR_DEFAULTS)
    defaults['Axis Microsteps per mm / Objective']['Y'] = 40000
    board = SimulatedMotorBoard(motorconfig_defaults=defaults)
    try:
        assert board.position_to_steps('X', 1000) == 20157
        assert board.position_to_steps('Y', 1000) == 40000
        assert board.steps_to_position('Y', 40000) == 1000
    finally:
        board.disconnect()
