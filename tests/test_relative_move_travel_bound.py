"""A relative move whose target is outside travel is refused, not clamped.

The absolute path refused an out-of-travel target, but the relative path
never looked at travel: it handed the offset to the driver, which turned
it into an absolute move and clamped it to the limit. A jog past the top
of Z then reported success and left the stage at the limit, somewhere
nobody asked for -- the same wrong-place-with-a-clean-log the absolute
refusal exists to prevent. Every jog, scroll-to-focus, click-to-center
and autofocus step is a relative move.
"""

import pytest

from modules.exceptions import PositionOutOfRangeError
from tests.scope_fakes import build_scope, home_sim_scope


@pytest.fixture(scope='module')
def motion():
    """One homed simulated scope for the module: every test places its
    axis before it jogs, and a refused jog leaves the axis where it was."""
    scope = home_sim_scope(build_scope(simulate=True))
    scope._motion_driver.set_timing_mode('instant')
    yield scope.motion
    scope.disconnect()


def test_a_jog_past_the_top_of_travel_is_refused_and_nothing_moves(motion):
    motion.move_absolute('Z', 2000.0)
    z_max = motion.get_axis_limits('Z')['max']
    before = motion.get_current_position('Z')

    with pytest.raises(PositionOutOfRangeError):
        motion.move_relative('Z', z_max)

    assert motion.get_current_position('Z') == before


def test_a_jog_past_the_bottom_of_travel_is_refused_and_nothing_moves(motion):
    motion.move_absolute('X', 500.0)
    before = motion.get_current_position('X')

    with pytest.raises(PositionOutOfRangeError):
        motion.move_relative('X', -1000.0)

    assert motion.get_current_position('X') == before


def test_a_jog_inside_travel_still_moves(motion):
    motion.move_absolute('Z', 2000.0)

    motion.move_relative('Z', 1000.0)

    assert motion.get_current_position('Z') == pytest.approx(3000.0)


def test_a_jog_that_lands_exactly_on_the_limit_moves(motion):
    """The limit is inside travel, as the absolute refusal's bounds are."""
    z_max = motion.get_axis_limits('Z')['max']
    motion.move_absolute('Z', z_max - 100.0)

    motion.move_relative('Z', 100.0)

    assert motion.get_current_position('Z') == pytest.approx(z_max)
