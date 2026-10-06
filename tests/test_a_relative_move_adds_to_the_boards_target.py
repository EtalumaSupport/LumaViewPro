# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A relative move adds its offset to the board's own target.

The API range-checked and published its own base -- the last move's
target while one was in flight, the position cache at rest -- while the
driver drove the board's target plus the offset. The two agree only
while the cache holds what the board says, and the cache can hold a
position the motion monitor never read. One number is now range-checked,
published and driven: the board's target plus the offset.
"""

import pytest

from modules.exceptions import MoveNotCompletedError, PositionOutOfRangeError
from tests.scope_fakes import build_scope, home_sim_scope


@pytest.fixture
def scope():
    scope = home_sim_scope(build_scope(simulate=True))
    yield scope
    scope.motion._disconnect()


def test_the_offset_is_added_to_the_boards_target_not_the_cache(scope):
    motion = scope.motion
    driver = scope._motion_driver
    motion.move_absolute('X', 5000.0)
    with motion._pos_cache_lock:
        motion._pos_cache['X'] = 0.0  # a number the board never reported

    motion.move_relative('X', 10.0)

    assert driver.target_pos('X') == pytest.approx(5010.0, abs=0.1)


def test_travel_is_checked_against_the_target_that_is_driven(scope):
    """With the board near the end of travel and the cache saying the
    origin, a jog past travel is refused; checking the cache let it drive
    beyond travel."""
    motion = scope.motion
    driver = scope._motion_driver
    x_max = motion.get_axis_limits('X')['max']
    motion.move_absolute('X', x_max - 50.0)
    with motion._pos_cache_lock:
        motion._pos_cache['X'] = 0.0

    with pytest.raises(PositionOutOfRangeError):
        motion.move_relative('X', 100.0)

    assert driver.target_pos('X') == pytest.approx(x_max - 50.0, abs=0.1)


def test_a_jog_during_a_move_adds_to_that_moves_target(scope):
    motion = scope.motion
    driver = scope._motion_driver
    driver.set_timing_mode('realistic')
    motion.start_move_absolute('X', 60000.0)

    motion.move_relative('X', 100.0)

    assert driver.target_pos('X') == pytest.approx(60100.0, abs=0.1)


def test_an_unreadable_target_drives_nothing(scope):
    motion = scope.motion
    driver = scope._motion_driver
    motion.move_absolute('X', 5000.0)
    driver._fail_on.add('TARGET_RX')

    with pytest.raises(MoveNotCompletedError) as exc:
        motion.move_relative('X', 10.0)

    assert exc.value.reason == 'driver_failed'
    driver._fail_on.discard('TARGET_RX')
    assert driver.target_pos('X') == pytest.approx(5000.0, abs=0.1)
