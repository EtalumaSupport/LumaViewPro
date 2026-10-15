# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A position event carries the axis's state and position as ``axis_positions`` answers them.

The position listener used to be called with ``(axis, position, state)``,
the position read from the cache and the state under another lock: during
a home it reported the last number the axis had, or 0.0, beside
``'homing'``, where ``axis_positions`` answers None -- and the GUI's Z
readout showed that number while Z homed. The listener is now called
with ``(axis, AxisPosition)`` from the one snapshot ``axis_positions``
reads, so an axis whose position is not known is reported with none.
"""

import itertools
import threading

import pytest

from modules.lumascope_api import AxisPosition, AxisState
from tests.scope_fakes import build_scope

KNOWN = (AxisState.IDLE, AxisState.MOVING)


@pytest.fixture
def scope():
    scope = build_scope(simulate=True)
    yield scope
    scope.motion._disconnect()


def _listen(scope):
    heard = []

    def listener(axis, at):
        heard.append((axis, at))

    scope.motion.add_position_listener(listener)
    return heard


def test_during_a_home_no_event_carries_a_position_the_axis_does_not_know(scope):
    scope.motion.home()
    scope.motion.move_absolute('X', 1500.0)
    heard = _listen(scope)

    scope.motion.home()

    homing = [(axis, at) for axis, at in heard if at.state == AxisState.HOMING]
    assert homing, f'no event during the home said homing: {heard}'
    for axis, at in heard:
        assert isinstance(at, AxisPosition), (axis, at)
        assert (at.position is None) == (at.state not in KNOWN), (axis, at)


def test_a_settled_move_reports_the_position_axis_positions_answers(scope):
    scope.motion.home()
    heard = _listen(scope)

    scope.motion.move_absolute('X', 1500.0)

    last_x = [at for axis, at in heard if axis == 'X'][-1]
    assert last_x == scope.motion.axis_positions()['X']
    assert last_x.state == AxisState.IDLE
    assert last_x.position == pytest.approx(1500.0, abs=1.0)


def test_a_poll_tells_its_listeners_once_every_moving_axis_is_read(scope):
    """The stage draws from ``axis_positions``, so a poll that told X before reading Y drew a staircase.

    The monitor read X, told the listeners, then read Y: the stage drew
    X's new position beside Y's from the poll before, and a diagonal move
    crossed the plate in steps (LS850T SN 12075, 2026-10-14). A poll now
    reads every moving axis before it tells any listener, so the monitor's
    X event and the Y event after it see one snapshot.
    """
    motion = scope.motion
    motion.home()
    # Realistic timing: the axes travel across many polls, as a stage does.
    motion._driver.set_timing_mode('realistic')
    seen = []

    def listener(axis, _at):
        if threading.current_thread().name != 'motion-monitor':
            return
        now = motion.axis_positions()
        if now['X'].state == now['Y'].state == AxisState.MOVING:
            seen.append((axis, now['X'].position, now['Y'].position))

    motion.add_position_listener(listener)
    x = motion.start_move_absolute('X', 60000.0)
    y = motion.start_move_absolute('Y', 40000.0)
    x.result()
    y.result()

    told_together = [
        (a[1:], b[1:]) for a, b in itertools.pairwise(seen) if (a[0], b[0]) == ('X', 'Y')
    ]
    assert len(told_together) >= 3, f'too few polls with both axes moving: {seen}'
    apart = [pair for pair in told_together if pair[0] != pair[1]]
    assert not apart, f'Y was read between the X and Y events of a poll: {apart}'
