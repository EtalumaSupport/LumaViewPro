# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated motor answers a failed position read the way the board does.

The real board answers None when a target-position read fails, and its
relative move refuses on it. The simulator answered 0 -- a number the
layer above cannot tell from the origin -- so the whole failure class was
unreachable in the simulator and every consumer's handling of it went
untested there.
"""

import pytest

from drivers.exceptions import HardwareError
from modules.lumascope_api import Lumascope


@pytest.fixture
def scope():
    scope = Lumascope(simulate=True)
    yield scope
    scope.motion._disconnect()


class TestTheSimulatedBoardFailsLikeTheRealOne:
    def test_a_failed_position_read_answers_none(self, scope):
        scope._motion_driver._fail_on.add('TARGET_RX')
        assert scope._motion_driver.target_pos('X') is None

    def test_a_relative_move_without_a_readable_target_is_refused(self, scope):
        scope._motion_driver._fail_on.add('TARGET_RX')
        with pytest.raises(HardwareError):
            scope._motion_driver.move_rel_pos('X', 10.0)
