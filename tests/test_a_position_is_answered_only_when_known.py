# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A position is answered only while the scope knows it, and a home does not
say "known" before it has read the position.

The position cache keeps the last number an axis reported after its
reference is lost, and until now a home marked every axis IDLE first and
read the board second, so a reader sampling at frame rate could pair
"known" with the pre-home number for the length of the serial
round-trips -- and a failed read wrote 0.0 into the cache under an IDLE
state, which every consumer reads as the origin. These tests pin the
one member that answers state and position together, the order a home
uses, and what a failed read leaves behind.

Every test composes a REAL ``Lumascope(simulate=True)`` and injects the
failure through the simulator's own path (``_fail_on`` makes
``exchange_command`` return None, which is what a dead board does), so
the driver's real error handling runs.
"""

import pytest

from modules.lumascope_api import AxisState, Lumascope


@pytest.fixture
def scope(monkeypatch):
    import modules.notification_center as nc

    errors = []
    monkeypatch.setattr(
        nc.notifications,
        'error',
        lambda category, title, message, **k: errors.append((category, title, message)),
    )
    scope = Lumascope(simulate=True)
    scope.notifications_seen = errors
    yield scope
    scope.motion._disconnect()


def _home_and_park_x(scope, x_um=1500.0):
    assert scope.motion.home() is True
    scope.motion.move_absolute('X', x_um, wait_until_complete=True)
    assert scope.motion.get_current_position('X') == pytest.approx(x_um, abs=1.0)


class TestTheSnapshot:
    def test_an_idle_axis_answers_its_position(self, scope):
        _home_and_park_x(scope)
        x = scope.motion.axis_positions()['X']
        assert x.state == AxisState.IDLE
        assert x.position == pytest.approx(scope.motion.get_current_position('X'))

    def test_an_unknown_axis_answers_no_position_while_the_cache_holds_one(self, scope):
        _home_and_park_x(scope)
        scope._motion_driver._fail_on.add('HOME')
        assert scope.motion.home() is False
        x = scope.motion.axis_positions()['X']
        assert x.state == AxisState.UNKNOWN
        assert x.position is None
        # The cache still holds the pre-loss number; the snapshot does not hand it out.
        assert scope.motion.get_current_position('X') == pytest.approx(1500.0, abs=1.0)

    def test_every_present_axis_is_answered(self, scope):
        assert set(scope.motion.axis_positions()) == set(scope.capabilities.axes)


class TestAHomeReadsBeforeItSaysKnown:
    def test_the_position_is_read_while_the_axis_is_still_homing(self, scope):
        driver = scope._motion_driver
        real_target_pos = driver.target_pos
        seen = []

        def spy(axis):
            seen.append((axis, scope.motion.get_axis_state(axis)))
            return real_target_pos(axis)

        driver.target_pos = spy
        assert scope.motion.home() is True
        assert seen, 'the home never read a position'
        assert {state for _, state in seen} == {AxisState.HOMING}
        assert all(
            scope.motion.get_axis_state(ax) == AxisState.IDLE for ax in scope.capabilities.axes
        )

    def test_a_z_home_reads_z_while_it_is_still_homing(self, scope):
        assert scope.motion.home() is True
        driver = scope._motion_driver
        real_target_pos = driver.target_pos
        seen = []

        def spy(axis):
            seen.append((axis, scope.motion.get_axis_state(axis)))
            return real_target_pos(axis)

        driver.target_pos = spy
        assert scope.motion.home(axis='Z') is True
        assert ('Z', AxisState.HOMING) in seen
        assert scope.motion.get_axis_state('Z') == AxisState.IDLE


class TestAFailedReadLeavesTheAxisUnknown:
    def test_after_a_full_home(self, scope):
        _home_and_park_x(scope)
        scope._motion_driver._fail_on.add('TARGET_RX')

        assert scope.motion.home() is False

        assert scope.motion.get_axis_state('X') == AxisState.UNKNOWN
        assert scope.motion.axis_positions()['X'].position is None
        # The cache entry is untouched: neither 0.0 nor the board's answer.
        assert scope.motion.get_current_position('X') == pytest.approx(1500.0, abs=1.0)
        assert scope.motion.get_axis_state('Y') == AxisState.IDLE
        assert scope.motion.get_axis_state('Z') == AxisState.IDLE
        assert any('could not be read' in message for _, _, message in scope.notifications_seen)

    def test_after_a_z_home(self, scope):
        assert scope.motion.home() is True
        scope._motion_driver._fail_on.add('TARGET_RZ')

        assert scope.motion.home(axis='Z') is False

        assert scope.motion.get_axis_state('Z') == AxisState.UNKNOWN
        assert scope.motion.get_axis_state('X') == AxisState.IDLE

    def test_after_a_turret_home_no_slot_is_recorded(self, scope):
        assert scope.motion.home() is True
        scope._motion_driver._fail_on.add('TARGET_RT')

        assert scope.motion.home(axis='T') is False

        assert scope.motion.get_axis_state('T') == AxisState.UNKNOWN
        assert scope.motion._last_turret_position is None
