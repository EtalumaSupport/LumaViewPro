# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A position the board does not report is never answered with a number.

The drivers' position reads answered None on a failed exchange (the
simulator 0, the null board 0.0), and each reader took the stand-in for
a position: the Z move decided "not above the target" and skipped its
backlash leg, reporting arrived with the backlash taken the wrong way;
``get_actual_position`` answered 0.0. The reads now raise, and each
reader decides where it owns the decision.
"""

import time

import pytest

from drivers.exceptions import HardwareError
from modules.exceptions import HardwareCommandRefusedError, MoveNotCompletedError
from modules.lumascope_api.motion import AxisState
from tests.scope_fakes import build_scope, home_sim_scope


@pytest.fixture
def scope():
    scope = home_sim_scope(build_scope(simulate=True))
    yield scope
    scope.motion._disconnect()


@pytest.mark.parametrize(
    'read', ['current_pos', 'target_pos', 'current_pos_steps', 'target_pos_steps']
)
def test_a_failed_read_raises(scope, read):
    driver = scope._motion_driver
    register = 'ACTUAL_RX' if read.startswith('current') else 'TARGET_RX'
    driver._fail_on.add(register)
    with pytest.raises(HardwareError):
        getattr(driver, read)('X')


def test_a_z_move_whose_position_cannot_be_read_is_refused_with_nothing_written(scope):
    """Row 7: one failed ACTUAL_R skipped the backlash leg and the move
    reported arrived, approaching from the wrong side."""
    motion = scope.motion
    driver = scope._motion_driver
    motion.move_absolute('Z', 3000.0)
    sent = []
    real = driver.exchange_command

    def recorded(command, *args, **kwargs):
        sent.append(command)
        return real(command, *args, **kwargs)

    driver.exchange_command = recorded
    driver._fail_on.add('ACTUAL_RZ')

    with pytest.raises(HardwareCommandRefusedError) as exc:
        motion.move_absolute('Z', 1000.0, overshoot_enabled=True)

    assert exc.value.reason == 'position_unread'
    assert not [c for c in sent if c.startswith('TARGET_W')]
    assert motion.get_axis_state('Z') == AxisState.IDLE


def test_the_actual_position_raises_when_the_board_does_not_report_it(scope):
    scope._motion_driver._fail_on.add('ACTUAL_RZ')
    with pytest.raises(HardwareError):
        scope.motion.get_actual_position('Z')


def test_the_actual_position_is_refused_with_no_board(scope):
    scope.motion._disconnect()
    scope._motion_driver.disconnect()
    with pytest.raises(HardwareCommandRefusedError) as exc:
        scope.motion.get_actual_position('Z')
    assert exc.value.reason == 'not_connected'


def _move_x_and_lose_its_position(scope):
    """Start a long X move on the realistic simulator and fail every
    position read until the board has long reported it arrived."""
    motion = scope.motion
    driver = scope._motion_driver
    driver.set_timing_mode('realistic')
    handle = motion.start_move_absolute('X', 20000.0)
    driver._fail_on.add('ACTUAL_RX')
    deadline = time.monotonic() + 5.0
    while not driver.target_status('X'):
        assert time.monotonic() < deadline, 'the simulated move never arrived'
        time.sleep(0.02)
    time.sleep(0.2)  # ten monitor polls past the arrival
    return handle


def test_an_arrival_whose_position_was_not_read_is_not_written(scope):
    """The monitor wrote IDLE on the reached bit whatever its position read
    did, so the axis was 'known' at a number nobody read."""
    motion = scope.motion
    driver = scope._motion_driver
    handle = _move_x_and_lose_its_position(scope)

    assert motion.get_axis_state('X') == AxisState.MOVING

    driver._fail_on.discard('ACTUAL_RX')
    handle.wait()
    assert motion.get_axis_state('X') == AxisState.IDLE
    assert motion.get_current_position('X') == pytest.approx(20000.0, abs=0.1)


def test_an_arrival_never_read_is_given_up_as_that(scope, monkeypatch):
    motion = scope.motion
    monkeypatch.setattr(motion, '_MOTION_SETTLE_TIMEOUT_S', 1.0)
    real_wait = motion._wait_for_axis_to_stop
    monkeypatch.setattr(
        motion, '_wait_for_axis_to_stop', lambda axis, timeout_s: real_wait(axis, 5.0)
    )
    handle = _move_x_and_lose_its_position(scope)

    with pytest.raises(MoveNotCompletedError) as exc:
        handle.wait()

    assert exc.value.reason == 'position_unread'
    assert exc.value.title == 'Motor Position Unknown'
    assert motion.get_axis_state('X') == AxisState.UNKNOWN
    scope._motion_driver._fail_on.discard('ACTUAL_RX')
