# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A position the board does not report is never answered with a number.

The drivers' position reads answered None on a failed exchange (the
simulator 0, the null board 0.0), and each reader took the stand-in for
a position: the Z move decided "not above the target" and skipped its
backlash leg, reporting arrived with the backlash taken the wrong way;
``get_actual_position`` answered 0.0. The reads now raise, and each
reader decides where it owns the decision.
"""

import pytest

from drivers.exceptions import HardwareError
from modules.exceptions import HardwareCommandRefusedError
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
