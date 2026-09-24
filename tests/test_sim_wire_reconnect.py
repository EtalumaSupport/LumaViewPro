# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the API believes about the stage when the motor board's link fails,
against the real firmware.

A cable pulled mid-move: the axis must not stay MOVING forever, which
would wedge autofocus and a protocol run with no word to the user. The API
faults it to UNKNOWN after its disconnect deadline and says so once.

When a command fails on the wire the driver closes its port, and its next
command reconnects without the disconnect path. After a cable pull that is
right: the board stayed powered, its firmware is still homed, and the API
still knows every position. After a reboot it is not: the firmware has lost
its reference, and the API must stop answering that it knows where the
stage is, or a run is admitted at positions that are not true.

The reboot case fails today: the reconnect keeps the API's axis states and
the driver's homing latch (the triage track holds the fix). ``strict=True``
turns it red once the fix lands, so the marker is removed with it.
"""

import functools
import re
import sys
import time

import pytest

import drivers.sim_wire.backend as sim_backend
import modules.notification_center as notification_center

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )


@pytest.fixture
def scope():
    session = ScopeSession.create(
        complete_settings(simulator_tier='firmware', microscope='LS850T'), simulate=True
    )
    try:
        assert session.scope.motion.home()
        assert session.scope.motion.axes_without_position() == {}
        yield session.scope
    finally:
        session.shutdown()


def _silent_reconnect(scope) -> dict[str, bool]:
    """The driver meets the dropped link on its next command, closes, and
    reconnects on the one after. Returns what the firmware says is homed."""
    driver = scope._motion_driver
    assert driver.exchange_command('INFO') is None
    fullinfo = driver.exchange_command('FULLINFO')
    return {axis: value == 'True' for axis, value in re.findall(r'(\w) homed: (\w+)', fullinfo)}


def test_after_a_cable_pull_the_firmware_and_the_api_both_still_know_the_stage(scope):
    board = scope._motion_driver._backend.motor_board
    board.unplug()
    board.replug()
    assert _silent_reconnect(scope) == {'X': True, 'Y': True, 'Z': True, 'T': True}
    assert scope.motion.axes_without_position() == {}
    assert scope.motion.has_homed()


@pytest.mark.xfail(
    strict=True,
    reason='the silent auto-reconnect keeps the axis states and the homing latch '
    'across a board reboot; the triage track holds the fix',
)
def test_after_a_reboot_the_api_no_longer_claims_to_know_the_stage(scope):
    scope._motion_driver._backend.motor_board.reboot()
    assert _silent_reconnect(scope) == {'X': False, 'Y': False, 'Z': False, 'T': False}
    assert set(scope.motion.axes_without_position()) == {'X', 'Y', 'Z', 'T'}
    assert not scope.motion.has_homed()


def test_a_cable_pulled_mid_move_faults_the_axis_within_the_deadline_and_says_so_once(
    monkeypatch,
):
    # The pull must land while the stage is still travelling, so the board
    # runs in realistic timing: a session has no setting for it yet, so the
    # spec the scope builds its board from is given one.
    monkeypatch.setattr(
        sim_backend,
        'MotorBoardSpec',
        functools.partial(sim_backend.MotorBoardSpec, timing='realistic'),
    )
    errors = []
    monkeypatch.setattr(
        notification_center.notifications,
        'error',
        lambda category, title, message, **kwargs: errors.append((category, title)),
    )
    session = ScopeSession.create(
        complete_settings(simulator_tier='firmware', microscope='LS850T'), simulate=True
    )
    try:
        motion = session.scope.motion
        assert motion.home()
        motion.move_absolute('X', 5000.0)  # about 55 mm from home: over a second of travel
        time.sleep(0.3)
        assert motion.is_moving()

        session.scope._motion_driver._backend.motor_board.unplug()
        pulled = time.monotonic()
        while motion.is_moving():
            assert time.monotonic() - pulled < motion._DISCONNECT_FAULT_S + 2.0, 'X stayed MOVING'
            time.sleep(0.05)
        assert time.monotonic() - pulled >= motion._DISCONNECT_FAULT_S
        assert motion.axes_without_position() == {'X': 'unknown'}
        assert errors == [('Motion', 'Motor board disconnected')]
    finally:
        session.shutdown()


# A home after a cable pull: the firmware twin of the NullMotionBoard tests in
# test_lumascope_api.py, which model a board that was never there. Here a
# production board loses its port mid-session, and its next command meets it.

_HOMES = {
    'all': lambda motion: motion.home(),
    'T': lambda motion: motion.home(axis='T'),
    'Z': lambda motion: motion.home(axis='Z'),
}


def _pulled(scope, monkeypatch) -> list:
    errors = []
    monkeypatch.setattr(
        notification_center.notifications,
        'error',
        lambda category, title, message, **kwargs: errors.append(title),
    )
    scope._motion_driver._backend.motor_board.unplug()
    return errors


@pytest.mark.parametrize('home', list(_HOMES))
def test_a_home_after_a_cable_pull_fails_at_once(scope, monkeypatch, home):
    # #632: a disconnected motor must not hold the caller while the driver
    # waits out its timeouts and reconnect attempts.
    _pulled(scope, monkeypatch)
    started = time.monotonic()
    assert _HOMES[home](scope.motion) is False
    assert time.monotonic() - started < 0.5


@pytest.mark.xfail(
    strict=True,
    reason='the first home after a pull meets the dead port as a failed command and says '
    "'Homing Error', a homing fault; a turret home also says the safety Z move failed. "
    'Only the next call knows the motor is disconnected',
)
@pytest.mark.parametrize('home', list(_HOMES))
def test_a_home_after_a_cable_pull_says_the_motor_is_not_connected(scope, monkeypatch, home):
    errors = _pulled(scope, monkeypatch)
    _HOMES[home](scope.motion)
    assert errors == ['Motor Not Connected']
