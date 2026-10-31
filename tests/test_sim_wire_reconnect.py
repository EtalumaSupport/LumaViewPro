# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the API believes about the stage when the motor board's link fails,
against the real firmware.

A cable pulled mid-move: the axis must not stay MOVING forever, which
would wedge autofocus and a protocol run with no word to the user. The API
faults it to UNKNOWN after its disconnect deadline and says so once. A board
lost while every axis is idle reads as lost from the exchange that finds it.

After a reboot the firmware has lost its reference, and the API must stop
answering that it knows where the stage is, or a run is admitted at
positions that are not true.
"""

import functools
import time

import pytest

import drivers.sim_wire.backend as sim_backend
from fastapi.testclient import TestClient
from lib.lumascope_error import RefusalCause
from modules.exceptions import HardwareCommandRefusedError, HomingFailedError, MissingPart
from modules.lumascope_api._constants import AxisPosition, AxisState, TurretStatus
from modules.lumascope_api.diagnostics import MotorBoardInfo
from modules.notification_center import Severity
from modules.scope_session import ScopeSession
from rest.app import build_app
from tests.settings_fixtures import complete_settings

pytestmark = pytest.mark.firmware_tier


@pytest.fixture
def scope():
    session = ScopeSession.create(
        complete_settings(simulator_tier='firmware', microscope='LS850T'), simulate=True
    )
    try:
        session.scope.motion.home()
        assert session.scope.motion.axes_without_position() == {}
        yield session.scope
    finally:
        session.shutdown()


def test_after_a_reboot_the_api_no_longer_claims_to_know_the_stage(scope):
    """Nothing reconnects a rebooted board during a session: the move that
    meets the reboot is refused, every axis then reads unknown, and the next
    move is refused for a restart."""
    scope.simulation.reboot('motor')
    with pytest.raises(HardwareCommandRefusedError):
        scope.motion.move_z(100.0)

    assert scope.motion.axes_without_position() == dict.fromkeys('XYZT', AxisState.UNKNOWN)
    with pytest.raises(HardwareCommandRefusedError) as refused:
        scope.motion.move_z(100.0)
    assert (refused.value.reason, refused.value.cause) == ('not_connected', RefusalCause.RESTART)


@pytest.mark.slow
def test_a_cable_pulled_mid_move_faults_the_axis_within_the_deadline_and_says_so_once(
    monkeypatch, centre_posts
):
    # The pull must land while the stage is still travelling, so the board
    # runs in realistic timing: a session has no setting for it yet, so the
    # spec the scope builds its board from is given one.
    monkeypatch.setattr(
        sim_backend,
        'MotorBoardSpec',
        functools.partial(sim_backend.MotorBoardSpec, timing='realistic'),
    )
    session = ScopeSession.create(
        complete_settings(simulator_tier='firmware', microscope='LS850T'), simulate=True
    )
    try:
        motion = session.scope.motion
        motion.home()
        motion.start_move_xy(x_um=5000.0)  # about 55 mm from home: over a second of travel
        # MOVING from the start's return, so the pull lands with all of the
        # travel still ahead rather than what a sleep left of it.
        assert motion.is_any_axis_moving()

        # Read before the pull: the monitor's disconnect clock starts after
        # it, so the interval measured from here can only be the longer.
        pulled = time.monotonic()
        session.scope.simulation.unplug('motor')
        while motion.is_any_axis_moving():
            assert time.monotonic() - pulled < motion._DISCONNECT_FAULT_S + 2.0, 'X stayed MOVING'
            time.sleep(0.05)
        assert time.monotonic() - pulled >= motion._DISCONNECT_FAULT_S
        # The board is lost, so no axis knows its position, not only the one moving.
        assert motion.axes_without_position() == dict.fromkeys(('X', 'Y', 'Z', 'T'), 'unknown')
        errors = [(n.category, n.title) for n in centre_posts if n.severity == Severity.ERROR]
        assert errors == [('Motion', 'Motor Board Disconnected')]
    finally:
        session.shutdown()


@pytest.mark.parametrize('model', ('LS850T', 'LS720'))
def test_a_board_lost_while_every_axis_is_idle_reads_as_lost(model):
    """Once an exchange finds the board gone, every read of the axes says so:
    every axis unknown with no position, the turret unknown with no slot, the
    interlocks refused as a command is, the board's identity unknown."""
    session = ScopeSession.create(
        complete_settings(simulator_tier='firmware', microscope=model), simulate=True
    )
    try:
        scope = session.scope
        motion = scope.motion
        motion.home()
        scope.simulation.unplug('motor')
        with pytest.raises(HardwareCommandRefusedError):
            motion.move_z(100.0)  # the exchange that finds the board gone
        assert scope.motor_connected is False

        unknown = AxisPosition(AxisState.UNKNOWN, None)
        assert motion.axis_positions() == dict.fromkeys(('X', 'Y', 'Z'), unknown)
        assert session.status.axes == dict.fromkeys(('X', 'Y', 'Z'), unknown)
        turret = TurretStatus(AxisState.UNKNOWN, None) if 'T' in scope.capabilities.axes else None
        assert (motion.turret_status(), session.status.turret) == (turret, turret)
        assert motion.axes_without_position().keys() - {'T'}
        with pytest.raises(HardwareCommandRefusedError) as refused:
            motion.interlocks()
        assert (refused.value.reason, refused.value.missing) == (
            'not_connected',
            MissingPart.MOTOR_CONTROLLER,
        )
        assert scope.diagnostics.get_motor_info() == MotorBoardInfo(None, None, None, None, None)
    finally:
        session.shutdown()


# What finds each board's loss, and whether the scope says it is connected.
_LOSS = {
    'motor': (lambda scope: scope.motion.move_z(100.0), lambda scope: scope.motor_connected),
    'led': (lambda scope: scope.illumination.led_on('BF', 10.0), lambda scope: scope.led_connected),
}


@pytest.mark.parametrize('board', ('led', 'motor'))
def test_a_lost_board_stays_lost_through_a_replug_and_a_diagnostic(scope, board):
    """Nothing reconnects a board during a session, a diagnostic included: a
    diagnostic sent to it reached the serial layer's reopen, which brought the
    board back under the API with every position held from before the loss."""
    finds_it, connected = _LOSS[board]
    scope.simulation.unplug(board)
    with pytest.raises(HardwareCommandRefusedError):
        finds_it(scope)
    scope.simulation.replug(board)

    assert scope.diagnostics.send_diagnostic_command(board, 'INFO') == 'Board not connected'
    assert connected(scope) is False
    with pytest.raises(HardwareCommandRefusedError) as refused:
        finds_it(scope)
    assert (refused.value.reason, refused.value.cause) == ('not_connected', RefusalCause.RESTART)


def test_over_the_wire_a_lost_board_is_a_restart_refusal():
    session = ScopeSession.create(
        complete_settings(simulator_tier='firmware', microscope='LS850T'), simulate=True
    )
    try:
        session.scope.motion.home()
        session.scope.simulation.unplug('motor')
        with pytest.raises(HardwareCommandRefusedError):
            session.scope.motion.move_z(100.0)  # the exchange that finds the board gone
        with TestClient(build_app(session)) as client:
            answer = client.get('/api/v1/scope/motion/get_actual_position', params={'axis': 'Z'})
        assert (answer.status_code, answer.json()['reason']) == (503, 'not_connected')
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


def _pulled(scope) -> None:
    scope.simulation.unplug('motor')


# parametrize-justified: a cable pull is permanent for its scope
@pytest.mark.parametrize('home', list(_HOMES))
def test_a_home_after_a_cable_pull_fails_at_once(scope, home):
    # #632: a disconnected motor must not hold the caller while the driver
    # waits out its timeouts and reconnect attempts.
    _pulled(scope)
    started = time.monotonic()
    with pytest.raises((HomingFailedError, HardwareCommandRefusedError)):
        _HOMES[home](scope.motion)
    assert time.monotonic() - started < 0.5


_HOMING_FAULT = pytest.mark.xfail(
    strict=True,
    reason='the first home after a pull meets the dead port as a failed command and raises '
    'HomingFailedError, a homing fault. Only the next call knows the motor is disconnected',
)


# parametrize-justified: a cable pull is permanent for its scope. A turret
# home meets the dead port in its Z park, whose outcome leaves as itself.
@pytest.mark.parametrize(
    'home',
    [pytest.param('all', marks=_HOMING_FAULT), 'T', pytest.param('Z', marks=_HOMING_FAULT)],
)
def test_a_home_after_a_cable_pull_says_the_motor_is_not_connected(scope, home):
    _pulled(scope)
    with pytest.raises(HardwareCommandRefusedError) as raised:
        _HOMES[home](scope.motion)
    assert raised.value.reason == 'not_connected'
