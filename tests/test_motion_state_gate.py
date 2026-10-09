# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An axis whose position is unknown must refuse to move, not drive blind.

Three producers already set an axis to UNKNOWN -- a failed home, the
disconnect fault, the stall fault -- and nothing consumes any of them.
The result is #702's cascade: home fails, the axes are correctly marked
UNKNOWN, and the very next commanded move drives against a reference
frame the software already knows is invalid. #709 Half B is the same
defect one layer down: a dead-board target write returns None instead of
raising, so a move that never happened reports success to every layer
above.

This file pins the consumer. The invariant, in one sentence: no
commanded move reaches the driver while its axis is UNKNOWN unless the
caller passed ``force=True``, and a driver-side move failure lands the
axis back in UNKNOWN rather than leaving a stale IDLE.

Every test runs the production motor driver against the real board
firmware (the firmware simulator tier) and makes the hardware fail: a
stalled X motor, so the firmware's own home fails, and a pulled cable,
which is what a dead board is to the driver. The driver's and the API's
real error handling runs on the firmware's real failure.
"""

import sys
import time

import pytest

from drivers.exceptions import HardwareError
from drivers.sim_wire.mp import tmc5072
from modules.exceptions import (
    AxisStateUnknownError,
    HardwareCommandRefusedError,
    HomingFailedError,
    MoveNotCompletedError,
)
from modules.lumascope_api import AxisState
from modules.notification_center import Severity
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )


def _errors_posted(centre_posts):
    return [(n.category, n.title, n.message) for n in centre_posts if n.severity == Severity.ERROR]


def _wait_until(predicate, timeout=3.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@pytest.fixture
def session():
    """A simulated scope on the real firmware.

    An LS850T has X/Y/Z plus a turret, so the turret paths are exercised
    on the same instance as the stage paths.
    """
    session = ScopeSession.create(
        complete_settings(simulator_tier='firmware', microscope='LS850T'),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        yield session
    finally:
        session.shutdown()


@pytest.fixture
def scope(session):
    return session.scope


def _board(scope):
    return scope._motion_driver._backend.motor_board


def _fail_home(scope):
    """Stall the X motor: the stage never reaches its home switch, and the
    firmware's home fails."""
    _board(scope).inject('X', tmc5072.STALL)


def _home_and_fail(scope):
    """Run the production home body against an injected failure.

    The home raises the homing fault, which is what the orchestrator
    honors.
    """
    _fail_home(scope)
    with pytest.raises(HomingFailedError):
        scope.motion._home_impl()


# ---------------------------------------------------------------------------
# The precondition: the producers already work. If these break, the rest of
# the file is testing nothing.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_failed_home_marks_every_axis_unknown(scope, centre_posts):
    _home_and_fail(scope)
    for axis in scope.capabilities.axes:
        assert scope.motion._axis_state[axis] == AxisState.UNKNOWN, (
            f'{axis} must be UNKNOWN after a failed home'
        )
    assert scope.motion.has_homed() is False
    errors = _errors_posted(centre_posts)
    assert errors == [], f'the home raises and posts nothing; its caller reports it, got {errors}'


# ---------------------------------------------------------------------------
# B3: every commanded move refuses on an UNKNOWN axis.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_absolute_move_refuses_on_unknown_axis(scope):
    _home_and_fail(scope)
    with pytest.raises(AxisStateUnknownError) as exc:
        scope.motion._move_absolute_impl('Z', position=1000)
    assert exc.value.axis == 'Z'


@pytest.mark.slow
def test_relative_move_refuses_on_unknown_axis(scope):
    """The relative path does not route through the absolute one, so it
    needs its own gate."""
    _home_and_fail(scope)
    with pytest.raises(AxisStateUnknownError) as exc:
        scope.motion._move_relative_impl('X', distance=50)
    assert exc.value.axis == 'X'


@pytest.mark.slow
def test_turret_move_refuses_before_lowering_z(scope):
    """The turret move must refuse BEFORE the safety Z-retract.

    ``_move_turret_impl`` opens ``_safe_turret_move``, which drives Z to 0
    first. Gating only the inner absolute move would lower Z -- real
    motion against an unknown Z reference -- and only then refuse, so
    the refusal has to sit at the turret entry point.
    """
    _home_and_fail(scope)
    z_before = scope._motion_driver.target_pos('Z')
    with pytest.raises(AxisStateUnknownError) as exc:
        scope.motion._move_turret_impl(position=3)
    assert exc.value.axis == 'T'
    assert scope._motion_driver.target_pos('Z') == z_before, (
        'Z must not be driven by a turret move that was refused'
    )


def test_absolute_move_still_works_on_a_known_axis(scope):
    """The gate must refuse UNKNOWN only. A homed axis moves as before."""
    scope.motion._home_impl()
    scope.motion._move_absolute_impl('Z', position=1000)
    assert scope.motion._axis_state['Z'] in (AxisState.MOVING, AxisState.IDLE)


# ---------------------------------------------------------------------------
# B4: the recovery hatch. A gate with no hatch deadlocks its own recovery --
# _safe_turret_move lowers Z through the gated path, and after a failed home
# Z is exactly the axis that is UNKNOWN.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_forced_move_still_drives_on_unknown_axis(scope):
    _home_and_fail(scope)
    # A refusal raises here; a forced move drives, and its wait returns only
    # once Z arrived.
    scope.motion._move_absolute_impl('Z', position=0, force=True).wait()
    assert scope.motion._axis_state['Z'] == AxisState.IDLE


@pytest.mark.slow
def test_turret_home_recovers_from_unknown_z(scope):
    """A turret home after a failed home must not deadlock.

    ``_home_turret_impl`` lowers Z inside ``_safe_turret_move`` while Z is
    UNKNOWN. Without the hatch the gate refuses its own recovery path
    and the turret can never be re-homed without a restart.
    """
    _home_and_fail(scope)
    _board(scope).clear('X', tmc5072.STALL)
    # Turret homing must survive an UNKNOWN Z -- it is the recovery path.
    scope.motion._home_turret_impl()
    assert scope.motion._axis_state['T'] == AxisState.IDLE


# ---------------------------------------------------------------------------
# B2: one state store. The driver's turret-homed flag clears only on physical
# disconnect, so a stall or disconnect fault mid-turret-move leaves it True
# while _axis_state says UNKNOWN -- and turret_select's safety check reads the
# flag. That is a live bypass: the turret drives against an unknown reference.
# ---------------------------------------------------------------------------


def test_turret_fault_revokes_homed_state(scope):
    """A fault that makes T UNKNOWN must revoke the turret's known position.

    This is the state the stall fault and the disconnect fault leave
    behind: the board answered the home, then the move faulted. The
    driver flag alone cannot see that.
    """
    scope.motion._home_impl()
    assert scope.motion.position_is_known('T') is True, 'precondition: a good home homes the turret'

    scope.motion._set_axis_state('T', AxisState.UNKNOWN)

    assert scope.motion.position_is_known('T') is False, (
        "position_is_known('T') must follow the axis state, not a driver flag that "
        'clears only on physical disconnect'
    )
    with pytest.raises(AxisStateUnknownError):
        scope.motion._move_turret_impl(position=2)


def test_stage_fault_revokes_homed_state(scope):
    """Same defect on the stage half: has_homed() must follow the state."""
    scope.motion._home_impl()
    assert scope.motion.has_homed() is True

    scope.motion._set_axis_state('Z', AxisState.UNKNOWN)

    assert scope.motion.has_homed() is False, (
        'has_homed() must follow the axis state, not the driver latch'
    )


# ---------------------------------------------------------------------------
# B6: a dead-board move raises, and the API records the axis as UNKNOWN.
# ---------------------------------------------------------------------------


def _pull_the_cable(scope):
    """The board stops answering: the next target write gets nothing."""
    _board(scope).unplug()


def test_driver_move_raises_when_target_write_is_unanswered(scope):
    """``move()`` warned and returned None -- a jog invisible to every
    layer above it (#709 Half B)."""
    scope.motion._home_impl()
    _pull_the_cable(scope)

    with pytest.raises(HardwareError):
        scope._motion_driver.move('Z', 1000)


def test_api_marks_axis_unknown_when_the_driver_move_raises(scope, centre_posts):
    """The API half: on a driver raise the axis must land in UNKNOWN.

    The move paths re-raise with only a log line today, so the axis keeps
    its stale prior state -- commonly IDLE, i.e. "arrived" -- after a
    move that never happened. The #618 ordering (drive, then mark MOVING)
    means control never reaches the MOVING write on a raise, so the
    except path is where the state has to be set; the order itself must
    not change.
    """
    scope.motion._home_impl()
    _pull_the_cable(scope)

    # No backlash leg, so no position read: the target write is what fails.
    # A move that needs a read first is refused before anything is driven
    # (test_a_failed_position_read_is_never_a_position).
    with pytest.raises(MoveNotCompletedError) as failed:
        scope.motion._move_absolute_impl('Z', position=1000, overshoot_enabled=False)

    assert scope.motion._axis_state['Z'] == AxisState.UNKNOWN, (
        'a move that failed at the driver must leave the axis UNKNOWN, not IDLE'
    )
    # The user is told through the one typed fault the move raises, which
    # its caller shows; the move itself posts nothing.
    assert failed.value.reason == 'driver_failed'
    assert isinstance(failed.value.__cause__, HardwareError)
    assert _errors_posted(centre_posts) == []


def test_a_failed_move_then_refuses_the_next_one(scope):
    """The two halves compose: a dead-board move poisons the axis, and the
    follow-up is refused instead of driving blind again -- first for the
    controller the failed write found gone, which is the person's remedy."""
    scope.motion._home_impl()
    _pull_the_cable(scope)
    with pytest.raises(MoveNotCompletedError):
        scope.motion._move_absolute_impl('Z', position=1000, overshoot_enabled=False)

    assert scope.motion.get_axis_state('Z') == AxisState.UNKNOWN
    with pytest.raises(HardwareCommandRefusedError) as refused:
        scope.motion._move_absolute_impl('Z', position=2000)
    assert refused.value.reason == 'not_connected'
