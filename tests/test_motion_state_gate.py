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
commanded move reaches the driver while any axis of the scope is
UNKNOWN, whichever axis it drives -- only the whole-scope home and the Z
home run then -- and a driver-side move failure lands the axis back in
UNKNOWN rather than leaving a stale IDLE.

Every test runs the production motor driver against the real board
firmware (the firmware simulator tier) and makes the hardware fail: a
stalled X motor, so the firmware's own home fails, and a pulled cable,
which is what a dead board is to the driver. The driver's and the API's
real error handling runs on the firmware's real failure.
"""

import time

import pytest

from _lumascope.drivers.exceptions import HardwareError
from _lumascope.modules.exceptions import (
    AxisStateUnknownError,
    HardwareCommandRefusedError,
    HomingFailedError,
    MoveFaultError,
    MoveNotCompletedError,
)
from _lumascope.modules.axes import AxisState
from _lumascope.modules.notification_center import Severity
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings


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


@pytest.fixture
def stage_scope(tmp_path):
    """A homed simulated LS850T on the realistic-timing stage, where a move
    can be held part of the way and stall: what loses one axis alone."""
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'),
        simulate=True,
        warn_pre_release=False,
    )
    try:
        home_sim_scope(session.scope)
        session.scope.motion._driver.set_timing_mode('realistic')
        yield session.scope
    finally:
        session.shutdown()


def _fail_home(scope):
    """Stall the X motor: the stage never reaches its home switch, and the
    firmware's home fails."""
    scope.simulation.fail_home('X', 'stall')


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
@pytest.mark.firmware_tier
def test_failed_home_marks_every_axis_unknown(scope, centre_posts):
    _home_and_fail(scope)
    for axis in scope.capabilities.axes:
        assert scope.motion._axis_state[axis] == AxisState.UNKNOWN, (
            f'{axis} must be UNKNOWN after a failed home'
        )
    assert scope.motion.axes_without_position().keys() - {'T'}
    errors = _errors_posted(centre_posts)
    assert errors == [], f'the home raises and posts nothing; its caller reports it, got {errors}'


# ---------------------------------------------------------------------------
# B3: every commanded move refuses on an UNKNOWN axis.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.firmware_tier
def test_absolute_move_refuses_on_unknown_axis(scope):
    _home_and_fail(scope)
    with pytest.raises(AxisStateUnknownError) as exc:
        scope.motion._move_absolute_impl('Z', position=1000)
    assert exc.value.reason == 'position_unknown'
    assert set(exc.value.axes) == set(scope.capabilities.axes)


@pytest.mark.slow
@pytest.mark.firmware_tier
def test_relative_move_refuses_on_unknown_axis(scope):
    """The relative path does not route through the absolute one, so it
    needs its own gate."""
    _home_and_fail(scope)
    with pytest.raises(AxisStateUnknownError) as exc:
        scope.motion.move_xy_by(dx_um=50)
    assert exc.value.reason == 'position_unknown'
    assert set(exc.value.axes) == set(scope.capabilities.axes)


@pytest.mark.slow
@pytest.mark.firmware_tier
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
        scope.motion._move_turret_impl(slot=3)
    assert exc.value.reason == 'position_unknown'
    assert 'T' in exc.value.axes
    assert scope._motion_driver.target_pos('Z') == z_before, (
        'Z must not be driven by a turret move that was refused'
    )


@pytest.mark.firmware_tier
def test_absolute_move_still_works_on_a_known_axis(scope):
    """The gate must refuse UNKNOWN only. A homed axis moves as before."""
    scope.motion._home_impl()
    scope.motion._move_absolute_impl('Z', position=1000)
    assert scope.motion._axis_state['Z'] in (AxisState.MOVING, AxisState.IDLE)


# ---------------------------------------------------------------------------
# One answer for the whole scope: while any axis is UNKNOWN nothing moves --
# no move, jog, turret change or turret home, on any axis -- and only the
# whole-scope home and the Z home run. A turret change parks Z and restores
# it; with a hatch past the gate for that park, a lost Z came out of a
# turret home reading as known at a number nobody established.
# ---------------------------------------------------------------------------


_HOLD_WAIT_S = 5.0
# Past the motion monitor's stall window, with room for a slowed host.
_GIVE_UP_WAIT_S = 8.0


def _stall(scope, axis, start):
    """Hold ``axis`` part of the way so its move stalls: the monitor gives it
    up, and the axis is UNKNOWN with every other axis as it was."""
    held = scope.simulation.hold_axis(axis)
    operation = start()
    held.result(_HOLD_WAIT_S)
    return operation


def _lose_x(scope):
    """On a homed scope, an X move that stalls: only X is UNKNOWN."""
    moving = _stall(scope, 'X', lambda: scope.motion.start_move_xy(x_um=60000.0))
    with pytest.raises(MoveFaultError) as failed:
        moving.result(timeout=_GIVE_UP_WAIT_S)
    assert failed.value.reason == 'stalled'
    scope.simulation.clear()
    states = {axis: scope.motion.get_axis_state(axis) for axis in scope.capabilities.axes}
    assert states == {**dict.fromkeys(states, AxisState.IDLE), 'X': AxisState.UNKNOWN}, (
        f'precondition: only X is lost, got {states}'
    )


def _board_targets(scope):
    return {axis: scope._motion_driver.target_pos(axis) for axis in ('Y', 'Z', 'T')}


@pytest.mark.slow
def test_one_lost_axis_refuses_every_move_on_every_axis(stage_scope):
    """With only X lost, a Z move, a Z jog, a turret change and a turret home
    are each refused naming X, and nothing is driven."""
    _lose_x(stage_scope)
    before = _board_targets(stage_scope)

    for name, gesture in (
        ('Z move', lambda: stage_scope.motion.move_z(z_um=1000.0)),
        ('Z jog', lambda: stage_scope.motion.move_z_by(dz_um=10.0)),
        ('turret change', lambda: stage_scope.motion.move_turret(slot=2)),
        ('turret home', lambda: stage_scope.motion.home('T')),
    ):
        with pytest.raises(AxisStateUnknownError) as refused:
            gesture()
        assert refused.value.reason == 'position_unknown', name
        assert list(refused.value.axes) == ['X'], f'the {name} must be refused for X'

    assert _board_targets(stage_scope) == before, 'a refused gesture must drive nothing'


@pytest.mark.slow
def test_a_z_home_runs_while_an_axis_is_lost_and_motion_stays_refused(stage_scope):
    _lose_x(stage_scope)

    stage_scope.motion.home('Z')

    assert stage_scope.motion.get_axis_state('Z') == AxisState.IDLE
    with pytest.raises(AxisStateUnknownError) as refused:
        stage_scope.motion.move_z(z_um=1000.0)
    assert list(refused.value.axes) == ['X']


@pytest.mark.slow
def test_a_z_home_that_puts_back_still_recovers_z_while_an_axis_is_lost(stage_scope):
    """Asked to put back, the Z home recovers Z and leaves it at its home: the
    put-back would be refused for X, and the home itself succeeded."""
    stage_scope.motion.move_z(z_um=3000.0)
    _lose_x(stage_scope)
    z_before = _board_targets(stage_scope)['Z']

    stage_scope.motion.home('Z', put_back=True)

    assert stage_scope.motion.get_axis_state('Z') == AxisState.IDLE
    assert stage_scope.motion.get_axis_state('X') == AxisState.UNKNOWN
    assert _board_targets(stage_scope)['Z'] != z_before, 'Z stays where the home left it'


@pytest.mark.slow
def test_a_move_gesture_is_refused_for_an_axis_it_does_not_name(stage_scope):
    """The consult seam answers as the gate does, so a gesture's pre-check
    cannot pass a move the gate then refuses; a save still asks only about
    the axes it records."""
    _lose_x(stage_scope)

    with pytest.raises(AxisStateUnknownError) as refused:
        stage_scope.motion.refuse_unknown_positions(('Z',), recording=False, then='move it')
    assert list(refused.value.axes) == ['X']

    stage_scope.motion.refuse_unknown_positions(('Z',), recording=True, then='save the focus')


@pytest.mark.slow
@pytest.mark.firmware_tier
def test_a_turret_home_after_a_failed_home_is_refused_and_home_all_recovers(scope):
    """After a failed home, the turret home is refused before Z is parked,
    so Z is not made known by anything but a home; Home All then recovers
    every axis."""
    _home_and_fail(scope)
    scope.simulation.clear()
    targets = {axis: scope._motion_driver.target_pos(axis) for axis in ('Z', 'T')}

    with pytest.raises(AxisStateUnknownError) as refused:
        scope.motion.home('T')
    assert refused.value.reason == 'position_unknown'
    assert set(refused.value.axes) == set(scope.capabilities.axes)
    assert {axis: scope._motion_driver.target_pos(axis) for axis in ('Z', 'T')} == targets
    assert scope.motion.get_axis_state('Z') == AxisState.UNKNOWN

    scope.motion.home()

    for axis in scope.capabilities.axes:
        assert scope.motion.get_axis_state(axis) == AxisState.IDLE, axis
    scope.motion.move_z(z_um=1000.0)


@pytest.mark.slow
def test_a_turret_change_that_loses_the_turret_leaves_z_parked(stage_scope):
    """The Z restore is a move, and with the turret lost nothing moves: Z
    stays parked, and the turret's own failure is what the change raises,
    not the gate's refusal of the restore.

    The turret's target write goes unanswered, so the driver fails the move
    and no stop is sent: a stall would halt the stage, which leaves Z parked
    by the stop's own rule and never reaches this one.
    """
    stage_scope.motion.move_z(z_um=1000.0)
    # Slot 2's target on the simulated board's wire.
    stage_scope.simulation.drop_replies('motor', count=1, command='TARGET_WT80000')

    with pytest.raises(MoveFaultError) as failed:
        stage_scope.motion.move_turret(slot=2)

    assert failed.value.axis == 'T'
    assert failed.value.reason == 'driver_failed'
    assert stage_scope.motion.get_axis_state('T') == AxisState.UNKNOWN
    assert stage_scope._motion_driver.target_pos('Z') == 0, 'Z must stay parked'


@pytest.mark.slow
@pytest.mark.firmware_tier
def test_a_turret_home_that_fails_leaves_z_parked(scope):
    scope.motion._home_impl()
    scope.motion.move_z(z_um=1000.0)
    scope.simulation.fail_home('T', 'stall')

    with pytest.raises(HomingFailedError):
        scope.motion.home('T')

    assert scope.motion.get_axis_state('T') == AxisState.UNKNOWN
    assert scope._motion_driver.target_pos('Z') == 0, 'Z must stay parked'


# ---------------------------------------------------------------------------
# B2: one state store. The driver's turret-homed flag clears only on physical
# disconnect, so a stall or disconnect fault mid-turret-move leaves it True
# while _axis_state says UNKNOWN -- and turret_select's safety check reads the
# flag. That is a live bypass: the turret drives against an unknown reference.
# ---------------------------------------------------------------------------


@pytest.mark.firmware_tier
def test_turret_fault_revokes_homed_state(scope):
    """A fault that makes T UNKNOWN must revoke the turret's known position.

    This is the state the stall fault and the disconnect fault leave
    behind: the board answered the home, then the move faulted. The
    driver flag alone cannot see that.
    """
    scope.motion._home_impl()
    assert 'T' not in scope.motion.axes_without_position(), (
        'precondition: a good home homes the turret'
    )

    scope.motion._set_axis_state('T', AxisState.UNKNOWN)

    assert 'T' in scope.motion.axes_without_position(), (
        "axes_without_position() must follow T's axis state, not a driver flag that "
        'clears only on physical disconnect'
    )
    with pytest.raises(AxisStateUnknownError):
        scope.motion._move_turret_impl(slot=2)


@pytest.mark.firmware_tier
def test_stage_fault_revokes_homed_state(scope):
    """Same defect on the stage half: axes_without_position() must follow the state."""
    scope.motion._home_impl()
    assert not (scope.motion.axes_without_position().keys() - {'T'})

    scope.motion._set_axis_state('Z', AxisState.UNKNOWN)

    assert scope.motion.axes_without_position().keys() - {'T'}, (
        'axes_without_position() must follow the axis state, not the driver latch'
    )


# ---------------------------------------------------------------------------
# B6: a dead-board move raises, and the API records the axis as UNKNOWN.
# ---------------------------------------------------------------------------


def _pull_the_cable(scope):
    """The board stops answering: the next target write gets nothing."""
    scope.simulation.unplug('motor')


@pytest.mark.firmware_tier
def test_driver_move_raises_when_target_write_is_unanswered(scope):
    """``move()`` warned and returned None -- a jog invisible to every
    layer above it (#709 Half B)."""
    scope.motion._home_impl()
    _pull_the_cable(scope)

    with pytest.raises(HardwareError):
        scope._motion_driver.move('Z', 1000)


@pytest.mark.firmware_tier
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
    # its caller shows, naming the lost board; the move itself posts nothing.
    assert failed.value.reason == 'board_lost'
    assert isinstance(failed.value.__cause__, HardwareError)
    assert _errors_posted(centre_posts) == []


@pytest.mark.firmware_tier
def test_a_defect_in_the_drive_leaves_as_itself_with_the_axis_unknown(
    scope, monkeypatch, centre_posts
):
    scope.motion._home_impl()

    def defect(*_a, **_kw):
        # a stand-in by design: our own defect inside the driver's move,
        # which no simulated board fault produces
        raise TypeError('scripted defect')

    monkeypatch.setattr(scope.motion._driver, 'move_abs_pos', defect)

    with pytest.raises(TypeError, match='scripted defect'):
        scope.motion._move_absolute_impl('Z', position=1000, overshoot_enabled=False)

    assert scope.motion.get_axis_state('Z') == AxisState.UNKNOWN
    assert _errors_posted(centre_posts) == []


@pytest.mark.firmware_tier
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
