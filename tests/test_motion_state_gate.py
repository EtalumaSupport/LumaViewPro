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

Every test composes a REAL ``Lumascope(simulate=True)`` and injects
failure through the simulator's own paths (``_fail_on`` makes
``exchange_command`` return None, which is what a dead board does), so
the driver's real error handling runs rather than a replaced method.

Two seams are substituted, both deliberately:

* The startup test's two motion calls, supplied through
  ``start_application_session``'s ``home_fn`` / ``turret_fn`` parameters
  -- the same seam the Kivy app uses to pass its widget-flavored
  wrappers. The substitutes route straight to the production motion
  bodies and record what the orchestrator attempted, so the decision
  under test -- does startup attempt the turret move after a failed
  home? -- is observed exactly, while the hardware underneath stays the
  real simulator.
* ``exchange_command`` returning None on a target write, for the
  dead-board move. That is the serial boundary, where Rule 11 puts the
  only permitted canned value.
"""

import time

import pytest

from drivers.exceptions import HardwareError
from modules.exceptions import AxisStateUnknownError
from modules.lumascope_api import AxisState, Lumascope


def _silence_notifications(monkeypatch, sink):
    import modules.notification_center as nc

    monkeypatch.setattr(
        nc.notifications,
        'error',
        lambda category, title, message, **k: sink.append((category, title, message)),
    )


def _wait_until(predicate, timeout=3.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@pytest.fixture
def scope(monkeypatch):
    """A real simulated scope with notifications captured.

    The LS850T default gives X/Y/Z plus a turret, so the turret paths
    are exercised on the same instance as the stage paths.
    """
    errors = []
    _silence_notifications(monkeypatch, errors)
    scope = Lumascope(simulate=True)
    scope.notifications_seen = errors
    yield scope
    scope.motion._disconnect()


def _fail_home(scope):
    """Make the next home fail the way a dead board fails it."""
    scope._motion_driver._fail_on.add('HOME')


def _home_and_fail(scope):
    """Run the production home body against an injected failure.

    Returns the home result so a caller can assert on the bool the
    orchestrator is supposed to honor.
    """
    _fail_home(scope)
    return scope.motion._home_impl()


# ---------------------------------------------------------------------------
# The precondition: the producers already work. If these break, the rest of
# the file is testing nothing.
# ---------------------------------------------------------------------------


def test_failed_home_marks_every_axis_unknown(scope):
    assert _home_and_fail(scope) is False, 'a failed home must report False'
    for axis in scope.capabilities.axes:
        assert scope.motion._axis_state[axis] == AxisState.UNKNOWN, (
            f'{axis} must be UNKNOWN after a failed home'
        )
    assert scope.motion.has_homed() is False
    assert len(scope.notifications_seen) == 1, (
        f'exactly one home-failure notification expected, got {scope.notifications_seen}'
    )


# ---------------------------------------------------------------------------
# B3: every commanded move refuses on an UNKNOWN axis.
# ---------------------------------------------------------------------------


def test_absolute_move_refuses_on_unknown_axis(scope):
    _home_and_fail(scope)
    with pytest.raises(AxisStateUnknownError) as exc:
        scope.motion._move_absolute_impl('Z', position=1000)
    assert exc.value.axis == 'Z'


def test_relative_move_refuses_on_unknown_axis(scope):
    """The relative path does not route through the absolute one -- it
    calls ``move_rel_pos`` directly, so it needs its own gate."""
    _home_and_fail(scope)
    with pytest.raises(AxisStateUnknownError) as exc:
        scope.motion._move_relative_impl('X', distance=50)
    assert exc.value.axis == 'X'


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
    assert scope.motion._home_impl() is True
    scope.motion._move_absolute_impl('Z', position=1000)
    assert scope.motion._axis_state['Z'] in (AxisState.MOVING, AxisState.IDLE)


# ---------------------------------------------------------------------------
# B4: the recovery hatch. A gate with no hatch deadlocks its own recovery --
# _safe_turret_move lowers Z through the gated path, and after a failed home
# Z is exactly the axis that is UNKNOWN.
# ---------------------------------------------------------------------------


def test_forced_move_still_drives_on_unknown_axis(scope):
    _home_and_fail(scope)
    scope.motion._move_absolute_impl('Z', position=0, force=True)
    assert scope.motion._axis_state['Z'] == AxisState.MOVING, (
        'a forced move must actually drive, not refuse'
    )


def test_turret_home_recovers_from_unknown_z(scope):
    """A turret home after a failed home must not deadlock.

    ``_home_turret_impl`` lowers Z inside ``_safe_turret_move`` while Z is
    UNKNOWN. Without the hatch the gate refuses its own recovery path
    and the turret can never be re-homed without a restart.
    """
    _home_and_fail(scope)
    scope._motion_driver._fail_on.discard('HOME')
    assert scope.motion._home_turret_impl() is True, (
        'turret homing must survive an UNKNOWN Z -- it is the recovery path'
    )
    assert scope.motion._axis_state['T'] == AxisState.IDLE


# ---------------------------------------------------------------------------
# B2: one state store. The driver's has_turret_homed() flag clears only on physical
# disconnect, so a stall or disconnect fault mid-turret-move leaves it True
# while _axis_state says UNKNOWN -- and turret_select's safety check reads the
# flag. That is a live bypass: the turret drives against an unknown reference.
# ---------------------------------------------------------------------------


def test_turret_fault_revokes_homed_state(scope):
    """A fault that makes T UNKNOWN must revoke has_turret_homed().

    This is the state the stall fault and the disconnect fault leave
    behind: the board answered the home, then the move faulted. The
    driver flag alone cannot see that.
    """
    assert scope.motion._home_impl() is True
    assert scope.motion.has_turret_homed() is True, 'precondition: a good home homes the turret'

    scope.motion._set_axis_state('T', AxisState.UNKNOWN)

    assert scope.motion.has_turret_homed() is False, (
        'has_turret_homed() must follow the axis state, not a driver flag that '
        'clears only on physical disconnect'
    )
    with pytest.raises(AxisStateUnknownError):
        scope.motion._move_turret_impl(position=2)


def test_stage_fault_revokes_homed_state(scope):
    """Same defect on the stage half: has_homed() must follow the state."""
    assert scope.motion._home_impl() is True
    assert scope.motion.has_homed() is True

    scope.motion._set_axis_state('Z', AxisState.UNKNOWN)

    assert scope.motion.has_homed() is False, (
        'has_homed() must follow the axis state, not the driver latch'
    )


# ---------------------------------------------------------------------------
# B5: the orchestrator honors the home result.
# ---------------------------------------------------------------------------


def _startup_session(scope, monkeypatch):
    """Build a session and route its two motion calls to the production
    bodies, recording what startup attempted.

    See the module docstring for why these two are substituted.

    The substitutes go in through ``start_application_session``'s own
    ``home_fn`` / ``turret_fn`` parameters -- the same seam the GUI uses
    to supply its widget-flavored wrappers. An earlier version patched
    ``ui.ui_helpers`` attributes instead, which pinned the substitution
    MECHANISM rather than the invariant and stopped intercepting
    anything the moment the Session took its motion callables by
    injection.
    """
    from unittest.mock import MagicMock

    from modules.scope_session import ScopeSession

    session = ScopeSession(
        settings={},
        scope=scope,
        io_executor=MagicMock(),
        camera_executor=MagicMock(),
    )
    attempts = []

    # Both substitutes call the production body directly rather than the
    # async wrapper. The wrapper would hand the task to this session's
    # mock executor, which accepts it and never runs it -- a home that
    # silently never happens, which is exactly the failure this file
    # exists to catch. Running the body inline keeps the sequence ordered
    # and the home's real result observable.
    def _home_fn(axis):
        attempts.append(('home', axis))
        return scope.motion._home_impl()

    def _turret_fn(position):
        attempts.append(('move', 'T', position))
        scope.motion._move_absolute_impl('T', position)

    hooks = {'home_fn': _home_fn, 'turret_fn': _turret_fn}
    return session, attempts, hooks


def test_startup_skips_turret_positioning_after_failed_home(scope, monkeypatch):
    """The cascade in #702: startup homes, the home fails, and startup
    positions the turret anyway -- a real move against an unknown
    reference, and a second error popup on top of the home's own."""
    session, attempts, hooks = _startup_session(scope, monkeypatch)
    _fail_home(scope)

    session.start_application_session(**hooks)

    assert ('home', 'ALL') in attempts, 'startup must still attempt the home'
    turret_moves = [a for a in attempts if a[0] == 'move' and a[1] == 'T']
    assert turret_moves == [], (
        f'startup must not position the turret after a failed home, attempted {turret_moves}'
    )
    assert len(scope.notifications_seen) == 1, (
        f'the home failure notifies once; the skipped turret move must not add '
        f'a second popup, got {scope.notifications_seen}'
    )


def test_startup_positions_turret_after_successful_home(scope, monkeypatch):
    """The control: a good home must still position the turret. A gate
    that refuses everything would pass the test above."""
    session, attempts, hooks = _startup_session(scope, monkeypatch)

    session.start_application_session(**hooks)

    assert ('home', 'ALL') in attempts
    turret_moves = [a for a in attempts if a[0] == 'move' and a[1] == 'T']
    assert len(turret_moves) == 1, (
        f'a successful home must be followed by exactly one turret positioning '
        f'move, got {turret_moves}'
    )


# ---------------------------------------------------------------------------
# B6: a dead-board move raises, and the API records the axis as UNKNOWN.
# ---------------------------------------------------------------------------


def _deaden_target_writes(monkeypatch, scope):
    """Make target writes answer the way a dead board answers: nothing."""
    driver = scope._motion_driver
    real = driver.exchange_command

    def _dead(command, *args, **kwargs):
        if command.startswith('TARGET_W'):
            return None
        return real(command, *args, **kwargs)

    monkeypatch.setattr(driver, 'exchange_command', _dead)


def test_driver_move_raises_when_target_write_is_unanswered(scope, monkeypatch):
    """``move()`` warned and returned None -- a jog invisible to every
    layer above it (#709 Half B)."""
    assert scope.motion._home_impl() is True
    _deaden_target_writes(monkeypatch, scope)

    with pytest.raises(HardwareError):
        scope._motion_driver.move('Z', 1000)


def test_api_marks_axis_unknown_when_the_driver_move_raises(scope, monkeypatch):
    """The API half: on a driver raise the axis must land in UNKNOWN.

    The move paths re-raise with only a log line today, so the axis keeps
    its stale prior state -- commonly IDLE, i.e. "arrived" -- after a
    move that never happened. The #618 ordering (drive, then mark MOVING)
    means control never reaches the MOVING write on a raise, so the
    except path is where the state has to be set; the order itself must
    not change.
    """
    assert scope.motion._home_impl() is True
    _deaden_target_writes(monkeypatch, scope)

    with pytest.raises(HardwareError):
        scope.motion._move_absolute_impl('Z', position=1000)

    assert scope.motion._axis_state['Z'] == AxisState.UNKNOWN, (
        'a move that failed at the driver must leave the axis UNKNOWN, not IDLE'
    )
    assert any(c == 'Motion' for c, _t, _m in scope.notifications_seen), (
        'the user must be told the move failed'
    )


def test_a_failed_move_then_refuses_the_next_one(scope, monkeypatch):
    """The two halves compose: a dead-board move poisons the axis, and
    the gate then refuses the follow-up instead of driving blind again."""
    assert scope.motion._home_impl() is True
    _deaden_target_writes(monkeypatch, scope)
    with pytest.raises(HardwareError):
        scope.motion._move_absolute_impl('Z', position=1000)

    with pytest.raises(AxisStateUnknownError):
        scope.motion._move_absolute_impl('Z', position=2000)
