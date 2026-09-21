"""An absolute move beyond an axis's travel is refused, not clamped.

The driver clamps an out-of-travel target to the nearest limit and drives
there, reporting success at a position nobody asked for. A protocol step
saved beyond this scope's travel then images the wrong place, and the log
cannot tell that from a step that went where it was told.

These tests drive the real ``MotionAPI._move_absolute_impl``. The gate
raises before ``_pre_drive``, so the object needs only the two attributes
the path reads on the way there -- reimplementing the check in the test
would pass whether or not the production wiring exists.
"""

import pytest

from modules.exceptions import AxisStateUnknownError, PositionOutOfRangeError
from modules.lumascope_api.motion import MotionAPI


LIMITS = {
    'X': {'min': 0.0, 'max': 120000.0},
    'Y': {'min': 0.0, 'max': 80000.0},
    'Z': {'min': 0.0, 'max': 14000.0},
    # T carries no travel: its position is a slot, not a distance.
    'T': None,
}


class _ReachedPreDriveError(Exception):
    """The gate let the move through.

    _pre_drive is the first statement after the gate, so reaching it is
    how "allowed" is observed. It has to RAISE rather than record: the
    rest of the method needs a position cache and a live driver, and this
    fixture deliberately supplies neither.
    """


@pytest.fixture
def api():
    motion = MotionAPI.__new__(MotionAPI)
    # get_axis_limits is the seam the gate reads; _driver is a read-only
    # property, so the stub goes at the call the gate actually makes.
    motion.get_axis_limits = lambda axis: LIMITS.get(axis)
    # Sized to the present axes; the gate sits just after this check.
    motion._arrival_events = dict.fromkeys(('X', 'Y', 'Z', 'T'))

    def _reached(axis, force=False):
        raise _ReachedPreDriveError(axis)

    motion._pre_drive = _reached
    return motion


def test_it_is_a_valueerror_subclass():
    """Callers already catching ValueError from this call keep working."""
    assert issubclass(PositionOutOfRangeError, ValueError)


def test_the_message_names_the_axis_the_request_and_the_range():
    """The executor shows str(exception) verbatim as the popup body."""
    err = PositionOutOfRangeError('Y', 200000.0, 0.0, 80000.0)

    text = str(err)

    assert 'Y' in text
    assert '200000.0' in text
    assert '80000.0' in text
    assert err.axis == 'Y'
    assert err.position == 200000.0


def test_the_executor_shows_its_message_rather_than_a_generic_one():
    """A refusal a user can act on must not be flattened to 'action failed'.

    Pins membership from the consumer side: the tuple is rebuilt per
    failure inside the handler, so an import-time check would not see
    what the handler actually uses. Both arms matter -- the fallback runs
    when drivers.exceptions is unavailable.
    """
    import inspect

    from modules import sequential_io_executor

    src = inspect.getsource(sequential_io_executor)

    assert src.count('PositionOutOfRangeError') >= 3, (
        'must be imported and present in BOTH typed tuples'
    )
    assert src.count('AxisStateUnknownError') >= 3, (
        'must be imported and present in BOTH typed tuples'
    )


@pytest.mark.parametrize(
    'axis,position',
    [
        ('X', 120000.1),
        ('Y', 80000.1),
        ('Z', 14000.1),
        ('X', -0.1),
        ('Y', -1.0),
        ('Z', -0.5),
    ],
)
def test_out_of_travel_is_refused(api, axis, position):
    with pytest.raises(PositionOutOfRangeError) as caught:
        api._move_absolute_impl(axis, position)

    assert caught.value.axis == axis


@pytest.mark.parametrize(
    'axis,position',
    [('X', 0.0), ('X', 120000.0), ('Y', 40000.0), ('Z', 14000.0)],
)
def test_in_travel_and_the_boundaries_are_allowed(api, axis, position):
    """The limits are inclusive; refusing an endpoint would break homing.

    Reaching _pre_drive is the pass condition: it is the statement
    immediately after the gate, so catching its sentinel is how "the gate
    allowed this" is observed.
    """
    with pytest.raises(_ReachedPreDriveError):
        api._move_absolute_impl(axis, position)


def test_the_turret_is_checked_against_its_slots_not_against_travel(api):
    """The turret publishes no travel, so the um range check above refuses
    nothing for it -- but it does have a real bound, and this is the door.

    A real slot must pass: without that the gate would raise on every
    turret move. A slot that does not exist must be refused here and not
    only at move_turret, because the generic mover is reachable directly
    by an L2 caller and the motor's answer to slot 99 is to drive 24.5
    revolutions. The refusal must also name SLOTS -- telling someone 99 is
    outside a metre-scale safety limit points them at a number that means
    nothing for a turret.
    """
    with pytest.raises(_ReachedPreDriveError):
        api._move_absolute_impl('T', 3)

    with pytest.raises(PositionOutOfRangeError) as caught:
        api._move_absolute_impl('T', 99)

    assert caught.value.axis == 'T'
    assert 'turret slots' in str(caught.value)
    assert 'safety limit' not in str(caught.value)


def test_ignore_limits_does_not_open_the_turret(api):
    """The hatch is for driving outside TRAVEL deliberately, not for
    handing the motor a slot the turret does not have -- the same reason
    the coarse safety ceiling is not gated on it either."""
    with pytest.raises(PositionOutOfRangeError):
        api._move_absolute_impl('T', 99, ignore_limits=True)


def test_ignore_limits_still_bypasses(api):
    """A documented public bypass; the new refusal must not silently void it."""
    with pytest.raises(_ReachedPreDriveError):
        api._move_absolute_impl('X', 999999.0, ignore_limits=True)


def test_an_absent_axis_is_still_a_silent_no_op(api):
    """The present-axis check precedes the gate, so a Z-only scope does not
    start refusing the moves it used to ignore."""
    del api._arrival_events['X']

    api._move_absolute_impl('X', 999999.0)


def test_axis_state_unknown_is_a_separate_failure():
    """The two refusals are distinct; neither should catch the other."""
    assert not issubclass(AxisStateUnknownError, PositionOutOfRangeError)
    assert not issubclass(PositionOutOfRangeError, AxisStateUnknownError)


class _ReachedPreDriveOnTurretError(Exception):
    """The turret slot bound let the command through.

    ``_move_turret_impl``'s first statement after the bound is
    ``_pre_drive('T')``, so catching this is how "the bound allowed this
    slot" is observed without a driver or a position cache.
    """


@pytest.fixture
def turret():
    """The real ``_move_turret_impl``, stopped at its first side effect.

    The bound has to be exercised on the production method: a check
    rebuilt in the test would pass whether or not the wiring exists.
    """
    motion = MotionAPI.__new__(MotionAPI)

    def _reached(axis, force=False):
        raise _ReachedPreDriveOnTurretError(axis)

    motion._pre_drive = _reached
    # Never equal to a slot under test, so the same-position short-circuit
    # cannot be what stops the call.
    motion._last_turret_position = None
    return motion


@pytest.mark.parametrize('slot', [0, 5, 99, -1, 2.5, True, '3', None])
def test_a_slot_the_turret_does_not_have_is_refused(turret, slot):
    """The motor accepts 99 and drives 24.5 revolutions; the API must not.

    ``_move_absolute_impl`` cannot catch this -- the turret publishes no
    travel, which the case above pins -- so the refusal lives here.
    """
    with pytest.raises(PositionOutOfRangeError) as caught:
        turret._move_turret_impl(slot)

    assert caught.value.axis == 'T'
    assert caught.value.position == slot


@pytest.mark.parametrize('slot', [1, 2, 3, 4])
def test_every_real_slot_is_allowed(turret, slot):
    """All four inclusive; refusing an endpoint would strand slot 1 or 4."""
    with pytest.raises(_ReachedPreDriveOnTurretError):
        turret._move_turret_impl(slot)


def test_the_refusal_names_the_slot_range_and_not_a_travel_range(turret):
    """Two bounds can refuse a move, and naming the wrong one sends the
    user to a number that means nothing for a turret."""
    with pytest.raises(PositionOutOfRangeError) as caught:
        turret._move_turret_impl(99)

    text = str(caught.value)
    assert 'turret slots' in text
    assert '1 to 4' in text
    assert 'travel range' not in text
