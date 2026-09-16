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


def test_an_axis_without_travel_is_not_range_checked(api):
    """The turret's position is a slot; get_axis_limits returns None for it.

    Without this case the gate raises on every turret move.
    """
    for slot in (3, 99):
        with pytest.raises(_ReachedPreDriveError):
            api._move_absolute_impl('T', slot)


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
