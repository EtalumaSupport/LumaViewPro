"""An absolute move beyond an axis's travel is refused, not clamped.

The motion API holds the one travel check; the drivers have none, and a
driver handed a target past the travel drives there. A target outside
the travel is refused with ``PositionOutOfRangeError``, the stage stays
where it was and the previous target stands, so a protocol step saved
beyond this scope's travel cannot image the wrong place while the log
says it went where it was told. The turret has no travel: its bound is
its four slots, and a refusal names them, never a distance.

Every case drives the simulated LS850T through the public movers,
``move_absolute`` and ``move_turret``, so the travel the check reads is
what the scope's own driver publishes.
"""

import pytest

from modules.exceptions import (
    AxisStateUnknownError,
    HardwareCommandRefusedError,
    PositionOutOfRangeError,
)
from modules.lumascope_api._constants import MOTOR_POSITION_LIMIT
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

# The LS850T's travel, in um, from the shipped motor defaults. Written out
# rather than read back so a defaults change fails this file loudly instead
# of silently retargeting every case; the first test pins the agreement.
TRAVEL = {'X': (0.0, 120000.0), 'Y': (0.0, 80000.0), 'Z': (0.0, 14000.0)}


@pytest.fixture(scope='module')
def motion(sim_turreted_session):
    return sim_turreted_session.scope.motion


def test_the_simulated_scope_publishes_the_travel_these_cases_assume(motion):
    published = {
        axis: (limits['min'], limits['max'])
        for axis, limits in ((axis, motion.get_axis_limits(axis)) for axis in TRAVEL)
    }
    assert published == TRAVEL
    # T carries no travel: its position is a slot, not a distance.
    assert motion.get_axis_limits('T') is None


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
def test_out_of_travel_is_refused_and_nothing_is_driven(motion, axis, position):
    """A refused move writes no target, so the previous one stands."""
    target_before = motion.get_target_position(axis)

    with pytest.raises(PositionOutOfRangeError) as caught:
        motion.move_absolute(axis, position)

    assert (caught.value.axis, caught.value.bound) == (axis, 'travel range')
    assert motion.get_target_position(axis) == target_before


@pytest.mark.parametrize(
    'axis,position',
    [('X', 0.0), ('X', 120000.0), ('Y', 40000.0), ('Z', 14000.0)],
)
def test_in_travel_and_the_boundaries_are_allowed(motion, axis, position):
    """The limits are inclusive; refusing an endpoint would strand the stage at it.

    ``move_absolute`` returns once the axis has arrived, so returning is
    how "allowed" is observed; the target it wrote is the number asked for.
    """
    motion.move_absolute(axis, position)

    assert motion.get_target_position(axis) == position
    assert motion.get_current_position(axis) == pytest.approx(position, abs=1.0)


def test_an_absent_axis_is_refused_before_its_travel_is_judged(tmp_path):
    """A Z-only scope says it has no X, not that the target is outside X's travel."""
    settings = complete_settings(microscope='LS820', live_folder=str(tmp_path))
    session = ScopeSession.create(settings, simulate=True)
    try:
        with pytest.raises(HardwareCommandRefusedError) as caught:
            session.scope.motion.move_absolute('X', 999999.0)

        assert caught.value.reason == 'axis_absent'
    finally:
        session.shutdown()


def test_axis_state_unknown_is_a_separate_failure():
    """The two refusals are distinct; neither should catch the other."""
    assert not issubclass(AxisStateUnknownError, PositionOutOfRangeError)
    assert not issubclass(PositionOutOfRangeError, AxisStateUnknownError)


# A slot that is no whole number at all is refused at move_turret's argument
# door (tests/test_an_argument_of_another_type_is_refused_at_the_door.py).
@pytest.mark.parametrize('slot', [0, 5, 99, -1, MOTOR_POSITION_LIMIT + 1], ids=repr)
def test_a_slot_the_turret_does_not_have_is_refused_before_z_is_parked(motion, slot):
    """The motor accepts 99 and drives 24.5 revolutions; the API refuses it.

    The refusal names the slots, whatever the magnitude of the request:
    naming slots for 5 and a metre-scale safety limit for 1000001 would
    give a user two answers for one mistake, and the second points at a
    number that means nothing for a turret. It is refused before the Z
    park that precedes a real turret move, so Z stays where it was and
    the slot on record is unchanged.
    """
    motion.move_absolute('Z', 1000.0)
    slot_before = motion.get_turret_slot()

    with pytest.raises(PositionOutOfRangeError) as caught:
        motion.move_turret(slot)

    assert (caught.value.axis, caught.value.position, caught.value.bound) == (
        'T',
        slot,
        'turret slots',
    )
    assert '1 to 4' in str(caught.value)
    assert 'travel range' not in str(caught.value)
    assert 'safety limit' not in str(caught.value)
    assert motion.get_current_position('Z') == pytest.approx(1000.0, abs=1.0)
    assert motion.get_turret_slot() == slot_before


@pytest.mark.parametrize('slot', [1, 2, 3, 4])
def test_every_real_slot_is_taken(motion, slot):
    """All four inclusive; refusing an endpoint would strand slot 1 or 4."""
    motion.move_turret(slot)

    assert motion.get_turret_slot() == slot
