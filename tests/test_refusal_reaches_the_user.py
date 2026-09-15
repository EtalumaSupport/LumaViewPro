"""A refusal a user can provoke reaches that user, with its own words.

Two defects found by driving the app in the simulator, 2026-09-15:

The safety-limit check refused a nonsense magnitude with a message that
said exactly what was wrong -- and raised a bare ValueError, which is not
in the interactive executor's user-facing set, so the message was
replaced by "The '_move_absolute_impl' background operation failed."
Typing 13246567 into a position box produced that; typing 141323 into the
same box produced a clear refusal, because it fell under the safety
ceiling and reached the travel check instead. Same box, same kind of
mistake, two different qualities of answer.

And every failure routed through one executor shared a notification
title -- '<executor name> task failed' -- while notifications dedup on
(category, title) for ten seconds. A refused stage move and an unrelated
camera failure seconds apart were one dedup key, so the second never
reached the user at all. Measured: of nine refusals in that session,
four produced no popup.
"""

import pytest

from modules.exceptions import PositionOutOfRangeError
from modules.lumascope_api._constants import MOTOR_POSITION_LIMIT
from modules.lumascope_api.motion import MotionAPI


class _ReachedPreDriveError(Exception):
    """The gate let the move through."""


@pytest.fixture
def api():
    motion = MotionAPI.__new__(MotionAPI)
    motion.get_axis_limits = lambda axis: {'min': 0.0, 'max': 80000.0}
    motion._arrival_events = dict.fromkeys(('X', 'Y', 'Z', 'T'))

    def _reached(axis, force=False):
        raise _ReachedPreDriveError(axis)

    motion._pre_drive = _reached
    return motion


def test_an_absurd_position_is_refused_by_name_not_as_a_bare_valueerror(api):
    """A bare ValueError is not in the executor's user-facing set, so its
    message never reaches the user."""
    with pytest.raises(PositionOutOfRangeError) as caught:
        api._move_absolute_impl('X', MOTOR_POSITION_LIMIT + 1)

    assert caught.value.bound == 'safety limit'


def test_the_safety_refusal_says_which_limit_refused(api):
    """Naming the travel range here would point the user at the wrong
    number: the entry never got as far as the axis."""
    with pytest.raises(PositionOutOfRangeError) as caught:
        api._move_absolute_impl('X', 13246567.0)

    text = str(caught.value)
    assert 'safety limit' in text
    assert '13246567.0' in text
    assert 'travel range' not in text


def test_the_travel_refusal_still_says_travel_range(api):
    """The two bounds must stay distinguishable in the user's words."""
    with pytest.raises(PositionOutOfRangeError) as caught:
        api._move_absolute_impl('Y', 90000.0)

    assert 'travel range' in str(caught.value)
    assert 'safety limit' not in str(caught.value)


def test_a_relative_move_of_absurd_distance_is_refused_the_same_way(api):
    """The sibling check on the relative path had the identical defect."""
    with pytest.raises(PositionOutOfRangeError) as caught:
        api._move_relative_impl('X', MOTOR_POSITION_LIMIT + 1)

    assert caught.value.bound == 'safety limit'
    assert 'distance' in str(caught.value)


def test_both_safety_refusals_are_in_the_executors_user_facing_set():
    """Pinned from the consumer side: the tuple is rebuilt per failure
    inside the handler, so an import-time check would not see it."""
    import inspect

    from modules import sequential_io_executor

    src = inspect.getsource(sequential_io_executor)

    assert src.count('PositionOutOfRangeError') >= 3


def test_distinct_failures_do_not_suppress_each_other():
    """Notifications dedup on (category, title) for ten seconds. A title
    naming the EXECUTOR made one key for everything it runs, so a refused
    move and an unrelated failure seconds apart were the same event.
    """
    from modules.notification_center import NotificationCenter, Severity

    centre = NotificationCenter(dedup_window_s=10.0)
    seen = []
    centre.add_listener(seen.append, min_severity=Severity.ERROR)

    centre.error('Task', '_move_absolute_impl failed', 'Y position ... travel range')
    centre.error('Task', '_set_gain_db_impl failed', 'the camera refused the gain')

    assert len(seen) == 2, (
        'two unrelated failures were collapsed into one; the second never reaches the user'
    )


def test_a_genuine_repeat_of_one_failure_still_dedups():
    """The spam the dedup exists to stop must still be stopped -- one
    failure repeating is not two failures."""
    from modules.notification_center import NotificationCenter, Severity

    centre = NotificationCenter(dedup_window_s=10.0)
    seen = []
    centre.add_listener(seen.append, min_severity=Severity.ERROR)

    for _ in range(5):
        centre.error('Task', '_move_absolute_impl failed', 'same failure again')

    assert len(seen) == 1


def test_the_executor_titles_its_notification_with_the_action():
    """The title has to identify the failure, because it is half the
    dedup key. Naming the executor gave every one of its failures the
    same identity."""
    import inspect

    from modules import sequential_io_executor

    src = inspect.getsource(sequential_io_executor)

    assert "f'{action_name} task failed'" in src, (
        'the notification title must name the failed ACTION, not the executor'
    )
    assert "f'{self.name} task failed'" not in src, (
        'a title naming the executor makes one dedup key for everything it runs'
    )
