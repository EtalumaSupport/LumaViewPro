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


# T publishes no travel: its position is a slot, not a distance.
LIMITS = {'X': {'min': 0.0, 'max': 80000.0}, 'Y': {'min': 0.0, 'max': 80000.0}, 'T': None}


@pytest.fixture
def api():
    motion = MotionAPI.__new__(MotionAPI)
    motion.get_axis_limits = lambda axis: LIMITS.get(axis)
    motion._arrival_events = dict.fromkeys(('X', 'Y', 'Z', 'T'))

    def _reached(axis, force=False):
        raise _ReachedPreDriveError(axis)

    motion._pre_drive = _reached
    return motion


@pytest.mark.parametrize('position', [90000.0, 13246567.0, MOTOR_POSITION_LIMIT + 1, 1e30])
def test_one_mistake_gets_one_answer_whatever_its_magnitude(api, position):
    """An axis that publishes travel always answers with TRAVEL.

    Eric, 2026-09-15: *"why are there two limits? Why does a user care? If
    you are outside the travel limits (80/120mm) it should say you are out
    of the travel limit."* Before this, a value a little past travel named
    the travel range and a larger one named a 1 m ceiling, so one kind of
    mistake produced two unrelated answers depending on how wrong it was.
    Travel lies inside the ceiling for any axis that has travel, so the
    ceiling can only ever fire on a value that is ALSO outside travel --
    it had nothing of its own to say and was stealing the better message.
    """
    with pytest.raises(PositionOutOfRangeError) as caught:
        api._move_absolute_impl('Y', position)

    assert caught.value.bound == 'travel range'
    assert 'safety limit' not in str(caught.value)


def test_the_turret_answers_in_one_vocabulary_however_absurd_the_request(api):
    """The turret publishes no travel, but it is not unbounded: it has four
    slots, and that bound answers before the coarse ceiling.

    The ceiling used to be the only thing that could refuse this axis. It
    is not any more, and it must not take the question back for large
    values -- that would give a user two different answers for one
    mistake, naming slots for 5 and a metre-scale limit for 2000000, which
    is the same inconsistency the travel-before-ceiling order exists to
    prevent for the axes that do publish travel.
    """
    for absurd in (5, MOTOR_POSITION_LIMIT + 1):
        with pytest.raises(PositionOutOfRangeError) as caught:
            api._move_absolute_impl('T', absurd)

        assert caught.value.bound == 'turret slots'
        assert 'safety limit' not in str(caught.value)


def test_an_in_range_move_on_a_limitless_axis_is_allowed(api):
    """The ceiling must not start refusing ordinary turret slots."""
    with pytest.raises(_ReachedPreDriveError):
        api._move_absolute_impl('T', 3)


def test_ignore_limits_bypasses_travel_but_not_the_ceiling(api):
    """The hatch exists to drive outside TRAVEL deliberately. It is not a
    licence to hand the motor an arbitrary number."""
    with pytest.raises(_ReachedPreDriveError):
        api._move_absolute_impl('Y', 90000.0, ignore_limits=True)

    with pytest.raises(PositionOutOfRangeError) as caught:
        api._move_absolute_impl('Y', MOTOR_POSITION_LIMIT + 1, ignore_limits=True)
    assert caught.value.bound == 'safety limit'


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


def test_each_failure_has_its_own_identity_and_none_of_it_is_a_symbol():
    """Each failure must carry its own dedup identity, and none of it may
    be a Python symbol.

    Two properties, one mechanism. The identity has to differ per action --
    naming the executor gave everything it runs one key, so an unrelated
    second failure never reached the user. And none of it may be spelled
    from the callable: a symbol is not English, and for a partial or lambda
    it is a repr carrying a heap address, which can never match itself and
    so silently disables the dedup it was supposed to serve.

    Driven through the real executor rather than read out of the source,
    because an assertion on the source text goes stale the moment the
    wording changes and says nothing about what the user is shown.
    """
    import re

    import modules.sequential_io_executor as sio
    from modules.notification_center import NotificationCenter, Severity
    from modules.sequential_io_executor import IOTask, SequentialIOExecutor

    def _grind_beans():
        raise ValueError('burr jammed')

    def _pull_shot():
        raise ValueError('portafilter empty')

    centre = NotificationCenter(dedup_window_s=10.0)
    seen = []
    centre.add_listener(seen.append, min_severity=Severity.ERROR)

    original = sio.notifications
    try:
        sio.notifications = centre
        for action in (_grind_beans, _pull_shot, _grind_beans):
            executor = SequentialIOExecutor(name='TEST')
            task = IOTask(action)
            task.set_name(executor.executor_name)
            executor.queue.put(task)
            executor.queue.get()
            result, exception = task.run()
            executor._on_task_done(task, result, exception)
    finally:
        sio.notifications = original

    # Two distinct actions reach the user; the repeat of the first dedups.
    assert len(seen) == 2, (
        f'distinct failures must keep distinct dedup identities, and a repeat '
        f'of one must collapse: {[(n.category, n.title) for n in seen]}'
    )
    assert seen[0].category != seen[1].category, (
        'one dedup identity for everything the executor runs is the bug this '
        'guards: an unrelated second failure never reaches the user'
    )

    # Nothing the USER reads may be spelled from the callable.
    for n in seen:
        assert '_grind_beans' not in n.title and '_pull_shot' not in n.title, (
            f'the popup title is a Python symbol: {n.title!r}'
        )
        assert not re.search(r'0x[0-9a-f]{6,}', f'{n.title} {n.message}'), (
            f'a heap address reached the user, and makes the dedup key unmatchable: {n.title!r}'
        )
        assert n.title[:1].isupper(), f'the popup title is not prose: {n.title!r}'
