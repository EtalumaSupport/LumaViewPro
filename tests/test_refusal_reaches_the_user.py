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


@pytest.fixture(scope='module')
def motion(sim_turreted_session):
    """The simulated LS850T's motion API; Y's travel is 0 to 80000 um."""
    return sim_turreted_session.scope.motion


@pytest.mark.parametrize('position', [90000.0, 13246567.0, MOTOR_POSITION_LIMIT + 1, 1e30])
def test_one_mistake_gets_one_answer_whatever_its_magnitude(motion, position):
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
        motion.move_absolute('Y', position)

    assert caught.value.bound == 'travel range'
    assert 'safety limit' not in str(caught.value)


def test_ignore_limits_bypasses_travel_but_not_the_ceiling(motion):
    """The hatch exists to drive outside TRAVEL deliberately. It is not a
    licence to hand the motor an arbitrary number."""
    motion.move_absolute('Y', 90000.0, ignore_limits=True)
    assert motion.get_target_position('Y') == 90000.0

    with pytest.raises(PositionOutOfRangeError) as caught:
        motion.move_absolute('Y', MOTOR_POSITION_LIMIT + 1, ignore_limits=True)
    assert caught.value.bound == 'safety limit'
    assert motion.get_target_position('Y') == 90000.0

    motion.move_absolute('Y', 40000.0)


def test_a_relative_move_of_absurd_distance_is_refused_the_same_way(motion):
    """The sibling check on the relative path had the identical defect."""
    with pytest.raises(PositionOutOfRangeError) as caught:
        motion.move_relative('X', MOTOR_POSITION_LIMIT + 1)

    assert (caught.value.bound, caught.value.quantity) == ('safety limit', 'distance')
    assert 'distance' in str(caught.value)


@pytest.mark.parametrize('bound', ['travel range', 'safety limit'])
def test_both_safety_refusals_reach_the_user_in_their_own_words(bound, centre_posts):
    """Driven through the real executor: a refusal raised in a background
    task is shown with its own message, whichever limit refused it."""
    from modules.sequential_io_executor import IOTask, SequentialIOExecutor

    error = PositionOutOfRangeError('Y', 1e30, 0.0, 80000.0, bound=bound)

    def _move_absolute_impl():
        raise error

    executor = SequentialIOExecutor(name='TEST')
    task = IOTask(_move_absolute_impl)
    task.set_name(executor.executor_name)
    executor.queue.put(task)
    executor.queue.get()
    result, exception = task.run()
    executor._on_task_done(task, result, exception)

    assert [n.message for n in centre_posts] == [str(error)]


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


def test_each_failure_has_its_own_identity_and_none_of_it_is_a_symbol(centre_posts):
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

    from modules.sequential_io_executor import IOTask, SequentialIOExecutor

    def _grind_beans():
        raise ValueError('burr jammed')

    def _pull_shot():
        raise ValueError('portafilter empty')

    for action in (_grind_beans, _pull_shot, _grind_beans):
        executor = SequentialIOExecutor(name='TEST')
        task = IOTask(action)
        task.set_name(executor.executor_name)
        executor.queue.put(task)
        executor.queue.get()
        result, exception = task.run()
        executor._on_task_done(task, result, exception)

    # Two distinct actions reach the user; the repeat of the first dedups.
    seen = [n for n in centre_posts if n.shown]
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
