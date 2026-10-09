# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refusal raised inside a background task is shown and logged as a refusal.

The executor had two answers for a task's exception: a failure or a cancel.
A refusal -- the scope declining a request, nothing broken -- took the
failure answer: an ERROR with a full traceback in the errors log, and a
popup titled "Background operation failed" over the refusal's own words.
Found by driving the app in the simulator: Go To a Y bookmark on a scope
that had not homed. Every one-axis GUI move reaches the executor that way
(the bookmark and focus Go To buttons, the jogs, the Z slider, a typed
position, the turret buttons), because the motion API's pre-drive gate
raises without notifying and leaves the showing to the executor.

A refusal now says what it is and carries its title; the executor shows it
as a warning under that title and logs one line for it with no traceback,
wherever the task ran.
"""

from __future__ import annotations

import logging

import pytest

import modules.sequential_io_executor as sio
from modules.exceptions import AxisStateUnknownError, PositionOutOfRangeError
from modules.notification_center import REFUSAL_OPERATION_KEY, Severity
from modules.sequential_io_executor import IOTask, SequentialIOExecutor

# Each case builds its refusal fresh, as a press does: an outcome is logged
# once and shown at most once per exception object, so one object shared by
# every test would be spent by the first that reported it.
REFUSALS = [
    pytest.param(
        lambda: AxisStateUnknownError({'Y': 'unknown'}),
        'Scope Not Homed',
        id='unknown_position',
    ),
    pytest.param(
        lambda: PositionOutOfRangeError('X', 90000.0, 0.0, 80000.0),
        'Position Out of Range',
        id='outside_travel',
    ),
]


@pytest.fixture(autouse=True)
def _executor_log(monkeypatch):
    # The suite mocks lvp_logger, so the executor's own lines go nowhere;
    # a real logger lets these tests read what it writes.
    monkeypatch.setattr(sio, 'logger', logging.getLogger('LVP.test_executor'))


def _move_absolute_impl(error):
    raise error


def _shown(centre_posts):
    return [n for n in centre_posts if n.shown]


def _run_task(action, *args, protocol=False):
    """Run one fire-and-forget task through a real executor's worker path
    and epilogue. ``protocol`` is the worker's own mark for a task it took
    off the run's queue."""
    executor = SequentialIOExecutor(name='TEST')
    task = IOTask(action, args=args)
    task.set_name(executor.executor_name)
    lane = executor.protocol_queue if protocol else executor.queue
    lane.put(task)
    lane.get()
    task.protocol = protocol
    result, exception = task.run()
    executor._on_task_done(task, result, exception)


def _run_on_the_lane(centre_posts, action, *args):
    """One fire-and-forget task; what the user was shown."""
    _run_task(action, *args)
    return _shown(centre_posts)


# The notification centre also writes an INFO forensic line of what the
# user was shown; it records the popup, not the refusal.
FORENSIC_LOGGER = 'LVP.gui_interactions'


def _task_records(caplog, action='_move_absolute_impl'):
    return [r for r in caplog.records if action in r.getMessage() and r.name != FORENSIC_LOGGER]


@pytest.mark.parametrize(('make_error', 'title'), REFUSALS)
class TestAFireAndForgetRefusal:
    def test_is_shown_once_as_a_warning_under_its_own_title_in_its_own_words(
        self, make_error, title, centre_posts
    ):
        error = make_error()
        shown = _run_on_the_lane(centre_posts, _move_absolute_impl, error)

        assert [(n.severity, n.title, n.message) for n in shown] == [
            (Severity.WARNING, title, str(error))
        ]

    def test_is_logged_as_a_warning_with_no_traceback(self, make_error, title, caplog):
        with caplog.at_level(logging.DEBUG):
            _run_task(_move_absolute_impl, make_error())

        # A shown refusal's record is the notification's own line, which
        # names the action in its category: one WARNING, no traceback.
        records = _task_records(caplog)
        assert [(r.levelno, bool(r.exc_info)) for r in records] == [(logging.WARNING, False)], [
            (r.name, r.levelname, r.getMessage()) for r in records
        ]


@pytest.fixture
def scope():
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(), simulate=True)
    try:
        yield session.scope
    finally:
        session.shutdown()


@pytest.mark.parametrize(('make_error', 'title'), REFUSALS)
def test_a_waited_refusal_reaches_its_caller_and_is_logged_once_without_a_traceback(
    scope, caplog, make_error, title
):
    error = make_error()
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(type(error)):
            scope.motion._dispatch_motion(
                _move_absolute_impl, 'move_absolute', args=(error,), timeout_s=5.0
            )
        scope._io_executor.put(IOTask(action=lambda: None), return_future=True).result(timeout=5.0)

    # The refusal is its waiter's: the raise reaches the caller, and the
    # caller -- here the test -- is where it is reported. The lane logs
    # nothing for it, so it is never logged twice.
    assert _task_records(caplog) == []


@pytest.mark.parametrize(('make_error', 'title'), REFUSALS)
def test_every_refused_press_is_shown_however_soon_it_repeats(make_error, title, centre_posts):
    """Eric, 2026-09-23: *"i do not want a 10 second filter on user buttons.
    Every time you try to go out of range, you should get the dialog."*
    A task outside a run was asked for by someone, so its refusal is an
    answer, and an answer is never filtered as a repeat."""
    for _ in range(3):
        _run_task(_move_absolute_impl, make_error())

    shown = _shown(centre_posts)
    assert [n.title for n in shown] == [title, title, title]
    # Each replaces the last refusal popup rather than stacking on it.
    assert {n.operation_key for n in shown} == {REFUSAL_OPERATION_KEY}


@pytest.mark.parametrize(('make_error', 'title'), REFUSALS)
def test_a_refusal_in_a_runs_own_task_keeps_the_runs_mute(
    make_error, title, centre_posts, unattended_run
):
    # Mid-run only a fatal error may pop up; the run owns its refusals.
    _run_task(_move_absolute_impl, make_error(), protocol=True)

    assert _shown(centre_posts) == []


def test_a_one_axis_move_on_an_unhomed_scope_is_refused_as_not_homed(
    scope, monkeypatch, centre_posts
):
    """The sim finding, end to end: a person's move on an axis that has not
    homed reaches them as the same refusal every gesture gives."""
    from ui import ui_helpers

    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))

    ui_helpers.submit_reported(
        lambda: scope.motion.move_absolute('Y', 1000.0),
        None,
        'MOVE_Y',
        lane=scope._io_executor,
    )
    scope._io_executor.put(IOTask(action=lambda: None), return_future=True).result(timeout=5.0)

    assert [(n.severity, n.title) for n in centre_posts] == [(Severity.WARNING, 'Scope Not Homed')]
    assert 'Y position is unknown' in centre_posts[0].message
