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
from modules.notification_center import REFUSAL_OPERATION_KEY, NotificationCenter, Severity
from modules.sequential_io_executor import IOTask, SequentialIOExecutor

REFUSALS = [
    pytest.param(
        AxisStateUnknownError({'Y': 'unknown'}),
        'Scope Not Homed',
        id='unknown_position',
    ),
    pytest.param(
        PositionOutOfRangeError('X', 90000.0, 0.0, 80000.0),
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


def _watched_centre():
    centre = NotificationCenter(dedup_window_s=10.0)
    shown = []
    centre.add_listener(shown.append, min_severity=Severity.INFO)
    return centre, shown


def _run_task(centre, action, *args, protocol=False):
    """Run one fire-and-forget task through a real executor's worker path
    and epilogue, posting to ``centre``. ``protocol`` is the worker's own
    mark for a task it took off the run's queue."""
    original = sio.notifications
    try:
        sio.notifications = centre
        executor = SequentialIOExecutor(name='TEST')
        task = IOTask(action, args=args)
        task.set_name(executor.executor_name)
        lane = executor.protocol_queue if protocol else executor.queue
        lane.put(task)
        lane.get()
        task.protocol = protocol
        result, exception = task.run()
        executor._on_task_done(task, result, exception)
    finally:
        sio.notifications = original


def _run_on_the_lane(action, *args):
    """One fire-and-forget task on a fresh centre; what the user was shown."""
    centre, shown = _watched_centre()
    _run_task(centre, action, *args)
    return shown


# The notification centre also writes an INFO forensic line of what the
# user was shown; it records the popup, not the refusal.
FORENSIC_LOGGER = 'LVP.gui_interactions'


def _task_records(caplog, action='_move_absolute_impl'):
    return [r for r in caplog.records if action in r.getMessage() and r.name != FORENSIC_LOGGER]


@pytest.mark.parametrize(('error', 'title'), REFUSALS)
class TestAFireAndForgetRefusal:
    def test_is_shown_once_as_a_warning_under_its_own_title_in_its_own_words(self, error, title):
        shown = _run_on_the_lane(_move_absolute_impl, error)

        assert [(n.severity, n.title, n.message) for n in shown] == [
            (Severity.WARNING, title, str(error))
        ]

    def test_is_logged_as_a_warning_with_no_traceback(self, error, title, caplog):
        with caplog.at_level(logging.DEBUG):
            _run_on_the_lane(_move_absolute_impl, error)

        records = _task_records(caplog)
        assert records, 'the refusal left no log line'
        assert all(r.levelno == logging.WARNING for r in records), [
            (r.levelname, r.getMessage()) for r in records
        ]
        assert not any(r.exc_info for r in records), 'a refusal was logged with a traceback'
        raised = [r for r in records if r.name != 'LVP.notifications']
        assert len(raised) == 1, 'one line where it was raised, naming the action'


@pytest.fixture
def scope():
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(), simulate=True)
    try:
        yield session.scope
    finally:
        session.shutdown()


@pytest.mark.parametrize(('error', 'title'), REFUSALS)
def test_a_waited_refusal_reaches_its_caller_and_is_logged_once_without_a_traceback(
    scope, caplog, error, title
):
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(type(error)):
            scope.motion._dispatch_motion(
                _move_absolute_impl, 'move_absolute', args=(error,), timeout_s=5.0
            )
        scope._io_executor.put(IOTask(action=lambda: None), return_future=True).result(timeout=5.0)

    records = _task_records(caplog)
    assert [r.levelno for r in records] == [logging.WARNING]
    assert not records[0].exc_info


@pytest.mark.parametrize(('error', 'title'), REFUSALS)
def test_a_refusal_with_no_executor_still_leaves_one_line(scope, caplog, monkeypatch, error, title):
    # A script on a bare scope runs the task on its own thread and nothing
    # posts a notification, so the line where it was raised is the record.
    monkeypatch.setattr(scope, '_io_executor', None)

    with caplog.at_level(logging.DEBUG), pytest.raises(type(error)):
        scope.motion._submit_motion(_move_absolute_impl, 'move_absolute', kwargs={'error': error})

    records = _task_records(caplog)
    assert [r.levelno for r in records] == [logging.WARNING]
    assert not records[0].exc_info


@pytest.mark.parametrize(('error', 'title'), REFUSALS)
def test_every_refused_press_is_shown_however_soon_it_repeats(error, title):
    """Eric, 2026-09-23: *"i do not want a 10 second filter on user buttons.
    Every time you try to go out of range, you should get the dialog."*
    A task outside a run was asked for by someone, so its refusal is an
    answer, and an answer is never filtered as a repeat."""
    centre, shown = _watched_centre()

    for _ in range(3):
        _run_task(centre, _move_absolute_impl, error)

    assert [n.title for n in shown] == [title, title, title]
    # Each replaces the last refusal popup rather than stacking on it.
    assert {n.operation_key for n in shown} == {REFUSAL_OPERATION_KEY}


@pytest.mark.parametrize(('error', 'title'), REFUSALS)
def test_a_refusal_in_a_runs_own_task_keeps_the_runs_mute(error, title):
    # Mid-run only a fatal error may pop up; the run owns its refusals.
    centre, shown = _watched_centre()
    centre.set_unattended_run(True)

    _run_task(centre, _move_absolute_impl, error, protocol=True)

    assert shown == []


def test_a_failure_is_still_a_failure(caplog):
    def _move_absolute_impl_broke():
        raise RuntimeError('the board stopped answering')

    with caplog.at_level(logging.DEBUG):
        shown = _run_on_the_lane(_move_absolute_impl_broke)

    assert [(n.severity, n.title) for n in shown] == [
        (Severity.ERROR, 'Background operation failed')
    ]
    raised = [
        r
        for r in _task_records(caplog, '_move_absolute_impl_broke')
        if r.name != 'LVP.notifications'
    ]
    assert raised and raised[0].levelno == logging.ERROR and raised[0].exc_info


def test_a_one_axis_move_on_an_unhomed_scope_is_refused_as_not_homed(scope, monkeypatch):
    """The sim finding, end to end: a fire-and-forget move on an axis that
    has not homed reaches the user as the same refusal every gesture gives."""
    centre = NotificationCenter(dedup_window_s=10.0)
    shown = []
    centre.add_listener(shown.append, min_severity=Severity.INFO)
    monkeypatch.setattr(sio, 'notifications', centre)

    scope.motion.move_absolute_async('Y', 1000.0)
    scope._io_executor.put(IOTask(action=lambda: None), return_future=True).result(timeout=5.0)

    assert [(n.severity, n.title) for n in shown] == [(Severity.WARNING, 'Scope Not Homed')]
    assert 'Y position is unknown' in shown[0].message
