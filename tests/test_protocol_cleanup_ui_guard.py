# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run event's handler that raises is reported once, as itself, and never
takes the application down.

A handler handed to the UI scheduler runs on a later Clock tick, after the
call that scheduled it has returned, and the app's crash guard re-raises
anything it cannot attribute to a plugin. So the catch has to sit inside the
function the dispatcher runs, around the handler: a ``try`` around the
scheduling call catches nothing on the GUI host. The report is the
handler's own exception, under the event's name: a refusal stays a refusal
and a fault a fault, never a run cleanup failure with LED, camera and stage
advice.

The re-raise is the right DEFAULT and is left alone everywhere else; a
core bug should be loud.
"""

import contextlib

import pytest

from modules import kivy_utils
from modules.exceptions import ProtocolRunRefusedError, RunCleanupFailedError
from modules.notification_center import notifications
from modules.run_events import deliver
from modules.scope_session import ScopeSession


@pytest.fixture
def immediate_gui_dispatcher():
    """Stand in for Clock.schedule_once, which runs callbacks unguarded."""
    ScopeSession.set_ui_dispatcher(
        kivy_utils.UiDispatcher(schedule=lambda func, timeout: func(timeout), thread=None)
    )
    yield
    ScopeSession.set_ui_dispatcher(None)


@pytest.fixture
def deferring_gui_dispatcher():
    """A dispatcher that runs what it is handed only when the test says: a later tick."""
    queued = []
    ScopeSession.set_ui_dispatcher(
        kivy_utils.UiDispatcher(schedule=lambda func, timeout: queued.append(func), thread=None)
    )
    yield queued
    ScopeSession.set_ui_dispatcher(None)


@pytest.fixture
def reported(monkeypatch):
    seen = []
    monkeypatch.setattr(
        notifications, 'report_outcome', lambda ex, *a, **k: seen.append((ex, a, k))
    )
    return seen


def _raise(ex):
    def _handler(*_args):
        raise ex

    return _handler


def test_a_raising_handler_does_not_reach_the_event_loop(immediate_gui_dispatcher, reported):
    """This is the crash: the exception used to escape to Kivy and exit."""
    deliver(_raise(RuntimeError('handler exploded')), 'run_ended', None, None, None)


def test_a_fault_is_reported_once_as_that_fault_under_the_events_name(
    immediate_gui_dispatcher, reported
):
    fault = RuntimeError('handler exploded')
    deliver(_raise(fault), 'files_written', None, 'written')

    assert len(reported) == 1, reported
    ex, _args, kwargs = reported[0]
    assert ex is fault, "the report is the handler's own exception, not a wrapper"
    assert not isinstance(ex, RunCleanupFailedError)
    assert kwargs == {'solicited': False, 'category': 'files_written'}


def test_a_refusal_is_reported_once_as_that_refusal(immediate_gui_dispatcher, reported):
    refusal = ProtocolRunRefusedError(reason='run_not_live', title='Not Live', message='No.')
    deliver(_raise(refusal), 'run_ended', None, None, None)

    assert [ex for ex, _a, _k in reported] == [refusal]


def test_a_handler_that_raises_after_the_scheduling_returned_is_still_caught(
    deferring_gui_dispatcher, reported
):
    fault = RuntimeError('raised on a later tick')
    deliver(_raise(fault), 'step_started', 0)
    assert reported == [], 'nothing has run yet: the dispatcher deferred it'

    (queued,) = deferring_gui_dispatcher
    queued(0)

    assert [ex for ex, _a, _k in reported] == [fault]


def test_a_healthy_handler_runs_with_its_arguments(immediate_gui_dispatcher, reported):
    seen = []
    deliver(lambda *args: seen.append(args), 'scan_started', 1, 2, 'interval')

    assert seen == [(1, 2, 'interval')]
    assert reported == []


def test_the_delivery_is_left_once_the_handler_and_its_report_are_done(
    deferring_gui_dispatcher, reported
):
    marks = []

    @contextlib.contextmanager
    def _delivery():
        marks.append('entered')
        yield
        marks.append('left')

    deliver(_raise(RuntimeError('x')), 'run_ended', None, None, None, delivery=_delivery())
    assert marks == [], 'the delivery is entered on the thread that runs the handler'
    deferring_gui_dispatcher[0](0)
    assert marks == ['entered', 'left'] and len(reported) == 1


def test_no_handler_still_enters_and_leaves_the_delivery(reported):
    marks = []

    @contextlib.contextmanager
    def _delivery():
        marks.append('entered')
        yield
        marks.append('left')

    deliver(None, 'run_ended', None, None, None, delivery=_delivery())
    assert marks == ['entered', 'left'], 'a wait must never wait on a delivery that will not come'


def test_the_unguarded_scheduler_still_propagates(immediate_gui_dispatcher):
    """Why the delivery's catch has to exist, and that it stays narrowly scoped.

    Raw schedule_ui on the GUI branch propagates -- that is the documented
    app-wide policy and the delivery deliberately does not touch it.
    """
    with pytest.raises(RuntimeError):
        kivy_utils.schedule_ui(_raise(RuntimeError('boom')), 0)
