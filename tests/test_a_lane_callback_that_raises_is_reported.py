# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A task callback's raise, and a scheduled callback's, is reported, not only logged.

A lane runs a task's callback once the task is done and its waiter answered,
and a headless host runs a scheduled UI callback in place: in both, no
caller is left for the raise to reach. The lane logged it, its default
dispatcher at DEBUG, below the default log level; now each is reported.
"""

import pytest

from modules import kivy_utils
from modules.notification_center import notifications
from modules.sequential_io_executor import IOTask, SequentialIOExecutor, _direct_dispatch


class _CallbackError(RuntimeError):
    pass


def _raising(*args, **kwargs):
    raise _CallbackError('the callback fell over')


@pytest.fixture
def reported(monkeypatch):
    seen = []
    monkeypatch.setattr(notifications, 'report_outcome', lambda ex, **kw: seen.append((ex, kw)))
    return seen


def test_a_task_callback_that_raises_is_reported(reported):
    lane = SequentialIOExecutor(name='CALLBACK_TEST')
    lane.start()
    try:
        future = lane.put(IOTask(action=lambda: 'done', callback=_raising), return_future=True)
        assert future.result(timeout=5.0) == 'done'
        lane.wait_for_idle(5.0)
    finally:
        lane.shutdown()

    [(_ex, kw)] = [r for r in reported if isinstance(r[0], _CallbackError)]
    assert kw['solicited'] is False


def test_the_default_dispatcher_reports_what_it_runs(reported):
    _direct_dispatch(lambda dt: _raising())

    [(ex, _kw)] = reported
    assert isinstance(ex, _CallbackError)


def test_a_headless_scheduled_callback_is_reported(reported):
    previous = kivy_utils._ui_dispatcher
    kivy_utils.set_ui_dispatcher(None)
    try:
        kivy_utils.schedule_ui(lambda dt: _raising())
    finally:
        kivy_utils.set_ui_dispatcher(previous)

    [(ex, _kw)] = reported
    assert isinstance(ex, _CallbackError)
