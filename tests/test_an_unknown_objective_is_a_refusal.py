# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An unknown objective raised inside a background task is shown and logged as a refusal.

Nothing is broken when no one can say which objective is in the light
path: the scope declines to answer with one it cannot vouch for, and the
person can home the turret or assign the objective. Unmarked, a lane task
that raised it read as a crash -- an ERROR with a traceback, and a popup
titled "Background operation failed" over the refusal's own words.
"""

from __future__ import annotations

import logging

import pytest

import modules.sequential_io_executor as sio
from modules.exceptions import ObjectiveUnknownError
from modules.notification_center import NotificationCenter, Severity
from modules.sequential_io_executor import IOTask, SequentialIOExecutor

# The notification centre also writes an INFO forensic line of what the
# user was shown; it records the popup, not the refusal.
FORENSIC_LOGGER = 'LVP.gui_interactions'


@pytest.fixture(autouse=True)
def _executor_log(monkeypatch):
    # The suite mocks lvp_logger, so the executor's own lines go nowhere;
    # a real logger lets these tests read what it writes.
    monkeypatch.setattr(sio, 'logger', logging.getLogger('LVP.test_executor'))


def _jog_step():
    raise ObjectiveUnknownError('slot_unassigned', slot=2)


def _run_on_the_lane(monkeypatch):
    """Run one fire-and-forget task through a real executor's worker path
    and epilogue; what the person was shown."""
    centre = NotificationCenter(dedup_window_s=10.0)
    shown = []
    centre.add_listener(shown.append, min_severity=Severity.INFO)
    monkeypatch.setattr(sio, 'notifications', centre)
    executor = SequentialIOExecutor(name='TEST')
    task = IOTask(_jog_step)
    task.set_name(executor.executor_name)
    executor.queue.put(task)
    executor.queue.get()
    result, exception = task.run()
    executor._on_task_done(task, result, exception)
    return shown


def test_is_shown_once_as_a_warning_titled_objective_unknown(monkeypatch):
    shown = _run_on_the_lane(monkeypatch)

    assert [(n.severity, n.title, n.message) for n in shown] == [
        (
            Severity.WARNING,
            'Objective Unknown',
            str(ObjectiveUnknownError('slot_unassigned', slot=2)),
        )
    ]


def test_is_logged_as_a_warning_with_no_traceback(monkeypatch, caplog):
    with caplog.at_level(logging.DEBUG):
        _run_on_the_lane(monkeypatch)

    records = [
        r for r in caplog.records if '_jog_step' in r.getMessage() and r.name != FORENSIC_LOGGER
    ]
    assert records, 'the refusal left no log line'
    assert all(r.levelno == logging.WARNING for r in records), [
        (r.levelname, r.getMessage()) for r in records
    ]
    assert not any(r.exc_info for r in records), 'a refusal was logged with a traceback'
