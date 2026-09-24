# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An outcome is logged once and shown at most once, as its type says.

The kind -- refusal, quiet or fault -- the words, the title and the log level
come from the exception; whoever reports it says only whether a person just
asked and which category it belongs to. Reporting the same exception object a
second time, from any thread, adds nothing.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import CancelledError

import pytest

from modules.exceptions import (
    CaptureError,
    HardwareCommandRefusedError,
    ObjectiveUnknownError,
    RunAlreadyEndedError,
)
from modules.notification_center import REFUSAL_OPERATION_KEY, NotificationCenter, Severity

OUTCOMES = 'LVP.outcomes'
NOTIFICATIONS = 'LVP.notifications'


@pytest.fixture
def centre():
    c = NotificationCenter(dedup_window_s=10.0)
    c.shown = []
    c.add_listener(c.shown.append, min_severity=Severity.INFO)
    return c


def _records(caplog):
    return [
        (r.name, r.levelno, bool(r.exc_info))
        for r in caplog.records
        if r.name in (OUTCOMES, NOTIFICATIONS)
    ]


def _raised(exc):
    # A reported exception carries the traceback of where it was raised.
    try:
        raise exc
    except type(exc) as e:
        return e


class TestARefusal:
    def test_shown_as_a_warning_under_its_title_in_its_words_with_one_warning_line(
        self, centre, caplog
    ):
        refusal = ObjectiveUnknownError('slot_unassigned', slot=2)
        with caplog.at_level(logging.DEBUG):
            centre.report_outcome(refusal, solicited=True, category='UI:PICK')

        assert [
            (n.severity, n.title, n.message, n.operation_key, n.solicited) for n in centre.shown
        ] == [(Severity.WARNING, 'Objective Unknown', str(refusal), REFUSAL_OPERATION_KEY, True)]
        assert _records(caplog) == [(NOTIFICATIONS, logging.WARNING, False)]

    def test_not_shown_is_one_warning_naming_its_reason_without_a_traceback(self, centre, caplog):
        refusal = ObjectiveUnknownError('slot_unknown')
        with caplog.at_level(logging.DEBUG):
            centre.report_outcome(refusal, solicited=False, category='Task:x', log_only=True)

        assert centre.shown == []
        assert _records(caplog) == [(OUTCOMES, logging.WARNING, False)]
        assert 'slot_unknown' in caplog.records[-1].getMessage()


class TestAFault:
    def test_typed_is_shown_in_its_own_words_and_logged_with_its_traceback(self, centre, caplog):
        fault = _raised(CaptureError('the camera stopped delivering frames', 'no_frame'))
        with caplog.at_level(logging.DEBUG):
            centre.report_outcome(
                fault, solicited=False, category='Task:grab', fault_title='Grab failed'
            )

        assert [(n.severity, n.title, n.message) for n in centre.shown] == [
            (Severity.ERROR, 'Grab failed', 'the camera stopped delivering frames')
        ]
        assert _records(caplog) == [
            (OUTCOMES, logging.ERROR, True),
            (NOTIFICATIONS, logging.ERROR, False),
        ]

    def test_untyped_is_shown_in_the_generic_sentence(self, centre):
        centre.report_outcome(_raised(KeyError('slot')), solicited=False, category='Task:x')

        assert [n.message for n in centre.shown] == [
            'The operation did not complete. Check the main log for details.'
        ]
        assert [n.title for n in centre.shown] == ['Operation failed']


@pytest.mark.parametrize(
    'quiet',
    [RunAlreadyEndedError('the run has ended'), CancelledError()],
    ids=['quiet_marker', 'cancel'],
)
def test_a_quiet_outcome_is_logged_at_info_and_never_shown(centre, caplog, quiet):
    with caplog.at_level(logging.DEBUG):
        centre.report_outcome(quiet, solicited=True, category='UI:STOP')

    assert centre.shown == []
    assert _records(caplog) == [(OUTCOMES, logging.INFO, False)]


def test_the_same_object_reported_again_adds_nothing(centre, caplog):
    refusal = HardwareCommandRefusedError('exclusive_activity_running', 'move', 'protocol')
    with caplog.at_level(logging.DEBUG):
        centre.report_outcome(refusal, solicited=True, category='Task:move')
        centre.report_outcome(refusal, solicited=True, category='UI:JOG')

    assert len(centre.shown) == 1
    assert _records(caplog) == [(NOTIFICATIONS, logging.WARNING, False)]


def test_a_later_report_may_show_what_an_earlier_one_only_logged(centre):
    refusal = HardwareCommandRefusedError('exclusive_activity_running', 'move', 'protocol')
    centre.report_outcome(refusal, solicited=False, category='Task:move', log_only=True)
    centre.report_outcome(refusal, solicited=True, category='UI:JOG')

    assert [n.title for n in centre.shown] == ['Microscope Busy']


def test_two_threads_reporting_one_object_show_it_once(centre):
    fault = _raised(RuntimeError('board gone'))
    barrier = threading.Barrier(8)

    def _report():
        barrier.wait()
        centre.report_outcome(fault, solicited=False, category='Task:x')

    threads = [threading.Thread(target=_report) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(2.0)

    assert len(centre.shown) == 1


def test_a_listener_that_reports_does_not_deadlock(centre):
    inner = _raised(RuntimeError('reported from a listener'))
    done = threading.Event()

    def _listener(notification):
        if notification.category == 'Task:outer':
            centre.report_outcome(inner, solicited=False, category='Task:inner')

    centre.add_listener(_listener, min_severity=Severity.INFO)

    def _run():
        centre.report_outcome(
            _raised(RuntimeError('outer')), solicited=False, category='Task:outer'
        )
        done.set()

    threading.Thread(target=_run, daemon=True).start()
    assert done.wait(2.0), 'the reporter held the centre lock while notifying'
    assert [n.category for n in centre.shown] == ['Task:outer', 'Task:inner']
