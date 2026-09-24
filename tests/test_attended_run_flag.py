# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Autofocus button is an ATTENDED run, so its failures reach the user.

Bench 2026-09-14 (bundle SNNone-2026-09-14-151051): three flat-curve
autofocus failures wrote `NOTIFICATION ERROR | Autofocus/Autofocus Failed`
to gui_interactions.log and produced ZERO popups, while every notification
posted after the last run's cleanup rendered normally.

Cause: the manual Autofocus button runs through SequencedCaptureRunner,
which told the notification centre a protocol was running for EVERY run
kind. The autofocus run therefore suppressed its own failure popup, ~0.5 s
before cleanup lowered the flag again.

The existing four tests in test_autofocus_notify_gate.py all pass with that
bug in place: they stop at "was notifications.error called", which is the
segment that still worked. These tests cross the notification centre, which
is where the defect lived.
"""

from __future__ import annotations

import logging

import pytest

from modules.notification_center import NotificationCenter, Severity, notifications


# --------------------------------------------------------------------------
# 1. The run kind decides. The Autofocus button is the attended one.
# --------------------------------------------------------------------------


def _record_flag_calls(monkeypatch):
    """Spy every set_unattended_run() call made during a drive.

    A spy rather than reading the flag after start(): an unattended run
    lowers it again at cleanup, so reading the live value races the
    teardown and passes or fails on timing.
    """
    calls: list[bool] = []
    monkeypatch.setattr(notifications, 'set_unattended_run', lambda value: calls.append(value))
    return calls


@pytest.mark.parametrize(
    ('trigger', 'expected_unattended'),
    [
        ('autofocus', False),  # the Autofocus button -- the bug
        ('scan', True),
        ('protocol', True),
        ('zstack', True),
        ('composite', True),
        ('api_protocol', True),
        ('test', True),  # the harness default keeps its current behaviour
    ],
)
def test_start_declares_attendedness_from_the_run_kind(monkeypatch, trigger, expected_unattended):
    """A real SequencedCaptureRunner, driven through prepare() -> start().

    Not a mock of the runner: mocking the runner is precisely what let this
    ship -- every existing test asserted on the notification call and never
    on what the runner told the centre.
    """
    from tests.protocol_drives import bare_capture_runner, scr_run_kwargs

    calls = _record_flag_calls(monkeypatch)

    runner = bare_capture_runner()
    plan = runner.prepare(**scr_run_kwargs(run_trigger_source=trigger))
    runner.start(plan)

    assert calls, 'start() told the notification centre nothing about this run'
    assert calls[0] is expected_unattended, (
        f'run kind {trigger!r} declared unattended={calls[0]}, expected {expected_unattended}'
    )


# --------------------------------------------------------------------------
# 2. What the flag does at the centre.
# --------------------------------------------------------------------------


def _listening_centre():
    received = []
    nc = NotificationCenter()
    nc.add_listener(received.append, min_severity=Severity.NOTICE)
    return nc, received


def test_unattended_run_suppresses_a_non_fatal_notification():
    nc, received = _listening_centre()
    nc.set_unattended_run(True)

    nc.error('Autofocus', 'Autofocus Failed', 'Focus curve is flat or invalid')

    assert received == [], 'a non-fatal notification escaped an unattended run'


def test_attended_run_delivers_a_non_fatal_notification():
    """The bench case: a flat curve on the Autofocus button must reach the
    listener that renders popups."""
    nc, received = _listening_centre()
    nc.set_unattended_run(False)

    nc.error('Autofocus', 'Autofocus Failed', 'Focus curve is flat or invalid')

    assert len(received) == 1, 'the attended autofocus failure never reached the listener'
    assert received[0].title == 'Autofocus Failed'


@pytest.mark.parametrize('unattended', [True, False])
def test_a_fatal_notification_is_delivered_either_way(unattended):
    """fatal means run-aborting; it crosses the bridge regardless. This is
    the clause that keeps a lost connection visible mid-protocol."""
    nc, received = _listening_centre()
    nc.set_unattended_run(unattended)

    nc.error('Protocol', 'Protocol Aborted', 'Motion timeout', fatal=True)

    assert len(received) == 1


# --------------------------------------------------------------------------
# 3. A suppressed notification leaves a record.
# --------------------------------------------------------------------------


def test_suppression_names_its_reason_in_the_log(caplog):
    """Without this line a customer log cannot answer "did the user see
    this?": notify() writes its forensic gui_interactions entry BEFORE it
    decides whether to dispatch, so that entry means "posted", never "seen".

    Diagnosing the 2026-09-14 bench defect took a bench session, a recovered
    bundle and a 14-row hand census for want of this one line.
    """
    nc, _received = _listening_centre()
    nc.set_unattended_run(True)

    with caplog.at_level(logging.INFO, logger='LVP.notifications'):
        nc.error('Autofocus', 'Autofocus Failed', 'Focus curve is flat or invalid')

    assert any('unattended_run' in r.message for r in caplog.records), (
        'a suppressed notification left no record of WHY it was suppressed; '
        f'saw: {[r.message for r in caplog.records]}'
    )


def test_dedup_suppression_also_names_its_reason(caplog):
    nc, _received = _listening_centre()

    nc.error('Camera', 'Camera not connected', 'Check USB')
    with caplog.at_level(logging.INFO, logger='LVP.notifications'):
        nc.error('Camera', 'Camera not connected', 'Check USB')

    assert any('dedup' in r.message for r in caplog.records), (
        f'dedup suppression left no reason; saw: {[r.message for r in caplog.records]}'
    )
