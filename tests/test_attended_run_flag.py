# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A one-position run is ATTENDED, so its failures reach the user.

Bench 2026-09-14 (bundle SNNone-2026-09-14-151051): three flat-curve
autofocus failures wrote `NOTIFICATION ERROR | Autofocus/Autofocus Failed`
to gui_interactions.log and produced ZERO popups, while every notification
posted after the last run's cleanup rendered normally.

Cause: the manual Autofocus button runs through SequencedCaptureRunner,
which told the notification centre a protocol was running for EVERY run
kind. The autofocus run therefore suppressed its own failure popup, ~0.5 s
before cleanup lowered the flag again.

The fix keyed the mute on the Autofocus button's trigger string, so a REST
or scripted autofocus, composite or z-stack -- the same operation -- was
muted while the button's was not. What the run does decides now: the
one-position kinds are attended, whoever started them.

A test that stops at "was notifications.error called" passes with that bug
in place, since that segment still worked. These tests cross the
notification centre, which is where the defect lived.
"""

from __future__ import annotations

import logging

import pytest

from modules.notification_center import NotificationCenter, Severity, notifications
from modules.protocol_state_machine import SequencedCaptureRunMode


# --------------------------------------------------------------------------
# 1. What the run does decides, never who started it.
# --------------------------------------------------------------------------


def _record_scope_opens(monkeypatch):
    """Spy every open_run_scope() call made during a drive.

    A spy rather than reading the scope after start(): cleanup closes it
    again, so reading the live value races the teardown and passes or fails
    on timing.
    """
    opened: list[bool] = []
    monkeypatch.setattr(
        notifications, 'open_run_scope', lambda *, attended: opened.append(attended)
    )
    return opened


# Each run kind with the trigger its GUI control passes and its API entry's
# default: the same kind must answer the same whichever started it.
_RUN_KINDS = [
    (SequencedCaptureRunMode.SINGLE_AUTOFOCUS, ('autofocus', 'api_autofocus'), True),
    (SequencedCaptureRunMode.SINGLE_COMPOSITE, ('composite', 'api_composite'), True),
    (SequencedCaptureRunMode.SINGLE_ZSTACK, ('zstack', 'api_zstack'), True),
    (SequencedCaptureRunMode.SINGLE_SCAN, ('scan', 'api_scan'), False),
    (SequencedCaptureRunMode.FULL_PROTOCOL, ('protocol', 'api_protocol'), False),
    (
        SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
        ('autofocus_scan', 'api_autofocus_scan'),
        False,
    ),
]


@pytest.mark.parametrize(
    ('run_mode', 'trigger', 'attended'),
    [(mode, trigger, attended) for mode, triggers, attended in _RUN_KINDS for trigger in triggers],
    ids=lambda v: v.value if isinstance(v, SequencedCaptureRunMode) else str(v),
)
def test_start_declares_attendedness_from_the_run_kind(monkeypatch, run_mode, trigger, attended):
    """A real SequencedCaptureRunner, driven through prepare() -> start().

    Not a mock of the runner: mocking the runner is precisely what let the
    Autofocus button's own popup be muted -- every test asserted on the
    notification call and never on what the runner told the centre.
    """
    from tests.protocol_drives import bare_capture_runner, scr_run_kwargs

    opened = _record_scope_opens(monkeypatch)

    runner = bare_capture_runner()
    kwargs = scr_run_kwargs(run_mode=run_mode, run_trigger_source=trigger)
    # A composite is refused unless two channels are set to capture.
    kwargs['protocol'].steps.return_value.__getitem__.return_value.nunique.return_value = 2
    plan = runner.prepare(**kwargs)
    runner.start(plan)

    assert opened, 'start() told the notification centre nothing about this run'
    assert opened[0] is attended, (
        f'{run_mode.value} started by {trigger!r} declared attended={opened[0]}, '
        f'expected {attended}'
    )


def test_a_run_under_a_borrowed_claim_is_unattended(monkeypatch):
    """A diagnostic's autofocus is one step of the diagnostic: its outcome
    goes back to the diagnostic through the handle, and the diagnostic
    decides what to show. The engineering plugin's characterization runs
    autofocus this way, in loops, and records a failure as data."""
    from tests.protocol_drives import bare_capture_runner, scr_run_kwargs

    opened = _record_scope_opens(monkeypatch)

    runner = bare_capture_runner()
    diagnostic = runner._activity_claim.try_claim('diagnostic')
    plan = runner.prepare(
        **scr_run_kwargs(
            run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS,
            run_trigger_source='api_autofocus',
            borrowed_claim=diagnostic.lend(),
        )
    )
    runner.start(plan)

    assert opened == [False]


@pytest.mark.parametrize(
    ('run_mode', 'leds_state_at_end'),
    [(mode, 'return_to_original' if attended else 'off') for mode, _, attended in _RUN_KINDS],
    ids=lambda v: v.value if isinstance(v, SequencedCaptureRunMode) else v,
)
def test_the_same_declaration_decides_how_the_leds_are_left(run_mode, leds_state_at_end):
    """A one-position run hands the LEDs back as it found them; a plate
    traverse ends dark. One fact about the run, read in one place."""
    assert run_mode.leds_state_at_end == leds_state_at_end


# --------------------------------------------------------------------------
# 2. What the run scope does at the centre.
# --------------------------------------------------------------------------


def _listening_centre():
    received = []
    nc = NotificationCenter()
    nc.add_listener((lambda n: n.shown and received.append(n)), min_severity=Severity.NOTICE)
    return nc, received


@pytest.mark.parametrize('attended', [True, False])
def test_a_fatal_notification_is_delivered_either_way(attended):
    """fatal means run-aborting; it crosses the bridge regardless. This is
    the clause that keeps a lost connection visible mid-protocol."""
    nc, received = _listening_centre()
    nc.open_run_scope(attended=attended)

    nc.error('Protocol', 'Protocol Aborted', 'Motion timeout', fatal=True)

    assert len(received) == 1


def _every_post_centre():
    heard = []
    nc = NotificationCenter(dedup_window_s=0.0)
    nc.add_listener(heard.append, min_severity=Severity.NOTICE)
    return nc, heard


def test_an_attended_run_shows_an_identical_post_once():
    """A refused gain repeats on every slice of a z-stack. Shown once; heard
    every time, so a listener that records keeps them all. The dedup window
    is zeroed so only the run's own rule can be what suppresses."""
    nc, heard = _every_post_centre()
    nc.open_run_scope(attended=True)

    for _ in range(3):
        nc.warning('Camera', 'Camera Setting Not Applied', 'The camera rejected 12 dB.')

    assert [n.shown for n in heard] == [True, False, False]


def test_an_attended_run_shows_different_posts_that_share_a_title():
    """A refused gain and a refused exposure carry one title; each is its own
    fault, and each is shown."""
    nc, heard = _every_post_centre()
    nc.open_run_scope(attended=True)

    nc.warning('Camera', 'Camera Setting Not Applied', 'The camera rejected 12 dB.')
    nc.warning('Camera', 'Camera Setting Not Applied', 'The camera rejected 40 ms.')

    assert [n.shown for n in heard] == [True, True]


def test_a_fatal_repeat_is_not_judged_by_the_run():
    nc, heard = _every_post_centre()
    nc.open_run_scope(attended=True)

    nc.error('Protocol', 'Protocol Aborted', 'Motion timeout', fatal=True)
    nc.error('Protocol', 'Protocol Aborted', 'Motion timeout', fatal=True)

    assert [n.shown for n in heard] == [True, True]


def test_a_solicited_post_is_shown_and_counts_as_shown_in_the_run():
    """Someone asked: the answer is shown, and the same words arriving later
    unasked are what that person has already read."""
    nc, heard = _every_post_centre()
    nc.open_run_scope(attended=True)

    nc.warning(
        'Protocol', 'Already Running', 'The Scan run is using the microscope.', solicited=True
    )
    nc.warning('Protocol', 'Already Running', 'The Scan run is using the microscope.')

    assert [n.shown for n in heard] == [True, False]


def test_opening_a_scope_starts_clean_and_closing_ends_it():
    """A missed close is healed by the next run, not carried into it; and a
    post after the close is judged as outside any run."""
    nc, heard = _every_post_centre()
    nc.open_run_scope(attended=True)
    nc.warning('Camera', 'Camera Setting Not Applied', 'The camera rejected 12 dB.')

    nc.open_run_scope(attended=True)
    nc.warning('Camera', 'Camera Setting Not Applied', 'The camera rejected 12 dB.')
    nc.close_run_scope()
    nc.close_run_scope()
    nc.warning('Camera', 'Camera Setting Not Applied', 'The camera rejected 12 dB.')

    assert [n.shown for n in heard] == [True, True, True]


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
    nc.open_run_scope(attended=False)

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
