# Copyright Etaluma, Inc.
"""A run refusal is one report: one WARNING line naming its reason, one warning.

The refusal funnels raise and report nothing; the boundary the caller sits
behind hands the refusal to the one reporter, which logs it once and shows
it once, so whoever reports it again -- a lane's epilogue -- changes
nothing. A check that crashed is a fault instead: raised with the crash
chained, reported by its caller, logged once with both tracebacks.
"""

import logging

import pytest

from modules.exceptions import (
    ProtocolRunRefusedError,
    RecordingRefusedError,
    Refusal,
    RunCheckFailedError,
)
from modules.notification_center import Severity, notifications


def _refuse_through_the_runner():
    from modules.sequenced_capture_runner import SequencedCaptureRunner

    runner = object.__new__(SequencedCaptureRunner)
    runner._refuse(
        reason='already_running',
        message='A run is using the microscope.',
    )


def _refuse_through_the_protocols_api():
    from modules.lumascope_api.protocols import ProtocolsAPI

    api = object.__new__(ProtocolsAPI)
    api._refuse(
        reason='objective_not_mounted',
        message='The protocol names an objective no turret slot holds.',
    )


@pytest.mark.parametrize('refuse', [_refuse_through_the_runner, _refuse_through_the_protocols_api])
def test_a_funnel_refusal_is_one_warning_line_naming_its_reason(centre_posts, caplog, refuse):
    """Raised by the funnel, reported by the GUI's boundary a person's press sits behind."""
    from ui.ui_helpers import run_reported

    refused = []

    def press():
        try:
            refuse()
        except ProtocolRunRefusedError as refusal:
            refused.append(refusal)
            raise

    with caplog.at_level(logging.DEBUG):
        run_reported(press, None, 'RUN')

    [refusal] = refused

    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(records) == 1, [r.getMessage() for r in records]
    assert records[0].levelno == logging.WARNING
    assert f'({refusal.reason})' in records[0].getMessage(), (
        'the one line a shown refusal leaves must name its reason code'
    )
    assert len(centre_posts) == 1
    assert centre_posts[0].severity == Severity.WARNING
    assert centre_posts[0].solicited is True
    assert centre_posts[0].title == refusal.title


def test_a_refusals_words_are_its_sentence():
    run = ProtocolRunRefusedError(reason='already_running', message='Stop it first.')
    recording = RecordingRefusedError(reason='camera_exposure_unknown', message='No exposure.')
    for refusal in (run, recording):
        assert isinstance(refusal, Refusal)
        assert str(refusal) == refusal.message, 'str() must be the sentence, not reason: message'


def test_a_crashed_check_is_one_error_with_both_tracebacks(centre_posts, caplog):
    try:
        try:
            raise OSError('objectives.json missing')
        except OSError as crash:
            raise RunCheckFailedError(
                reason='validation_crashed',
                message='Pre-run validation could not run.',
            ) from crash
    except RunCheckFailedError as raised:
        fault = raised
    with caplog.at_level(logging.DEBUG):
        notifications.report_outcome(fault, solicited=True, category='UI:RUN')

    assert not isinstance(fault, Refusal)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    traced = [r for r in errors if r.exc_info]
    assert len(traced) == 1, 'a fault is logged once with its traceback'
    assert traced[0].exc_info[1].__cause__.args == ('objectives.json missing',), (
        'the crash that stopped the check must travel with the fault'
    )
    assert len(centre_posts) == 1
    assert centre_posts[0].severity == Severity.ERROR
    assert centre_posts[0].title == 'Cannot validate protocol'
    assert centre_posts[0].message == 'Pre-run validation could not run.'
