# Copyright Etaluma, Inc.
"""A run refusal is one report: one WARNING line naming its reason, one warning.

The refusal funnels hand the refusal to the one reporter and raise that same
object, so whoever reports it again -- the GUI boundary, a lane's epilogue --
changes nothing. A check that crashed is a fault instead: raised with the crash
chained, reported by its caller, logged once with both tracebacks.
"""

import logging

import pytest

from modules.exceptions import (
    ProtocolRunRefusedError,
    RecordingRefusedError,
    Refusal,
    RunCheckFailedError,
    SettingsSaveRefusedError,
)
from modules.notification_center import NotificationCenter, Severity


@pytest.fixture
def centre(monkeypatch):
    import modules.notification_center as nc

    centre = NotificationCenter(dedup_window_s=10.0)
    seen = []
    centre.add_listener(seen.append, min_severity=Severity.DEBUG)
    monkeypatch.setattr(nc, 'notifications', centre)
    centre.seen = seen
    return centre


def _refuse_through_the_runner():
    from modules.sequenced_capture_runner import SequencedCaptureRunner

    runner = object.__new__(SequencedCaptureRunner)
    with pytest.raises(ProtocolRunRefusedError) as raised:
        runner._refuse(
            reason='already_running',
            title='Already Running',
            message='A run is using the microscope.',
        )
    return raised.value


def _refuse_through_the_protocols_api():
    from modules.lumascope_api.protocols import ProtocolsAPI

    api = object.__new__(ProtocolsAPI)
    with pytest.raises(ProtocolRunRefusedError) as raised:
        api._refuse(
            reason='objective_not_mounted',
            title='Objective Not On The Turret',
            message='The protocol names an objective no turret slot holds.',
        )
    return raised.value


@pytest.mark.parametrize('refuse', [_refuse_through_the_runner, _refuse_through_the_protocols_api])
def test_a_funnel_refusal_is_one_warning_line_naming_its_reason(centre, caplog, refuse):
    with caplog.at_level(logging.DEBUG):
        refusal = refuse()

    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(records) == 1, [r.getMessage() for r in records]
    assert records[0].levelno == logging.WARNING
    assert f'({refusal.reason})' in records[0].getMessage(), (
        'the one line a shown refusal leaves must name its reason code'
    )
    assert len(centre.seen) == 1
    assert centre.seen[0].severity == Severity.WARNING
    assert centre.seen[0].solicited is True
    assert centre.seen[0].title == refusal.title


@pytest.mark.parametrize('refuse', [_refuse_through_the_runner, _refuse_through_the_protocols_api])
def test_reporting_a_funnel_refusal_again_changes_nothing(centre, caplog, refuse):
    refusal = refuse()
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        centre.report_outcome(refusal, solicited=True, category='UI:RUN')

    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
    assert len(centre.seen) == 1, 'a second report of the same refusal was shown'


def test_a_refusals_words_are_its_sentence():
    run = ProtocolRunRefusedError(reason='already_running', title='T', message='Stop it first.')
    recording = RecordingRefusedError(reason='camera_inactive', title='T', message='No camera.')
    for refusal in (run, recording):
        assert isinstance(refusal, Refusal)
        assert str(refusal) == refusal.message, 'str() must be the sentence, not reason: message'


@pytest.mark.parametrize('reason', ['settings_provisional', 'no_hardware'])
def test_a_refused_settings_save_says_why_in_a_sentence(reason):
    refusal = SettingsSaveRefusedError(reason=reason, file='./data/current.json')

    assert isinstance(refusal, Refusal)
    assert refusal.title == 'Settings Not Saved'
    assert str(refusal).startswith('The settings were not saved to ./data/current.json:')
    assert reason not in str(refusal), 'the code is for machines; the sentence is for people'


def test_a_crashed_check_is_one_error_with_both_tracebacks(centre, caplog):
    try:
        try:
            raise OSError('objectives.json missing')
        except OSError as crash:
            raise RunCheckFailedError(
                reason='validation_crashed',
                title='Cannot validate protocol',
                message='Pre-run validation could not run.',
            ) from crash
    except RunCheckFailedError as raised:
        fault = raised
    with caplog.at_level(logging.DEBUG):
        centre.report_outcome(fault, solicited=True, category='UI:RUN')

    assert not isinstance(fault, Refusal)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    traced = [r for r in errors if r.exc_info]
    assert len(traced) == 1, 'a fault is logged once with its traceback'
    assert traced[0].exc_info[1].__cause__.args == ('objectives.json missing',), (
        'the crash that stopped the check must travel with the fault'
    )
    assert len(centre.seen) == 1
    assert centre.seen[0].severity == Severity.ERROR
    assert centre.seen[0].title == 'Cannot validate protocol'
    assert centre.seen[0].message == 'Pre-run validation could not run.'
