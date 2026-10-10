# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An outcome whose own words cannot be written is still reported, and the reporter never raises.

``report_outcome`` is where every outcome ends its flight: lanes, listeners,
a run's cleanup and the claim all hand it what they caught. It formatted
the exception's words unguarded, so an exception whose ``__str__`` raised
made the reporter raise, and the failure escaped into whatever was
reporting it. Now the words are built once through a guarded ``str``, and
the outcome is logged and shown with a sentence naming its type.
"""

import logging

from modules.exceptions import Quiet, Refusal, RefusalCause
from modules.notification_center import Severity, notifications


class _FaultError(Exception):
    def __str__(self):
        raise RuntimeError('the message broke')


class _RefusedError(Refusal, Exception):
    cause = RefusalCause.STATE
    title = 'Refused'

    def __str__(self):
        raise RuntimeError('the message broke')


class _QuietError(Quiet, Exception):
    cause = RefusalCause.STATE

    def __str__(self):
        raise RuntimeError('the message broke')


FALLBACK = '(its message could not be written: RuntimeError)'


def _shown(centre_posts):
    told = {Severity.ERROR, Severity.WARNING}
    return [(n.severity.name.lower(), n.message) for n in centre_posts if n.severity in told]


def _logged(caplog):
    return [r.getMessage() for r in caplog.records if r.name == 'LVP.outcomes']


def test_a_fault_is_logged_in_words_naming_its_type_and_shown(caplog, centre_posts):
    with caplog.at_level(logging.DEBUG, logger='LVP.outcomes'):
        notifications.report_outcome(_FaultError(), solicited=True, category='probe')

    assert any(f'_FaultError {FALLBACK}' in line for line in _logged(caplog))
    assert [method for method, _ in _shown(centre_posts)] == ['error']


def test_a_refusal_is_shown_in_words_naming_its_type(caplog, centre_posts):
    with caplog.at_level(logging.DEBUG, logger='LVP.outcomes'):
        notifications.report_outcome(_RefusedError(), solicited=True, category='probe')

    assert _shown(centre_posts) == [('warning', f'_RefusedError {FALLBACK}')]


def test_a_quiet_outcome_is_logged_in_words_naming_its_type(caplog, centre_posts):
    with caplog.at_level(logging.DEBUG, logger='LVP.outcomes'):
        notifications.report_outcome(_QuietError(), solicited=True, category='probe')

    assert any(f'_QuietError {FALLBACK}' in line for line in _logged(caplog))
    assert _shown(centre_posts) == []


def test_a_refusal_that_is_only_logged_is_logged_in_words_naming_its_type(caplog, centre_posts):
    with caplog.at_level(logging.DEBUG, logger='LVP.outcomes'):
        notifications.report_outcome(
            _RefusedError(), solicited=True, category='probe', log_only=True
        )

    assert any(f'_RefusedError {FALLBACK}' in line for line in _logged(caplog))
    assert _shown(centre_posts) == []
