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

import pytest

from modules.exceptions import Quiet, Refusal
from modules.notification_center import notifications


class _FaultError(Exception):
    def __str__(self):
        raise RuntimeError('the message broke')


class _RefusedError(Refusal, Exception):
    title = 'Refused'

    def __str__(self):
        raise RuntimeError('the message broke')


class _QuietError(Quiet, Exception):
    def __str__(self):
        raise RuntimeError('the message broke')


FALLBACK = '(its message could not be written: RuntimeError)'


@pytest.fixture
def shown(monkeypatch):
    posts = []
    for method in ('error', 'warning'):

        def post(category, title, message, _method=method, **kw):
            posts.append((_method, message))
            return True

        monkeypatch.setattr(notifications, method, post)
    return posts


def _logged(caplog):
    return [r.getMessage() for r in caplog.records if r.name == 'LVP.outcomes']


def test_a_fault_is_logged_in_words_naming_its_type_and_shown(caplog, shown):
    with caplog.at_level(logging.DEBUG, logger='LVP.outcomes'):
        notifications.report_outcome(_FaultError(), solicited=True, category='probe')

    assert any(f'_FaultError {FALLBACK}' in line for line in _logged(caplog))
    assert [method for method, _ in shown] == ['error']


def test_a_refusal_is_shown_in_words_naming_its_type(caplog, shown):
    with caplog.at_level(logging.DEBUG, logger='LVP.outcomes'):
        notifications.report_outcome(_RefusedError(), solicited=True, category='probe')

    assert shown == [('warning', f'_RefusedError {FALLBACK}')]


def test_a_quiet_outcome_is_logged_in_words_naming_its_type(caplog, shown):
    with caplog.at_level(logging.DEBUG, logger='LVP.outcomes'):
        notifications.report_outcome(_QuietError(), solicited=True, category='probe')

    assert any(f'_QuietError {FALLBACK}' in line for line in _logged(caplog))
    assert shown == []


def test_a_refusal_that_is_only_logged_is_logged_in_words_naming_its_type(caplog, shown):
    with caplog.at_level(logging.DEBUG, logger='LVP.outcomes'):
        notifications.report_outcome(
            _RefusedError(), solicited=True, category='probe', log_only=True
        )

    assert any(f'_RefusedError {FALLBACK}' in line for line in _logged(caplog))
    assert shown == []
