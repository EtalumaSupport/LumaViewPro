# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The log level and the autofocus score log are the session's settings, on every host.

``logging.default.level`` was read a second time from the settings files by
a GUI-only loader, and put on that loader's own module logger: every other
module logged at the level ``debug_mode`` chose, on every host, and a
stored level outside its range logged a traceback on top of the
preparation's one report (#832). The session now applies both settings
from its prepared settings, when it is created and when either is written,
to the loggers that write the main log; ``debug_mode``, the installation's
switch for a detailed log, holds the level at DEBUG.
"""

import logging

import numpy as np
import pytest

import modules.autofocus_functions as autofocus_functions
from modules.scope_session import ScopeSession
from tests.log_capture import records_of
from tests.settings_fixtures import complete_settings


# The conftest's _no_log_setting_outlives_its_test puts both settings back
# after each test.
_MAIN_LOG = logging.getLogger('LVP')


def _session(level='INFO', autofocus=False, debug_mode=False):
    return ScopeSession.create(
        complete_settings(
            logging={'default': {'level': level}, 'autofocus': autofocus}, debug_mode=debug_mode
        ),
        simulate=True,
    )


def _logs_a_focus_score(caplog):
    caplog.clear()
    autofocus_functions.focus_function(np.arange(64, dtype=np.uint8).reshape(8, 8))
    return any(
        r.levelno == logging.INFO and 'Focus Score' in r.getMessage()
        for r in records_of(caplog, autofocus_functions)
    )


@pytest.mark.parametrize(
    ('stored', 'applied'), [('debug', logging.DEBUG), ('Error', logging.ERROR)]
)
def test_the_stored_level_is_the_main_logs_whatever_its_case(stored, applied):
    session = _session(level=stored)
    try:
        assert _MAIN_LOG.level == applied
        assert logging.getLogger('LVP.modules.scope_session').getEffectiveLevel() == applied
    finally:
        session.shutdown()


def test_a_written_level_takes_effect_as_it_is_written():
    session = _session(level='INFO')
    try:
        session.update_settings('logging.default.level', 'DEBUG')
        assert _MAIN_LOG.level == logging.DEBUG
    finally:
        session.shutdown()


def test_debug_mode_puts_the_main_log_at_debug_whatever_the_level_says():
    # The installation's one switch for a detailed log: on, a written level
    # does not take it below DEBUG.
    session = _session(level='INFO', debug_mode=True)
    try:
        assert _MAIN_LOG.level == logging.DEBUG
        session.update_settings('logging.default.level', 'WARNING')
        assert _MAIN_LOG.level == logging.DEBUG
    finally:
        session.shutdown()


def test_the_autofocus_score_log_follows_its_setting(caplog):
    session = _session(autofocus=True)
    try:
        assert _logs_a_focus_score(caplog)
        session.update_settings('logging.autofocus', False)
        assert not _logs_a_focus_score(caplog)
    finally:
        session.shutdown()
