# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The log level and the autofocus score log are the session's settings, on every host.

``logging.default.level`` was read a second time from the settings files by
a GUI-only loader, and put on that loader's own module logger: every other
module logged at the level ``debug_mode`` chose, on every host, and a
stored level outside its range logged a traceback on top of the
preparation's one report (#832). The session now applies both settings
from its prepared settings, when it is created and when either is written,
to the loggers that write the main log.
"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

import modules.autofocus_functions as autofocus_functions
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def restored_logging():
    lvp = logging.getLogger('LVP')
    before = (lvp.level, autofocus_functions._enable_af_score_logging)
    yield lvp
    lvp.setLevel(before[0])
    autofocus_functions.enable_af_score_logging(enable=before[1])


def _session(level='INFO', autofocus=False):
    return ScopeSession.create(
        complete_settings(logging={'default': {'level': level}, 'autofocus': autofocus}),
        simulate=True,
    )


def _logs_a_focus_score(monkeypatch):
    logged = MagicMock()
    monkeypatch.setattr(autofocus_functions, 'logger', logged)
    autofocus_functions.focus_function(np.arange(64, dtype=np.uint8).reshape(8, 8))
    return any('Focus Score' in str(call) for call in logged.info.call_args_list)


@pytest.mark.parametrize(
    ('stored', 'applied'), [('debug', logging.DEBUG), ('Error', logging.ERROR)]
)
def test_the_stored_level_is_the_main_logs_whatever_its_case(restored_logging, stored, applied):
    session = _session(level=stored)
    try:
        assert restored_logging.level == applied
        assert logging.getLogger('LVP.modules.scope_session').getEffectiveLevel() == applied
    finally:
        session.shutdown()


def test_a_written_level_takes_effect_as_it_is_written(restored_logging):
    session = _session(level='INFO')
    try:
        session.update_settings('logging.default.level', 'DEBUG')
        assert restored_logging.level == logging.DEBUG
    finally:
        session.shutdown()


def test_the_autofocus_score_log_follows_its_setting(restored_logging, monkeypatch):
    session = _session(autofocus=True)
    try:
        assert _logs_a_focus_score(monkeypatch)
        session.update_settings('logging.autofocus', False)
        assert not _logs_a_focus_score(monkeypatch)
    finally:
        session.shutdown()
