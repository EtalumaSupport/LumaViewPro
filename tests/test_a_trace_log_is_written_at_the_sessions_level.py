# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The serial, camera and api logs are written at the session's level.

Each of these trace logs set a fixed level of its own, so neither
``debug_mode`` nor ``logging.default.level`` reached it, and its routine
per-command lines were INFO: an overnight 1536-well run filled serial.log in
1.5 h. The session's level is now theirs (``log_helpers.apply_session_level``),
and the routine lines are VERBOSE.
"""

import logging

import pytest

from lib import log_helpers
from lib.log_helpers import VERBOSE
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


class _File(logging.Handler):
    # a stand-in by design: the rotating file is the trace log's
    # interchangeable collaborator; the subject is which records reach it.
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def lines(self) -> list[tuple[str, str]]:
        return [(r.levelname, r.getMessage()) for r in self.records]


@pytest.fixture
def trace_log(request):
    """A trace logger of this test's own, and its file, set up as lvp_logger sets serial.log."""
    traced = logging.getLogger(f'LVP.trace_test.{request.node.name}')
    traced.propagate = False
    file = _File()
    log_helpers.write_trace_log(traced, file)
    yield traced, file
    traced.handlers.clear()


def _session(level='INFO', debug_mode=False):
    return ScopeSession.create(
        complete_settings(
            logging={'default': {'level': level}, 'autofocus': False}, debug_mode=debug_mode
        ),
        simulate=True,
    )


def test_a_written_level_reaches_the_trace_logs(trace_log):
    traced, file = trace_log
    session = _session(level='INFO')
    try:
        session.run_application_session()
        session.update_settings('logging.default.level', 'VERBOSE')
        traced.log(VERBOSE, 'move -> answered')
        traced.debug('a developer internal')
    finally:
        session.shutdown()
    assert file.lines() == [('VERBOSE', 'move -> answered')]


def test_debug_mode_writes_the_trace_logs_at_debug(trace_log):
    traced, file = trace_log
    session = _session(level='INFO', debug_mode=True)
    try:
        session.run_application_session()
        traced.debug('a developer internal')
    finally:
        session.shutdown()
    assert file.lines() == [('DEBUG', 'a developer internal')]
