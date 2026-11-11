# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The serial, camera and api logs: written at the session's level; bring-up kept in bringup.log.

Each of these trace logs set a fixed level of its own, so neither
``debug_mode`` nor ``logging.default.level`` reached it, and its routine
per-command lines were INFO: an overnight 1536-well run filled serial.log in
1.5 h. The session's level is now its file's (``log_helpers.apply_session_level``),
from the first line to the last; the VERBOSE lines a file at INFO leaves out
are kept, the last few, and written before the next warning, so a fault
carries the commands that led to it. Bring-up's traffic, which a long run
rotates out of the rolling files hours before a bundle is taken, is kept in
a window file of its own (``LogWindow.BRING_UP``) at VERBOSE whatever the
session's level, from the session's creation until its startup motion
ends, appended below the bring-ups before it.
"""

import logging

import pytest

from _lumascope.drivers.exceptions import BoardNoReplyError
from _lumascope.lib import log_helpers
from _lumascope.lib.log_helpers import VERBOSE, LogWindow
from _lumascope.modules.lumascope_api.bring_up import Part
from _lumascope.modules.scope_session import ScopeSession
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
    """A trace logger of this test's own, its file and its bring-up window's
    file, set up as lvp_logger sets serial.log and bringup.log."""
    traced = logging.getLogger(f'LVP.trace_test.{request.node.name}')
    traced.propagate = False
    file, window = _File(), _File()
    log_helpers.write_trace_log(traced, file, held_lines=3)
    log_helpers.write_window(
        LogWindow.BRING_UP, window, trace_loggers=(traced,), budget_bytes=1024 * 1024
    )
    yield traced, file, window
    traced.handlers.clear()


def _window_lines(window: _File) -> list[str]:
    """The window's lines, its headers marked as such."""
    return [
        'HEADER' if message.endswith('window opened: a session is being brought up') else message
        for _, message in window.lines()
    ]


def _session(level='INFO', debug_mode=False):
    return ScopeSession.create(
        complete_settings(
            logging={'default': {'level': level}, 'autofocus': False}, debug_mode=debug_mode
        ),
        simulate=True,
    )


def test_bring_up_is_kept_in_its_window_and_the_rolling_file_at_the_sessions_level(trace_log):
    traced, file, window = trace_log
    session = _session(level='INFO')
    try:
        traced.log(VERBOSE, 'FULLINFO -> answered')
        session.run_application_session()
        traced.log(VERBOSE, 'move -> answered')
        traced.info('port reset')
    finally:
        session.shutdown()
    assert file.lines() == [('INFO', 'port reset')]
    assert _window_lines(window) == ['HEADER', 'FULLINFO -> answered']


def test_each_bring_up_is_appended_below_the_one_before(trace_log):
    traced, _, window = trace_log
    for command in ('first FULLINFO', 'second FULLINFO'):
        session = _session(level='INFO')
        try:
            traced.log(VERBOSE, command)
            session.run_application_session()
            traced.log(VERBOSE, 'move -> answered')
        finally:
            session.shutdown()
    assert _window_lines(window) == ['HEADER', 'first FULLINFO', 'HEADER', 'second FULLINFO']


def test_a_window_that_spends_its_budget_says_so_and_writes_no_more():
    traced = logging.getLogger('LVP.trace_test.budget')
    traced.propagate = False
    traced.setLevel(VERBOSE)
    window = _File()
    log_helpers.write_window(LogWindow.BRING_UP, window, trace_loggers=(traced,), budget_bytes=100)
    try:
        log_helpers.begin_bring_up()
        for n in range(10):
            traced.log(VERBOSE, f'exchange {n} ' + 'x' * 30)
    finally:
        traced.handlers.clear()
    messages = [message for _, message in window.lines()]
    assert messages[1:-1] == [f'exchange {n} ' + 'x' * 30 for n in range(3)]
    assert messages[-1] == (
        'bringup.log window closed: it wrote its budget of 100 bytes before its end'
    )


def test_a_written_level_reaches_the_trace_logs(trace_log):
    traced, file, _ = trace_log
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
    traced, file, _ = trace_log
    session = _session(level='INFO', debug_mode=True)
    try:
        session.run_application_session()
        traced.debug('a developer internal')
    finally:
        session.shutdown()
    assert file.lines() == [('DEBUG', 'a developer internal')]


def test_the_commands_before_a_missed_reply_are_written_before_its_warning(sim_scope):
    serial = logging.getLogger('LVP.serial')
    file = _File()
    handlers = list(serial.handlers)
    log_helpers.write_trace_log(serial, file, held_lines=2)
    log_helpers.apply_session_level('INFO')
    try:
        for command in ('INFO', 'STATUS', 'FULLINFO'):
            sim_scope.diagnostics.send_diagnostic_command('motor', command)
        sim_scope.simulation.drop_replies(Part.MOTOR, count=2, command='ACTUAL')
        for _ in range(2):
            with pytest.raises(BoardNoReplyError):
                sim_scope.diagnostics.send_diagnostic_command('motor', 'ACTUAL R X')
    finally:
        serial.handlers[:] = handlers
        log_helpers._trace_files.pop(serial)
    written = file.lines()
    assert [level for level, _ in written] == ['INFO', 'VERBOSE', 'VERBOSE', 'WARNING', 'WARNING']
    assert written[0][1] == 'The 2 VERBOSE lines before the WARNING that follows:'
    assert written[1][1].startswith('[XYZ Sim] STATUS -> ')
    assert written[2][1].startswith('[XYZ Sim] FULLINFO -> ')
    assert 'ACTUAL R X -> INJECTED TIMEOUT' in written[3][1]
