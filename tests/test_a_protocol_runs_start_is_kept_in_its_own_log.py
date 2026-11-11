# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol run's start is kept in run_start.log until its outcome settles.

On a 22 h field run the rolling trace logs had rotated the run's start away
hours before the bundle was taken. A protocol run (a full protocol or a
single scan) opens the run-start window at its commitment point
(``SequencedCaptureRunner.start``) and ends it where its outcome settles
(``protocol_cleanup.send_run_ended``); the window holds the serial, camera
and api lines in between. A one-position run (autofocus, composite,
Z-stack) and the autofocus scan open none, so pressing Autofocus after a run
leaves the run's start where it is. Only the run that opened the window ends
it: a run whose end comes after a successor opened it leaves it open.
"""

import logging

import pytest

from _lumascope.lib import log_helpers
from _lumascope.lib.log_helpers import VERBOSE, LogWindow
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

_TRACE_LOGGERS = ('LVP.serial', 'LVP.camera', 'LVP.api')


class _File(logging.Handler):
    # a stand-in by design: the rotating file is the window's interchangeable
    # collaborator; the subject is which records reach it.
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def run_start_log():
    """The run-start window on the three trace loggers, passing VERBOSE as
    the start sets them."""
    loggers = [logging.getLogger(name) for name in _TRACE_LOGGERS]
    levels = [lg.level for lg in loggers]
    file = _File()
    for lg in loggers:
        lg.setLevel(VERBOSE)
    log_helpers.write_window(
        LogWindow.RUN_START, file, trace_loggers=tuple(loggers), budget_bytes=1024 * 1024
    )
    keeper = log_helpers._windows[LogWindow.RUN_START]
    try:
        yield file
    finally:
        for lg, level in zip(loggers, levels, strict=True):
            lg.removeHandler(keeper)
            lg.setLevel(level)


def test_a_scans_lines_are_kept_from_its_start_until_its_outcome_settles(tmp_path, run_start_log):
    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        pending = runner.start_single_scan(
            protocol=_protocol([_step('P1', 0, x=20.0, gain=1.0)]), parent_dir=str(tmp_path)
        )
        assert pending.result(timeout=60).status == 'completed'
        written = len(run_start_log.messages)
        logging.getLogger('LVP.api').log(VERBOSE, 'after the run')

    assert run_start_log.messages[0].startswith('run_start.log window opened: a scan run of ')
    assert any(message.startswith('capture_still: frame') for message in run_start_log.messages)
    assert len(run_start_log.messages) == written
    assert 'after the run' not in run_start_log.messages


def test_an_autofocus_run_opens_no_run_start_window(tmp_path, run_start_log):
    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        # The simulated focus curve is flat, so the run fails; ran is enough.
        runner.start_autofocus(channel='BF').result(timeout=60)

    assert run_start_log.messages == []


def test_a_window_is_ended_only_by_the_run_that_opened_it(run_start_log):
    earlier, later = object(), object()
    log_helpers.open_window(LogWindow.RUN_START, 'the later run started', owner=later)

    ended_by_earlier = log_helpers.end_window(LogWindow.RUN_START, owner=earlier)
    logging.getLogger('LVP.serial').log(VERBOSE, 'the later run moves')
    ended_by_later = log_helpers.end_window(LogWindow.RUN_START, owner=later)

    assert (ended_by_earlier, ended_by_later) == (False, True)
    assert run_start_log.messages[-1] == 'the later run moves'
