# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A support zip is named for the serial number the motor board stated at connect.

The name comes from ``DiagnosticsAPI.motor_info().serial_number``, the
connect-time record, never from a question put to the board when the zip is
made: while a run or a control session holds the scope the raw command
channel is refused, and a field zip taken during a run lost its serial
(``SNlogs-``) and logged the refusal as an ERROR. With no serial stated, the
logs-only zip is ``SNlogs-`` and the full report ``SNUNKNOWN-TSR-``.
"""

import logging
from unittest.mock import patch

import pytest

from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.tech_support_report import TechSupportReport
from tests.protocol_drives import run_identity
from tests.settings_fixtures import complete_settings


def _session(tmp_path, microscope):
    return ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope=microscope),
        simulate=True,
    )


@pytest.fixture
def ls850(tmp_path):
    s = _session(tmp_path, 'LS850')
    yield s
    s.shutdown()


@pytest.fixture
def ls620(tmp_path):
    # Built without a motor board: no serial is ever stated.
    s = _session(tmp_path, 'LS620')
    yield s
    s.shutdown()


def _motor_commands_sent(session):
    """Record every command put to the motor board while the patch is held."""
    driver = session.scope._motion_driver
    sent = []
    single, multi = driver.exchange_command, driver.exchange_multiline

    def record_single(command, *a, **k):
        sent.append(command)
        return single(command, *a, **k)

    def record_multi(command, *a, **k):
        sent.append(command)
        return multi(command, *a, **k)

    patches = (
        patch.object(driver, 'exchange_command', record_single),
        patch.object(driver, 'exchange_multiline', record_multi),
    )
    return patches, sent


def _logs_only_while_held(session, tmp_path, hold, caplog):
    patches, sent = _motor_commands_sent(session)
    release = hold(session)
    try:
        with patches[0], patches[1], caplog.at_level(logging.ERROR):
            zip_path = TechSupportReport(session=session).generate_logs_only(
                output_dir=tmp_path / 'out'
            )
    finally:
        release()
    return zip_path, sent


def _a_run(session):
    run = session._activity_claim.try_claim('protocol', run=run_identity())
    return run.release


def _a_control_session(session):
    session_id = session.take_control('plate reader')
    return lambda: session.end_control(session_id)


@pytest.mark.parametrize('hold', [_a_run, _a_control_session], ids=['run', 'control_session'])
def test_a_logs_only_zip_taken_while_the_scope_is_held_names_its_serial(
    ls850, tmp_path, caplog, hold
):
    # parametrize-justified: a run and a control session hold the scope through
    # different gates, each its own refusal of the raw channel.
    serial = ls850.scope.diagnostics.motor_info().serial_number
    assert serial

    zip_path, sent = _logs_only_while_held(ls850, tmp_path, hold, caplog)

    assert zip_path.name.startswith(f'SN{serial}-')
    assert sent == [], 'the zip asked the board while the scope was held'
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], [r.getMessage() for r in errors]


def _full_report_name(report, tmp_path):
    """The full report's ZIP name: its scope steps' serial, through its namer.

    The scope steps and the namer, not ``generate``: its USB step enumerates
    the host's serial ports, which the test process does not reach.
    """
    steps = tmp_path / 'steps'
    steps.mkdir()
    sn = report._run_scope_steps(steps, lambda pct, msg: None, include_bw=False)
    return report._create_zip(steps, sn, tmp_path / 'out').name


def test_a_full_report_is_named_for_the_serial_stated_at_connect_not_a_reply_now(ls850, tmp_path):
    # The board does not answer FULLINFO when the report is made; the name
    # is the serial it stated at connect all the same.
    serial = ls850.scope.diagnostics.motor_info().serial_number
    ls850.scope.simulation.drop_replies('motor', count=None, command='FULLINFO')

    name = _full_report_name(TechSupportReport(session=ls850), tmp_path)

    assert name.startswith(f'SN{serial}-TSR-')


def test_a_scope_whose_board_stated_no_serial_keeps_each_zips_unknown_name(ls620, tmp_path, caplog):
    assert ls620.scope.diagnostics.motor_info().serial_number is None
    report = TechSupportReport(session=ls620)

    with caplog.at_level(logging.ERROR):
        logs_only = report.generate_logs_only(output_dir=tmp_path / 'logs')
        full = _full_report_name(report, tmp_path)

    assert logs_only.name.startswith('SNlogs-')
    assert full.startswith('SNUNKNOWN-TSR-')
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], [r.getMessage() for r in errors]
