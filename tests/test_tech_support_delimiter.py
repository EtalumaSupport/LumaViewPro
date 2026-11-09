# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
The report's log delimiter travels through the logging system only.

The root logger owns the main-log file handler, so the logger call IS
the write to lumaviewpro.log. The retired direct file append picked its
target by newest mtime, which under a live camera feed was the
camera.log firehose -- the marker polluted an unrelated log and never
reliably reached the one it was meant for.
"""

import logging

from _lumascope.modules.tech_support_report import TechSupportReport
from tests.installation_fixtures import process_logs_at


def test_delimiter_does_not_append_to_the_newest_log_file(
    sim_session, tmp_path, caplog, monkeypatch
):
    decoy = process_logs_at(monkeypatch, tmp_path) / 'camera.log'
    decoy.write_text('camera line\n')
    original = decoy.read_text()

    report = TechSupportReport(sim_session)
    with caplog.at_level(logging.INFO, logger='LVP.support_report'):
        report._write_log_delimiter()

    assert decoy.read_text() == original, 'no direct append into an arbitrary log file'
    assert any(
        'TECH SUPPORT REPORT GENERATION STARTED' in record.getMessage() for record in caplog.records
    ), 'the delimiter must travel through the logging system'


def test_the_delimiter_names_the_report_once(sim_session, caplog):
    report = TechSupportReport(sim_session)
    with caplog.at_level(logging.INFO, logger='LVP.support_report'):
        report._write_log_delimiter()

    message = next(
        record.getMessage()
        for record in caplog.records
        if 'TECH SUPPORT REPORT GENERATION STARTED' in record.getMessage()
    )
    assert message.count('TECH SUPPORT REPORT GENERATION STARTED') == 1
    assert '=' * 72 in message, 'a full rule of = sets the banner apart'
