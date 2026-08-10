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
from unittest.mock import patch

import modules.tech_support_report as tech_support_report
from modules.tech_support_report import TechSupportReport


def test_delimiter_does_not_append_to_the_newest_log_file(tmp_path, caplog):
    decoy = tmp_path / 'camera.log'
    decoy.write_text('camera line\n')
    original = decoy.read_text()

    report = TechSupportReport()
    with (
        patch.object(tech_support_report, '_get_lvp_logs_dir', return_value=tmp_path),
        caplog.at_level(logging.INFO, logger='modules.tech_support_report'),
    ):
        report._write_log_delimiter()

    assert decoy.read_text() == original, 'no direct append into an arbitrary log file'
    assert any(
        'TECH SUPPORT REPORT GENERATION STARTED' in record.getMessage() for record in caplog.records
    ), 'the delimiter must travel through the logging system'
