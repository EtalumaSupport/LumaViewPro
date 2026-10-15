# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run saves its protocol in its folder under the protocol's name.

The copy is named from the run's ``sequence_name``, with ``.tsv`` added to a
name without it. The check for the extension was case-sensitive, so a
protocol named ``plate1.TSV`` was copied as ``plate1.TSV.tsv``.
"""

from __future__ import annotations

from tests.test_a_late_write_records_its_frame import _protocol
from tests.test_a_run_reports_what_it_captured import _two_steps
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 60


def _copies(tmp_path, sequence_name):
    runs = tmp_path / 'runs'
    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        outcome = runner.start_single_scan(
            protocol=_protocol(_two_steps()),
            sequence_name=sequence_name,
            parent_dir=str(runs),
            enable_image_saving=False,
        ).result(timeout=WAIT_S)
    assert outcome.status == 'completed', outcome
    return sorted(p.name for p in runs.glob('*/plate1*'))


def test_a_name_ending_in_tsv_in_any_case_is_kept(tmp_path):
    assert _copies(tmp_path, 'plate1.TSV') == ['plate1.TSV']


def test_a_name_without_tsv_is_given_it(tmp_path):
    assert _copies(tmp_path, 'plate1') == ['plate1.tsv']
