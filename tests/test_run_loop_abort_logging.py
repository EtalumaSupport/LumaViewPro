# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: scan-summary log lines tell the truth about aborts.

Bug shape: the per-scan summary lines fired unconditionally after
scan_loop() returned, logging "completed" for a scan the user aborted
mid-flight. The run loop already computes the truth (the aborted event
drives run_status for cleanup); the summary wording must branch on it.

Source-text assertion (quote/wrap agnostic): the module must not log a
scan summary whose wording is unconditionally "completed".
"""

import pathlib
import re

import modules.protocol_run_loop as protocol_run_loop


def _source() -> str:
    return pathlib.Path(protocol_run_loop.__file__).read_text()


def test_scan_summary_wording_branches_on_abort_state():
    src = _source()
    # The INFO summary: any f-string logging "Protocol scan ... completed in"
    # as a fixed literal is abort-blind.
    unconditional = re.findall(r'Protocol scan \{[^}]+\}\s+completed in', src)
    assert not unconditional, (
        'the scan summary must branch its wording on the aborted state '
        f'(found unconditional wording: {unconditional})'
    )
    # The wording branch must exist: "aborted" appears in a scan-summary
    # context tied to the aborted event.
    assert re.search(r'aborted[^\n]{0,120}scan|scan[^\n]{0,120}aborted', src, re.IGNORECASE), (
        'expected an abort-aware scan summary wording'
    )


def test_debug_scan_counter_line_is_abort_aware():
    src = _source()
    unconditional = re.findall(r'Scan \{[^}]+\}/\{[^}]+\}\s+completed', src)
    assert not unconditional, (
        'the per-scan debug counter must not claim "completed" for an aborted scan '
        f'(found: {unconditional})'
    )
