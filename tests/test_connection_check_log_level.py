# Copyright Etaluma, Inc.
"""Regression test: periodic connection-check happy-path logs at debug.

are_all_connected() runs on a periodic timer during a protocol scan, so its
"Performing connection check" and "All components connected" lines flooded
the main log (~1.2k/soak). Those happy-path lines are debug (routine, per
Rule 5); the "<board> not connected" lines stay at info -- a board dropping
out mid-run is a real degraded-path event the user should see.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _are_all_connected_body():
    src = (REPO_ROOT / 'modules' / 'lumascope_api' / '_lumascope.py').read_text()
    start = src.find('def are_all_connected(')
    assert start != -1, 'are_all_connected not found'
    body = src[start : start + 1200]
    end = body.find('\n    def ', 1)
    return body if end == -1 else body[:end]


def test_happy_path_lines_are_debug():
    body = _are_all_connected_body()
    assert "logger.debug('[SCOPE API ] Performing connection check" in body, (
        'the periodic "Performing connection check" line must be debug'
    )
    assert "logger.debug('[SCOPE API ] Connection Check: All components connected" in body, (
        'the happy-path "All components connected" line must be debug'
    )


def test_not_connected_lines_stay_info():
    body = _are_all_connected_body()
    for board in ('LED Board', 'Motion Board', 'Camera'):
        assert f"logger.info('[SCOPE API ] Connection Check: {board} not connected" in body, (
            f'the "{board} not connected" degraded-path line must stay at info'
        )
