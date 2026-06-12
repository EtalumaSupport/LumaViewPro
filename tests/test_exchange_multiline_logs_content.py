# Copyright Etaluma, Inc.
"""Regression test: exchange_multiline logs the response content, not a count.

exchange_multiline's serial.log line recorded only "{command} -> {N} lines",
so a multi-line diagnostic / calibration reply was not recoverable from the
log. It now logs the joined response content as well.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _exchange_multiline_body():
    src = (REPO_ROOT / 'drivers' / 'serialboard.py').read_text()
    start = src.find('def exchange_multiline(')
    assert start != -1, 'exchange_multiline not found'
    end = src.find('\n    def ', start + 1)
    return src[start:end] if end != -1 else src[start:]


def test_success_log_includes_response_content():
    body = _exchange_multiline_body()
    # The success-path info log must carry the joined line content, not only
    # the line count.
    assert "' | '.join(lines)" in body, (
        'exchange_multiline must log the joined response content so a '
        'multi-line reply is recoverable from serial.log'
    )
    assert 'shown' in body and '{shown!r}' in body, (
        'the response content must appear in the serial.log success line'
    )
