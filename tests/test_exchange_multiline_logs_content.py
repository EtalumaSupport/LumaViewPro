# Copyright Etaluma, Inc.
"""Regression test: exchange_multiline logs the response content, not a count.

exchange_multiline's serial.log line recorded only "{command} -> {N} lines",
so a multi-line diagnostic / calibration reply was not recoverable from the
log. It now logs the joined response content as well. Driven against a mock
serial port; the LVP.serial record is the observable.
"""

import logging
import threading
from unittest.mock import MagicMock

import serial

from drivers.ledboard import LEDBoard


def _make_board(reply_lines):
    """LEDBoard with a mock serial port that plays back reply_lines then
    goes quiet (empty reads end the multiline loop)."""
    board = LEDBoard.__new__(LEDBoard)
    board._lock = threading.RLock()
    board._label = '[LED Class ]'
    driver = MagicMock(spec=serial.Serial)
    driver.timeout = 1.0
    driver.in_waiting = 0
    replies = [line.encode('utf-8') + b'\r\n' for line in reply_lines]
    driver.readline.side_effect = replies + [b''] * 20
    board.driver = driver
    return board


def test_success_log_includes_response_content(caplog):
    board = _make_board(['CAL line one', 'CAL line two', 'DONE'])
    with caplog.at_level(logging.INFO, logger='LVP.serial'):
        result = board.exchange_multiline('CALREAD', timeout=2)

    assert result is not None and 'CAL line one' in result

    summary_records = [
        r
        for r in caplog.records
        if r.name == 'LVP.serial' and 'CALREAD' in r.getMessage() and 'lines' in r.getMessage()
    ]
    assert summary_records, 'exchange_multiline must log a serial.log summary line'
    message = summary_records[-1].getMessage()
    assert 'CAL line one | CAL line two' in message, (
        'exchange_multiline must log the joined response content so a '
        f'multi-line reply is recoverable from serial.log; got: {message}'
    )
