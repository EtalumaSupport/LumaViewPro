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


class _RecordingPort:
    """A port that plays back reply lines and records every timeout change
    against what had been written so far. On Windows a timeout change
    reconfigures the port, which drops bytes the board is still sending."""

    def __init__(self, reply_lines):
        self._timeout = 0.1
        self.written = []
        self.timeout_sets = []
        self._pending = [line.encode('utf-8') + b'\r\n' for line in reply_lines]

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self.timeout_sets.append((value, len(self.written)))
        self._timeout = value

    @property
    def in_waiting(self):
        return sum(len(p) for p in self._pending) if self.written else 0

    def write(self, data):
        self.written.append(data)

    def readline(self):
        return self._pending.pop(0) if self._pending else b''

    def read(self, n):
        return b''


def test_the_port_timeout_is_not_changed_while_the_reply_arrives():
    board = LEDBoard.__new__(LEDBoard)
    board._lock = threading.RLock()
    board._label = '[LED Class ]'
    port = _RecordingPort(['Engineering Mode: Press q', 'board info', 'LED enable', 'LED on'])
    board.driver = port

    result = board.exchange_multiline('Y', timeout=5, end_markers=['Engineering'])

    assert 'LED on' in result
    during_reply = [value for value, writes in port.timeout_sets[1:-1]]
    assert port.timeout_sets[0][1] == 0, 'the call sets its window before it writes'
    assert during_reply == [], f'timeout changed mid-reply: {port.timeout_sets}'
    assert port.timeout == 0.1, 'the port timeout is restored'
