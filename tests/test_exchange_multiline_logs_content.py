# Copyright Etaluma, Inc.
"""Regression test: exchange_multiline logs the response content, not a count.

exchange_multiline's serial.log line recorded only "{command} -> {N} lines",
so a multi-line diagnostic / calibration reply was not recoverable from the
log. It now logs the joined response content as well. Driven against the
field LED firmware on the firmware-backed simulator; the LVP.serial record
is the observable.
"""

import logging

import pytest
from serial import SerialBase

from drivers.ledboard import LEDBoard
from drivers.sim_wire.backend import LedBoardSpec, SimWireBackend
from drivers.sim_wire.port import EmulatedPort
from tests.wire_boards import led_on_the_wire

pytestmark = pytest.mark.firmware_tier

_INFO_ENDS = ['CALIBRATION']


def test_success_log_includes_response_content(caplog):
    board, _sim = led_on_the_wire()
    with caplog.at_level(logging.INFO, logger='LVP.serial'):
        result = board.exchange_multiline('INFO', timeout=5, end_markers=_INFO_ENDS)

    lines = result.splitlines()
    assert len(lines) > 1, result

    summary_records = [
        r
        for r in caplog.records
        if r.name == 'LVP.serial' and 'INFO' in r.getMessage() and 'lines' in r.getMessage()
    ]
    assert summary_records, 'exchange_multiline must log a serial.log summary line'
    message = summary_records[-1].getMessage()
    assert ' | '.join(lines[:2]) in message, (
        'exchange_multiline must log the joined response content so a '
        f'multi-line reply is recoverable from serial.log; got: {message}'
    )


class _TimeoutRecordingPort(EmulatedPort):
    """The emulated port, recording every timeout change against how many
    writes had gone out. On Windows a timeout change reconfigures the port,
    which drops bytes the board is still sending; the emulated port drops
    nothing, so the change is what is observed."""

    def __init__(self, *args, **kwargs):
        self.writes = 0
        self.timeout_sets = []
        super().__init__(*args, **kwargs)

    @SerialBase.timeout.setter
    def timeout(self, value):
        self.timeout_sets.append((value, self.writes))
        SerialBase.timeout.fset(self, value)

    def write(self, data):
        self.writes += 1
        return super().write(data)


class _RecordingBackend(SimWireBackend):
    def open(self, **kwargs):
        self.port = _TimeoutRecordingPort(self.led_board, **kwargs)
        return self.port


def test_the_port_timeout_is_not_changed_while_the_reply_arrives():
    backend = _RecordingBackend(None, led=LedBoardSpec('LS850T'))
    board = LEDBoard(backend=backend)
    try:
        port = backend.port
        port.timeout_sets.clear()
        port.writes = 0
        port_timeout = port.timeout

        result = board.exchange_multiline('INFO', timeout=5, end_markers=_INFO_ENDS)

        assert 'Calibration' in result
        during_reply = [value for value, writes in port.timeout_sets[1:-1]]
        assert port.timeout_sets[0][1] == 0, 'the call sets its window before it writes'
        assert during_reply == [], f'timeout changed mid-reply: {port.timeout_sets}'
        assert port.timeout == port_timeout, 'the port timeout is restored'
    finally:
        board.disconnect()
