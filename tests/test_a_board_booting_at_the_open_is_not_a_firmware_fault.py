# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A board that answers the connect's wake late is identified, and no firmware fault is logged.

The connect wakes a board with a bare newline and drains its answer before
it asks INFO. The field firmwares answer every line that is not a command,
that newline included. A board still booting when its port is opened
answers after the drain, so the answer lands in INFO's reply ahead of the
board's identity. On 2026-11-09 an LS850T launched two seconds after its
board enumerated logged ``FIRMWARE ERROR: INFO`` for it, a fault that did
not happen. An INFO answered with an error and no identity is still one.
"""

import logging

import pytest

from _lumascope.drivers.serialboard import SerialBoard
from tests.scripted_port import LED_PID, LED_VID, ScriptedBackend, ScriptedLink
from tests.wire_boards import motor_on_the_wire

# Past the wake's 100 ms drain and INFO's first 0.5 s line read, as a
# booting board's answer was on the bench.
_BOOTING_S = 0.6


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.firmware_tier
def test_a_field_motor_board_answering_the_wake_late_is_identified_without_a_fault(caplog):
    with caplog.at_level(logging.INFO):
        board, _sim = motor_on_the_wire(first_reply_late_s=_BOOTING_S)

    assert board.firmware_responding is True
    assert board.firmware_date == '2024-09-10'
    assert _warnings(caplog) == []


def test_an_info_answered_with_an_error_and_no_identity_is_a_firmware_fault(caplog):
    # a stand-in by design: no simulated firmware answers INFO with an error
    link = ScriptedLink({'INFO': ["ERROR: command 'INFO' not found: "]})
    board = SerialBoard(LED_VID, LED_PID, '[Test]', backend=ScriptedBackend(link))
    with caplog.at_level(logging.INFO):
        board.connect()
    board.disconnect()

    faults = [r for r in _warnings(caplog) if 'FIRMWARE ERROR: INFO' in r.getMessage()]
    assert len(faults) == 1
