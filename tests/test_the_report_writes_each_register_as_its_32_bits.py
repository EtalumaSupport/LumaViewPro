# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support report writes each motor driver register as its 32 bits.

The EL-0940's field firmware answers a register read with the word as a
signed decimal, so a DRV_STATUS with its standstill bit (31) set read
-2130313175 in a field report, where the chip's datasheet and whoever
decodes a fault work in the unsigned word, 0x81060029. The report wrote
the reply as the board sent it, and every reader converted by hand.
"""

import re

from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.tech_support_report import TechSupportReport, _NotRead, _register_word
from tests.settings_fixtures import complete_settings


def test_a_reply_that_is_not_a_32_bit_word_is_written_as_the_board_sent_it():
    # Only a 32-bit word in decimal is converted: a read that did not go
    # through, the board's error words and a number no register can hold
    # are the board's answer, and a converted one would be a wrong word.
    not_read = _NotRead('no reply in 2 s')
    cases = {
        '-2130313175': '0x81060029',
        '501': '0x000001F5',
        '4294967295': '0xFFFFFFFF',
        '-2147483648': '0x80000000',
        'ERROR: command not found': 'ERROR: command not found',
        '4294967296': '4294967296',
        '-2147483649': '-2147483649',
        not_read: not_read,
    }
    wrong = {
        reply: _register_word(reply)
        for reply, want in cases.items()
        if _register_word(reply) != want
    }
    assert wrong == {}


def test_every_register_is_written_as_its_unsigned_word_in_hex(tmp_path):
    # The firmware tier runs the field firmware over a modelled TMC5072, the
    # one simulator whose register reads come back signed as the bench's do.
    session = ScopeSession.create(
        complete_settings(
            live_folder=str(tmp_path / 'live'), microscope='LS850T', simulator_tier='firmware'
        ),
        simulate=True,
    )
    try:
        TechSupportReport(session=session)._step_motor_driver_registers(tmp_path)
    finally:
        session.shutdown()
        session.scope.disconnect()

    text = (tmp_path / 'hardware_checks' / 'motor_driver_registers.txt').read_text()
    rows = re.findall(r'^  (\w+)\s*: (.*)$', text, re.M)
    assert len(rows) == 14, text
    not_a_word = [
        (name, value) for name, value in rows if not re.fullmatch(r'0x[0-9A-F]{8}', value)
    ]
    assert not_a_word == []
    # X's driver at standstill: the field firmware sends -2147287040.
    assert ('DRV_STATUS0', '0x80030000') in rows


def test_each_axiss_driver_status_is_written_as_its_word_in_hex(tmp_path):
    # The driver status is the same DRV_STATUS word the register read
    # writes, so a reader compares the two sections word for word.
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope='LS850T'),
        simulate=True,
    )
    try:
        TechSupportReport(session=session)._step_firmware_info(tmp_path)
    finally:
        session.shutdown()
        session.scope.disconnect()

    text = (tmp_path / 'firmware_info' / 'motor_status.txt').read_text()
    section = text.split('Motor Driver Status:\n')[1]
    rows = dict(re.findall(r'^  ([XYZT]): (.*)$', section, re.M))
    assert rows == dict.fromkeys('XYZT', '0x00000000'), text
