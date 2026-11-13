# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A motor axis's driver status is the word the firmware's DRVSTAT reply carries.

The 3.0 motor firmware answers ``DRVSTAT_<axis>`` with the TMC5072's
DRV_STATUS word and its decode, ``Z: raw=0x8104003F SG=63 CS=4 STANDSTILL
STALL``. The driver read the reply's first word as the number, so every
reply a 3.0 board sent was unreadable, and the read answered None, which
the support report writes as a firmware that does not offer the command.
A reply that does not carry the asked axis's word is a reply that cannot
be read, and says so; only the firmware's ``ERROR:`` answer means it has
no DRVSTAT (pinned by ``test_diagnostic_query_warning_suppression.py``).
"""

import re

import pytest

from _lumascope.drivers.exceptions import BoardReplyUnreadableError
from _lumascope.drivers.motorboard import MotorBoard
from _lumascope.drivers.sim_wire.backend import MotorBoardSpec, SimWireBackend
from _lumascope.drivers.sim_wire.mp import tmc5072
from _lumascope.modules.scope_session import ScopeSession
from tests.motorconfig_fixtures import SHIPPED_MOTOR_DEFAULTS
from tests.settings_fixtures import complete_settings

# The firmware's own reply, read off the board in a 2026-11-12 support report.
_REPLY = re.compile(r'Z: raw=0x[0-9A-F]{8} SG=\d+ CS=\d+( [A-Z_]+)*')


@pytest.fixture
def board_on_3_0():
    """(MotorBoard, the simulated board behind it) running the 3.0 motor firmware."""
    backend = SimWireBackend(MotorBoardSpec('LS850T', frozenset('XYZT'), dialect='3.0'))
    board = MotorBoard(motorconfig_defaults=SHIPPED_MOTOR_DEFAULTS, backend=backend)
    try:
        yield board, backend.motor_board
    finally:
        board.disconnect()


@pytest.mark.firmware_tier
def test_the_status_is_the_chips_word_as_the_firmware_sends_it(board_on_3_0):
    board, sim = board_on_3_0
    open_load = tmc5072.DRV_OLA | tmc5072.DRV_OLB
    present = board.read_drv_status('Z')
    sim.inject('Z', tmc5072.ABSENT)
    absent = board.read_drv_status('Z')
    assert (present & open_load, present & tmc5072.DRV_STST) == (0, tmc5072.DRV_STST)
    assert absent & open_load == open_load


class _Replies:
    """A board that answers DRVSTAT_<axis> with one reply.

    A stand-in by design: these are replies no simulated board sends, a
    garbled word and another axis's answer.
    """

    _diagnostic_query = MotorBoard._diagnostic_query
    read_drv_status = MotorBoard.read_drv_status

    def __init__(self, reply):
        self._reply = reply

    def exchange_command(self, command, **kwargs):
        return self._reply


def test_a_reply_without_the_axiss_word_is_unreadable_never_not_offered():
    replies = [
        'Z: raw=0x8104 SG=63 CS=4',
        'Y: raw=0x80030000 SG=0 CS=3 STANDSTILL',
        '0x80000000',
        '',
    ]
    read = {}
    for reply in replies:
        try:
            read[reply] = _Replies(reply).read_drv_status('Z')
        except BoardReplyUnreadableError as e:
            read[reply] = str(e).split(':')[0]
    assert read == dict.fromkeys(replies, 'DRVSTAT_Z')


def test_the_fast_tier_answers_drvstat_as_the_3_0_firmware_does(tmp_path):
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live'), microscope='LS850T'),
        simulate=True,
    )
    try:
        reply = session.scope.diagnostics.send_diagnostic_command('motor', 'DRVSTAT_Z')
    finally:
        session.shutdown()
    assert _REPLY.fullmatch(reply), reply
