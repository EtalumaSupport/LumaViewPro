# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An LED current read answers what the board said, never a stand-in for it.

``LEDBoard.read_led_current`` answered None for a reply it missed, one it
could not parse, and our own defect alike, so the support report's leakage
check could not tell a dead board from one with no current sensing. None now
means only the board senses no current; a missed reply raises the wire's
fault and an answer with no current in it is unreadable.

The board's wire is a stand-in by design: the simulated wire runs only the
field LED firmware, which is not v2 and senses no current, and the fast
simulated LED board does not run this driver.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from drivers.exceptions import BoardNoReplyError, BoardReplyUnreadableError
from drivers.ledboard import LEDBoard


def _v2_board(answer):
    # a stand-in by design: no simulated v2 LED board runs this driver
    board = LEDBoard.__new__(LEDBoard)
    board._lock = threading.RLock()
    board._label = '[LED Class ]'
    board.firmware_version = '3.0.8'
    board.exchange_command = MagicMock(
        side_effect=answer if isinstance(answer, Exception) else None, return_value=answer
    )
    return board


def test_a_reply_answers_its_current():
    board = _v2_board(['LED0 I_SENS  (AIN14): 1.2800V  ->   200.1 mA', 'LED_K 0.3', ''])
    assert board.read_led_current(0) == 200.1


def test_a_board_that_senses_no_current_answers_none_and_is_not_asked():
    board = _v2_board(['unused'])
    board.firmware_version = None
    assert board.read_led_current(0) is None
    board.exchange_command.assert_not_called()


def test_no_reply_raises_the_wire_s_fault():
    board = _v2_board(BoardNoReplyError('LEDREAD0: no reply from the board'))
    with pytest.raises(BoardNoReplyError, match=r'^LEDREAD0: no reply from the board$'):
        board.read_led_current(0)


def test_an_answer_with_no_current_in_it_is_unreadable():
    board = _v2_board(['LED0 I_SENS  (AIN14): 1.2800V  ->   ... mA', 'LED_K 0.3', ''])
    with pytest.raises(BoardReplyUnreadableError, match=r'^LEDREAD0: no current in the reply'):
        board.read_led_current(0)
