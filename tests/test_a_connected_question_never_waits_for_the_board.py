# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Whether a board is connected is answered while the board is busy talking.

A serial board's exchange holds its serial lock for as long as the board
takes to answer, and the EL-0940 motor board answers HOME only when the home
ends. A connected question that waited on that lock froze the window for a
whole home: the status bar asks `motor_connected` ten times a second while
the pointer is over the live image. Another thread holding the lock stands in
for the exchange in flight here; the question must be answered before it is
released.
"""

import sys
import threading

import pytest

import drivers.serialboard as sb
from tests.scope_fakes import build_scope

# Only a failure waits this long; a passing answer comes back at once.
_FAILURE_BOUND_S = 5.0


def _answer_while_the_lock_is_held(lock, ask):
    holding = threading.Event()
    release = threading.Event()

    def hold():
        with lock:
            holding.set()
            release.wait(_FAILURE_BOUND_S * 2)

    holder = threading.Thread(target=hold, name='exchange-in-flight', daemon=True)
    holder.start()
    assert holding.wait(_FAILURE_BOUND_S)

    answers = []
    asker = threading.Thread(target=lambda: answers.append(ask()), name='asker', daemon=True)
    asker.start()
    asker.join(_FAILURE_BOUND_S)
    answered_while_held = bool(answers)
    release.set()
    holder.join(_FAILURE_BOUND_S)
    asker.join(_FAILURE_BOUND_S)
    assert answered_while_held, 'the connected question waited for the exchange to end'
    return answers[0]


@pytest.mark.parametrize('connected', [True, False])
def test_a_serial_board_answers_while_an_exchange_holds_its_lock(connected):
    board = sb.SerialBoard(vid=0x1234, pid=0x5678, label='[Test]', port='test-port')
    board.driver = object() if connected else None

    assert _answer_while_the_lock_is_held(board._lock, board.is_connected) is connected


@pytest.mark.skipif(
    not (sys.platform == 'darwin' or sys.platform.startswith('linux')),
    reason='the firmware-backed simulator runs on macOS and Linux only',
)
@pytest.mark.parametrize(
    ('driver', 'question'),
    [('_motion_driver', 'motor_connected'), ('_led_driver', 'led_connected')],
)
def test_the_scope_answers_while_a_board_is_busy(driver, question):
    scope = build_scope(
        simulate=True,
        sim_tier='firmware',
        sim_model='LS850T',
        warn_pre_release=False,
        register_atexit=False,
    )
    board = getattr(scope, driver)
    assert isinstance(board, sb.SerialBoard)

    answer = _answer_while_the_lock_is_held(board._lock, lambda: getattr(scope, question))

    assert answer is True
