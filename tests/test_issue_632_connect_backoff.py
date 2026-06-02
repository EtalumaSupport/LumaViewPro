"""Regression for #632: a failed connect must not be immediately retried by
the next command, which produced a duplicate startup failure log.

At construction the board calls connect(); if the port is missing or held it
logs 'connect() failed' (#1) and leaves driver=None. The very next command
after construction -- CONFIG (motor) or LEDS_OFF (LED, from _safety_leds_off
in __init__) -- enters the auto-reconnect path, re-runs the full
open+reset+detect sequence, and logs the identical 'connect() failed' (#2).

A short reconnect backoff (reuse _error_log_interval) skips the immediate
re-attempt: a board that just failed to open won't appear within a couple
seconds. Genuine recovery of a re-plugged board happens past the window and
is unaffected.
"""

import time
from types import SimpleNamespace

import drivers.serialboard as sb


def _fake(last_fail, interval=2.0):
    return SimpleNamespace(_last_connect_fail_time=last_fail, _error_log_interval=interval)


def test_backoff_active_within_window():
    assert sb.SerialBoard._reconnect_backoff_active(_fake(time.monotonic())) is True


def test_backoff_inactive_after_window():
    assert sb.SerialBoard._reconnect_backoff_active(_fake(time.monotonic() - 5.0)) is False


def test_backoff_inactive_when_never_failed():
    assert sb.SerialBoard._reconnect_backoff_active(_fake(0.0)) is False


def test_command_after_failed_connect_does_not_reattempt():
    """A command right after a failed connect must NOT re-run connect (no
    duplicate failure) while the backoff window is open; after it expires it
    must reconnect again."""
    board = sb.SerialBoard(vid=0x1234, pid=0x5678, label='[Test]')

    open_calls = {'n': 0}

    def _raising_open():
        open_calls['n'] += 1
        raise OSError('no such port')

    board._open_serial = _raising_open

    board.connect()  # fails -> logs once, arms backoff
    assert open_calls['n'] == 1
    assert board.driver is None

    # Immediate command: backoff active -> no second connect attempt.
    board.exchange_command('LEDS_OFF')
    assert open_calls['n'] == 1, 'command right after a failed connect must not re-attempt'

    # Window expired: a later command does reconnect (recovery preserved).
    board._last_connect_fail_time = time.monotonic() - 5.0
    board.exchange_command('LEDS_OFF')
    assert open_calls['n'] == 2, 'after the backoff window, reconnect resumes'
