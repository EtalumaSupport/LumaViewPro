"""Regression for #539: a USB yank mid-operation must not flood the error log.

A disconnect while moving / autofocusing fires auto-reconnect for every queued
command (~73/sec measured), each previously logging a fresh full-stack error.
SerialBoard._log_reconnect_failure dedupes to one full error per error-class
per error-log-interval window; same-class repeats drop to debug. Both reconnect
sites (_exchange_command_impl and exchange_multiline) route through the one
helper, so a flood that bounces between command paths still dedupes against a
single window -- the multiline path (formerly an un-deduped error) is the bug
this closes.
"""

import logging
from types import SimpleNamespace

import drivers.serialboard as sb


_SERIAL_LOGGER = 'LVP.serial'


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _run(calls):
    """Run a sequence of (command, exception) through the helper on one fake
    board, capturing LVP.serial records. Returns (n_error, n_debug)."""
    fake = SimpleNamespace(
        _label='[Motor]',
        _error_log_interval=2.0,
        _last_reconnect_err_class=None,
        _last_reconnect_err_time=0.0,
    )
    log = logging.getLogger(_SERIAL_LOGGER)
    handler = _Capture()
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        for command, exc in calls:
            sb.SerialBoard._log_reconnect_failure(fake, command, exc)
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)
    n_error = sum(1 for r in handler.records if r.levelno == logging.ERROR)
    n_debug = sum(1 for r in handler.records if r.levelno == logging.DEBUG)
    return n_error, n_debug


def test_repeated_same_class_failures_collapse_to_one_error():
    exc = OSError('device not configured')
    # A burst of identical reconnect failures (the USB-yank flood).
    n_error, n_debug = _run([('HOME', exc)] * 50)
    assert n_error == 1, 'only the first failure should log a full error'
    assert n_debug == 49, 'same-class repeats within the window drop to debug'


def test_different_error_class_refires_full_error():
    n_error, _ = _run([('HOME', OSError('x')), ('HOME', ValueError('y'))])
    assert n_error == 2, 'a different error class is a distinct failure, log it fully'
