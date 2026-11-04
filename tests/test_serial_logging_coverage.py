# Copyright Etaluma, Inc.
"""Regression tests: serial-command dark paths reach a log (Rule 13).

Two send paths bypassed the logged serial wrapper and went dark:
  - raw_repl.py (firmware flash / bootloader / config backup) used a bare
    `getLogger(__name__)` ('drivers.raw_repl'), outside the LVP tree, so it
    had no handler -- a board-bricking-capable transport logging into the
    void.
  - the LED STIM capability probe scans multiple reply lines and so cannot
    use exchange_command; it wrote/read the port raw with no serial.log entry.

These assert both paths now log.
"""

import pytest


def test_raw_repl_logger_under_lvp_tree():
    """raw_repl's module logger must parent under 'LVP.' so its records
    propagate to the main + error handlers (bare __name__ has none)."""
    import drivers.raw_repl as raw_repl

    assert raw_repl.logger.name.startswith('LVP.'), (
        f'raw_repl logger is {raw_repl.logger.name!r}; it must be under the '
        f"'LVP.' tree or firmware-flash / bootloader I/O logs into the void"
    )


@pytest.mark.firmware_tier
def test_stim_probe_logs_to_serial_log(caplog):
    """The bespoke STIM capability probe must record its send + reply in
    serial.log -- it scans multiple reply lines and so bypasses
    exchange_command's own logging. Driven by the production driver over
    the field LED firmware on the firmware-backed simulator; the
    LVP.serial record is the observable."""
    import logging

    from tests.wire_boards import led_on_the_wire

    board, _sim = led_on_the_wire()
    with caplog.at_level(logging.INFO, logger='LVP.serial'):
        assert board.supports_firmware_stim() is False

    probe_records = [
        r for r in caplog.records if r.name == 'LVP.serial' and 'STIM 0 0 1 2 1' in r.getMessage()
    ]
    assert probe_records, (
        'supports_firmware_stim must log its raw STIM probe send/reply to '
        'the LVP.serial logger -- it does not go through exchange_command'
    )
    assert 'Command not recognized' in probe_records[0].getMessage(), (
        'the probe reply must be recoverable from the serial.log line'
    )


def test_the_lines_drained_after_a_reply_reach_the_serial_log(caplog):
    """A one-line read takes the first line of a firmware traceback and the
    exchange drains the rest before the next command; the drained lines
    must be in serial.log, or the board's exception is never read. Driven
    against a mock serial port; the LVP.serial record is the observable."""
    import logging
    import threading
    from unittest.mock import MagicMock, PropertyMock

    from drivers.motorboard import MotorBoard

    rest = b'  File "main.py", line 512, in home\r\nOSError: [Errno 5] EIO\r\n'
    # No simulated board answers with a firmware traceback (the sim port's
    # faults garble, delay or disarm), and the subject is the driver's read
    # loop over its port.
    # a stand-in by design: a fault the simulator cannot produce, as above
    board = MotorBoard.__new__(MotorBoard)
    board._lock = threading.RLock()
    board._label = '[XYZ Class ]'
    driver = MagicMock()  # a stand-in by design: the port, as above
    driver.timeout = 1.0
    driver.readline.return_value = b'Traceback (most recent call last):\r\n'
    # Nothing waiting before the write; the traceback's rest once the line is read.
    type(driver).in_waiting = PropertyMock(side_effect=[0, len(rest)])
    driver.read.return_value = rest
    board.driver = driver

    with caplog.at_level(logging.INFO, logger='LVP.serial'):
        reply = board.exchange_command('HOME')

    assert reply == 'Traceback (most recent call last):'
    logged = '\n'.join(r.getMessage() for r in caplog.records if r.name == 'LVP.serial')
    assert 'OSError: [Errno 5] EIO' in logged
