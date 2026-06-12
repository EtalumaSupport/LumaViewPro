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

def test_raw_repl_logger_under_lvp_tree():
    """raw_repl's module logger must parent under 'LVP.' so its records
    propagate to the main + error handlers (bare __name__ has none)."""
    import drivers.raw_repl as raw_repl

    assert raw_repl.logger.name.startswith('LVP.'), (
        f"raw_repl logger is {raw_repl.logger.name!r}; it must be under the "
        f"'LVP.' tree or firmware-flash / bootloader I/O logs into the void"
    )


def test_stim_probe_logs_to_serial_log(caplog, monkeypatch):
    """The bespoke STIM capability probe must record its send + reply in
    serial.log -- it scans multiple reply lines and so bypasses
    exchange_command's own logging. Driven against a mock serial port;
    the LVP.serial record is the observable."""
    import logging
    import threading
    import time
    from unittest.mock import MagicMock

    from drivers.ledboard import LEDBoard

    board = LEDBoard.__new__(LEDBoard)
    board._lock = threading.RLock()
    board._label = '[LED Class ]'
    driver = MagicMock()
    driver.timeout = 1.0
    driver.readline.return_value = b'STIM: mA must be > 0\r\n'
    driver.in_waiting = 0
    board.driver = driver

    monkeypatch.setattr(time, 'sleep', lambda s: None)
    with caplog.at_level(logging.INFO, logger='LVP.serial'):
        assert board.supports_firmware_stim() is True

    probe_records = [
        r for r in caplog.records
        if r.name == 'LVP.serial' and 'STIM 0 0 1 2 1' in r.getMessage()
    ]
    assert probe_records, (
        'supports_firmware_stim must log its raw STIM probe send/reply to '
        'the LVP.serial logger -- it does not go through exchange_command'
    )
    assert 'STIM: mA must be > 0' in probe_records[0].getMessage(), (
        'the probe reply must be recoverable from the serial.log line'
    )
