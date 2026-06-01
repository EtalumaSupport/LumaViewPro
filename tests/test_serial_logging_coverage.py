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

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_raw_repl_logger_under_lvp_tree():
    """raw_repl's module logger must parent under 'LVP.' so its records
    propagate to the main + error handlers (bare __name__ has none)."""
    import drivers.raw_repl as raw_repl

    assert raw_repl.logger.name.startswith('LVP.'), (
        f"raw_repl logger is {raw_repl.logger.name!r}; it must be under the "
        f"'LVP.' tree or firmware-flash / bootloader I/O logs into the void"
    )


def test_stim_probe_logs_to_serial_log():
    """The bespoke STIM capability probe must record its send + reply in
    serial.log (it bypasses exchange_command's own logging)."""
    src = (REPO_ROOT / 'drivers' / 'ledboard.py').read_text()
    start = src.find('def supports_firmware_stim(')
    assert start != -1, 'supports_firmware_stim not found'
    body = src[start : src.find('\n    def ', start + 1)]
    assert '_serial_log' in body, (
        'supports_firmware_stim must log its raw STIM probe send/reply to '
        'the LVP.serial logger -- it does not go through exchange_command'
    )
