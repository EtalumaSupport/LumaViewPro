# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: MotorBoard._diagnostic_query suppresses
FIRMWARE ERROR warnings on legacy firmware.

DRVSTAT_<axis> / FANSPEED / FAN:<duty> are diagnostic commands
added to motor firmware revisions after 2024-09-10. On the legacy
firmware still in the field, each of these returns
``ERROR: command 'X' not found``, and pre-fix the WARNING
"FIRMWARE ERROR: ..." fired from ``serialboard.exchange_command``
BEFORE ``_diagnostic_query`` caught the response and returned
None. The TSR was hitting these commands during its diagnostic
phase, surfacing a WARNING to the user-visible log per command on
every TSR generation against legacy firmware.

Fix: ``_diagnostic_query`` passes ``expect_unsupported=True`` to
``exchange_command``, so the WARNING is suppressed for these
probes. The helper still returns None to the caller and the caller
still treats None as "INCONCLUSIVE -- firmware does not support."

Same shape as ``motor_stop`` (commit 3f3553c) and tracked under
GitHub issue #654.

Driven against the real firmware on the firmware-backed simulator: the
field firmware (2024-09-10) has none of these commands, and the 3.0
firmware answers VOLTAGE. The warning is read off the serial log.
"""

from __future__ import annotations

import logging

import pytest

from drivers.exceptions import HardwareError
from tests.wire_boards import motor_on_the_wire

pytestmark = pytest.mark.firmware_tier


@pytest.fixture(scope='module')
def field_board():
    """One MotorBoard on the field firmware for the module: each test reads it
    or is refused, and leaves it as it found it."""
    board, _sim = motor_on_the_wire(dialect='field')
    yield board
    board.disconnect()


def _firmware_errors(caplog, command):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == 'LVP.serial'
        and r.levelno == logging.WARNING
        and f'FIRMWARE ERROR: {command} ->' in r.getMessage()
    ]


def _exchanged(caplog, command):
    return [
        r for r in caplog.records if r.name == 'LVP.serial' and f' {command} -> ' in r.getMessage()
    ]


class TestDiagnosticQueryCapabilityProbe:
    def test_an_unsupported_drvstat_is_no_reading_and_no_warning(self, field_board, caplog):
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            result = field_board.read_drv_status('Z')
        assert result is None, (
            'ERROR response must be swallowed and returned as None '
            '(unsupported-firmware indicator), not propagated.'
        )
        assert _exchanged(caplog, 'DRVSTAT_Z'), 'the probe never reached the board'
        assert _firmware_errors(caplog, 'DRVSTAT_Z') == []

    def test_an_unsupported_fanspeed_is_no_reading_and_no_warning(self, field_board, caplog):
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            result = field_board.read_fanspeed()
        assert result is None
        assert _exchanged(caplog, 'FANSPEED'), 'the probe never reached the board'
        assert _firmware_errors(caplog, 'FANSPEED') == []

    def test_a_refused_fan_write_raises_and_keeps_its_warning(self, field_board, caplog):
        # Support is supports_fan's answer, asked first: the write itself is
        # not a probe, so an ERROR to it is a fault and keeps its warning.
        with caplog.at_level(logging.INFO, logger='LVP.serial'), pytest.raises(HardwareError):
            field_board.set_fan_duty(0)
        assert len(_firmware_errors(caplog, 'FAN:0')) == 1

    def test_diagnostic_query_returns_response_when_supported(self):
        """When firmware supports the command, the response passes
        through unchanged (no None substitution): the 3.0 firmware's
        VOLTAGE line."""
        board, _sim = motor_on_the_wire(dialect='3.0')
        result = board._diagnostic_query('VOLTAGE')
        assert result.startswith('24V=')

    def test_diagnostic_query_returns_none_on_error_response(self, field_board):
        """When firmware rejects with ERROR prefix, the response is
        swallowed and None is returned (caller treats as inconclusive).

        On the field firmware, which has no DRVSTAT. The query itself, not
        read_drv_status: that read's own parse fallback also answers None
        for an ERROR line, so it cannot tell this clause from its own."""
        assert field_board._diagnostic_query('DRVSTAT_Z') is None
