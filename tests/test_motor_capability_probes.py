# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Motor firmware command-family capability probes.

The driver answers "does the connected firmware implement STOP / the
fan commands / the diagnostic queries" via probe-and-cache predicates
(supports_motor_stop / supports_fan / supports_diagnostics), mirroring
the LED board's firmware-stim probe, so callers gate on the probe
answer instead of firmware version strings or per-call ERROR handling.

Driven against the real firmware on the firmware-backed simulator: the
field firmware (2024-09-10) has neither FANSPEED nor VOLTAGE, the 3.0
firmware has VOLTAGE, and each answer is the firmware's own reply. The
probe's exchanges are read off the serial log, the log a support case
is read from. STOP's probe is `test_motor_stop_capability_probe.py`'s.
"""

from __future__ import annotations

import logging

import pytest

from drivers.exceptions import BoardNoReplyError
from tests.wire_boards import motor_on_the_wire

pytestmark = pytest.mark.firmware_tier


def _exchanges(caplog, command):
    return [
        r for r in caplog.records if r.name == 'LVP.serial' and f' {command} -> ' in r.getMessage()
    ]


def _firmware_errors(caplog):
    return [
        r
        for r in caplog.records
        if r.name == 'LVP.serial'
        and r.levelno == logging.WARNING
        and 'FIRMWARE ERROR' in r.getMessage()
    ]


class TestSupportsPredicates:
    def test_firmware_without_the_families_is_answered_unsupported_once_and_quietly(self, caplog):
        """The field firmware answers both probes with its not-found error:
        unsupported, asked once, and no FIRMWARE ERROR warning, since an
        unsupported command is the answer the probe asked for. The fan probe
        is the read-only FANSPEED, never a FAN:<duty> that would change the
        fan."""
        board, _sim = motor_on_the_wire(dialect='field')
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            assert board.supports_fan() is False
            assert board.supports_fan() is False
            assert board.supports_diagnostics() is False
            assert board.supports_diagnostics() is False
        assert len(_exchanges(caplog, 'FANSPEED')) == 1
        assert len(_exchanges(caplog, 'VOLTAGE')) == 1
        assert not [r for r in caplog.records if ' FAN:' in r.getMessage()]
        assert _firmware_errors(caplog) == []

    def test_firmware_with_the_family_is_answered_supported_once(self, caplog):
        board, _sim = motor_on_the_wire(dialect='3.0')
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            assert board.supports_diagnostics() is True
            assert board.supports_diagnostics() is True
        assert len(_exchanges(caplog, 'VOLTAGE')) == 1

    @pytest.mark.slow
    def test_no_reply_is_inconclusive_and_asked_again(self, caplog):
        """No reply (board absent / wedged) is not a capability answer:
        it raises and is not kept, so the next probe asks the firmware again.
        Slow: the lost reply is waited out for the read's own bound."""
        board, sim = motor_on_the_wire(dialect='3.0')
        sim.drop_next_reply()
        with (
            caplog.at_level(logging.INFO, logger='LVP.serial'),
            pytest.raises(BoardNoReplyError, match=r'^VOLTAGE: '),
        ):
            board.supports_diagnostics()
        assert board.supports_diagnostics() is True
        assert sim.replies_dropped == 1
