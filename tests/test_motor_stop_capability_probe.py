# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: MotorBoard.motor_stop probes firmware support
quietly.

Field firmware (e.g. EL-0940 2024-09-10) does not implement the
``STOP`` command and replies ``ERROR: command 'STOP' not found``.
LVP sends STOP on shutdown via ``Lumascope.stop_motion`` -> driver
``motor_stop`` -> wire ``STOP``. Pre-fix, the FIRMWARE ERROR
warning from ``exchange_command`` fired every shutdown:

  [WARNING] serialboard.py - [XYZ Class ] FIRMWARE ERROR:
  STOP -> ERROR: command 'STOP' not found:

Then ``motor_stop`` caught the response, cached
``_stop_supported=False``, and logged its OWN info-level "firmware
does not support STOP" message. The user saw both messages -- one
alarming (WARNING), one reassuring (INFO) -- for an expected,
handled condition.

Fix: ``motor_stop`` passes ``expect_unsupported=True`` to
``exchange_command``, which suppresses the WARNING for this probe.
The INFO-level message from ``motor_stop`` is the single, accurate
log line for the unsupported-firmware case.

Driven against the real firmware on the firmware-backed simulator: the
3.0 firmware implements STOP (it answers ``STOPPED``) and the field
firmware does not (it answers the ``not found`` error above), so each
case is the firmware's own reply rather than a hand-written one.
"""

from __future__ import annotations

import logging
import sys

import pytest

from drivers.motorboard import MotorBoard
from drivers.sim_wire.backend import MotorBoardSpec, SimWireBackend

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )


@pytest.fixture
def board(request):
    """A MotorBoard on the real firmware of the requested dialect."""
    b = MotorBoard(
        backend=SimWireBackend(MotorBoardSpec('LS850T', frozenset('XYZT'), dialect=request.param))
    )
    try:
        yield b
    finally:
        b.disconnect()


def _serial_records(caplog, predicate):
    return [r for r in caplog.records if r.name == 'LVP.serial' and predicate(r)]


def _firmware_errors(caplog):
    return _serial_records(
        caplog, lambda r: r.levelno == logging.WARNING and 'FIRMWARE ERROR' in r.getMessage()
    )


def _stop_exchanges(caplog):
    return _serial_records(caplog, lambda r: ' STOP -> ' in r.getMessage())


class TestMotorStopCapabilityProbe:
    @pytest.mark.parametrize('board', ['field'], indirect=True)
    def test_the_probe_on_firmware_without_stop_logs_no_firmware_error(self, board, caplog):
        """The probe must opt into FIRMWARE ERROR suppression so the
        warning doesn't fire on firmware that doesn't support STOP."""
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            board.motor_stop()
        assert len(_stop_exchanges(caplog)) == 1
        assert _firmware_errors(caplog) == [], (
            'motor_stop must pass expect_unsupported=True so the '
            'FIRMWARE ERROR warning is suppressed on the probe -- '
            'the unsupported case is logged at INFO instead.'
        )

    @pytest.mark.parametrize('board', ['field'], indirect=True)
    def test_motor_stop_caches_unsupported_on_error_response(self, board):
        """ERROR response -> cache unsupported -> return False."""
        assert board.motor_stop() is False
        assert board._supports_stop_cached is False

    @pytest.mark.parametrize('board', ['3.0'], indirect=True)
    def test_motor_stop_caches_supported_on_clean_response(self, board):
        """Non-ERROR response -> cache supported -> return True."""
        assert board.motor_stop() is True
        assert board._supports_stop_cached is True

    @pytest.mark.parametrize('board', ['field'], indirect=True)
    def test_motor_stop_skips_wire_when_cached_unsupported(self, board, caplog):
        """Cached unsupported: skip the wire call entirely."""
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            assert board.motor_stop() is False
            assert board.motor_stop() is False
        assert len(_stop_exchanges(caplog)) == 1, (
            'Cached unsupported state must skip the wire to avoid '
            're-probing the same firmware repeatedly.'
        )

    @pytest.mark.parametrize('board', ['field'], indirect=True)
    def test_motor_stop_shares_cache_with_supports_predicate(self, board, caplog):
        """motor_stop's verdict feeds supports_motor_stop without a
        second wire exchange."""
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            board.motor_stop()
            assert board.supports_motor_stop() is False
        assert len(_stop_exchanges(caplog)) == 1


class TestExchangeCommandExpectUnsupportedSuppresses:
    """exchange_command(expect_unsupported=True) must NOT fire the
    FIRMWARE ERROR warning when the response carries an ERROR token --
    and the warning must still fire for default callers. Driven through
    the real exchange_command against the field firmware's own reply."""

    @pytest.mark.parametrize('board', ['field'], indirect=True)
    def test_warning_fires_by_default_on_error_response(self, board, caplog):
        """Sanity: default exchange_command (no flag) DOES fire the
        warning when the response contains ERROR. This guards against
        the flag being inverted or always-suppressing."""
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            resp = board.exchange_command('STOP')
        assert resp == "ERROR: command 'STOP' not found:"
        records = _firmware_errors(caplog)
        assert len(records) == 1, (
            'a real firmware ERROR must fire the FIRMWARE ERROR warning '
            f'for non-probe callers; got {[r.getMessage() for r in caplog.records]}'
        )
        assert '[XYZ Class ]' in records[0].getMessage(), (
            'the warning must cite the board label so the log line '
            'identifies which board emitted the error'
        )

    @pytest.mark.parametrize('board', ['field'], indirect=True)
    def test_expect_unsupported_suppresses_warning(self, board, caplog):
        """The capability-probe shape: same ERROR reply, flag on -- no
        FIRMWARE ERROR warning."""
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            resp = board.exchange_command('STOP', expect_unsupported=True)
        assert resp == "ERROR: command 'STOP' not found:"
        assert _firmware_errors(caplog) == [], (
            'expect_unsupported=True must suppress the FIRMWARE ERROR '
            'warning -- the probe call site already handles the '
            'unsupported case'
        )
