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
"""

from __future__ import annotations

from unittest.mock import MagicMock


class TestMotorStopCapabilityProbe:
    def _make_board(self, response):
        """Build a MotorBoard stub with exchange_command monkeypatched."""
        from drivers.motorboard import MotorBoard

        board = MotorBoard.__new__(MotorBoard)
        board.exchange_command = MagicMock(return_value=response)
        return board

    def test_motor_stop_passes_expect_unsupported_flag(self):
        """The probe must opt into FIRMWARE ERROR suppression so the
        warning doesn't fire on firmware that doesn't support STOP."""
        board = self._make_board(response='OK')
        board.motor_stop()
        # Single exchange_command call with expect_unsupported=True
        assert board.exchange_command.call_count == 1
        _args, kwargs = board.exchange_command.call_args
        assert kwargs.get('expect_unsupported') is True, (
            'motor_stop must pass expect_unsupported=True so the '
            'FIRMWARE ERROR warning is suppressed on the probe -- '
            'the unsupported case is logged at INFO instead.'
        )

    def test_motor_stop_caches_unsupported_on_error_response(self):
        """ERROR response -> cache unsupported -> return False."""
        board = self._make_board(response="ERROR: command 'STOP' not found:")
        result = board.motor_stop()
        assert result is False
        assert board._supports_stop_cached is False

    def test_motor_stop_caches_supported_on_clean_response(self):
        """Non-ERROR response -> cache supported -> return True."""
        board = self._make_board(response='OK')
        result = board.motor_stop()
        assert result is True
        assert board._supports_stop_cached is True

    def test_motor_stop_skips_wire_when_cached_unsupported(self):
        """Cached unsupported: skip the wire call entirely."""
        board = self._make_board(response='OK')
        board._supports_stop_cached = False
        result = board.motor_stop()
        assert result is False
        assert board.exchange_command.call_count == 0, (
            'Cached unsupported state must skip the wire to avoid '
            're-probing the same firmware repeatedly.'
        )

    def test_motor_stop_shares_cache_with_supports_predicate(self):
        """motor_stop's verdict feeds supports_motor_stop without a
        second wire exchange."""
        board = self._make_board(response="ERROR: command 'STOP' not found:")
        board.motor_stop()
        assert board.supports_motor_stop() is False
        assert board.exchange_command.call_count == 1


class TestExchangeCommandExpectUnsupportedSuppresses:
    """exchange_command(expect_unsupported=True) must NOT fire the
    FIRMWARE ERROR warning when the response carries an ERROR token --
    and the warning must still fire for default callers. Driven through
    the real exchange_command against a mock serial port."""

    def _make_wire_board(self, reply):
        import threading

        import serial

        from drivers.motorboard import MotorBoard

        board = MotorBoard.__new__(MotorBoard)
        board._lock = threading.RLock()
        board._label = '[XYZ Class ]'
        driver = MagicMock(spec=serial.Serial)
        driver.timeout = 1.0
        driver.in_waiting = 0
        driver.readline.return_value = reply.encode('utf-8') + b'\r\n'
        board.driver = driver
        return board

    def _firmware_error_records(self, caplog):
        return [
            r
            for r in caplog.records
            if r.name == 'LVP.serial' and r.levelno == 30 and 'FIRMWARE ERROR' in r.getMessage()
        ]

    def test_warning_fires_by_default_on_error_response(self, caplog):
        """Sanity: default exchange_command (no flag) DOES fire the
        warning when the response contains ERROR. This guards against
        the flag being inverted or always-suppressing."""
        import logging

        board = self._make_wire_board("ERROR: command 'STOP' not found:")
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            resp = board.exchange_command('STOP')
        assert resp is not None and 'ERROR' in resp
        records = self._firmware_error_records(caplog)
        assert len(records) == 1, (
            'a real firmware ERROR must fire the FIRMWARE ERROR warning '
            f'for non-probe callers; got {[r.getMessage() for r in caplog.records]}'
        )
        assert '[XYZ Class ]' in records[0].getMessage(), (
            'the warning must cite the board label so the log line '
            'identifies which board emitted the error'
        )

    def test_expect_unsupported_suppresses_warning(self, caplog):
        """The capability-probe shape: same ERROR reply, flag on -- no
        FIRMWARE ERROR warning."""
        import logging

        board = self._make_wire_board("ERROR: command 'STOP' not found:")
        with caplog.at_level(logging.INFO, logger='LVP.serial'):
            resp = board.exchange_command('STOP', expect_unsupported=True)
        assert resp is not None and 'ERROR' in resp
        assert self._firmware_error_records(caplog) == [], (
            'expect_unsupported=True must suppress the FIRMWARE ERROR '
            'warning -- the probe call site already handles the '
            'unsupported case'
        )
