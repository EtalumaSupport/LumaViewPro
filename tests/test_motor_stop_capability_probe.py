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
            "motor_stop must pass expect_unsupported=True so the "
            "FIRMWARE ERROR warning is suppressed on the probe -- "
            "the unsupported case is logged at INFO instead."
        )

    def test_motor_stop_caches_unsupported_on_error_response(self):
        """ERROR response -> cache _stop_supported=False -> return False."""
        board = self._make_board(
            response="ERROR: command 'STOP' not found:"
        )
        result = board.motor_stop()
        assert result is False
        assert board._stop_supported is False

    def test_motor_stop_caches_supported_on_clean_response(self):
        """Non-ERROR response -> cache _stop_supported=True -> return True."""
        board = self._make_board(response='OK')
        result = board.motor_stop()
        assert result is True
        assert board._stop_supported is True

    def test_motor_stop_skips_wire_when_cached_unsupported(self):
        """Cached unsupported: skip the wire call entirely."""
        board = self._make_board(response='OK')
        board._stop_supported = False
        result = board.motor_stop()
        assert result is False
        assert board.exchange_command.call_count == 0, (
            "Cached unsupported state must skip the wire to avoid "
            "re-probing the same firmware repeatedly."
        )


class TestExchangeCommandExpectUnsupportedSuppresses:
    """exchange_command(expect_unsupported=True) must NOT fire the
    FIRMWARE ERROR warning when the response carries an ERROR token."""

    def test_warning_fires_by_default_on_error_response(self):
        """Sanity: default exchange_command (no flag) DOES fire the
        warning when response contains ERROR. This guards against the
        flag being inverted or always-suppressing."""
        # Static source check -- the warning line exists with the
        # `if not expect_unsupported` guard.
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / 'drivers' / 'serialboard.py').read_text()
        assert 'if not expect_unsupported:' in src, (
            "serialboard.py must gate the FIRMWARE ERROR warning on "
            "`if not expect_unsupported:` so callers opting into "
            "the probe shape can suppress the false alarm."
        )
        assert "_serial_log.warning(f'{self._label} FIRMWARE ERROR:" in src, (
            "FIRMWARE ERROR warning must still fire for non-probe "
            "callers -- the warning is the diagnostic for real "
            "firmware errors."
        )
