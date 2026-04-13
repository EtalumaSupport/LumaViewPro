# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for Lumascope API behavior.

Issue #616 (xyhome on Z-only / no-hardware units):
  User reported a "Homing Failed" notification popup on every startup when
  running on an LS820 bench board that has only Z wired (no XY stage), and
  also when no motor hardware is plugged in at all. Root cause: lumaviewpro
  on_start() scheduled move_home('XY') unconditionally, and Lumascope.xyhome()
  had no precondition check — it called motion.xyhome() regardless of which
  axes were physically present, so firmware returned "ERROR: X not present"
  and the scope emitted a user-facing error.

  Fix: Lumascope.xyhome() reads self.motion.detect_present_axes() and silently
  no-ops when X or Y is missing. NullMotionBoard gained a detect_present_axes
  method returning [] so the null-drop-in path produces the same behavior as
  a real board with no XY stage.
"""

import sys
from unittest.mock import MagicMock

# Mock heavy deps before importing
sys.modules.setdefault('userpaths', MagicMock())
sys.modules.setdefault('requests', MagicMock())
sys.modules.setdefault('requests.structures', MagicMock())
_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = MagicMock()
_mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
_mock_lvp_logger.unpause_thread = MagicMock()
_mock_lvp_logger.pause_thread = MagicMock()
sys.modules.setdefault('lvp_logger', _mock_lvp_logger)

import pytest

from modules.lumascope_api import Lumascope
from modules.notification_center import notifications, Severity
from drivers.null_motorboard import NullMotionBoard


class TestNullMotionBoardCapabilities:
    """NullMotionBoard must be a faithful drop-in for the motor driver interface."""

    def test_detect_present_axes_returns_empty(self):
        """Null board has no physically present axes."""
        null = NullMotionBoard()
        assert null.detect_present_axes() == []

    def test_detect_present_axes_callable_without_args(self):
        """Callers must not need to pass any arguments."""
        null = NullMotionBoard()
        # Must not raise
        result = null.detect_present_axes()
        assert isinstance(result, list)


class TestXYHomePresentGuard:
    """#616: xyhome must skip silently when X/Y are not physically present."""

    def _capture_errors(self):
        """Attach a listener that records every ERROR notification emitted
        during the test. Returns the list; clear it between tests."""
        received = []
        notifications.add_listener(
            lambda n: received.append(n),
            min_severity=Severity.ERROR,
        )
        return received

    def test_xyhome_noop_with_null_motion_board(self):
        """xyhome on a scope with NullMotionBoard must not raise a notification."""
        received = self._capture_errors()
        scope = Lumascope(simulate=True)
        # Force the motion board to be the null drop-in
        scope.motion = NullMotionBoard()

        # Call xyhome — must return without raising and without emitting
        # any ERROR notification.
        scope.xyhome()

        assert received == [], (
            f"xyhome on NullMotionBoard emitted unexpected notifications: {received}"
        )

    def test_xyhome_noop_with_z_only_hardware(self):
        """xyhome on a Z-only unit (bench board) must skip silently."""
        received = self._capture_errors()
        scope = Lumascope(simulate=True)

        # Monkey-patch the motion board's detect_present_axes to report Z only,
        # simulating an LS820 bench board without an XY stage.
        scope.motion.detect_present_axes = lambda: ['Z']

        # Also spy on motion.xyhome to verify it is NOT called — the guard
        # should prevent the hardware call entirely.
        original_xyhome = scope.motion.xyhome
        xyhome_calls = []

        def spy_xyhome(*args, **kwargs):
            xyhome_calls.append((args, kwargs))
            return original_xyhome(*args, **kwargs)

        scope.motion.xyhome = spy_xyhome

        scope.xyhome()

        assert received == [], f"xyhome on Z-only unit emitted notifications: {received}"
        assert xyhome_calls == [], (
            f"xyhome on Z-only unit incorrectly called motion.xyhome: {xyhome_calls}"
        )

    def test_xyhome_noop_when_only_x_present(self):
        """Edge: missing Y alone should still skip (can't home XY stage with one axis)."""
        received = self._capture_errors()
        scope = Lumascope(simulate=True)
        scope.motion.detect_present_axes = lambda: ['X', 'Z']

        xyhome_calls = []
        scope.motion.xyhome = lambda *a, **k: xyhome_calls.append(True)

        scope.xyhome()

        assert received == []
        assert xyhome_calls == []

    def test_xyhome_runs_on_full_xyz_hardware(self):
        """Sanity: xyhome on a simulated LS850-style scope (X+Y+Z) must execute."""
        scope = Lumascope(simulate=True)
        # Simulator normally reports full axes; confirm the precondition.
        present = scope.motion.detect_present_axes()
        assert 'X' in present and 'Y' in present, (
            f"Precondition failed: simulated motor should report X+Y, got {present}"
        )

        xyhome_called = []
        original_xyhome = scope.motion.xyhome

        def spy_xyhome():
            xyhome_called.append(True)
            return original_xyhome()

        scope.motion.xyhome = spy_xyhome

        # Must not raise
        scope.xyhome()
        assert xyhome_called, "xyhome on full XYZ hardware should call motion.xyhome"

    def test_xyhome_handles_detect_present_axes_exception(self):
        """Defensive: if detect_present_axes raises, xyhome must still skip
        silently (not propagate the exception to the caller)."""
        received = self._capture_errors()
        scope = Lumascope(simulate=True)

        def raise_err():
            raise RuntimeError("motor board disconnected")

        scope.motion.detect_present_axes = raise_err
        xyhome_calls = []
        scope.motion.xyhome = lambda *a, **k: xyhome_calls.append(True)

        # Must not raise
        scope.xyhome()
        assert received == []
        assert xyhome_calls == []
