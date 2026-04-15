# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for Lumascope API behavior.

Issue #616 / #618 follow-up — the rename of `xyhome` to `home`:

  Original report (#616): "Homing Failed" notification popup on every
  startup on an LS820 bench board that has only Z wired (no XY stage).
  Root cause: the host-side method was named `xyhome()` even though the
  firmware command it sends (HOME) homes Z, T, and X/Y in the same
  routine. The misleading name led to a session-12 fix that added a
  host-side precondition check skipping `motion.xyhome()` entirely when
  X/Y were missing — which silenced the popup but broke Z homing on
  LS820 boards (#618 follow-up). Discovered in office testing
  2026-04-14: backlash characterization ran against an unhomed Z and
  produced nonsense.

  Structural fix (this commit): rename `xyhome` -> `home` everywhere
  (driver, API, callers, tests) and push the partial-home recognition
  into the driver layer where the firmware response is already known.
  - drivers/motorboard.py::home() recognizes 'ERROR: X not present' /
    'Y not present' as a partial-home success and returns True.
  - Lumascope.home() trusts the driver's verdict — no host-side
    presence check, no False interpretation.
  - The misnamed method was the conceptual trap that allowed both #616
    and #618 to land. The rename retires the trap.
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

from modules.lumascope_api import Lumascope, AxisState
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
        result = null.detect_present_axes()
        assert isinstance(result, list)

    def test_home_returns_true(self):
        """Null home() is a no-op success."""
        null = NullMotionBoard()
        assert null.home() is True

    def test_has_homed_returns_true(self):
        """Null reports homed = True so callers depending on the flag
        don't get stuck waiting."""
        null = NullMotionBoard()
        assert null.has_homed() is True


class TestLumascopeHome:
    """#616 / #618 follow-up: Lumascope.home() must reach the firmware so
    the firmware can home every axis the board has, and must not emit a
    Homing Failed popup when the only failure is X/Y missing on a Z-only
    board (the driver layer recognizes this case and returns True)."""

    def _capture_errors(self):
        """Attach a listener that records every ERROR notification."""
        received = []
        notifications.add_listener(
            lambda n: received.append(n),
            min_severity=Severity.ERROR,
        )
        return received

    def test_home_with_null_motion_board_no_notification(self):
        """home() on a scope with NullMotionBoard must not raise a
        notification. NullMotionBoard.home() returns True, so this hits
        the success branch with an empty axes_present list."""
        received = self._capture_errors()
        scope = Lumascope(simulate=True)
        scope.motion = NullMotionBoard()

        scope.home()

        assert received == [], (
            f"home() on NullMotionBoard emitted unexpected notifications: {received}"
        )

    def test_home_on_z_only_board_marks_z_idle(self):
        """LS820 (Z-only): the driver recognizes the 'X not present'
        firmware response as a partial-home success. The API trusts
        the driver's True return and marks present axes IDLE."""
        received = self._capture_errors()
        scope = Lumascope(simulate=True)

        # Simulate LS820: only Z physically present.
        scope.motion.detect_present_axes = lambda: ['Z']

        # Driver returns True (the partial-home case is its responsibility).
        home_calls = []
        def fake_home(*args, **kwargs):
            home_calls.append((args, kwargs))
            return True
        scope.motion.home = fake_home

        scope.home()

        assert home_calls, (
            "Lumascope.home() must call motion.home() so firmware can "
            "home the axes the board has (#618 follow-up)"
        )
        assert received == [], (
            f"home() on Z-only board with True driver return must not "
            f"notify: {received}"
        )
        assert scope.get_axis_state('Z') == AxisState.IDLE, (
            f"Z must be marked IDLE on success, got {scope.get_axis_state('Z')}"
        )

    def test_home_real_failure_DOES_notify(self):
        """Negative test: when motion.home() returns False, that means a
        REAL failure (no response, hardware error, partial home aborted
        by Z/T error). The API must raise the Homing Failed popup."""
        received = self._capture_errors()
        scope = Lumascope(simulate=True)

        scope.motion.home = lambda *a, **k: False

        scope.home()

        assert received, (
            "Real homing failure (driver returned False) must raise the "
            "Homing Failed notification"
        )
        for ax in scope.axes_present():
            assert scope.get_axis_state(ax) == AxisState.UNKNOWN, (
                f"{ax} must be UNKNOWN after real homing failure"
            )

    def test_home_full_xyz_success(self):
        """Sanity: home() on a simulated LS850-style scope (X+Y+Z) must
        execute and mark all present axes IDLE on the success path."""
        scope = Lumascope(simulate=True)
        present = scope.motion.detect_present_axes()
        assert 'X' in present and 'Y' in present

        home_called = []
        original_home = scope.motion.home
        def spy_home():
            home_called.append(True)
            return original_home()
        scope.motion.home = spy_home

        scope.home()

        assert home_called, "home() on full XYZ hardware must call motion.home"
        for ax in ('X', 'Y', 'Z'):
            assert scope.get_axis_state(ax) == AxisState.IDLE


class TestMotorBoardHomePartialResponse:
    """drivers/motorboard.py::home() must recognize the firmware partial-
    home response ('ERROR: X not present' on LS820) as a success rather
    than a failure. This is the structural fix for #618 follow-up — the
    API layer trusts the driver's verdict, so the partial-home logic
    must live in the driver where the firmware response is already known.

    These tests cover the driver in isolation. test_serial_safety.py
    has the wire-level versions that go through exchange_command.
    """

    def test_partial_home_x_not_present_returns_true(self):
        """LS820: firmware homes Z, then returns 'ERROR: X not present'.
        Driver must return True — Z is at its reference position."""
        from drivers.motorboard import MotorBoard

        board = MotorBoard.__new__(MotorBoard)
        import threading
        board._state_lock = threading.Lock()
        board.initial_homing_complete = False
        board.exchange_command = MagicMock(return_value="ERROR: X not present")

        result = board.home()

        assert result is True, (
            "Driver must treat 'ERROR: X not present' as partial-home "
            "success — firmware homed Z (and T) before reporting missing X"
        )
        assert board.initial_homing_complete is True

    def test_partial_home_y_not_present_returns_true(self):
        """Same as above for missing Y (one-axis bench config)."""
        from drivers.motorboard import MotorBoard

        board = MotorBoard.__new__(MotorBoard)
        import threading
        board._state_lock = threading.Lock()
        board.initial_homing_complete = False
        board.exchange_command = MagicMock(return_value="ERROR: Y not present")

        assert board.home() is True
        assert board.initial_homing_complete is True

    def test_full_home_complete_returns_true(self):
        """Full XYZ board: firmware returns 'XYZ home complete'."""
        from drivers.motorboard import MotorBoard

        board = MotorBoard.__new__(MotorBoard)
        import threading
        board._state_lock = threading.Lock()
        board.initial_homing_complete = False
        board.exchange_command = MagicMock(return_value="XYZ home complete")

        assert board.home() is True
        assert board.initial_homing_complete is True

    def test_real_failure_returns_false(self):
        """Non-partial errors (timeout, hardware fault, Z homing aborted)
        must return False — these are the cases that should raise the
        Homing Failed popup at the API layer."""
        from drivers.motorboard import MotorBoard

        board = MotorBoard.__new__(MotorBoard)
        import threading
        board._state_lock = threading.Lock()
        board.initial_homing_complete = False
        board.exchange_command = MagicMock(return_value="ERROR: timeout")

        assert board.home() is False
        assert board.initial_homing_complete is False

    def test_no_response_returns_false(self):
        """No response (None) means disconnect/timeout — real failure."""
        from drivers.motorboard import MotorBoard

        board = MotorBoard.__new__(MotorBoard)
        import threading
        board._state_lock = threading.Lock()
        board.initial_homing_complete = False
        board.exchange_command = MagicMock(return_value=None)

        assert board.home() is False
        assert board.initial_homing_complete is False


class TestFrameValidityDuringHoming:
    """Issue #609: the frame valid marker was showing green during homing
    because zhome/home/thome never called frame_validity.invalidate().
    The settle-check callback correctly rejects HOMING state, but only if
    the source is actually in _pending — which requires invalidate().

    These tests capture scope.frame_validity.is_valid at the moment the
    motion driver method is executing (axis state is HOMING, motion is in
    progress). They fail before the fix and pass after.
    """

    def test_zhome_marks_frame_invalid_during_motion(self):
        scope = Lumascope(simulate=True)
        captured = {}

        def fake_zhome():
            captured['is_valid'] = scope.frame_validity.is_valid
            captured['z_state'] = scope.get_axis_state('Z')
            captured['pending'] = dict(scope.frame_validity.pending_sources)
            return True
        scope.motion.zhome = fake_zhome

        scope.zhome()

        assert captured['z_state'] == AxisState.HOMING
        assert 'z_move' in captured['pending'], (
            "zhome() must invalidate 'z_move' so frame_validity "
            "can consult the settle-check callback (#609)"
        )
        assert captured['is_valid'] is False, (
            "frame_validity.is_valid must be False while Z is homing — "
            "the frame valid marker should not be green during homing"
        )

    def test_home_marks_frame_invalid_during_motion_full_xyz(self):
        scope = Lumascope(simulate=True)
        present = scope.motion.detect_present_axes()
        assert 'X' in present and 'Y' in present and 'Z' in present
        captured = {}

        original_home = scope.motion.home
        def spy_home():
            captured['is_valid'] = scope.frame_validity.is_valid
            captured['pending'] = dict(scope.frame_validity.pending_sources)
            captured['x_state'] = scope.get_axis_state('X')
            captured['z_state'] = scope.get_axis_state('Z')
            return original_home()
        scope.motion.home = spy_home

        scope.home()

        assert captured['x_state'] == AxisState.HOMING
        assert captured['z_state'] == AxisState.HOMING
        assert 'xy_move' in captured['pending']
        assert 'z_move' in captured['pending']
        assert captured['is_valid'] is False, (
            "frame_validity.is_valid must be False while XYZ are homing"
        )

    def test_home_marks_frame_invalid_z_only_board(self):
        """LS820: only Z present. home() must invalidate z_move only,
        not xy_move or turret (those sources aren't in motion)."""
        scope = Lumascope(simulate=True)
        scope.motion.detect_present_axes = lambda: ['Z']
        captured = {}

        def fake_home(*args, **kwargs):
            captured['pending'] = dict(scope.frame_validity.pending_sources)
            captured['is_valid'] = scope.frame_validity.is_valid
            return True
        scope.motion.home = fake_home

        scope.home()

        assert 'z_move' in captured['pending']
        assert 'xy_move' not in captured['pending']
        assert 'turret' not in captured['pending']
        assert captured['is_valid'] is False

    def test_thome_marks_frame_invalid_during_motion(self):
        scope = Lumascope(simulate=True)
        captured = {}

        original_thome = scope.motion.thome
        def spy_thome():
            captured['is_valid'] = scope.frame_validity.is_valid
            captured['t_state'] = scope.get_axis_state('T')
            captured['pending'] = dict(scope.frame_validity.pending_sources)
            return original_thome()
        scope.motion.thome = spy_thome

        scope.thome()

        assert captured['t_state'] == AxisState.HOMING
        assert 'turret' in captured['pending'], (
            "thome() must invalidate 'turret' so the frame valid marker "
            "goes red while the turret is rotating (#609)"
        )
        assert captured['is_valid'] is False
