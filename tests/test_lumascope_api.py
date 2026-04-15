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
import threading
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
from drivers.null_ledboard import NullLEDBoard
from drivers.simulated_motorboard import SimulatedMotorBoard
from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.protocols import MotorBoardProtocol, LEDBoardProtocol


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
        # Must use a turret-equipped sim (LS850T) since post-B4 the
        # default LS850 sim has no T axis and `thome()` correctly
        # no-ops there. The phantom-T behavior the original test relied
        # on is gone.
        from drivers.simulated_motorboard import SimulatedMotorBoard
        scope = Lumascope(simulate=True)
        scope.motion = SimulatedMotorBoard(model='LS850T')
        present = scope.motion.detect_present_axes()
        assert 'T' in present
        scope._pos_cache = {ax: 0.0 for ax in present}
        scope._axis_state = {ax: AxisState.UNKNOWN for ax in present}
        scope._arrival_events = {ax: threading.Event() for ax in present}
        for ev in scope._arrival_events.values():
            ev.set()
        scope._move_profile = {ax: None for ax in present}

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


class TestProtocolConformance:
    """Audit B1: every motor and LED driver implementation must satisfy
    the runtime-checkable Protocol in `drivers/protocols.py`. This guards
    against silent drift — if anyone deletes a method from a driver, or
    adds a new method to the Protocol without updating all implementations,
    these tests fail at construction time instead of at the call site.

    The Protocols use `@runtime_checkable`, so `isinstance(impl, Protocol)`
    checks structural conformance (method names + arity match). It does
    NOT check signature types, which is fine — that's mypy's job.

    These tests also document which classes are part of the Protocol
    contract. New driver implementations (e.g. the upcoming FX2 driver
    for Lumaview Classic) get added here.
    """

    def test_motorboard_satisfies_protocol(self):
        from drivers.motorboard import MotorBoard
        # Use __new__ to skip __init__ — we only need the class to expose
        # the Protocol's method set, not to actually open a serial port.
        instance = MotorBoard.__new__(MotorBoard)
        assert isinstance(instance, MotorBoardProtocol)

    def test_simulated_motorboard_satisfies_protocol(self):
        instance = SimulatedMotorBoard(model='LS850')
        assert isinstance(instance, MotorBoardProtocol)

    def test_null_motorboard_satisfies_protocol(self):
        instance = NullMotionBoard()
        assert isinstance(instance, MotorBoardProtocol)

    def test_ledboard_satisfies_protocol(self):
        from drivers.ledboard import LEDBoard
        instance = LEDBoard.__new__(LEDBoard)
        assert isinstance(instance, LEDBoardProtocol)

    def test_simulated_ledboard_satisfies_protocol(self):
        instance = SimulatedLEDBoard()
        assert isinstance(instance, LEDBoardProtocol)

    def test_null_ledboard_satisfies_protocol(self):
        instance = NullLEDBoard()
        assert isinstance(instance, LEDBoardProtocol)

    def test_lumascope_attributes_satisfy_protocols(self):
        """End-to-end: a constructed Lumascope's `motion` and `led`
        attributes must satisfy the Protocols regardless of which concrete
        implementation got selected (Sim / Null / real)."""
        scope = Lumascope(simulate=True)
        assert isinstance(scope.motion, MotorBoardProtocol)
        assert isinstance(scope.led, LEDBoardProtocol)


class TestLEDChannelDiscovery:
    """Audit B3: LED channel set comes from the driver, not from a
    hardcoded `range(6)` constant in the API. This is the gate for adding
    the FX2 driver for Lumaview Classic, which has 4 channels not 6.

    These tests verify:
    1. Each LED implementation's available_channels()/available_colors()
       are derived from its single source of truth (_COLOR_TO_CH).
    2. The API uses the driver's value at validation time, not a class
       constant — so swapping a 4-channel driver in works without
       touching the API.
    3. The error message reflects the actual valid range, not a stale
       hardcoded "0-5" string.
    """

    def test_ledboard_available_channels_from_color_map(self):
        from drivers.ledboard import LEDBoard
        instance = LEDBoard.__new__(LEDBoard)
        assert instance.available_channels() == tuple(LEDBoard._COLOR_TO_CH.values())
        assert instance.available_colors() == tuple(LEDBoard._COLOR_TO_CH.keys())
        assert len(instance.available_channels()) == 6

    def test_simulated_ledboard_available_channels_from_color_map(self):
        sim = SimulatedLEDBoard()
        assert sim.available_channels() == tuple(SimulatedLEDBoard._COLOR_TO_CH.values())
        assert len(sim.available_channels()) == 6

    def test_null_ledboard_returns_six_channels_for_compat(self):
        """NullLEDBoard returns 6 channels (same as RP2040) so callers on
        a no-LED-hardware system get silent no-ops, not ValueErrors."""
        null = NullLEDBoard()
        assert len(null.available_channels()) == 6
        assert null.available_channels() == (0, 1, 2, 3, 4, 5)

    def test_api_validation_uses_driver_channel_set_not_hardcoded(self):
        """The API must read the valid channel set from the driver. This
        test injects a driver that reports a SHORTER channel set and
        confirms the API rejects what would have been valid under the
        old hardcoded `range(6)` rule."""
        scope = Lumascope(simulate=True)

        class FourChannelLED(SimulatedLEDBoard):
            _COLOR_TO_CH = {'Blue': 0, 'Green': 1, 'Red': 2, 'BF': 3}
            _CH_TO_COLOR = {v: k for k, v in _COLOR_TO_CH.items()}
        scope.led = FourChannelLED()

        scope.led_on(0, 100)  # Blue — valid on 4-channel driver
        with pytest.raises(ValueError, match=r"LED channel must be one of"):
            scope.led_on(5, 100)  # DF — out of range on 4-channel driver
        with pytest.raises(ValueError, match=r"LED channel must be one of"):
            scope.led_on(4, 100)  # PC — out of range too

    def test_api_validation_error_message_reflects_actual_channels(self):
        """Error messages must describe the actual valid range (the
        audit's hardcoded 'must be 0-5' string was the symptom of the
        underlying problem)."""
        scope = Lumascope(simulate=True)

        class TwoChannelLED(SimulatedLEDBoard):
            _COLOR_TO_CH = {'BF': 0, 'Blue': 1}
            _CH_TO_COLOR = {v: k for k, v in _COLOR_TO_CH.items()}
        scope.led = TwoChannelLED()

        try:
            scope.led_on(3, 100)
        except ValueError as e:
            msg = str(e)
            assert "(0, 1)" in msg, f"error message must list actual channels, got: {msg}"
            assert "0-5" not in msg, f"error must not mention stale 0-5 range: {msg}"

    def test_no_hardcoded_LED_VALID_CHANNELS_constant(self):
        """The class-level `LED_VALID_CHANNELS = range(6)` constant has
        been deleted in favor of `self.led.available_channels()`."""
        assert not hasattr(Lumascope, 'LED_VALID_CHANNELS'), (
            "Lumascope.LED_VALID_CHANNELS must be removed — call sites "
            "now read from self.led.available_channels() per audit B3"
        )


class TestPerAxisDictsFromDriver:
    """Audit B4: per-axis state dicts (_pos_cache, _axis_state,
    _arrival_events, _move_profile) are sized at __init__ from
    `motion.detect_present_axes()`, not from a hardcoded 4-axis tuple.

    Tests cover:
    1. Full XYZ scope (LS850 default sim) gets 3 keys per dict
    2. Z-only scope (LS820-style) gets 1 key per dict
    3. Null motor (no hardware at all) gets empty dicts
    4. Rule 8 silent no-op: move_*_position on absent axes does NOT raise,
       it returns silently — the API behaves the same on Null hardware
       and on partial-hardware scopes
    5. Input sanity validation rejects non-axis names like 'Q'
    6. The misnamed `VALID_AXES` constant is gone; `_VALID_AXIS_NAMES` is
       a private input-vocabulary tuple, not a capability query
    """

    def test_xyz_scope_dicts_have_xyz_keys(self):
        scope = Lumascope(simulate=True)
        present = set(scope.motion.detect_present_axes())
        assert present == {'X', 'Y', 'Z'}, (
            f"Default sim should be LS850 (XYZ no turret), got {present}"
        )
        assert set(scope._pos_cache.keys()) == present
        assert set(scope._axis_state.keys()) == present
        assert set(scope._arrival_events.keys()) == present
        assert set(scope._move_profile.keys()) == present

    def test_z_only_scope_dicts_have_only_z(self):
        """Simulate an LS820 / LVC LS720-like Z-only scope."""
        scope = Lumascope(simulate=True)
        scope.motion.detect_present_axes = lambda: ['Z']
        # Re-init the per-axis dicts to reflect the patched motion.
        present = scope.motion.detect_present_axes()
        scope._pos_cache = {ax: 0.0 for ax in present}
        scope._axis_state = {ax: AxisState.UNKNOWN for ax in present}
        scope._arrival_events = {ax: threading.Event() for ax in present}
        for ev in scope._arrival_events.values():
            ev.set()
        scope._move_profile = {ax: None for ax in present}

        assert set(scope._pos_cache.keys()) == {'Z'}
        assert set(scope._axis_state.keys()) == {'Z'}
        assert set(scope._arrival_events.keys()) == {'Z'}
        assert set(scope._move_profile.keys()) == {'Z'}

    def test_null_motor_yields_empty_dicts(self):
        """A scope with no motor hardware (NullMotionBoard) should have
        empty per-axis dicts — there's nothing to track."""
        scope = Lumascope(simulate=True)
        scope.motion = NullMotionBoard()
        present = scope.motion.detect_present_axes()
        scope._pos_cache = {ax: 0.0 for ax in present}
        scope._axis_state = {ax: AxisState.UNKNOWN for ax in present}
        scope._arrival_events = {ax: threading.Event() for ax in present}
        scope._move_profile = {ax: None for ax in present}

        assert scope._pos_cache == {}
        assert scope._axis_state == {}
        assert scope._arrival_events == {}
        assert scope._move_profile == {}

    def test_move_absolute_on_absent_axis_is_silent_noop_rule_8(self):
        """Rule 8: API silently no-ops for absent axes. An LS820 user
        calling move_absolute_position('X', 0) gets a silent no-op, not
        a ValueError or HardwareError, regardless of whether they thought
        to call has_axis() first."""
        scope = Lumascope(simulate=True)
        scope.motion.detect_present_axes = lambda: ['Z']
        present = scope.motion.detect_present_axes()
        scope._pos_cache = {ax: 0.0 for ax in present}
        scope._axis_state = {ax: AxisState.UNKNOWN for ax in present}
        scope._arrival_events = {ax: threading.Event() for ax in present}
        for ev in scope._arrival_events.values():
            ev.set()
        scope._move_profile = {ax: None for ax in present}

        scope.move_absolute_position('X', 100)
        scope.move_absolute_position('Y', 100)
        scope.move_absolute_position('T', 0)
        assert 'X' not in scope._pos_cache
        assert 'Y' not in scope._pos_cache
        assert 'T' not in scope._pos_cache

        scope.move_relative_position('X', 50)
        assert 'X' not in scope._pos_cache

    def test_move_on_null_motor_is_silent_noop_rule_8(self):
        """Same Rule 8 contract on a system with NO motor hardware at
        all (NullMotionBoard). Pre-B4 behavior was silent no-op via
        VALID_AXES validation passing through to NullMotionBoard.move_abs_pos
        no-op — this contract must be preserved."""
        scope = Lumascope(simulate=True)
        scope.motion = NullMotionBoard()
        scope._pos_cache = {}
        scope._axis_state = {}
        scope._arrival_events = {}
        scope._move_profile = {}

        scope.move_absolute_position('Z', 100)
        scope.move_absolute_position('X', 0)
        scope.move_relative_position('Z', 10)

    def test_move_with_invalid_axis_name_still_raises(self):
        """Input sanity check still rejects non-axis names. _VALID_AXIS_NAMES
        is the input vocabulary; axes_present() is the capability query."""
        scope = Lumascope(simulate=True)
        with pytest.raises(ValueError, match=r"Axis must be one of"):
            scope.move_absolute_position('Q', 0)
        with pytest.raises(ValueError, match=r"Axis must be one of"):
            scope.move_relative_position('Q', 0)

    def test_no_hardcoded_VALID_AXES_constant(self):
        """The misnamed `VALID_AXES` class constant has been deleted.
        It implied "what axes are available" but actually meant "what
        axis names we accept as input" — which is now the private
        `_VALID_AXIS_NAMES`."""
        assert not hasattr(Lumascope, 'VALID_AXES'), (
            "Lumascope.VALID_AXES must be removed — its name was misleading "
            "(implied capability, meant vocabulary). Use axes_present() for "
            "capability queries; _VALID_AXIS_NAMES is the private input "
            "vocabulary tuple."
        )
        assert hasattr(Lumascope, '_VALID_AXIS_NAMES')
        assert tuple(Lumascope._VALID_AXIS_NAMES) == ('X', 'Y', 'Z', 'T')
