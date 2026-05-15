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

import threading
from unittest.mock import MagicMock

# Heavy deps are mocked by tests/conftest.py at module-import time.

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


@pytest.fixture(autouse=True)
def _clear_notification_dedup():
    """Reset notification dedup state between tests so multiple tests
    that fire the same (category, title) within 10 s each see their
    own listener-fire. Without this, NotificationCenter's 10 s dedup
    window suppresses the second+ test's notification."""
    notifications._dedup.clear()
    yield
    notifications._dedup.clear()


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

    def test_home_with_null_motion_board_notifies_motor_not_connected(self):
        """home() on a scope with NullMotionBoard must surface a clear
        Rule 14 'Motor Not Connected' notification.

        Contract change 2026-04-25 (issue #632 cluster): the prior
        contract was "silent no-op when motor is null." That left the
        user in the dark when Thonny held the motor port — they'd click
        Home and either nothing visible happened or, worse, they got
        "Homing Failed" implying a homing-mechanics issue rather than
        the actual cause (motor disconnected). The structural fix is to
        short-circuit at the API layer with the right diagnostic, per
        the hardware-absent audit
        (`docs/AUDIT_HARDWARE_ABSENT_STRUCTURAL_2026-04-24.md`).
        """
        received = self._capture_errors()
        scope = Lumascope(simulate=True)
        scope._motion_driver = NullMotionBoard()

        scope.home()

        assert received, (
            "home() on NullMotionBoard must notify 'Motor Not Connected' "
            "rather than silently no-op. User needs to know why nothing "
            "happened so they can fix the cause (port held, USB unplugged, etc.)."
        )
        assert any('Motor Not Connected' in n.title for n in received), (
            f"expected 'Motor Not Connected' notification, got: "
            f"{[n.title for n in received]}"
        )

    def test_home_short_circuits_on_disconnected_motor(self):
        """home() must return immediately when motor is not connected —
        no 30-second exchange_command timeout, no auto-reconnect retry
        burning the IO_WORKER. Issue #632 'spinning beachball' — user
        had to force-quit the app while home() was blocked."""
        import time
        received = self._capture_errors()
        scope = Lumascope(simulate=True)
        scope._motion_driver = NullMotionBoard()

        t0 = time.monotonic()
        scope.home()
        elapsed = time.monotonic() - t0

        assert elapsed < 0.5, (
            f"home() on disconnected motor took {elapsed:.2f}s — must be "
            f"< 0.5s. Beachball regression."
        )
        assert received, (
            "home() short-circuit must still fire the Rule 14 notification."
        )

    def test_thome_short_circuits_on_disconnected_motor(self):
        """Same contract as home() — thome must fail-fast with a clear
        notification rather than letting exchange_command burn its
        15s timeout."""
        import time
        received = self._capture_errors()
        scope = Lumascope(simulate=True)
        scope._motion_driver = NullMotionBoard()

        t0 = time.monotonic()
        scope.thome()
        elapsed = time.monotonic() - t0

        assert elapsed < 0.5, (
            f"thome() on disconnected motor took {elapsed:.2f}s — must be "
            f"< 0.5s."
        )
        assert any('Motor Not Connected' in n.title for n in received), (
            f"thome() must notify 'Motor Not Connected', got: "
            f"{[n.title for n in received]}"
        )

    def test_home_on_z_only_board_marks_z_idle(self):
        """LS820 (Z-only): the driver recognizes the 'X not present'
        firmware response as a partial-home success. The API trusts
        the driver's True return and marks present axes IDLE."""
        received = self._capture_errors()
        scope = Lumascope(simulate=True)

        # Simulate LS820: only Z physically present.
        scope._motion_driver.detect_present_axes = lambda: ['Z']

        # Driver returns True (the partial-home case is its responsibility).
        home_calls = []
        def fake_home(*args, **kwargs):
            home_calls.append((args, kwargs))
            return True
        scope._motion_driver.home = fake_home

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

        scope._motion_driver.home = lambda *a, **k: False

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
        present = scope._motion_driver.detect_present_axes()
        assert 'X' in present and 'Y' in present

        home_called = []
        original_home = scope._motion_driver.home
        def spy_home():
            home_called.append(True)
            return original_home()
        scope._motion_driver.home = spy_home

        scope.home()

        assert home_called, "home() on full XYZ hardware must call motion.home"
        for ax in ('X', 'Y', 'Z'):
            assert scope.get_axis_state(ax) == AxisState.IDLE


class TestMotorBoardGetMicroscopeModelDisconnect:
    """drivers/motorboard.py::get_microscope_model() must NOT raise when
    the board is disconnected (self._fullinfo is None). Issue #632
    crash 1: caller did `info['model']` against cached None, blew up
    with TypeError 'NoneType is not subscriptable', killed the app's
    settings-load on first launch with Thonny holding the motor port.
    """

    def test_returns_none_when_fullinfo_not_cached(self):
        from drivers.motorboard import MotorBoard
        board = MotorBoard.__new__(MotorBoard)
        import threading
        board._state_lock = threading.Lock()
        board._fullinfo = None
        # Must return None, must not raise. Caller (UI) treats None
        # as "use saved settings" — safe path.
        assert board.get_microscope_model() is None

    def test_returns_model_from_cached_fullinfo(self):
        from drivers.motorboard import MotorBoard
        board = MotorBoard.__new__(MotorBoard)
        import threading
        board._state_lock = threading.Lock()
        board._fullinfo = {'model': 'LS850', 'serial': '12074'}
        assert board.get_microscope_model() == 'LS850'

    def test_returns_none_when_model_key_missing(self):
        # Defense-in-depth: malformed cache shouldn't crash either.
        from drivers.motorboard import MotorBoard
        board = MotorBoard.__new__(MotorBoard)
        import threading
        board._state_lock = threading.Lock()
        board._fullinfo = {'serial': '12074'}  # no 'model' key
        assert board.get_microscope_model() is None


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

    def test_real_failure_raises_hardware_error(self):
        """Non-partial errors (timeout, hardware fault, Z homing aborted)
        must raise HardwareError -- the API layer catches and raises the
        Homing Failed popup. (Wave 2 / D1: Rule 29 typed-exception migration.)"""
        from drivers.motorboard import MotorBoard
        from drivers.exceptions import HardwareError

        board = MotorBoard.__new__(MotorBoard)
        import threading
        board._state_lock = threading.Lock()
        board.initial_homing_complete = False
        board.exchange_command = MagicMock(return_value="ERROR: timeout")

        with pytest.raises(HardwareError, match="firmware error"):
            board.home()
        assert board.initial_homing_complete is False

    def test_no_response_raises_hardware_error(self):
        """No response (None) means disconnect/timeout -- raises HardwareError.
        (Wave 2 / D1: Rule 29 typed-exception migration.)"""
        from drivers.motorboard import MotorBoard
        from drivers.exceptions import HardwareError

        board = MotorBoard.__new__(MotorBoard)
        import threading
        board._state_lock = threading.Lock()
        board.initial_homing_complete = False
        board.exchange_command = MagicMock(return_value=None)

        with pytest.raises(HardwareError, match="no response"):
            board.home()
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
        scope._motion_driver.zhome = fake_zhome

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
        present = scope._motion_driver.detect_present_axes()
        assert 'X' in present and 'Y' in present and 'Z' in present
        captured = {}

        original_home = scope._motion_driver.home
        def spy_home():
            captured['is_valid'] = scope.frame_validity.is_valid
            captured['pending'] = dict(scope.frame_validity.pending_sources)
            captured['x_state'] = scope.get_axis_state('X')
            captured['z_state'] = scope.get_axis_state('Z')
            return original_home()
        scope._motion_driver.home = spy_home

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
        from modules.scope_capabilities import ScopeCapabilities
        scope = Lumascope(simulate=True)
        scope._motion_driver.detect_present_axes = lambda: ['Z']
        # Rebuild the capability snapshot after patching the driver —
        # post-B7, `axes_present()` reads from `capabilities.axes` (a
        # frozen snapshot built at init), so the test needs to
        # re-snapshot to reflect the patched motion.
        scope.capabilities = ScopeCapabilities.from_drivers(
            motion=scope._motion_driver, led=scope._led_driver, camera=scope._camera_driver,
            led_max_ma=Lumascope.LED_MAX_MA,
        )
        captured = {}

        def fake_home(*args, **kwargs):
            captured['pending'] = dict(scope.frame_validity.pending_sources)
            captured['is_valid'] = scope.frame_validity.is_valid
            return True
        scope._motion_driver.home = fake_home

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
        scope._motion_driver = SimulatedMotorBoard(model='LS850T')
        present = scope._motion_driver.detect_present_axes()
        assert 'T' in present
        scope._pos_cache = {ax: 0.0 for ax in present}
        scope._axis_state = {ax: AxisState.UNKNOWN for ax in present}
        scope._arrival_events = {ax: threading.Event() for ax in present}
        for ev in scope._arrival_events.values():
            ev.set()
        scope._move_profile = {ax: None for ax in present}

        captured = {}
        original_thome = scope._motion_driver.thome
        def spy_thome():
            captured['is_valid'] = scope.frame_validity.is_valid
            captured['t_state'] = scope.get_axis_state('T')
            captured['pending'] = dict(scope.frame_validity.pending_sources)
            return original_thome()
        scope._motion_driver.thome = spy_thome

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
        assert isinstance(scope._motion_driver, MotorBoardProtocol)
        assert isinstance(scope._led_driver, LEDBoardProtocol)


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
        scope._led_driver = FourChannelLED()

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
        scope._led_driver = TwoChannelLED()

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
        present = set(scope._motion_driver.detect_present_axes())
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
        scope._motion_driver.detect_present_axes = lambda: ['Z']
        # Re-init the per-axis dicts to reflect the patched motion.
        present = scope._motion_driver.detect_present_axes()
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
        scope._motion_driver = NullMotionBoard()
        present = scope._motion_driver.detect_present_axes()
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
        scope._motion_driver.detect_present_axes = lambda: ['Z']
        present = scope._motion_driver.detect_present_axes()
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
        scope._motion_driver = NullMotionBoard()
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


class TestRunGrabLifecycleBenchmark:
    """CAM-1 step (0a): regression tests for ``Lumascope.run_grab_lifecycle_benchmark``.

    The API method shipped 2026-05-04 (LVP `56f094b`) without tests. This
    class pins the contract: dict shape, num_cycles loop count, slow-cycle
    accounting, vary_settings alternation, and the inactive-camera guard.
    """

    def _scope_with_camera(self):
        """Return a Lumascope with simulated camera ready for stop/start cycles."""
        scope = Lumascope(simulate=True)
        # SimulatedCamera is wired by the registry; ensure it is in the
        # active grabbing state the benchmark expects.
        if scope._camera_driver and not scope._camera_driver.is_grabbing():
            scope._camera_driver.start_grabbing()
        return scope

    def test_returns_required_dict_keys(self):
        scope = self._scope_with_camera()
        r = scope.run_grab_lifecycle_benchmark(num_cycles=3,
                                                inter_cycle_delay_ms=0)
        for k in ('num_cycles', 'inter_cycle_delay_ms', 'vary_settings',
                  'slow_threshold_s', 'slow_cycle_count', 'slow_cycles',
                  'cycle_p50_s', 'cycle_p95_s', 'cycle_p99_s',
                  'stop_p50_s', 'stop_p95_s', 'stop_p99_s',
                  'start_p50_s', 'start_p95_s', 'start_p99_s',
                  'total_elapsed_s', 'camera_model', 'pylon_version',
                  'errors', 'written_to'):
            assert k in r, f'Missing key: {k}'
        assert r['num_cycles'] == 3
        assert r['inter_cycle_delay_ms'] == 0
        assert r['vary_settings'] is False
        assert r['slow_threshold_s'] == 3.0  # default

    def test_inactive_camera_returns_error(self):
        """When self.camera is None or inactive, the method must surface
        an error instead of crashing or silently returning empty results."""
        scope = Lumascope(simulate=True)
        scope._camera_driver = None
        r = scope.run_grab_lifecycle_benchmark(num_cycles=3)
        assert r['errors'], (
            'Inactive-camera path must populate errors so the operator '
            'sees why the benchmark produced no data')
        assert any('not active' in e.lower() for e in r['errors'])
        # No samples means percentile fields stay at defaults.
        assert r['cycle_p50_s'] == 0.0
        assert r['slow_cycle_count'] == 0

    def test_slow_cycle_detection_with_zero_threshold(self):
        """slow_threshold_s=0.0 forces every cycle to count as slow,
        verifying the slow-cycle accounting + 50-entry cap."""
        scope = self._scope_with_camera()
        r = scope.run_grab_lifecycle_benchmark(num_cycles=4,
                                                inter_cycle_delay_ms=0,
                                                slow_threshold_s=0.0)
        assert r['slow_cycle_count'] == 4
        assert len(r['slow_cycles']) == 4
        for entry in r['slow_cycles']:
            for field in ('idx', 'cycle_s', 'stop_s', 'start_s'):
                assert field in entry, f'Missing field {field} in slow-cycle entry'

    def test_vary_settings_alternates_gain(self):
        """vary_settings=True alternates gain between 1.0 dB (even cycles)
        and 4.0 dB (odd cycles)."""
        scope = self._scope_with_camera()
        gain_calls = []
        original_set_gain = scope.set_gain

        def _track(gain):
            gain_calls.append(gain)
            return original_set_gain(gain)
        scope.set_gain = _track

        scope.run_grab_lifecycle_benchmark(num_cycles=4,
                                            inter_cycle_delay_ms=0,
                                            vary_settings=True)
        # 4 in-loop calls + restore at end (if vary_settings AND original_gain)
        # The benchmark restores original gain after the loop, so total may be 5.
        in_loop = gain_calls[:4]
        assert in_loop == [1.0, 4.0, 1.0, 4.0], \
            f'vary_settings should alternate 1.0/4.0; got {in_loop}'

    def test_writes_json_artifact(self, tmp_path, monkeypatch):
        """Persists results to data/camera_timing/. Filename includes camera
        model + SDK + delay + timestamp so a delay sweep produces one file
        per data point."""
        import json
        import os
        scope = self._scope_with_camera()
        r = scope.run_grab_lifecycle_benchmark(num_cycles=2,
                                                inter_cycle_delay_ms=0)
        assert r['written_to'] is not None, \
            f'JSON persistence path empty; errors: {r.get("errors")}'
        assert os.path.exists(r['written_to']), \
            f'Promised JSON not at {r["written_to"]}'
        with open(r['written_to']) as f:
            persisted = json.load(f)
        assert persisted['num_cycles'] == 2
        # Cleanup test artifact so it doesn't accumulate in the repo dir.
        try:
            os.remove(r['written_to'])
        except OSError:
            pass


class TestScopeCapabilities:
    """Audit B7: ScopeCapabilities is the single source of truth for
    "what does this scope have" — a frozen snapshot built at init from
    the three drivers. Pre-B7, capability questions were answered
    piecemeal by `axes_present()`, `has_turret()`, ad-hoc isinstance
    checks, and direct reads of `led.available_channels()` /
    `camera.profile.*`. This is Rule 9 enforcement.

    Runtime connection state (`motor_connected`, `led_connected`) stays
    live on Lumascope — it deliberately isn't on the frozen dataclass
    because disconnects need to reflect immediately.
    """

    def test_capabilities_built_at_init(self):
        scope = Lumascope(simulate=True)
        from modules.scope_capabilities import ScopeCapabilities
        assert isinstance(scope.capabilities, ScopeCapabilities)

    def test_ls850_default_sim_capabilities(self):
        """Default sim is LS850: X/Y/Z, no turret, 6-channel LED."""
        scope = Lumascope(simulate=True)
        caps = scope.capabilities
        assert caps.axes == ('X', 'Y', 'Z')
        assert caps.has_focus is True
        assert caps.has_xy_stage is True
        assert caps.has_turret is False
        assert len(caps.led_channels) == 6
        assert caps.led_max_ma == Lumascope.LED_MAX_MA

    def test_ls850t_capabilities_has_turret(self):
        from drivers.simulated_motorboard import SimulatedMotorBoard
        from modules.scope_capabilities import ScopeCapabilities
        scope = Lumascope(simulate=True)
        scope._motion_driver = SimulatedMotorBoard(model='LS850T')
        scope.capabilities = ScopeCapabilities.from_drivers(
            motion=scope._motion_driver, led=scope._led_driver, camera=scope._camera_driver,
            led_max_ma=Lumascope.LED_MAX_MA,
        )
        assert scope.capabilities.axes == ('X', 'Y', 'Z', 'T')
        assert scope.capabilities.has_turret is True
        assert scope.capabilities.has_xy_stage is True

    def test_z_only_sim_capabilities(self):
        """LS820 / LVC LS620-style Z-only scope."""
        from modules.scope_capabilities import ScopeCapabilities
        scope = Lumascope(simulate=True)
        scope._motion_driver.detect_present_axes = lambda: ['Z']
        scope.capabilities = ScopeCapabilities.from_drivers(
            motion=scope._motion_driver, led=scope._led_driver, camera=scope._camera_driver,
            led_max_ma=Lumascope.LED_MAX_MA,
        )
        assert scope.capabilities.axes == ('Z',)
        assert scope.capabilities.has_focus is True
        assert scope.capabilities.has_xy_stage is False
        assert scope.capabilities.has_turret is False

    def test_null_motor_capabilities_empty_axes(self):
        from modules.scope_capabilities import ScopeCapabilities
        caps = ScopeCapabilities.from_drivers(
            motion=NullMotionBoard(), led=NullLEDBoard(), camera=None,
            led_max_ma=1000,
        )
        assert caps.axes == ()
        assert caps.has_focus is False
        assert caps.has_xy_stage is False
        assert caps.has_turret is False

    def test_null_led_still_reports_six_channels_for_compat(self):
        """Per B3 compat: NullLEDBoard reports 6 channels so Rule 8
        silent no-ops work on channels 0-5. Capabilities mirrors that."""
        from modules.scope_capabilities import ScopeCapabilities
        caps = ScopeCapabilities.from_drivers(
            motion=NullMotionBoard(), led=NullLEDBoard(), camera=None,
            led_max_ma=1000,
        )
        assert len(caps.led_channels) == 6
        assert caps.led_channels == (0, 1, 2, 3, 4, 5)

    def test_four_channel_led_capabilities(self):
        """An FX2-style 4-channel LED driver propagates through."""
        from modules.scope_capabilities import ScopeCapabilities

        class FourChannelLED(SimulatedLEDBoard):
            _COLOR_TO_CH = {'Blue': 0, 'Green': 1, 'Red': 2, 'BF': 3}
            _CH_TO_COLOR = {v: k for k, v in _COLOR_TO_CH.items()}

        caps = ScopeCapabilities.from_drivers(
            motion=NullMotionBoard(), led=FourChannelLED(), camera=None,
            led_max_ma=1000,
        )
        assert caps.led_channels == (0, 1, 2, 3)
        assert set(caps.led_colors) == {'Blue', 'Green', 'Red', 'BF'}

    def test_capabilities_is_frozen(self):
        """The dataclass is frozen — any attempt to mutate a field
        raises FrozenInstanceError. This enforces the "snapshot" contract."""
        import dataclasses
        scope = Lumascope(simulate=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            scope.capabilities.axes = ('X',)  # type: ignore[misc]

    def test_backward_compat_methods_delegate_to_capabilities(self):
        """The existing methods (`axes_present`, `has_turret`, `has_axis`)
        must return values matching `scope.capabilities.*`. If they drift,
        callers using the old methods see different answers than callers
        using the new field — exactly the fragmentation B7 aims to retire."""
        scope = Lumascope(simulate=True)
        assert scope.axes_present() == list(scope.capabilities.axes)
        assert scope.has_turret() == scope.capabilities.has_turret
        assert scope.has_axis('Z') == ('Z' in scope.capabilities.axes)
        assert scope.has_axis('Q') is False

    def test_motor_connected_stays_live_not_in_capabilities(self):
        """Runtime connection state must NOT be a field on capabilities —
        it needs to reflect disconnects at runtime, which a frozen
        snapshot can't do. Lumascope.motor_connected / led_connected
        remain live properties."""
        scope = Lumascope(simulate=True)
        from modules.scope_capabilities import ScopeCapabilities
        cap_fields = {f.name for f in dataclasses_fields(ScopeCapabilities)}
        assert 'motor_connected' not in cap_fields
        assert 'led_connected' not in cap_fields
        assert 'camera_connected' not in cap_fields
        # Runtime properties still work as they did
        assert isinstance(scope.motor_connected, bool)
        assert isinstance(scope.led_connected, bool)


def dataclasses_fields(cls):
    """Helper — imported late to keep the imports tidy."""
    import dataclasses as _dc
    return _dc.fields(cls)


class TestSetExposureTimeValueWarningSuppression:
    """`set_exposure_time` warns at < 0.1 ms exposures because the
    L1-researcher failure mode is typing 0.05 thinking microseconds and
    getting a black image (lumascope_api.py:3761). Sweep-style internal
    callers (camera characterization dynamic_range / linearity stages)
    walk that range deliberately and need a way to silence the warning.
    The `suppress_value_warnings()` context manager flips an instance
    flag that gates the warning. Tests verify the gate, the flag's
    restore-on-exit semantics (including exception path), and that the
    warning still fires by default for L1-typed values.
    """

    def _patch_logger(self, monkeypatch):
        from unittest.mock import MagicMock
        import modules.lumascope_api._lumascope as lapi
        mock = MagicMock()
        monkeypatch.setattr(lapi, 'logger', mock)
        return mock

    def test_warning_fires_by_default_at_sub_0_1_ms(self, monkeypatch):
        mock_logger = self._patch_logger(monkeypatch)
        scope = Lumascope(simulate=True)
        scope.set_exposure_time(0.05)
        # Find the warning among any other logger calls
        warn_msgs = [str(c) for c in mock_logger.warning.call_args_list]
        assert any('set_exposure_time(0.05ms)' in m and 'very low' in m
                   for m in warn_msgs), (
            f'expected sub-0.1ms warning but got: {warn_msgs}')

    def test_warning_suppressed_inside_context_manager(self, monkeypatch):
        mock_logger = self._patch_logger(monkeypatch)
        scope = Lumascope(simulate=True)
        with scope.suppress_value_warnings():
            scope.set_exposure_time(0.05)
        warn_msgs = [str(c) for c in mock_logger.warning.call_args_list]
        assert not any('set_exposure_time(0.05ms)' in m
                       for m in warn_msgs), (
            f'expected no sub-0.1ms warning inside context, got: {warn_msgs}')

    def test_flag_restored_after_normal_exit(self):
        scope = Lumascope(simulate=True)
        assert scope._suppress_value_warnings is False
        with scope.suppress_value_warnings():
            assert scope._suppress_value_warnings is True
        assert scope._suppress_value_warnings is False

    def test_flag_restored_after_exception_in_context(self):
        scope = Lumascope(simulate=True)
        assert scope._suppress_value_warnings is False
        with pytest.raises(RuntimeError, match='boom'):
            with scope.suppress_value_warnings():
                assert scope._suppress_value_warnings is True
                raise RuntimeError('boom')
        assert scope._suppress_value_warnings is False

    def test_nested_context_managers_restore_to_outer_value(self):
        """Nested `with` blocks restore to prior, not unconditionally False --
        an outer `with` followed by an inner-then-exit must leave True."""
        scope = Lumascope(simulate=True)
        with scope.suppress_value_warnings():
            with scope.suppress_value_warnings():
                assert scope._suppress_value_warnings is True
            # After inner exit, outer is still suppressing
            assert scope._suppress_value_warnings is True
        assert scope._suppress_value_warnings is False

    def test_warning_does_not_fire_at_or_above_threshold(self, monkeypatch):
        mock_logger = self._patch_logger(monkeypatch)
        scope = Lumascope(simulate=True)
        scope.set_exposure_time(0.5)
        scope.set_exposure_time(20.0)
        warn_msgs = [str(c) for c in mock_logger.warning.call_args_list]
        assert not any('very low' in m for m in warn_msgs), (
            f'no sub-0.1ms warning expected, got: {warn_msgs}')
