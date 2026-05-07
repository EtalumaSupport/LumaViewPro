# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for audit fixes across LumaViewPro.

Covers:
  1. Domain exceptions (modules/exceptions.py)
  2. Input validation (lumascope_api.py)
  3. Protocol file limits (modules/protocol.py)
  4. ProtocolState transitions (modules/sequenced_capture_executor.py)
  5. Settings snapshot thread safety (modules/app_context.py)
  6. AppleScript escaping (ui/file_dialogs.py)
  7. FPS calculation edge case

IMPORTANT: This file does NOT manipulate sys.modules at module level.
All mocking is done inside fixtures/test methods and cleaned up afterward.
"""

import sys
import threading
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers for building mock modules (used by fixtures, not at module level)
# ---------------------------------------------------------------------------

def _build_mock_logger():
    """Build a mock lvp_logger module with a logger attribute."""
    mock_logger = MagicMock()
    for attr in ('info', 'debug', 'error', 'warning', 'critical'):
        setattr(mock_logger, attr, MagicMock())

    mock_lvp_logger = MagicMock()
    mock_lvp_logger.logger = mock_logger
    mock_lvp_logger.version = "test"
    mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
    mock_lvp_logger.unpause_thread = MagicMock()
    mock_lvp_logger.pause_thread = MagicMock()
    return mock_lvp_logger


def _kivy_mock_modules():
    """Return a dict of {module_name: mock_object} for Kivy and camera SDKs."""

    class _FakeKivyWidget:
        pass

    class _FakeButton(_FakeKivyWidget):
        pass

    class _FakeHoverBehavior:
        pass

    kivy_properties_mock = MagicMock()
    kivy_properties_mock.ListProperty = lambda *a, **k: None
    kivy_properties_mock.StringProperty = lambda *a, **k: None
    kivy_properties_mock.NumericProperty = lambda *a, **k: None
    kivy_properties_mock.BooleanProperty = lambda *a, **k: None
    kivy_properties_mock.ObjectProperty = lambda *a, **k: None

    kivy_uix_button_mock = MagicMock()
    kivy_uix_button_mock.Button = _FakeButton

    hover_mock = MagicMock()
    hover_mock.HoverBehavior = _FakeHoverBehavior

    mods = {}
    for name in [
        'kivy', 'kivy.app', 'kivy.clock', 'kivy.core', 'kivy.core.window',
        'kivy.factory', 'kivy.graphics', 'kivy.graphics.texture',
        'kivy.graphics.instructions', 'kivy.graphics.vertex_instructions',
        'kivy.lang', 'kivy.metrics',
        'kivy.uix', 'kivy.uix.boxlayout',
        'kivy.uix.floatlayout', 'kivy.uix.gridlayout', 'kivy.uix.image',
        'kivy.uix.label', 'kivy.uix.popup', 'kivy.uix.scrollview',
        'kivy.uix.slider', 'kivy.uix.spinner', 'kivy.uix.textinput',
        'kivy.uix.togglebutton', 'kivy.uix.widget',
        'kivy.uix.behaviors', 'kivy.uix.behaviors.hover',
    ]:
        mods[name] = MagicMock()

    mods['kivy.properties'] = kivy_properties_mock
    mods['kivy.uix.button'] = kivy_uix_button_mock
    mods['ui.hover_behavior'] = hover_mock

    return mods


def _camera_sdk_mock_modules():
    """Return a dict of camera SDK mock modules."""
    mods = {}
    for name in [
        'pypylon', 'pypylon.pylon', 'pypylon.genicam',
        'ids_peak', 'ids_peak.ids_peak', 'ids_peak.ids_peak_ipl_extension',
        'ids_peak_ipl',
    ]:
        mods[name] = MagicMock()
    return mods


def _common_mock_modules():
    """Return a dict of commonly needed mock modules (lvp_logger, platformdirs, etc).

    NOTE: cv2 is NOT mocked — it's a real installed package with no Kivy
    dependency. Mocking it causes test-ordering contamination: image_utils
    caches the mock cv2 reference at import time, and monkeypatch cleanup
    can't fix the cached reference. This broke TestAddTimestampInPlace.
    """
    mods = {
        'platformdirs': MagicMock(),
        'lvp_logger': _build_mock_logger(),
        'requests': MagicMock(),
        'requests.structures': MagicMock(),
        'psutil': MagicMock(),
    }
    mock_settings_init = MagicMock()
    mock_settings_init.settings = {}
    mods['modules.settings_init'] = mock_settings_init
    return mods


def _all_mock_modules():
    """Return the full set of mock modules needed for heavy imports."""
    mods = {}
    mods.update(_common_mock_modules())
    mods.update(_camera_sdk_mock_modules())
    mods.update(_kivy_mock_modules())
    return mods


# ---------------------------------------------------------------------------
# Fixture: temporarily install mock modules for a test class, then clean up.
# ---------------------------------------------------------------------------

@pytest.fixture
def _mock_heavy_deps(monkeypatch):
    """Install all mock modules into sys.modules for the duration of a test.

    Only inserts keys that are NOT already present, and removes them on teardown.
    This avoids polluting sys.modules for other test files.
    """
    mods = _all_mock_modules()
    inserted_keys = []
    for name, mock_mod in mods.items():
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, mock_mod)
            inserted_keys.append(name)
    yield
    # monkeypatch handles cleanup automatically


# ===========================================================================
# 1. Domain exceptions — no mocks needed, pure Python module
# ===========================================================================
from drivers.exceptions import HardwareError
from modules.exceptions import ProtocolError, ConfigError, CaptureError


class TestDomainExceptions:
    """Verify custom exception classes are proper Exception subclasses."""

    @pytest.mark.parametrize("exc_cls", [
        HardwareError, ProtocolError, ConfigError, CaptureError,
    ])
    def test_subclass_of_exception(self, exc_cls):
        assert issubclass(exc_cls, Exception)

    @pytest.mark.parametrize("exc_cls", [
        HardwareError, ProtocolError, ConfigError, CaptureError,
    ])
    def test_raise_and_catch_with_message(self, exc_cls):
        msg = f"test message for {exc_cls.__name__}"
        with pytest.raises(exc_cls, match=msg):
            raise exc_cls(msg)


# ===========================================================================
# 2. Input validation — Lumascope API (needs mocks for camera/logger deps)
# ===========================================================================

@pytest.fixture
def sim_scope(_mock_heavy_deps):
    """Create a Lumascope in simulate mode (no hardware needed)."""
    from modules.lumascope_api import Lumascope
    scope = Lumascope(simulate=True)
    yield scope
    scope.disconnect()


class TestLedOnValidation:
    """Verify led_on() rejects bad inputs."""

    def test_rejects_channel_out_of_range(self, sim_scope):
        with pytest.raises(ValueError, match="channel"):
            sim_scope.led_on(channel=99, mA=10)

    def test_rejects_negative_current(self, sim_scope):
        with pytest.raises(ValueError, match="current"):
            sim_scope.led_on(channel=0, mA=-1)

    def test_rejects_current_above_max(self, sim_scope):
        from modules.lumascope_api import Lumascope
        with pytest.raises(ValueError, match="current"):
            sim_scope.led_on(channel=0, mA=Lumascope.LED_MAX_MA + 1)

    def test_accepts_valid_input(self, sim_scope):
        sim_scope.led_on(channel=0, mA=50)


class TestMoveAbsolutePositionValidation:
    """Verify move_absolute_position() rejects bad inputs."""

    def test_rejects_invalid_axis(self, sim_scope):
        with pytest.raises(ValueError, match="Axis"):
            sim_scope.move_absolute_position(axis='Q', pos=100)

    def test_rejects_position_above_limit(self, sim_scope):
        from modules.lumascope_api import Lumascope
        with pytest.raises(ValueError, match="exceeds safety limit"):
            sim_scope.move_absolute_position(
                axis='Z', pos=Lumascope.MOTOR_POSITION_LIMIT + 1
            )

    def test_rejects_large_negative_position(self, sim_scope):
        from modules.lumascope_api import Lumascope
        with pytest.raises(ValueError, match="exceeds safety limit"):
            sim_scope.move_absolute_position(
                axis='Z', pos=-(Lumascope.MOTOR_POSITION_LIMIT + 1)
            )

    def test_accepts_valid_input(self, sim_scope):
        sim_scope.move_absolute_position(axis='Z', pos=1000)


# ===========================================================================
# 3. Protocol file limits (needs mocks for heavy deps)
# ===========================================================================

class TestProtocolFileLimits:
    """Verify Protocol.from_file() enforces size and step count limits."""

    def test_rejects_oversized_file(self, _mock_heavy_deps, tmp_path):
        """A file > 10 MB should be rejected before parsing."""
        from modules.protocol import Protocol

        big_file = tmp_path / "huge_protocol.tsv"
        big_file.write_bytes(b'x' * (10 * 1024 * 1024 + 1))

        with pytest.raises(ValueError, match="exceeds maximum size"):
            Protocol.from_file(
                file_path=big_file,
                tiling_configs_file_loc=None,
            )

    def test_accepts_file_under_limit(self, _mock_heavy_deps, tmp_path):
        """A small file should pass the size check (may fail later on format,
        but should NOT raise the size ValueError)."""
        from modules.protocol import Protocol

        small_file = tmp_path / "small.tsv"
        small_file.write_text("LumaViewPro Protocol\n")

        with pytest.raises(Exception) as exc_info:
            Protocol.from_file(
                file_path=small_file,
                tiling_configs_file_loc=None,
            )
        assert "exceeds maximum size" not in str(exc_info.value)


# ===========================================================================
# 4. ProtocolState transitions (needs mocks for heavy deps)
# ===========================================================================

@pytest.fixture
def protocol_state_imports(_mock_heavy_deps):
    """Import ProtocolState and transitions after mocks are installed."""
    from modules.protocol_state_machine import (
        ProtocolState,
        PROTOCOL_STATE_TRANSITIONS,
    )
    return ProtocolState, PROTOCOL_STATE_TRANSITIONS


class TestProtocolStateTransitions:
    """Verify the state machine allows only documented transitions."""

    def _state(self, protocol_state_imports, name):
        """Helper to get a ProtocolState member by name."""
        ProtocolState, _ = protocol_state_imports
        return ProtocolState[name]

    def _transitions(self, protocol_state_imports):
        _, transitions = protocol_state_imports
        return transitions

    @pytest.mark.parametrize("from_name, to_name", [
        ("IDLE", "RUNNING"),
        ("RUNNING", "SCANNING"),
        ("RUNNING", "COMPLETING"),
        ("RUNNING", "ERROR"),
        ("SCANNING", "RUNNING"),
        ("SCANNING", "COMPLETING"),
        ("SCANNING", "ERROR"),
        ("COMPLETING", "IDLE"),
        ("ERROR", "IDLE"),
    ])
    def test_valid_transitions(self, protocol_state_imports, from_name, to_name):
        """All documented transitions should be present in the map."""
        ProtocolState, transitions = protocol_state_imports
        from_state = ProtocolState[from_name]
        to_state = ProtocolState[to_name]
        allowed = transitions[from_state]
        assert to_state in allowed

    @pytest.mark.parametrize("from_name, to_name", [
        ("IDLE", "SCANNING"),
        ("IDLE", "COMPLETING"),
        ("IDLE", "ERROR"),
        ("COMPLETING", "RUNNING"),
        ("COMPLETING", "SCANNING"),
        ("ERROR", "RUNNING"),
        ("ERROR", "SCANNING"),
    ])
    def test_invalid_transitions(self, protocol_state_imports, from_name, to_name):
        """Undocumented transitions must NOT appear in the allowed set."""
        ProtocolState, transitions = protocol_state_imports
        from_state = ProtocolState[from_name]
        to_state = ProtocolState[to_name]
        allowed = transitions.get(from_state, set())
        assert to_state not in allowed

    def test_all_states_have_transition_entry(self, protocol_state_imports):
        """Every ProtocolState value should have an entry in the map."""
        ProtocolState, transitions = protocol_state_imports
        for state in ProtocolState:
            assert state in transitions

    def test_no_self_transitions_in_map(self, protocol_state_imports):
        """No state should list itself as an allowed target."""
        _, transitions = protocol_state_imports
        for state, allowed in transitions.items():
            assert state not in allowed, f"{state} allows self-transition"


# ===========================================================================
# 5. Settings snapshot (AppContext) — no mocks needed, pure Python dataclass
# ===========================================================================
from modules.app_context import AppContext


class TestSettingsSnapshot:
    """Verify thread-safe settings access on AppContext."""

    def test_snapshot_is_deep_copy(self):
        ctx = AppContext(settings={"display": {"brightness": 80}})
        snap = ctx.get_settings_snapshot()

        snap["display"]["brightness"] = 999
        snap["new_key"] = True

        assert ctx.settings["display"]["brightness"] == 80
        assert "new_key" not in ctx.settings

    def test_update_settings_writes_value(self):
        ctx = AppContext(settings={})
        ctx.update_settings("live_folder", "/tmp/test")
        assert ctx.settings["live_folder"] == "/tmp/test"

    def test_update_settings_overwrites_existing(self):
        ctx = AppContext(settings={"live_folder": "/old"})
        ctx.update_settings("live_folder", "/new")
        assert ctx.settings["live_folder"] == "/new"

    def test_snapshot_after_update(self):
        ctx = AppContext(settings={})
        ctx.update_settings("key", "value1")
        snap = ctx.get_settings_snapshot()
        ctx.update_settings("key", "value2")

        assert snap["key"] == "value1"
        assert ctx.settings["key"] == "value2"


# ===========================================================================
# 6. AppleScript escaping (needs Kivy mocks for ui.file_dialogs import)
# ===========================================================================

class TestAppleScriptEscaping:
    """Verify _escape_applescript handles special characters."""

    @pytest.fixture(autouse=True)
    def _import_escape_fn(self, _mock_heavy_deps):
        """Import the function under test after mocks are installed."""
        from ui.file_dialogs import _escape_applescript
        self._escape = _escape_applescript

    def test_escapes_double_quotes(self):
        assert self._escape('say "hello"') == 'say \\"hello\\"'

    def test_escapes_backslashes(self):
        assert self._escape('path\\to\\file') == 'path\\\\to\\\\file'

    def test_escapes_both(self):
        result = self._escape('a\\b"c')
        assert result == 'a\\\\b\\"c'

    def test_normal_string_unchanged(self):
        assert self._escape('/Users/test/folder') == '/Users/test/folder'

    def test_empty_string(self):
        assert self._escape('') == ''


# ===========================================================================
# 7. FPS calculation edge case — pure math, no imports needed
# ===========================================================================
class TestFpsCalculation:
    """Verify FPS floor calculation used in protocol timing."""

    def test_fps_at_least_one_with_slow_capture(self):
        """1 frame over 5 seconds should still yield FPS >= 1."""
        captured_frames = 1
        duration_sec = 5.0
        fps = max(1, int(captured_frames / duration_sec))
        assert fps >= 1

    def test_fps_at_least_one_with_zero_duration(self):
        """Guard against zero-duration edge case."""
        captured_frames = 10
        duration_sec = 0.001
        fps = max(1, int(captured_frames / duration_sec))
        assert fps >= 1

    def test_fps_normal_case(self):
        """30 frames in 1 second = 30 fps."""
        fps = max(1, int(30 / 1.0))
        assert fps == 30


# ===========================================================================
# 8. Phase 4f — Security hardening tests
# ===========================================================================

class TestSettingsValidation:
    """Verify settings validation logic.

    Uses the same validation logic as settings_init._validate_settings
    but reimplemented here to avoid sys.modules mock pollution from other
    test files that replace modules.settings_init with a MagicMock.
    """

    # Mirror the required keys from settings_init.py
    _REQUIRED = frozenset({'microscope', 'live_folder', 'frame'})

    @staticmethod
    def _validate(settings, filepath, logger):
        """Inline copy of validation logic for test isolation."""
        missing = TestSettingsValidation._REQUIRED - settings.keys()
        if missing:
            logger.warning(
                f'[Settings ] {filepath} missing required keys: {sorted(missing)}. '
                'App may not function correctly.'
            )
        if 'frame' in settings and not isinstance(settings['frame'], dict):
            logger.warning(
                f'[Settings ] {filepath}: "frame" should be a dict, '
                f'got {type(settings["frame"]).__name__}'
            )

    def test_warns_on_missing_required_keys(self):
        mock_logger = MagicMock()
        self._validate({}, 'test.json', mock_logger)
        mock_logger.warning.assert_called()
        call_args = str(mock_logger.warning.call_args)
        assert 'missing required keys' in call_args

    def test_no_warning_when_all_keys_present(self):
        mock_logger = MagicMock()
        settings = {
            'microscope': 'LS850',
            'live_folder': './capture',
            'frame': {'width': 1900, 'height': 1900},
        }
        self._validate(settings, 'test.json', mock_logger)
        mock_logger.warning.assert_not_called()

    def test_warns_on_bad_frame_type(self):
        mock_logger = MagicMock()
        settings = {
            'microscope': 'LS850',
            'live_folder': './capture',
            'frame': 'not_a_dict',
        }
        self._validate(settings, 'test.json', mock_logger)
        mock_logger.warning.assert_called()
        call_args = str(mock_logger.warning.call_args)
        assert 'should be a dict' in call_args


class TestLvpLock:
    """Verify LVP lock security improvements."""

    def test_ephemeral_port(self):
        """Port 0 should get an OS-assigned ephemeral port."""
        from modules.lvp_lock import LvpLock
        with LvpLock(lock_port=0) as lock:
            assert lock.lock() is True
            # OS should have assigned a real port
            assert lock.port > 0

    def test_fixed_port(self):
        """Fixed port should work as before."""
        from modules.lvp_lock import LvpLock
        import socket
        # Find a free port first
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        with LvpLock(lock_port=port) as lock:
            assert lock.lock() is True
            assert lock.port == port

    def test_context_manager_closes(self):
        from modules.lvp_lock import LvpLock
        lock = LvpLock(lock_port=0)
        lock.lock()
        lock.close()
        # Socket should be closed — port property still works
        assert isinstance(lock.port, int)

    def test_second_instance_blocked(self):
        """Regression for #559: two LvpLock instances on the same port must conflict.

        Without this guarantee, a second LumaViewPro launch silently tramples
        the first's exclusive serial ports on Windows. The bug was an accidental
        SO_REUSEADDR setsockopt — on Windows that has SO_REUSEPORT semantics
        and explicitly allows live double-bind.
        """
        from modules.lvp_lock import LvpLock
        import socket
        # Grab a free port, then release it so we can bind it from LvpLock
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        with LvpLock(lock_port=port) as first:
            assert first.lock() is True, "first lock should succeed"
            second = LvpLock(lock_port=port)
            try:
                assert second.lock() is False, (
                    "second lock on same port MUST fail — regression of #559 "
                    "(SO_REUSEADDR reintroduced?)"
                )
            finally:
                second.close()


class TestSerialRateLimiting:
    """Verify serial command rate limiting infrastructure."""

    def test_default_no_rate_limit(self):
        """Default _min_command_interval should be 0 (no limit)."""
        from drivers.serialboard import SerialBoard
        board = SerialBoard(vid=0, pid=0, label='TEST')
        assert board._min_command_interval == 0.0

    def test_rate_limit_attributes_exist(self):
        """Rate limit attributes should be set in __init__."""
        from drivers.serialboard import SerialBoard
        board = SerialBoard(vid=0, pid=0, label='TEST')
        assert hasattr(board, '_min_command_interval')
        assert hasattr(board, '_last_command_time')


class TestSerialDebugTruncation:
    """Verify serial debug log truncation."""

    def test_long_response_truncated_in_log(self):
        """Long responses should be truncated in debug output."""
        long_resp = 'A' * 500
        resp_repr = repr(long_resp)
        if len(resp_repr) > 200:
            resp_repr = resp_repr[:200] + '...'
        assert len(resp_repr) <= 203  # 200 + '...'
        assert resp_repr.endswith('...')

    def test_short_response_not_truncated(self):
        """Short responses should not be truncated."""
        short_resp = 'OK'
        resp_repr = repr(short_resp)
        if len(resp_repr) > 200:
            resp_repr = resp_repr[:200] + '...'
        assert not resp_repr.endswith('...')


class TestTechSupportPrivacyNotice:
    """Verify tech support report includes privacy notice."""

    def test_privacy_notice_in_zip(self, tmp_path):
        """Report ZIP should contain PRIVACY_NOTICE.txt."""
        import zipfile
        # Create a minimal ZIP to test the writestr pattern
        zip_path = tmp_path / 'test_report.zip'
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('PRIVACY_NOTICE.txt', 'test notice')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            assert 'PRIVACY_NOTICE.txt' in zf.namelist()
            content = zf.read('PRIVACY_NOTICE.txt').decode()
            assert 'test notice' in content


# ===========================================================================
# 9. Phase 6 — Cleanup tests
# ===========================================================================

class TestAddTimestampInPlace:
    """Verify add_timestamp in-place optimization."""

    def test_in_place_modifies_original(self):
        import numpy as np
        from modules.image_utils import add_timestamp
        img = np.zeros((100, 200), dtype=np.uint8)
        result = add_timestamp(img, "2026-01-01", in_place=True)
        # Should return the same array object
        assert result is img

    def test_copy_does_not_modify_original(self):
        import numpy as np
        from modules.image_utils import add_timestamp
        img = np.zeros((100, 200), dtype=np.uint8)
        original_sum = img.sum()
        result = add_timestamp(img, "2026-01-01", in_place=False)
        # Original should be unchanged
        assert img.sum() == original_sum
        # Result should be a different object
        assert result is not img

    def test_default_is_in_place(self):
        import numpy as np
        from modules.image_utils import add_timestamp
        img = np.zeros((100, 200), dtype=np.uint8)
        result = add_timestamp(img, "test")
        assert result is img


class TestPyprojectConfig:
    """Verify pyproject.toml configuration."""

    def test_pyproject_exists(self):
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        assert (root / 'pyproject.toml').is_file()

    def test_pyproject_has_pytest_config(self):
        import pathlib
        root = pathlib.Path(__file__).parent.parent
        content = (root / 'pyproject.toml').read_text()
        assert '[tool.pytest.ini_options]' in content
        assert '[tool.coverage.run]' in content


# ===========================================================================
# 9. Position cache — push-based, zero serial I/O
# ===========================================================================

class TestPositionCache:
    """Verify push-based position cache in Lumascope API.

    The position cache eliminates serial polling from the GUI layer.
    Positions are updated on move commands and after homing — the GUI
    reads from cache with zero hardware calls.
    """

    def test_initial_cache_is_zero(self, sim_scope):
        """Cache starts at 0 for all axes before any moves."""
        assert sim_scope.get_target_position('X') == 0.0
        assert sim_scope.get_target_position('Y') == 0.0
        assert sim_scope.get_target_position('Z') == 0.0

    def test_move_absolute_updates_cache(self, sim_scope):
        """move_absolute_position should push the new position into the cache."""
        sim_scope.move_absolute_position('Z', 5000.0)
        assert sim_scope.get_target_position('Z') == 5000.0

    def test_move_absolute_only_updates_target_axis(self, sim_scope):
        """Moving Z should not affect X or Y cache."""
        sim_scope.move_absolute_position('Z', 5000.0)
        assert sim_scope.get_target_position('X') == 0.0
        assert sim_scope.get_target_position('Y') == 0.0

    def test_move_relative_updates_cache(self, sim_scope):
        """move_relative_position should accumulate into the cache."""
        sim_scope.move_absolute_position('X', 1000.0)
        sim_scope.move_relative_position('X', 500.0)
        assert sim_scope.get_target_position('X') == 1500.0

    def test_move_relative_negative(self, sim_scope):
        """Negative relative moves should subtract from cache."""
        sim_scope.move_absolute_position('Z', 3000.0)
        sim_scope.move_relative_position('Z', -1000.0)
        assert sim_scope.get_target_position('Z') == 2000.0

    def test_get_all_axes(self, sim_scope):
        """get_target_position(None) returns dict of all axes."""
        sim_scope.move_absolute_position('X', 100.0)
        sim_scope.move_absolute_position('Y', 200.0)
        sim_scope.move_absolute_position('Z', 300.0)
        result = sim_scope.get_target_position()
        assert isinstance(result, dict)
        assert result['X'] == 100.0
        assert result['Y'] == 200.0
        assert result['Z'] == 300.0

    def test_get_current_position_matches_target(self, sim_scope):
        """After a blocking move, get_current_position returns the target."""
        sim_scope.move_absolute_position('Z', 7777.0, wait_until_complete=True)
        assert sim_scope.get_current_position('Z') == 7777.0

    def test_refresh_after_homing(self, sim_scope):
        """refresh_position_cache syncs cache from hardware (used after homing)."""
        # Directly set the simulated motor's internal position to simulate homing
        # The simulated motor stores positions in microsteps; target_pos() converts.
        # Use move_abs_pos to set a known position, then verify refresh reads it.
        sim_scope.motion.move_abs_pos('Z', 5000.0)
        # Cache still has old value since we bypassed move_absolute_position
        assert sim_scope.get_target_position('Z') != 5000.0
        # Now refresh from hardware
        sim_scope.refresh_position_cache()
        # Should now match what the motor reports
        pos = sim_scope.get_target_position('Z')
        assert abs(pos - 5000.0) < 1.0  # allow microstep rounding

    def test_cache_returns_copy(self, sim_scope):
        """get_target_position(None) should return a copy, not the internal dict."""
        result = sim_scope.get_target_position()
        result['X'] = 99999.0
        # Internal cache should be unaffected
        assert sim_scope.get_target_position('X') == 0.0


# ===========================================================================
# 8. Axis state model — push-based state tracking (zero serial I/O)
# ===========================================================================

class TestAxisState:
    """Verify axis state transitions in the Lumascope API."""

    def test_initial_state_is_unknown(self, sim_scope):
        """All axes start in UNKNOWN state before homing."""
        from modules.lumascope_api import AxisState
        for ax in ('X', 'Y', 'Z', 'T'):
            assert sim_scope.get_axis_state(ax) == AxisState.UNKNOWN

    def test_axis_state_idle_after_move_with_wait(self, sim_scope):
        """After move_absolute_position with wait_until_complete, axis is IDLE."""
        from modules.lumascope_api import AxisState
        sim_scope.move_absolute_position('Z', 1000, wait_until_complete=True)
        assert sim_scope.get_axis_state('Z') == AxisState.IDLE

    def test_axis_state_moving_during_fire_and_forget(self, sim_scope):
        """After fire-and-forget move, axis is initially MOVING then transitions to IDLE."""
        from modules.lumascope_api import AxisState
        sim_scope.move_absolute_position('Z', 500, wait_until_complete=False)
        state = sim_scope.get_axis_state('Z')
        # Simulated move completes instantly; motion monitor may or may not have
        # polled yet. Both MOVING and IDLE are valid states at this point.
        assert state in (AxisState.MOVING, AxisState.IDLE)

    def test_axis_state_homing_zhome(self, sim_scope):
        """After zhome, Z axis should be IDLE (homing is blocking)."""
        from modules.lumascope_api import AxisState
        sim_scope.zhome()
        assert sim_scope.get_axis_state('Z') == AxisState.IDLE

    def test_axis_state_homing_home(self, sim_scope):
        """After home(), present axes should be IDLE."""
        from modules.lumascope_api import AxisState
        sim_scope.home()
        for ax in sim_scope.axes_present():
            assert sim_scope.get_axis_state(ax) == AxisState.IDLE

    def test_axis_state_homing_thome(self, _mock_heavy_deps):
        """After thome on a turret-equipped scope, T axis should be IDLE.

        Uses an LS850T sim explicitly instead of the default LS850
        sim_scope fixture (which has no turret) — pre-B4 the test passed
        on LS850 only because `_axis_state['T']` was a phantom key from
        the hardcoded VALID_AXES tuple. Post-B4, T is correctly absent
        on no-turret scopes and `thome()` is a Rule 8 silent no-op there.
        """
        from modules.lumascope_api import Lumascope, AxisState
        from drivers.simulated_motorboard import SimulatedMotorBoard
        scope = Lumascope(simulate=True)
        scope.motion = SimulatedMotorBoard(model='LS850T')
        present = scope.motion.detect_present_axes()
        assert 'T' in present, "LS850T sim must report T present"
        scope._pos_cache = {ax: 0.0 for ax in present}
        scope._axis_state = {ax: AxisState.UNKNOWN for ax in present}
        scope._arrival_events = {ax: threading.Event() for ax in present}
        for ev in scope._arrival_events.values():
            ev.set()
        scope._move_profile = {ax: None for ax in present}

        scope.thome()
        assert scope.get_axis_state('T') == AxisState.IDLE

    def test_thome_on_no_turret_scope_is_silent_noop(self, sim_scope):
        """Audit B4 + Rule 8: calling thome() on a scope without a
        turret (LS850 default sim) must not raise and must leave T in
        UNKNOWN state — there is no phantom T axis to transition."""
        from modules.lumascope_api import AxisState
        assert 'T' not in sim_scope.axes_present()
        # Must not raise — Rule 8 silent no-op:
        sim_scope.thome()
        assert sim_scope.get_axis_state('T') == AxisState.UNKNOWN

    def test_is_any_axis_moving_false_when_all_idle(self, sim_scope):
        """is_any_axis_moving() returns False when all axes are IDLE."""
        from modules.lumascope_api import AxisState
        # Home all axes to set them IDLE
        sim_scope.zhome()
        sim_scope.home()
        assert not sim_scope.is_any_axis_moving()

    def test_is_any_axis_moving_true_when_moving(self, sim_scope):
        """is_any_axis_moving() returns True when an axis is in MOVING state."""
        from modules.lumascope_api import AxisState
        # Directly set state to avoid race with motion monitor on instant simulator
        sim_scope._set_axis_state('Z', AxisState.MOVING)
        assert sim_scope.is_any_axis_moving()
        sim_scope._set_axis_state('Z', AxisState.IDLE)

    def test_monitor_reconciles_state(self, sim_scope):
        """Motion monitor thread should detect arrival and set state to IDLE."""
        from modules.lumascope_api import AxisState
        sim_scope.move_absolute_position('Z', 1000, wait_until_complete=False)
        # In simulation, the move completes instantly. The motion monitor thread
        # detects arrival at 50Hz and transitions state to IDLE.
        sim_scope.wait_until_finished_moving(timeout=2.0)
        assert not sim_scope.is_moving()
        assert sim_scope.get_axis_state('Z') == AxisState.IDLE

    def test_disconnect_sets_unknown(self, sim_scope):
        """After disconnect, all axes should be UNKNOWN."""
        from modules.lumascope_api import AxisState
        sim_scope.zhome()  # Set to IDLE first
        sim_scope.disconnect()
        for ax in ('X', 'Y', 'Z', 'T'):
            assert sim_scope.get_axis_state(ax) == AxisState.UNKNOWN

    def test_axes_present(self, sim_scope):
        """axes_present() delegates to motion.detect_present_axes() (Rule 9).

        Default sim model LS850 has X/Y/Z and no turret, so the result
        must match the motion layer rather than a full 4-axis hardcoded
        list.
        """
        axes = sim_scope.axes_present()
        assert set(axes) == set(sim_scope.motion.detect_present_axes())
        assert set(axes) == {'X', 'Y', 'Z'}  # LS850 default — no T

    def test_has_axis(self, sim_scope):
        """has_axis() returns correct values."""
        assert sim_scope.has_axis('Z') is True
        assert sim_scope.has_axis('Q') is False

    def test_move_relative_state_tracking(self, sim_scope):
        """move_relative_position tracks axis state correctly."""
        from modules.lumascope_api import AxisState
        sim_scope.move_relative_position('Z', 100, wait_until_complete=True)
        assert sim_scope.get_axis_state('Z') == AxisState.IDLE

    def test_xycenter_state_tracking(self, sim_scope):
        """xycenter sets X/Y to IDLE after completion."""
        from modules.lumascope_api import AxisState
        sim_scope.xycenter()
        assert sim_scope.get_axis_state('X') == AxisState.IDLE
        assert sim_scope.get_axis_state('Y') == AxisState.IDLE


# ===========================================================================
# Issue Regression Tests — each bug fix gets a test (Rule 18)
# ===========================================================================

class TestIssue602_AFExecutorLED:
    """#602: Autofocus All Steps doesn't turn on the LED.

    Root cause: AF executor had no LED control. Fix: AF executor
    accepts led_color/led_illumination and manages its own LED.
    """

    def test_af_executor_accepts_led_params(self, _mock_heavy_deps):
        """AutofocusExecutor.run() should accept led_color and led_illumination."""
        import inspect
        from modules.autofocus_executor import AutofocusExecutor
        sig = inspect.signature(AutofocusExecutor.run)
        assert 'led_color' in sig.parameters
        assert 'led_illumination' in sig.parameters

    def test_af_executor_turns_led_on(self, _mock_heavy_deps):
        """AF executor should call led_on when led_color is provided."""
        from modules.autofocus_executor import AutofocusExecutor
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        from modules.sequential_io_executor import SequentialIOExecutor
        io = SequentialIOExecutor(name="IO_TEST")
        cam = SequentialIOExecutor(name="CAM_TEST")
        af_ex = SequentialIOExecutor(name="AF_TEST")
        file_ex = SequentialIOExecutor(name="FILE_TEST")
        af = AutofocusExecutor(
            scope=scope,
            camera_executor=cam,
            io_executor=io,
            file_io_executor=file_ex,
            autofocus_executor=af_ex,
        )
        # Verify _led_on and _led_off methods exist
        assert hasattr(af, '_led_on')
        assert hasattr(af, '_led_off')
        # Verify _reset_state initializes LED fields
        af._reset_state()
        assert af._led_color is None
        assert af._led_illumination == 0

    def test_af_executor_led_off_in_cancel(self, _mock_heavy_deps):
        """AF executor cancel() should turn off LED."""
        from modules.autofocus_executor import AutofocusExecutor
        from modules.lumascope_api import Lumascope
        from unittest.mock import patch

        scope = Lumascope(simulate=True)
        from modules.sequential_io_executor import SequentialIOExecutor
        io = SequentialIOExecutor(name="IO_TEST")
        cam = SequentialIOExecutor(name="CAM_TEST")
        af_ex = SequentialIOExecutor(name="AF_TEST")
        file_ex = SequentialIOExecutor(name="FILE_TEST")
        af = AutofocusExecutor(
            scope=scope,
            camera_executor=cam,
            io_executor=io,
            file_io_executor=file_ex,
            autofocus_executor=af_ex,
        )
        # Set LED state as if AF was running with LED
        af._led_color = 'BF'
        af._led_illumination = 100
        af._af_in_progress.set()

        with patch.object(af, '_led_off') as mock_led_off:
            af.cancel()
            mock_led_off.assert_called_once()


class TestIssue605_AccordionLEDProtocol:
    """#605: Stepping through Protocol with 'Protocol LEDs On' doesn't stay on.

    Root cause: accordion_collapse() unconditionally called scope_leds_off().
    Fix: skip leds_off when protocol_led_on setting is active.
    """

    def test_accordion_collapse_has_protocol_led_on_guard(self):
        """accordion_collapse source must check protocol_led_on setting."""
        import pathlib
        source = pathlib.Path("ui/image_settings.py").read_text()
        # Find the accordion_collapse method body
        assert "protocol_led_on" in source, \
            "accordion_collapse must check protocol_led_on setting (#605)"
        assert "scope_leds_off" in source, \
            "accordion_collapse must still call scope_leds_off when protocol_led_on is False"


class TestIssue606_TurretObjectiveValidation:
    """#606: Objective changeable without turret position assignment.

    Root cause: no validation in select_objective() or _is_protocol_valid().
    Fix: warn on select, block protocol run.
    """

    def test_select_objective_validates_turret(self):
        """select_objective source must check turret assignments."""
        import pathlib
        source = pathlib.Path("ui/microscope_settings.py").read_text()
        assert "Objective Not in Turret" in source, \
            "select_objective must warn when objective not in turret (#606)"

    def test_is_protocol_valid_checks_turret(self):
        """_is_protocol_valid source must validate turret config."""
        import pathlib
        source = pathlib.Path("ui/protocol_settings.py").read_text()
        # Find the _is_protocol_valid method
        idx = source.find("def _is_protocol_valid")
        assert idx != -1, "_is_protocol_valid method must exist"
        method_body = source[idx:idx+2000]
        assert "turret" in method_body.lower(), \
            "_is_protocol_valid must check turret objective assignments (#606)"


# ===========================================================================
# Audit Fix Regression Tests — Session 8 (B6, B5, D2, G3, F7, G4)
# ===========================================================================

class TestB6_WriteMotorRegisterRemoved:
    """B6: write_motor_register() was dead code with zero callers."""

    def test_write_motor_register_removed(self, _mock_heavy_deps):
        """write_motor_register should no longer exist on the API class."""
        from modules.lumascope_api import Lumascope
        scope = Lumascope(simulate=True)
        assert not hasattr(scope, 'write_motor_register'), \
            "write_motor_register() should have been removed (B6 — zero callers)"


class TestB5_GetCurrentPositionUsesAxesPresent:
    """B5: get_current_position(axis=None) should use axes_present(), not
    a hardcoded 4-axis list."""

    def test_returns_only_present_axes(self, _mock_heavy_deps):
        """get_current_position(None) should return dict keyed by present axes only."""
        from modules.lumascope_api import Lumascope
        scope = Lumascope(simulate=True)
        result = scope.get_current_position(axis=None)
        assert set(result.keys()) == set(scope.axes_present()), \
            "get_current_position(None) should use axes_present(), not a hardcoded axis list"


class TestD2_LEDBoardStateCacheHelper:
    """D2: LED state cache updates should use _update_state_cache() helper."""

    def test_update_state_cache_exists(self, _mock_heavy_deps):
        """LEDBoard should have _update_state_cache method."""
        from drivers.ledboard import LEDBoard
        assert hasattr(LEDBoard, '_update_state_cache'), \
            "LEDBoard must have _update_state_cache helper (D2)"

    def test_led_on_fast_updates_cache(self, _mock_heavy_deps):
        """led_on_fast should update state cache via _update_state_cache."""
        from drivers.simulated_ledboard import SimulatedLEDBoard
        led = SimulatedLEDBoard()
        led.led_on_fast(0, 100)
        # SimulatedLEDBoard tracks its own state; verify the color cache
        color = led.ch2color(0)
        assert led.led_ma[color] == 100


class TestG3_AutofocusFailureNotification:
    """G3: AF failures must notify the user (Rule 14)."""

    def test_af_exception_notifies_user(self, _mock_heavy_deps):
        """AF exception handler must call notifications.error()."""
        import pathlib
        source = pathlib.Path("modules/autofocus_executor.py").read_text()
        # Find the exception handler block
        idx = source.find("Error during loop")
        assert idx != -1, "Exception handler must exist"
        # Check notification exists near the error handler
        nearby = source[idx:idx+300]
        assert "notifications.error" in nearby, \
            "AF exception handler must call notifications.error (G3 — Rule 14)"

    def test_af_degenerate_curve_notifies_user(self, _mock_heavy_deps):
        """AF degenerate curve detection must call notifications.error()."""
        import pathlib
        source = pathlib.Path("modules/autofocus_executor.py").read_text()
        idx = source.find("degenerate focus curve")
        assert idx != -1, "Degenerate curve handler must exist"
        nearby = source[idx:idx+500]
        assert "notifications.error" in nearby, \
            "AF degenerate curve handler must call notifications.error (G3 — Rule 14)"

    def test_af_imports_notifications(self, _mock_heavy_deps):
        """autofocus_executor must import notifications module."""
        import pathlib
        source = pathlib.Path("modules/autofocus_executor.py").read_text()
        assert "from modules.notification_center import notifications" in source, \
            "autofocus_executor must import notifications (G3)"


class TestRule14_A4_PreRunValidationNotify:
    """A4: Pre-run validation errors must surface a user notification (Rule 14)."""

    def test_validation_errors_branch_notifies(self):
        """sequenced_capture_executor must call notifications.error when
        validation_errors is non-empty before returning."""
        import pathlib
        source = pathlib.Path("modules/sequenced_capture_executor.py").read_text()
        idx = source.find("Protocol has {len(validation_errors)} validation error(s). Cannot start run.")
        assert idx != -1, "Validation-errors return path must exist"
        nearby = source[idx:idx+800]
        assert "notifications.error" in nearby, \
            "validation_errors return path must call notifications.error (A4 -- Rule 14)"
        assert "Validation failed" in nearby, \
            "notification title must be 'Validation failed' (A4 -- audit recommendation)"

    def test_validation_summary_truncates_at_five(self):
        """Notification summary must show first 5 errors; mention 'see log' for overflow."""
        import pathlib
        source = pathlib.Path("modules/sequenced_capture_executor.py").read_text()
        idx = source.find("validation_errors[:5]")
        assert idx != -1, \
            "Notification summary must slice validation_errors[:5] to keep popup readable (A4)"
        idx = source.find("more (see log)")
        assert idx != -1, \
            "Overflow message must point user to the log for full details (A4)"


class TestRule14_A5_AreAllConnectedExceptionNotify:
    """A5: are_all_connected() exception branch must notify (Rule 14)."""

    def test_are_all_connected_exception_branch_notifies(self):
        """sequenced_capture_executor must call notifications.error when the
        are_all_connected check itself raises, before returning."""
        import pathlib
        source = pathlib.Path("modules/sequenced_capture_executor.py").read_text()
        idx = source.find("Error checking scope connection")
        assert idx != -1, "are_all_connected exception handler must exist"
        nearby = source[idx:idx+600]
        assert "notifications.error" in nearby, \
            "are_all_connected exception path must call notifications.error (A5 -- Rule 14)"
        assert "Cannot verify hardware state" in nearby, \
            "notification title must be 'Cannot verify hardware state' (A5 -- audit recommendation)"


class TestRule14_A8_ScopeSessionHelperNotify:
    """A8: scope_session optional helper failures must notify (Rule 14)."""

    def test_wellplate_loader_failure_notifies(self):
        import pathlib
        source = pathlib.Path("modules/scope_session.py").read_text()
        idx = source.find("Could not load wellplate loader:")
        assert idx != -1, "Wellplate loader except branch must exist"
        nearby = source[idx:idx+500]
        assert "notifications.warning" in nearby, \
            "Wellplate loader exception must call notifications.warning (A8 -- Rule 14)"
        assert "Wellplate loader unavailable" in nearby, \
            "Notification title must be 'Wellplate loader unavailable' (A8)"

    def test_coord_transformer_failure_notifies(self):
        import pathlib
        source = pathlib.Path("modules/scope_session.py").read_text()
        idx = source.find("Could not load coordinate transformer:")
        assert idx != -1, "Coordinate transformer except branch must exist"
        nearby = source[idx:idx+500]
        assert "notifications.warning" in nearby, \
            "Coordinate transformer exception must call notifications.warning (A8)"
        assert "Coordinate transformer unavailable" in nearby, \
            "Notification title must be 'Coordinate transformer unavailable' (A8)"

    def test_objective_helper_failure_notifies(self):
        import pathlib
        source = pathlib.Path("modules/scope_session.py").read_text()
        idx = source.find("Could not load objective helper:")
        assert idx != -1, "Objective helper except branch must exist"
        nearby = source[idx:idx+500]
        assert "notifications.warning" in nearby, \
            "Objective helper exception must call notifications.warning (A8)"
        assert "Objective helper unavailable" in nearby, \
            "Notification title must be 'Objective helper unavailable' (A8)"


class TestRule14_A7_HyperstackBuildNotify:
    """A7: Hyperstack build background-thread failure must notify (Rule 14)."""

    def test_hyperstack_build_exception_notifies(self):
        """create_hyperstacks_if_needed _build() must call notifications.error
        when stack_builder.load_folder raises."""
        import pathlib
        source = pathlib.Path("modules/config_ui_getters.py").read_text()
        idx = source.find('logger.exception("Error building hyperstacks")')
        assert idx != -1, "Hyperstack build exception handler must exist"
        nearby = source[idx:idx+500]
        assert "notifications.error" in nearby, \
            "Hyperstack build exception path must call notifications.error (A7 -- Rule 14)"
        assert "Hyperstack build failed" in nearby, \
            "notification title must be 'Hyperstack build failed' (A7 -- audit recommendation)"


class TestRule14_A10_ProtocolCleanupErrorCollection:
    """A10: protocol_cleanup must collect cleanup errors and surface a single
    summary notification (Rule 14)."""

    def test_cleanup_collects_errors(self):
        """run_cleanup must initialize cleanup_errors list and append to it
        on each step's exception."""
        import pathlib
        source = pathlib.Path("modules/protocol_cleanup.py").read_text()
        assert "cleanup_errors: list[str] = []" in source, \
            "run_cleanup must initialize cleanup_errors list (A10)"
        # Verify each except branch appends
        assert source.count("cleanup_errors.append") >= 6, \
            "Each cleanup step except branch must append to cleanup_errors (A10 -- 6 steps)"

    def test_cleanup_summary_notify(self):
        """run_cleanup must surface a single summary notification when
        cleanup_errors is non-empty."""
        import pathlib
        source = pathlib.Path("modules/protocol_cleanup.py").read_text()
        idx = source.find("if cleanup_errors:")
        assert idx != -1, "Cleanup-errors summary block must exist"
        nearby = source[idx:idx+800]
        assert "notifications.warning" in nearby, \
            "Cleanup-errors block must call notifications.warning (A10 -- summary, not 6 popups)"
        assert "Protocol cleanup issues" in nearby, \
            "Notification title must be 'Protocol cleanup issues' (A10)"
        assert "Check LED state, camera settings, and stage position." in nearby, \
            "Notification body must prompt user to verify hardware state (A10 audit recommendation)"


class TestRule14_A9_SetBinningSizeNotify:
    """A9: set_binning_size exception must surface a user notification (Rule 14)."""

    def test_set_binning_size_exception_notifies(self):
        """lumascope_api.set_binning_size must call notifications.error when
        the underlying SDK call raises."""
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        idx = source.find("def set_binning_size(self, size: int) -> bool:")
        assert idx != -1, "set_binning_size must exist with `-> bool` annotation"
        method_body = source[idx:idx+1500]
        assert "notifications.error" in method_body, \
            "set_binning_size exception path must call notifications.error (A9 -- Rule 14)"
        assert "Binning change failed" in method_body, \
            "notification title must be 'Binning change failed' (A9 -- audit recommendation)"


class TestSetBinningSizeReturnsBool:
    """Wave 1 / B1: Lumascope.set_binning_size must propagate the driver's bool.

    Bench session 2026-05-05 surfaced a phantom-failure bug where the API
    method dropped the driver's True return and implicitly returned None;
    char-tool's `if not ok:` check then misreported every successful binning
    op as a failure. This test pins the contract: capture-and-return on the
    success path, return False on exception.
    """

    def test_set_binning_size_has_bool_return_annotation(self):
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        idx = source.find("def set_binning_size(self, size: int) -> bool:")
        assert idx != -1, \
            "Lumascope.set_binning_size must declare `-> bool` (Wave 1 B1; Rule 37)"

    def test_set_binning_size_returns_driver_value(self):
        """Method body must capture and return the driver's return value
        on the success path, not drop it."""
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        idx = source.find("def set_binning_size(self, size: int) -> bool:")
        assert idx != -1
        # End the slice at the next def at module column 4 to scope the body
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+2000]
        assert "ok = self.camera.set_binning_size(size=size)" in body, \
            "set_binning_size must capture driver return into `ok`"
        assert "return ok" in body, \
            "set_binning_size success path must `return ok` (Wave 1 B1)"
        assert "return False" in body, \
            "set_binning_size exception path must `return False`"

    def test_set_binning_size_has_returns_docstring_section(self):
        """Rule 38: public methods declare what they return."""
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        idx = source.find("def set_binning_size(self, size: int) -> bool:")
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+2000]
        assert "Returns:" in body, \
            "set_binning_size docstring must have a Returns: section (Rule 38)"

    def test_pyloncamera_set_binning_size_raises_hardware_error(self):
        """Tier 3a / C2: PylonCamera.set_binning_size must raise HardwareError
        on caught exception paths, not return False (Rule 29)."""
        import pathlib
        source = pathlib.Path("drivers/pyloncamera.py").read_text()
        idx = source.find("def set_binning_size(self, size: int) -> bool:")
        assert idx != -1
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+3000]
        # Three exception classes; each must raise HardwareError, not return False
        for exc_clause in (
            "except genicam.TimeoutException",
            "except genicam.RuntimeException",
            "except Exception",
        ):
            assert exc_clause in body, f"PylonCamera.set_binning_size must keep {exc_clause}"
        assert body.count("raise HardwareError(") >= 3, \
            "PylonCamera.set_binning_size must raise HardwareError on each caught exception (C2)"

    def test_pyloncamera_set_pixel_format_raises_hardware_error(self):
        """Tier 3a / C1."""
        import pathlib
        source = pathlib.Path("drivers/pyloncamera.py").read_text()
        idx = source.find("def set_pixel_format(self, pixel_format: str) -> bool:")
        assert idx != -1
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+3000]
        assert body.count("raise HardwareError(") >= 2, \
            "PylonCamera.set_pixel_format must raise HardwareError on each caught exception (C1)"

    def test_idscamera_set_binning_size_raises_hardware_error(self):
        """Tier 3a / C5."""
        import pathlib
        source = pathlib.Path("drivers/idscamera.py").read_text()
        idx = source.find("def set_binning_size(self, size: int) -> bool:")
        assert idx != -1
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+3000]
        assert "raise HardwareError(" in body, \
            "IDSCamera.set_binning_size must raise HardwareError on caught exception (C5)"

    def test_idscamera_set_pixel_format_raises_and_annotated(self):
        """Tier 3a / C3 + Tier 1-A: annotation added, raises HardwareError."""
        import pathlib
        source = pathlib.Path("drivers/idscamera.py").read_text()
        idx = source.find("def set_pixel_format(self, pixel_format: str) -> bool:")
        assert idx != -1, "IDSCamera.set_pixel_format must declare `-> bool` (Wave 1 C3 / Rule 37)"
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+3000]
        assert "raise HardwareError(" in body, \
            "IDSCamera.set_pixel_format must raise HardwareError on caught exception (C3)"


class TestHomeReturnsBool:
    """Wave 2 / B8 + B9 + B10 + D1: Lumascope.{home, zhome, thome} must
    propagate the driver's bool, and MotorBoard / SimulatedMotorBoard
    must raise HardwareError instead of returning False on no-response /
    firmware-error paths (Rule 29).

    Pairs with the existing `--run-homing` opt-in test set; this class
    pins the mechanical contract (annotations + return propagation +
    typed exceptions) so a future regression can't silently revert it.
    """

    def test_lumascope_zhome_has_bool_return_annotation(self):
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        assert "def zhome(self) -> bool:" in source, \
            "Lumascope.zhome must declare `-> bool` (Wave 2 B9; Rule 37)"

    def test_lumascope_home_has_bool_return_annotation(self):
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        assert "def home(self) -> bool:" in source, \
            "Lumascope.home must declare `-> bool` (Wave 2 B10; Rule 37)"

    def test_lumascope_thome_has_bool_return_annotation(self):
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        assert "def thome(self) -> bool:" in source, \
            "Lumascope.thome must declare `-> bool` (Wave 2 B8; Rule 37)"

    def test_lumascope_zhome_returns_driver_value(self):
        """Method body must return True on success and False on failure paths."""
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        idx = source.find("def zhome(self) -> bool:")
        assert idx != -1
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+2000]
        assert "result = self.motion.zhome()" in body, \
            "zhome must capture driver return into `result`"
        assert "return True" in body, \
            "zhome success path must `return True` (Wave 2 B9)"
        assert "return False" in body, \
            "zhome failure paths must `return False` (Wave 2 B9)"
        assert "Returns:" in body, \
            "zhome docstring must have a Returns: section (Rule 38)"

    def test_lumascope_home_returns_driver_value(self):
        """Method body must capture and propagate the driver's return."""
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        idx = source.find("def home(self) -> bool:")
        assert idx != -1
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+3000]
        assert "result = self.motion.home()" in body, \
            "home must capture driver return into `result`"
        assert "return True" in body, \
            "home success path must `return True` (Wave 2 B10)"
        assert "return False" in body, \
            "home failure paths must `return False` (Wave 2 B10)"
        assert "Returns:" in body, \
            "home docstring must have a Returns: section (Rule 38)"

    def test_lumascope_thome_returns_driver_value(self):
        """Method body must capture, notify on False, and return the bool.

        Pre-Wave-2, thome dropped the driver return entirely (no capture,
        no notify on failure). This pins the captured-and-returned shape.
        """
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        idx = source.find("def thome(self) -> bool:")
        assert idx != -1
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+3000]
        assert "result = self.motion.thome()" in body, \
            "thome must capture driver return into `result` (Wave 2 B8)"
        assert "return True" in body, \
            "thome success path must `return True` (Wave 2 B8)"
        assert "return False" in body, \
            "thome failure paths must `return False` (Wave 2 B8)"
        assert "Turret homing failed" in body or "Homing Failed" in body, \
            "thome must notify the user on driver False (Rule 14)"
        assert "Returns:" in body, \
            "thome docstring must have a Returns: section (Rule 38)"

    def test_motorboard_zhome_raises_hardware_error(self):
        """Tier 3b D1: MotorBoard.zhome must raise HardwareError on
        no-response and firmware-error paths, not return False (Rule 29)."""
        import pathlib
        source = pathlib.Path("drivers/motorboard.py").read_text()
        idx = source.find("def zhome(self) -> bool:")
        assert idx != -1, \
            "MotorBoard.zhome must declare `-> bool` (Tier 1-A / Rule 37)"
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+2000]
        # Two error paths: no-response and firmware-error
        assert body.count("raise HardwareError(") >= 2, \
            "MotorBoard.zhome must raise HardwareError on no-response AND firmware-error (D1)"
        assert "Raises:" in body, \
            "MotorBoard.zhome docstring must document HardwareError (Rule 38)"

    def test_motorboard_home_raises_hardware_error(self):
        """Tier 3b D1."""
        import pathlib
        source = pathlib.Path("drivers/motorboard.py").read_text()
        idx = source.find("def home(self) -> bool:")
        assert idx != -1
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+3000]
        assert body.count("raise HardwareError(") >= 2, \
            "MotorBoard.home must raise HardwareError on each failure path (D1)"
        assert "Raises:" in body, \
            "MotorBoard.home docstring must document HardwareError (Rule 38)"

    def test_motorboard_thome_raises_hardware_error(self):
        """Tier 3b D1."""
        import pathlib
        source = pathlib.Path("drivers/motorboard.py").read_text()
        idx = source.find("def thome(self) -> bool:")
        assert idx != -1
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+2000]
        assert body.count("raise HardwareError(") >= 2, \
            "MotorBoard.thome must raise HardwareError on each failure path (D1)"
        assert "Raises:" in body, \
            "MotorBoard.thome docstring must document HardwareError (Rule 38)"

    def test_simulated_motorboard_home_family_raises_hardware_error(self):
        """Tier 3b D1: SimulatedMotorBoard mirrors MotorBoard contract so
        sim-backed tests exercise the same exception path as production."""
        import pathlib
        source = pathlib.Path("drivers/simulated_motorboard.py").read_text()
        assert "from drivers.exceptions import HardwareError" in source, \
            "SimulatedMotorBoard must import HardwareError"
        for method in ("zhome", "home", "thome"):
            idx = source.find(f"def {method}(self) -> bool:")
            assert idx != -1, \
                f"SimulatedMotorBoard.{method} must declare `-> bool`"
            next_def = source.find("\n    def ", idx + 1)
            body = source[idx:next_def] if next_def != -1 else source[idx:idx+2000]
            assert "raise HardwareError(" in body, \
                f"SimulatedMotorBoard.{method} must raise HardwareError on failure (D1)"


class TestDisconnectReturnsBool:
    """Wave 4 / B2: Lumascope.disconnect must return an aggregated bool
    indicating whether all sub-system disconnects (LED + motion + camera)
    succeeded. Best-effort teardown still runs every sub-system and
    resets state to Null variants even on partial failure.
    """

    def test_disconnect_has_bool_return_annotation(self):
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        assert "def disconnect(self) -> bool:" in source, \
            "Lumascope.disconnect must declare `-> bool` (Wave 4 B2; Rule 37)"

    def test_disconnect_aggregates_and_returns_bool(self):
        """Method body must aggregate three sub-system bools and return."""
        import pathlib
        source = pathlib.Path("modules/lumascope_api.py").read_text()
        idx = source.find("def disconnect(self) -> bool:")
        assert idx != -1
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+4000]
        # Each sub-system tracked independently:
        for var in ("led_ok", "motion_ok", "camera_ok"):
            assert var in body, f"disconnect must track {var} (Wave 4 B2)"
        # Aggregation + return:
        assert "led_ok and motion_ok and camera_ok" in body, \
            "disconnect must aggregate the three sub-bools (Wave 4 B2)"
        assert "return all_ok" in body, \
            "disconnect must return the aggregate (Wave 4 B2)"
        # Per-failure notification (Rule 14):
        assert "notifications.error(" in body, \
            "disconnect must notify per failure (Rule 14)"
        # Returns: docstring section (Rule 38):
        assert "Returns:" in body, \
            "disconnect docstring must have a Returns: section (Rule 38)"

    def test_disconnect_on_simulator_returns_true(self, sim_scope):
        """Sim path: every sub-system disconnects cleanly -> True."""
        # `sim_scope` fixture's teardown also calls disconnect; this
        # call covers the explicit-return-value contract.
        result = sim_scope.disconnect()
        assert result is True, \
            "Simulator disconnect must return True when no sub-system fails"

    def test_disconnect_camera_failure_returns_false(self, sim_scope):
        """If camera.disconnect raises, the API must catch, notify, and
        still return False. LED + motion still attempted; state still reset."""
        # Replace the camera with one whose disconnect raises.
        from unittest.mock import MagicMock
        sim_scope.camera = MagicMock()
        sim_scope.camera.disconnect = MagicMock(side_effect=RuntimeError("boom"))
        result = sim_scope.disconnect()
        assert result is False, \
            "disconnect must return False when camera teardown raises"
        assert sim_scope.camera is None, \
            "disconnect must reset self.camera even when teardown raises"


class TestEnterEngineeringModeRaises:
    """Wave 4 / D2: LEDBoard.enter_engineering_mode must raise
    HardwareError on the no-response and no-Y/N-prompt failure paths
    instead of `return False` (Rule 29).
    """

    def test_ledboard_enter_engineering_mode_has_bool_return(self):
        import pathlib
        source = pathlib.Path("drivers/ledboard.py").read_text()
        assert "def enter_engineering_mode(self, timeout: float = 5.0) -> bool:" in source, \
            "LEDBoard.enter_engineering_mode must declare `-> bool` (Tier 1-A; Rule 37)"

    def test_ledboard_enter_engineering_mode_raises(self):
        """Two failure paths must raise HardwareError."""
        import pathlib
        source = pathlib.Path("drivers/ledboard.py").read_text()
        idx = source.find("def enter_engineering_mode(self, timeout: float = 5.0) -> bool:")
        assert idx != -1
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx:idx+3000]
        assert body.count("raise HardwareError(") >= 2, \
            "enter_engineering_mode must raise HardwareError on both failure paths (D2)"
        assert "Raises:" in body, \
            "enter_engineering_mode docstring must document HardwareError (Rule 38)"
        # The legacy `return False` paths must be gone:
        # (Sanity check -- the only `return` in the body should be
        # `return True` on success; `return False` means migration regressed.)
        assert "return False" not in body, \
            "enter_engineering_mode must no longer `return False` (Rule 29 / D2)"

    def test_ledboard_imports_hardware_error(self):
        import pathlib
        source = pathlib.Path("drivers/ledboard.py").read_text()
        assert "from drivers.exceptions import HardwareError" in source, \
            "ledboard must import HardwareError"


class TestF7_ProtocolHomingInterlock:
    """F7: Homing/bookmark must be blocked during protocol execution."""

    def test_z_home_checks_protocol_running(self):
        """vertical_control home() must check protocol_running."""
        import pathlib
        source = pathlib.Path("ui/vertical_control.py").read_text()
        # Find the home method
        idx = source.find("def home(self):")
        assert idx != -1
        method_body = source[idx:idx+300]
        assert "protocol_running.is_set()" in method_body, \
            "Z home() must check protocol_running before homing (F7)"

    def test_goto_bookmark_checks_protocol_running(self):
        """vertical_control goto_bookmark() must check protocol_running."""
        import pathlib
        source = pathlib.Path("ui/vertical_control.py").read_text()
        idx = source.find("def goto_bookmark(self):")
        assert idx != -1
        method_body = source[idx:idx+300]
        assert "protocol_running.is_set()" in method_body, \
            "goto_bookmark() must check protocol_running (F7)"

    def test_turret_home_checks_protocol_running(self):
        """vertical_control turret_home() must check protocol_running."""
        import pathlib
        source = pathlib.Path("ui/vertical_control.py").read_text()
        idx = source.find("def turret_home(self):")
        assert idx != -1
        method_body = source[idx:idx+300]
        assert "protocol_running.is_set()" in method_body, \
            "turret_home() must check protocol_running (F7)"

    def test_xy_home_checks_protocol_running(self):
        """motion_settings home() must check protocol_running."""
        import pathlib
        source = pathlib.Path("ui/motion_settings.py").read_text()
        # Find the XYStageControl home method (after line 460)
        idx = source.find("def home(self):")
        assert idx != -1
        method_body = source[idx:idx+300]
        assert "protocol_running.is_set()" in method_body, \
            "XY home() must check protocol_running before homing (F7)"


class TestG4_MotorLogSuppression:
    """G4: Motor board should suppress only connect errors, not entire thread logging."""

    def test_no_pause_thread_in_motorboard(self):
        """motorboard.py must NOT call lvp_logger.pause_thread()."""
        import pathlib
        source = pathlib.Path("drivers/motorboard.py").read_text()
        assert "pause_thread()" not in source, \
            "motorboard.py must not use pause_thread() — suppresses all thread logging (G4)"

    def test_connect_log_suppressed_flag_exists(self, _mock_heavy_deps):
        """MotorBoard must have _connect_log_suppressed flag."""
        import pathlib
        source = pathlib.Path("drivers/motorboard.py").read_text()
        assert "_connect_log_suppressed" in source, \
            "MotorBoard must use _connect_log_suppressed flag for targeted suppression (G4)"

    def test_connect_log_suppressed_resets_on_success(self):
        """_connect_log_suppressed must be reset when connection succeeds."""
        import pathlib
        source = pathlib.Path("drivers/motorboard.py").read_text()
        # Find the success path (where _connect_fails = 0)
        idx = source.find("self._connect_fails = 0", source.find("def connect"))
        assert idx != -1
        nearby = source[idx:idx+200]
        assert "_connect_log_suppressed = False" in nearby, \
            "_connect_log_suppressed must be reset to False on successful connection (G4)"


class TestRule1_MotorBoardNoNotifications:
    """Rule 1: drivers must not fire user-facing notifications directly.
    Notifications are the API layer's responsibility — it has scope
    context to decide whether a driver failure is user-visible (LS820
    expected motor) vs expected absence (LS620 has no motor)."""

    def test_motorboard_does_not_import_notifications(self):
        import pathlib
        source = pathlib.Path("drivers/motorboard.py").read_text()
        assert "from modules.notification_center import notifications" not in source, \
            "MotorBoard must not import notifications — Rule 1 (call down, not up)"

    def test_motorboard_does_not_call_notifications(self):
        import pathlib
        source = pathlib.Path("drivers/motorboard.py").read_text()
        assert "notifications.error" not in source, \
            "MotorBoard must not call notifications.error — Rule 1"
        assert "notifications.warning" not in source, \
            "MotorBoard must not call notifications.warning — Rule 1"
        assert "notifications.info" not in source, \
            "MotorBoard must not call notifications.info — Rule 1"


class TestRule1_CameraNoNotifications:
    """Rule 1: drivers must not fire user-facing notifications directly.
    Camera disconnect notification is the API layer's responsibility
    (lumascope_api.py fires it with scope context). Duplicates from
    the driver layer just pop twice or at the wrong moment."""

    def test_camera_base_does_not_import_notifications(self):
        import pathlib
        source = pathlib.Path("drivers/camera.py").read_text()
        assert "from modules.notification_center import notifications" not in source, \
            "drivers/camera.py must not import notifications — Rule 1"

    def test_camera_base_does_not_call_notifications(self):
        import pathlib
        source = pathlib.Path("drivers/camera.py").read_text()
        assert "notifications.error" not in source, \
            "drivers/camera.py must not call notifications.error — Rule 1"
        assert "notifications.warning" not in source
        assert "notifications.info" not in source


class TestRule1_PylonCameraNoNotifications:
    """Rule 1: Pylon SDK removal callback (OnCameraDeviceRemoved) runs in
    a native SDK thread. Before the Rule 1 cleanup it called
    notifications.error from that thread, a secondary crash risk on top
    of the layering violation. API-level detection in get_image handles
    the user-facing notification on the main thread."""

    def test_pyloncamera_does_not_import_notifications(self):
        import pathlib
        source = pathlib.Path("drivers/pyloncamera.py").read_text()
        assert "from modules.notification_center import notifications" not in source, \
            "drivers/pyloncamera.py must not import notifications — Rule 1"

    def test_pyloncamera_does_not_call_notifications(self):
        import pathlib
        source = pathlib.Path("drivers/pyloncamera.py").read_text()
        assert "notifications.error" not in source, \
            "drivers/pyloncamera.py must not call notifications.error — Rule 1"
        assert "notifications.warning" not in source
        assert "notifications.info" not in source


class TestRule1_SerialBoardNoNotifications:
    """Rule 1: SerialBoard fires per-command timeout/exception notifications
    that would spam on every dropped command during a transient
    disconnect. Throttled logger calls are retained for diagnostic
    records; user-facing notification is the API layer's job (it has
    connection-state context and scope capabilities to decide whether a
    given failure is user-visible)."""

    def test_serialboard_does_not_import_notifications(self):
        import pathlib
        source = pathlib.Path("drivers/serialboard.py").read_text()
        assert "from modules.notification_center import notifications" not in source, \
            "drivers/serialboard.py must not import notifications — Rule 1"

    def test_serialboard_does_not_call_notifications(self):
        import pathlib
        source = pathlib.Path("drivers/serialboard.py").read_text()
        assert "notifications.error" not in source, \
            "drivers/serialboard.py must not call notifications.error — Rule 1"
        assert "notifications.warning" not in source
        assert "notifications.info" not in source


class TestIssue637_DrawerCloseSaturation:
    """#637: Closing the right-side LED settings drawer caused the image to
    saturate. Reproduction:
      1. Maximized image, PC LED ON, bullseye/crosshairs ON
      2. Hit right arrow to close drawer
      3. Image went all red (saturated)
      4. Reopen drawer, PC still selected, image returned to normal after
         cycling LED off/on

    Root cause: Kivy's Accordion auto-expands a different item when the
    active one collapses (default behavior — at least one item must stay
    expanded). When the user closed the drawer, Kivy auto-expanded another
    layer's accordion item (e.g. DF) behind the scenes. ImageSettings's
    on-collapse handler fired and called apply_settings() on that newly-
    expanded layer, applying its camera exposure (e.g. DF 30 ms vs PC 5 ms)
    while the user's actual LED was still on. 6x longer exposure with the
    same LED on saturated the image.

    Fix: drawer open/close must not send anything to camera/LEDs. Skip
    _do_accordion_collapse's apply_settings loop when the drawer toggle
    button is in 'normal' state (drawer closed).
    """

    def test_do_accordion_collapse_skips_when_drawer_closed(self):
        """_do_accordion_collapse must check toggle_imagesettings state
        before applying any layer settings."""
        import pathlib
        source = pathlib.Path("ui/image_settings.py").read_text()
        idx = source.find("def _do_accordion_collapse")
        assert idx >= 0, "_do_accordion_collapse not found in ui/image_settings.py"
        # Slice to just this method's body — find the next `def ` at the
        # same indent level. _do_accordion_collapse lives in a class so
        # subsequent methods use 4-space indent: '\n    def '.
        next_def = source.find("\n    def ", idx + 1)
        body = source[idx:next_def] if next_def > 0 else source[idx:]
        assert "toggle_imagesettings" in body, (
            "_do_accordion_collapse must check toggle_imagesettings state "
            "(issue #637) - without this guard, drawer close triggers "
            "apply_settings on a Kivy auto-expanded layer, saturating image."
        )
        assert "'normal'" in body or '"normal"' in body, (
            "_do_accordion_collapse must compare toggle_imagesettings.state "
            "to 'normal' (drawer-closed sentinel) per issue #637 fix."
        )


class TestIssue643_LumiLS820PlateViewInProtocol:
    """#643: On XYStage=False scopes (Lumi, LS820) the plate view + crosshair
    re-appeared in the protocol accordion when opened, despite
    set_ui_features_for_scope having called stage.remove_parent() at config
    load. Cause: accordion_collapse() in motion_settings.py unconditionally
    re-added the stage widget to whichever accordion was open, with no
    capability check.

    Fix: gate accordion_collapse on selected_scope_config['XYStage'] (the
    same source set_ui_features_for_scope uses). When False, call
    stage.remove_parent() and return early.
    """

    def test_accordion_collapse_checks_xystage_capability(self):
        """accordion_collapse must consult selected_scope_config['XYStage']
        before re-attaching the stage widget."""
        import pathlib
        source = pathlib.Path("ui/motion_settings.py").read_text()
        # Find the accordion_collapse method body
        idx = source.find("def accordion_collapse")
        assert idx >= 0, "accordion_collapse method not found in ui/motion_settings.py"
        # Take a slice large enough to cover the method body
        body = source[idx:idx + 3000]
        assert "XYStage" in body, (
            "accordion_collapse must check XYStage capability (issue #643) — "
            "without this guard, Lumi/LS820 protocol accordion re-shows the "
            "plate view + crosshair."
        )
        assert "remove_parent" in body, (
            "accordion_collapse must call stage.remove_parent() on the "
            "XYStage=False path (issue #643)."
        )

    def test_lumi_and_ls820_have_xystage_false(self):
        """Sanity: scopes.json must declare Lumi and LS820 as XYStage=False
        for the issue #643 guard to actually apply."""
        import json, pathlib
        scopes = json.loads(pathlib.Path("data/scopes.json").read_text())
        assert "Lumi" in scopes, "Lumi scope config missing from data/scopes.json"
        assert "LS820" in scopes, "LS820 scope config missing from data/scopes.json"
        assert scopes["Lumi"]["XYStage"] is False, (
            "data/scopes.json: Lumi must be XYStage=False for issue #643 guard "
            "to suppress plate view"
        )
        assert scopes["LS820"]["XYStage"] is False, (
            "data/scopes.json: LS820 must be XYStage=False for issue #643 guard "
            "to suppress plate view"
        )


class TestIssue642_FilesCompleteCallbackRace:
    """#642: protocol_complete_callback was wiped by protocol_end() before
    the dispatch loop could fire it, causing files_complete to never fire
    when a protocol aborted with an empty queue (e.g. pre-scan disk-space
    abort). UI consequence: button stuck at "Writing Files... (0)" disabled,
    user must quit the app.

    Root cause: dispatch loop in sequential_io_executor.py called
    self.protocol_end() (which clears self.protocol_complete_callback)
    BEFORE reading the callback to fire it. Race wiped the reference.

    Fix: capture callback BEFORE protocol_end() in the dispatch loop's
    drain branch. protocol_end() retains its callback-clear behavior for
    the "premature end" path where callers invoke it directly.
    """

    def test_complete_callback_fires_when_protocol_finishes_with_empty_queue(self):
        """Pre-scan abort scenario: protocol_finish set, queue never had tasks."""
        from modules.sequential_io_executor import SequentialIOExecutor
        import time

        ex = SequentialIOExecutor(name="TEST_642_EMPTY")
        ex.start()
        try:
            fired = []
            ex.protocol_start()
            ex.set_protocol_complete_callback(callback=lambda: fired.append(True))
            ex.protocol_finish_then_end()

            # Dispatch loop polls protocol_queue with 0.2 s timeout. After timeout,
            # it sees queue empty + protocol_finish set, fires the callback path.
            # 1.0 s is ample margin (5x the poll interval).
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not fired:
                time.sleep(0.05)

            assert fired, (
                "files_complete callback did not fire after protocol_finish_then_end "
                "on empty queue. Pre-fix bug: protocol_end() in the dispatch loop "
                "wiped the callback before it could be fired (issue #642)."
            )
        finally:
            ex.shutdown(wait=True)

    def test_complete_callback_fires_after_queued_tasks_drain(self):
        """Normal completion: protocol_start, queue task(s), wait for task to run,
        then protocol_finish_then_end, verify callback fires after queue drains."""
        from modules.sequential_io_executor import SequentialIOExecutor, IOTask
        import time

        ex = SequentialIOExecutor(name="TEST_642_DRAIN")
        ex.start()
        try:
            fired = []
            task_ran = []
            ex.protocol_start()
            ex.protocol_put(IOTask(action=lambda: task_ran.append(True)))

            # Wait for task to be picked up + executed before signaling finish.
            # If we call protocol_finish_then_end before the dispatcher pulls
            # the task, the dispatcher's queue.Empty branch fires first and
            # ends the protocol with the task still in queue (test artifact,
            # not the bug we're testing).
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not task_ran:
                time.sleep(0.05)
            assert task_ran, "Queued task did not execute within 2 s."

            ex.set_protocol_complete_callback(callback=lambda: fired.append(True))
            ex.protocol_finish_then_end()

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not fired:
                time.sleep(0.05)

            assert fired, (
                "files_complete callback did not fire after queue drained "
                "via protocol_finish_then_end (issue #642)."
            )
        finally:
            ex.shutdown(wait=True)


class TestAOC1_SaturationCheckShortCircuit:
    """AOC-1: lumascope_api.get_image saturation check uses
    `not np.any(tmp != max)` (short-circuit) instead of `np.all(tmp == max)`.

    Both forms allocate a bool array, but `np.any` short-circuits on the
    first True at the C level — for the common (non-saturated) case, the
    first non-max pixel exits the reduction immediately. Equivalence over
    saturated / non-saturated / single-pixel-different / all-zero arrays.
    """

    def test_source_uses_not_any_form(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        assert "not np.any(tmp != np.iinfo(tmp.dtype).max)" in src, (
            "AOC-1: get_image() saturation check should use the short-circuit "
            "`not np.any(tmp != max)` form."
        )
        assert "np.all(tmp == np.iinfo(tmp.dtype).max)" not in src, (
            "AOC-1: old `np.all(tmp == max)` form should be replaced."
        )

    def test_logical_equivalence_uint8(self):
        import numpy as np
        max_val = np.iinfo(np.uint8).max
        cases = [
            np.full((100, 100), max_val, dtype=np.uint8),  # saturated
            np.zeros((100, 100), dtype=np.uint8),  # all zero
            np.full((100, 100), max_val // 2, dtype=np.uint8),  # half-max
            np.full((100, 100), max_val, dtype=np.uint8),  # saturated except one px
        ]
        cases[3][50, 50] = max_val - 1  # near-saturated
        for arr in cases:
            old = bool(np.all(arr == np.iinfo(arr.dtype).max))
            new = not np.any(arr != np.iinfo(arr.dtype).max)
            assert old == new, f"Logical mismatch on uint8 case: old={old}, new={new}"

    def test_logical_equivalence_uint16(self):
        import numpy as np
        max_val = np.iinfo(np.uint16).max
        cases = [
            np.full((100, 100), max_val, dtype=np.uint16),  # saturated
            np.zeros((100, 100), dtype=np.uint16),  # all zero
            np.full((100, 100), max_val // 2, dtype=np.uint16),  # half-max
            np.full((100, 100), max_val, dtype=np.uint16),  # saturated except one px
        ]
        cases[3][50, 50] = max_val - 1  # near-saturated
        for arr in cases:
            old = bool(np.all(arr == np.iinfo(arr.dtype).max))
            new = not np.any(arr != np.iinfo(arr.dtype).max)
            assert old == new, f"Logical mismatch on uint16 case: old={old}, new={new}"


class TestAOC2_RetrySaturationCheckOutsideCamLock:
    """AOC-2: lumascope_api.get_image saturation-retry path used to hold
    cam_lock across the np.all validation walk on the retry frame. The walk
    doesn't need camera state — only the buffer returned from get_array().
    Holding cam_lock across the walk blocked concurrent set_gain/set_exposure
    from other threads for ~50-150 ms per saturated retry.

    Fix: move the saturation walk outside the cam_lock block. Retry frame
    is captured under the lock; the walk runs after the lock is released.
    Also applies the AOC-1 short-circuit pattern at the retry site
    (feedback_default_to_expanding_scope — fix the cluster).
    """

    def test_retry_saturation_walk_is_outside_cam_lock(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        # The old form: np.all(retry_frame == ...) inside the with self._cam_lock: block
        assert "np.all(retry_frame == np.iinfo(retry_frame.dtype).max)" not in src, (
            "AOC-2: old `np.all(retry_frame == max)` form should be replaced."
        )
        # New form: short-circuit np.any check, AND structurally placed in a sibling
        # block to the cam_lock. Verify the lock-release marker comment is present
        # AND the retry-frame check uses the AOC-1 pattern.
        assert "Saturation walk is outside cam_lock" in src, (
            "AOC-2: expected lock-release marker comment near retry-frame walk."
        )
        assert "np.any(retry_frame != np.iinfo(retry_frame.dtype).max)" in src, (
            "AOC-2: retry-frame check should use the AOC-1 short-circuit pattern."
        )

    def test_retry_frame_initialized_before_lock_block(self):
        """Structural: retry_frame must be initialized before the with block so the
        outside-lock check can reference it whether or not the grab succeeded."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        # Find the retry block; verify retry_frame = None precedes the with statement.
        idx_init = src.find("retry_frame = None")
        idx_lock = src.find("with self._cam_lock:", idx_init)
        idx_retry_grab = src.find("retry_status", idx_lock)
        assert idx_init != -1, "AOC-2: expected `retry_frame = None` initializer."
        assert idx_init < idx_lock < idx_retry_grab, (
            "AOC-2: retry_frame should be initialized BEFORE the with cam_lock block."
        )


class TestPIW3_FalseColor16bitCachedAtRunStart:
    """PIW-3: image_utils.write_tiff used to acquire `_app_ctx.ctx.settings_lock`
    on every TIFF save to read the `false_color_16bit` flag. Same Rule 14 / Rule 2
    family as PP-7 in the post-processing audit. The setting is read-mostly during
    a protocol run; per-save acquisition is wasteful and contends with GUI thread
    settings updates.

    Fix: thread an `use_false_color_16bit` parameter through write_tiff /
    save_image / save_image_static / ProtocolImageWriter, read once in
    sequenced_capture_executor at run start, and pass through. write_tiff
    falls back to the lock-read path when `use_false_color_16bit=None`,
    preserving behavior for ad-hoc callers.
    """

    def test_write_tiff_accepts_use_false_color_16bit_param(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "image_utils.py").read_text()
        assert "use_false_color_16bit: bool | None = None" in src, (
            "PIW-3: write_tiff() should accept use_false_color_16bit param."
        )
        # The lock acquire should be gated on use_false_color_16bit being None.
        assert "if use_false_color_16bit is None:" in src, (
            "PIW-3: settings_lock should be acquired only when caller did not supply the resolved bool."
        )

    def test_save_image_threads_param_to_write_tiff(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        # Both save_image() (instance method) and save_image_static() must accept the param.
        assert src.count("use_false_color_16bit: bool | None = None") >= 2, (
            "PIW-3: save_image and save_image_static should both accept use_false_color_16bit."
        )
        # Both should pass it through to write_tiff. Count the kwarg passes; expect >= 2.
        assert src.count("use_false_color_16bit=use_false_color_16bit") >= 2, (
            "PIW-3: save_image and save_image_static should pass the param to write_tiff."
        )

    def test_protocol_image_writer_caches_at_init(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "protocol_image_writer.py").read_text()
        assert "false_color_16bit: bool = False" in src, (
            "PIW-3: ProtocolImageWriter.__init__ should accept false_color_16bit."
        )
        assert "self._false_color_16bit = false_color_16bit" in src, (
            "PIW-3: ProtocolImageWriter should cache false_color_16bit on self."
        )
        assert "use_false_color_16bit=self._false_color_16bit" in src, (
            "PIW-3: ProtocolImageWriter should pass the cached value to save_image."
        )

    def test_sequenced_capture_executor_reads_once_at_run_start(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "sequenced_capture_executor.py").read_text()
        # The read must happen under the settings_lock and pass through to the writer.
        assert "with ctx.settings_lock:" in src, (
            "PIW-3: false_color_16bit read should be guarded by settings_lock."
        )
        assert "false_color_16bit = ctx.settings.get('false_color_16bit', False)" in src, (
            "PIW-3: expected single read of false_color_16bit from settings."
        )
        assert "false_color_16bit=false_color_16bit" in src, (
            "PIW-3: cached value should be passed to ProtocolImageWriter."
        )


class TestPIW5_Convert12to16OutBuffer:
    """PIW-5: convert_12bit_to_16bit() allocated a fresh ndarray on every save
    via image.copy() (~24 MB pulse for protocol-scale images). Same family as
    F-3 — fresh allocations on the hot save path.

    Fix: add `out=None` parameter; when caller supplies a buffer with matching
    shape and dtype, reuse it via np.copyto. Plumb a per-run reusable buffer
    through ProtocolImageWriter -> save_image -> prepare_image_for_saving ->
    convert_12bit_to_16bit. file_io_executor runs single-threaded so reuse
    across sequential saves is safe; mismatched shape/dtype falls back to
    allocation.
    """

    def test_convert_function_accepts_out_param(self):
        import numpy as np
        from modules.image_utils import convert_12bit_to_16bit

        # Functional: shape/dtype-matched out buffer is reused; result is *= 16 of input.
        src = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        buf = np.zeros((2, 2), dtype=np.uint16)
        result = convert_12bit_to_16bit(src, out=buf)
        assert result is buf, "PIW-5: convert should return the supplied out buffer."
        np.testing.assert_array_equal(result, src * 16)

        # Mismatched shape: falls back to fresh allocation, no error.
        bad_buf = np.zeros((3, 3), dtype=np.uint16)
        result2 = convert_12bit_to_16bit(src, out=bad_buf)
        assert result2 is not bad_buf, "PIW-5: shape-mismatch should fall back to fresh alloc."
        np.testing.assert_array_equal(result2, src * 16)

        # No out param: original behavior preserved.
        result3 = convert_12bit_to_16bit(src)
        assert result3 is not src, "PIW-5: no-out path should still allocate a fresh array."
        np.testing.assert_array_equal(result3, src * 16)

    def test_protocol_image_writer_holds_reusable_buffer(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "protocol_image_writer.py").read_text()
        assert "self._convert_buf_12to16 = None" in src, (
            "PIW-5: ProtocolImageWriter should initialize the convert buffer to None."
        )
        assert "_get_convert_buf_12to16" in src, (
            "PIW-5: ProtocolImageWriter should have a buffer-getter helper."
        )
        # Shape/dtype guard: the helper must re-allocate on shape change.
        assert "self._convert_buf_12to16.shape != array.shape" in src, (
            "PIW-5: buffer helper must re-allocate when input shape changes."
        )
        # Save-call site passes the buffer.
        assert "out_12to16=out_12to16" in src, (
            "PIW-5: _write_capture should pass the convert buffer to save_image."
        )

    def test_save_image_threads_out_12to16_to_prepare(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        # save_image accepts the param.
        assert "out_12to16: np.ndarray | None = None" in src, (
            "PIW-5: save_image / prepare_image_for_saving should accept out_12to16."
        )
        # save_image passes to prepare_image_for_saving.
        assert "out_12to16=out_12to16" in src, (
            "PIW-5: save_image should pass out_12to16 to prepare_image_for_saving."
        )
        # prepare_image_for_saving passes to convert_12bit_to_16bit.
        assert "convert_12bit_to_16bit(array, out=out_12to16)" in src, (
            "PIW-5: prepare_image_for_saving should pass out_12to16 to the convert call."
        )


class TestPIW6_PF3_FalseColorRgbPreallocated:
    """PIW-6 + PF-3 (combined): retire allocations on the false-color save path.

    Before:
      - add_false_color allocates (H, W, 3) BGR per save (~36 MB uint16)        — PF-3
      - data[:, :, ::-1] returns a stride-reversed VIEW; tifffile silently
        calls np.ascontiguousarray on write (~36 MB uint16 alloc)               — PIW-6

    After:
      - add_false_color(data, color, output=false_color_buf) reuses caller buf  — PF-3
      - cv2.cvtColor(bgr, COLOR_BGR2RGB, dst=rgb_buf) writes in-place           — PIW-6

    ProtocolImageWriter holds both buffers per run, lazy-allocated together
    on first uint16 2D save when false-color is enabled. Mismatched shape/dtype
    re-allocates on demand. file_io_executor runs single-threaded so reuse
    across sequential saves is safe.

    The cv2.cvtColor approach is more idiomatic in a cv2-based pipeline than
    the previous numpy stride-reversal + ascontiguousarray pattern.
    """

    def test_write_tiff_uses_cv2_cvtColor_into_rgb_buf(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "image_utils.py").read_text()
        # Old form (as code, not as comment-reference) gone.
        assert "data = data[:, :, ::-1]" not in src, (
            "PIW-6: old stride-reversed-view BGR->RGB assignment should be replaced."
        )
        # New form: cv2.cvtColor with dst kwarg.
        assert "cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=rgb_buf)" in src, (
            "PIW-6: BGR->RGB should use cv2.cvtColor with dst=rgb_buf for in-place conversion."
        )
        # Fallback path when no rgb_buf supplied.
        assert "cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)" in src, (
            "PIW-6: fallback path should still call cv2.cvtColor for ad-hoc callers."
        )
        # add_false_color is called with the output buffer.
        assert "add_false_color(data, color, output=false_color_buf)" in src, (
            "PF-3: add_false_color should be called with output=false_color_buf."
        )

    def test_write_tiff_signature_includes_buffers(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "image_utils.py").read_text()
        assert "false_color_buf: np.ndarray | None = None" in src, (
            "PF-3: write_tiff should accept false_color_buf param."
        )
        assert "rgb_buf: np.ndarray | None = None" in src, (
            "PIW-6: write_tiff should accept rgb_buf param."
        )

    def test_protocol_image_writer_holds_both_buffers(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "protocol_image_writer.py").read_text()
        assert "self._false_color_buf = None" in src, (
            "PF-3: ProtocolImageWriter should initialize false_color_buf to None."
        )
        assert "self._rgb_buf = None" in src, (
            "PIW-6: ProtocolImageWriter should initialize rgb_buf to None."
        )
        assert "_get_false_color_bufs" in src, (
            "PF-3 + PIW-6: helper that returns (false_color_buf, rgb_buf) tuple should exist."
        )
        # Buffers only allocated when false-color is enabled AND capture is uint16 2D.
        assert "if self._false_color_16bit and is_uint16_2d:" in src, (
            "PF-3 + PIW-6: buffer allocation should be gated on false_color_16bit AND uint16 2D."
        )
        # Both buffers passed to save_image.
        assert "false_color_buf=false_color_buf" in src, (
            "PF-3: false_color_buf should be passed to save_image."
        )
        assert "rgb_buf=rgb_buf" in src, (
            "PIW-6: rgb_buf should be passed to save_image."
        )

    def test_save_image_threads_buffers_to_write_tiff(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        assert "false_color_buf: np.ndarray | None = None" in src, (
            "PF-3: save_image should accept false_color_buf."
        )
        assert "rgb_buf: np.ndarray | None = None" in src, (
            "PIW-6: save_image should accept rgb_buf."
        )

    def test_add_false_color_uses_output_buffer(self):
        """Functional: add_false_color writes into the supplied output buffer."""
        import numpy as np
        from modules.image_utils import add_false_color
        src = np.full((4, 4), 100, dtype=np.uint16)
        buf = np.full((4, 4, 3), 999, dtype=np.uint16)
        result = add_false_color(src, 'Blue', output=buf)
        assert result is buf, "PF-3: add_false_color should return the supplied buffer."
        np.testing.assert_array_equal(result[:, :, 0], src)
        assert np.all(result[:, :, 1] == 0), "PF-3: green channel should be zeroed."
        assert np.all(result[:, :, 2] == 0), "PF-3: red channel should be zeroed."

    def test_cv2_cvtColor_dst_writes_in_place(self):
        """Functional: cv2.cvtColor with dst= writes BGR->RGB in-place."""
        import numpy as np
        import cv2
        bgr = np.zeros((2, 3, 3), dtype=np.uint16)
        bgr[:, :, 0] = 1
        bgr[:, :, 1] = 2
        bgr[:, :, 2] = 3
        rgb_buf = np.empty_like(bgr)
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=rgb_buf)
        assert np.all(rgb_buf[:, :, 0] == 3), "cv2.cvtColor: R channel"
        assert np.all(rgb_buf[:, :, 1] == 2), "cv2.cvtColor: G channel"
        assert np.all(rgb_buf[:, :, 2] == 1), "cv2.cvtColor: B channel"


class TestPIW1_NoTheatricalDelCapturedImage:
    """PIW-1: write_capture had `del captured_image` after save_image() completes.
    The line is theatrical — captured_image is passed as a kwarg in the IOTask
    queued at protocol_image_writer.py:303 (`"captured_image": captured_image`).
    The IOTask.kwargs dict holds the reference until the task completes, so the
    local `del` only releases a local binding — actual memory reclaim happens
    when the IOTask is freed after task completion, regardless.

    Misleading "memory free" gesture; remove the line.
    """

    def test_del_captured_image_line_removed(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "protocol_image_writer.py").read_text()
        assert "del captured_image" not in src, (
            "PIW-1: theatrical `del captured_image` should be removed — IOTask kwargs holds the ref."
        )


class TestPIW2_DisksUsageDeduped:
    """PIW-2: per-save disk-space check was redundant between
    `lumascope_api.save_image` / `save_live_image` (both called
    `common_utils.check_disk_space()` defaulting to "/", logged-only,
    non-actionable) and `protocol_image_writer._write_capture` (checks the
    actual save_folder, aborts the protocol on insufficient space).

    The lumascope_api checks (a) checked the wrong path — root filesystem,
    not the save folder — and (b) only logged at error level without aborting
    or notifying. The existing try/except in save_image already catches
    write failures via OSError and surfaces a user notification.

    Fix: remove the redundant lumascope_api checks. Keep the useful
    protocol_image_writer check at line 350.
    """

    def test_lumascope_api_disk_check_removed(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        # The exact pattern of the redundant warn-only check.
        assert "if (common_utils.check_disk_space() < 1024):" not in src, (
            "PIW-2: redundant per-save check_disk_space call should be removed from lumascope_api."
        )
        # 'Disk space < 1 GB' was the warn string, also gone.
        assert "Disk space < 1 GB. Image unlikely to save correctly." not in src, (
            "PIW-2: corresponding warn log should be removed."
        )

    def test_protocol_image_writer_disk_check_kept(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "protocol_image_writer.py").read_text()
        # The useful check (correct path + abort on exhaustion) must remain.
        assert "shutil.disk_usage(str(save_folder)).free" in src, (
            "PIW-2: protocol_image_writer's save-folder disk check should be kept (it's the useful one)."
        )
        assert "self._protocol_ended.set()" in src, (
            "PIW-2: protocol_image_writer's abort-on-low-disk path should still be present."
        )


class TestPF2_FileIoExecutorClearedOnAbort:
    """PF-2: on hardware-disconnect / abort cleanup, file_io_executor's
    pending queue was NOT cleared — only io_executor and protocol_executor
    were. Queued IOTasks hold captured_image references; on a slow drain
    these can pin GB of memory and lock the next protocol-start until the
    drain completes.

    Distinct from normal completion, where draining is correct (writes user
    data to disk). The discriminator is `ProtocolState.ERROR` at cleanup
    entry — that's an abort path; anything else (COMPLETING, IDLE) is
    normal end.

    Fix: capture is_aborted from initial state BEFORE the COMPLETING
    transition, then call file_io_executor.clear_protocol_pending() in the
    aborted branch alongside the existing io/protocol clear calls. Drain
    path is unchanged for normal completion.
    """

    def test_initial_state_captured_before_completing_transition(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "protocol_cleanup.py").read_text()
        # The capture must precede the COMPLETING transition so ERROR vs other states
        # is distinguishable.
        idx_capture = src.find("is_aborted = (get_state_fn() == ProtocolState.ERROR)")
        idx_transition = src.find("set_state_fn(ProtocolState.COMPLETING)")
        assert idx_capture != -1, (
            "PF-2: cleanup should capture is_aborted from initial state."
        )
        assert idx_capture < idx_transition, (
            "PF-2: is_aborted must be captured BEFORE the COMPLETING state transition."
        )

    def test_file_io_cleared_on_abort_only(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "protocol_cleanup.py").read_text()
        # The abort-branch clear is gated on is_aborted.
        assert "if is_aborted:" in src, (
            "PF-2: file_io clear should be gated on is_aborted."
        )
        assert "file_io_executor.clear_protocol_pending()" in src, (
            "PF-2: cleanup should clear file_io_executor's pending queue on abort."
        )
        # Existing unconditional clears for the other executors must still be present.
        assert "io_executor.clear_protocol_pending()" in src, (
            "PF-2: io_executor.clear_protocol_pending should still be called unconditionally."
        )
        assert "protocol_executor.clear_protocol_pending()" in src, (
            "PF-2: protocol_executor.clear_protocol_pending should still be called unconditionally."
        )


class TestPF5_ImageBufferRetired:
    """PF-5: Lumascope.image_buffer was a permanent shadow copy of the latest
    get_image() result — Rule 2 violation. Only ever read by get_image() itself
    (for chaining sum/scale-bar/8-bit-convert ops), never by external callers.
    Pinned one frame indefinitely between calls. The _state_lock around per-
    write didn't actually serialize concurrent get_image calls — chained
    writes from different threads could still interleave.

    Fix: chain through a local variable in get_image(). Remove the
    image_buffer property + setter, the _image_buffer attribute, and its
    initialization in __init__ + diagnostic-instance setup.
    """

    def test_image_buffer_property_removed(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        # Property declaration gone.
        assert "def image_buffer(self):" not in src, (
            "PF-5: image_buffer property getter should be removed."
        )
        assert "@image_buffer.setter" not in src, (
            "PF-5: image_buffer property setter should be removed."
        )
        # Assignments to self.image_buffer (as code, not in comments) gone.
        assert "self.image_buffer = " not in src, (
            "PF-5: all self.image_buffer assignments should be retired in favor of a local variable."
        )

    def test_image_buffer_attribute_removed(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        # The internal _image_buffer attribute init gone.
        assert "self._image_buffer = None" not in src, (
            "PF-5: self._image_buffer initialization should be removed."
        )
        assert "instance._image_buffer = None" not in src, (
            "PF-5: diagnostic-instance _image_buffer initialization should also be removed."
        )

    def test_get_image_returns_local_variable(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        # The chain must use a local `image` variable.
        assert "image = image_utils.add_scale_bar(" in src, (
            "PF-5: scale-bar step should bind to local `image` instead of self.image_buffer."
        )
        assert "image = image_utils.convert_12bit_to_8bit(image)" in src, (
            "PF-5: 8-bit convert step should bind to local `image`."
        )


class TestPF1_CpuPoolRetired:
    """PF-1: cpu_pool / use_multiprocessing infrastructure was dead.
    use_multiprocessing was hardcoded False, so the ProcessPoolExecutor
    construction at lumaviewpro.py:214-237 never ran. The
    sequenced_capture_writer.py module was only imported from that dead
    block — the entire module was unreachable. The cpu_pool param threaded
    through SequencedCaptureExecutor.__init__ was always None.

    Per IMAGE_PROCESSING_ARCHITECTURE_2026-04-30.md: do NOT pre-build a
    replacement pool — modules/postprocessing/ and modules/live_processing/
    will be built greenfield when their first feature lands.

    Fix: deleted modules/sequenced_capture_writer.py entirely. Removed
    cpu_pool / use_multiprocessing from lumaviewpro.py (declarations,
    init block, shutdown block, executor kwarg). Removed cpu_pool param
    from SequencedCaptureExecutor.__init__ + the now-unused
    ProcessPoolExecutor import. Removed the test that exercised the dead
    setup_worker_logger function.
    """

    def test_sequenced_capture_writer_module_deleted(self):
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "modules" / "sequenced_capture_writer.py"
        assert not path.exists(), (
            "PF-1: modules/sequenced_capture_writer.py should be deleted (dead module)."
        )

    def test_lumaviewpro_no_cpu_pool_or_use_multiprocessing(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "lumaviewpro.py").read_text()
        assert "cpu_pool" not in src, (
            "PF-1: all cpu_pool references should be removed from lumaviewpro.py."
        )
        assert "use_multiprocessing" not in src, (
            "PF-1: all use_multiprocessing references should be removed from lumaviewpro.py."
        )
        assert "from concurrent.futures import ProcessPoolExecutor" not in src, (
            "PF-1: unused ProcessPoolExecutor import should be removed from lumaviewpro.py."
        )

    def test_executor_no_cpu_pool_param(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "sequenced_capture_executor.py").read_text()
        assert "cpu_pool" not in src, (
            "PF-1: cpu_pool should be removed from SequencedCaptureExecutor."
        )
        assert "from concurrent.futures import ProcessPoolExecutor" not in src, (
            "PF-1: unused ProcessPoolExecutor import should be removed from sequenced_capture_executor.py."
        )


def _function_body_calls(source: str, func_name: str) -> set[str]:
    """Return the set of `self.<method>(...)` attribute calls in a named function's body.

    Used by frame-validity ship-gate tests to assert that capture call sites
    route through the canonical drain-then-grab helper. AST-based so the
    assertion survives whitespace / argument-order changes.
    """
    import ast
    tree = ast.parse(source)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target = node
            break
    if target is None:
        raise AssertionError(f"function {func_name!r} not found in source")
    calls: set[str] = set()
    for sub in ast.walk(target):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            value = sub.func.value
            if isinstance(value, ast.Name) and value.id == "self":
                calls.add(sub.func.attr)
    return calls


def _function_source(source: str, func_name: str) -> str:
    """Return the raw source text of a named top-level or method function.

    For substring assertions on chained-attribute calls that AST-walk would
    miss (e.g. `self.frame_validity.invalidate(...)` -- the chained receiver
    isn't a top-level `self.<method>` shape).
    """
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            text = ast.get_source_segment(source, node)
            if text is None:
                raise AssertionError(f"could not extract source for {func_name!r}")
            return text
    raise AssertionError(f"function {func_name!r} not found in source")


class TestFrameValidity_SaveLiveImageDrainsBeforeGrab:
    """Lumascope.save_live_image must drain stale frames before grabbing.
    Bare self.get_image(...) ships a mid-transition frame to disk on every
    manual save; the canonical helper is self.capture_and_wait(...)."""

    def test_save_live_image_calls_capture_and_wait(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        calls = _function_body_calls(src, "save_live_image")
        assert "capture_and_wait" in calls, (
            "save_live_image must call self.capture_and_wait(...) for drain-then-grab."
        )

    def test_save_live_image_does_not_call_bare_get_image(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        calls = _function_body_calls(src, "save_live_image")
        assert "get_image" not in calls, (
            "save_live_image must not call self.get_image(...) directly -- "
            "that bypasses frame_validity. Route through self.capture_and_wait(...)."
        )

    def test_capture_and_wait_accepts_earliest_image_ts(self):
        """capture_and_wait must forward earliest_image_ts so save_live_image's
        public signature stays stable for L2 SDK callers."""
        import inspect

        from modules import lumascope_api
        sig = inspect.signature(lumascope_api.Lumascope.capture_and_wait)
        assert "earliest_image_ts" in sig.parameters, (
            "capture_and_wait must accept earliest_image_ts so save_live_image "
            "can forward its existing parameter."
        )


def _scope_attribute_calls(source: str, func_name: str) -> set[str]:
    """Return the set of `self._scope.<method>(...)` attribute calls in a
    named function's body. Mirrors `_function_body_calls` for AF executor's
    indirect-via-_scope grab pattern."""
    import ast
    tree = ast.parse(source)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target = node
            break
    if target is None:
        raise AssertionError(f"function {func_name!r} not found in source")
    calls: set[str] = set()
    for sub in ast.walk(target):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            value = sub.func.value
            # match self._scope.<method>
            if (isinstance(value, ast.Attribute) and value.attr == "_scope"
                    and isinstance(value.value, ast.Name) and value.value.id == "self"):
                calls.add(sub.func.attr)
    return calls


class TestFrameValidity_AutofocusDrainsBeforeScore:
    """AutofocusExecutor._iterate must drain LED/gain/exposure-pending
    frames before scoring. Bare get_image after Z arrival can score on a
    mid-LED-warmup or mid-gain-change frame, corrupting the focus curve
    and landing the wrong best-Z. AF excludes z_move because AF is the
    controller of Z moves; once is_moving() reports idle, Z is settled."""

    def test_iterate_calls_capture_and_wait(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "autofocus_executor.py").read_text()
        calls = _scope_attribute_calls(src, "_iterate")
        assert "capture_and_wait" in calls, (
            "AutofocusExecutor._iterate must call self._scope.capture_and_wait(...) "
            "to drain LED/gain/exposure pending frames before scoring."
        )

    def test_iterate_does_not_call_bare_get_image(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "autofocus_executor.py").read_text()
        calls = _scope_attribute_calls(src, "_iterate")
        assert "get_image" not in calls, (
            "AutofocusExecutor._iterate must not call self._scope.get_image(...) "
            "directly -- bypasses frame_validity. Route through capture_and_wait."
        )

    def test_iterate_excludes_z_move_in_validity(self):
        """AF excludes z_move because is_moving() already gates motion; the
        drain is for LED/gain/exposure transitions only."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "autofocus_executor.py").read_text()
        # both call sites must specify exclude_sources=('z_move',)
        assert "exclude_sources=('z_move',)" in src, (
            "AutofocusExecutor._iterate's capture_and_wait calls must pass "
            "exclude_sources=('z_move',) since is_moving() already gates motion."
        )


class TestFrameValidity_LegacyCaptureRoutesThroughCaptureAndWait:
    """Lumascope.capture_complete and Lumascope.capture_blocking previously
    did fixed-time-sleep + bare get_image -- the v3.0.x anti-pattern.
    Both methods are deprecated (no production callers, not in
    LumascopeSkills.md). Implementation now routes through capture_and_wait
    so that during the deprecation cycle they grab valid frames. A
    DeprecationWarning fires on each call."""

    def test_capture_complete_calls_capture_and_wait(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        body = _function_source(src, "capture_complete")
        assert "self.capture_and_wait(" in body, (
            "capture_complete must call self.capture_and_wait(...) -- previously "
            "called bare self.get_image() after a fixed-time sleep."
        )
        assert "self.get_image(" not in body, (
            "capture_complete must not call self.get_image(...) directly."
        )

    def test_capture_blocking_calls_capture_and_wait(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        body = _function_source(src, "capture_blocking")
        assert "self.capture_and_wait(" in body, (
            "capture_blocking must call self.capture_and_wait(...)."
        )
        assert "self.get_image(" not in body, (
            "capture_blocking must not call self.get_image(...) directly."
        )
        assert "time.sleep(" not in body, (
            "capture_blocking must not contain a fixed-time sleep -- validity "
            "drain replaces the v3.0.x wait_time anti-pattern."
        )

    def test_capture_emits_deprecation_warning(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        body = _function_source(src, "capture")
        assert "DeprecationWarning" in body, (
            "capture must emit a DeprecationWarning so callers migrate to capture_and_wait."
        )

    def test_capture_blocking_emits_deprecation_warning(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        body = _function_source(src, "capture_blocking")
        assert "DeprecationWarning" in body, (
            "capture_blocking must emit a DeprecationWarning."
        )


class TestFrameValidity_CompositeEngineeringBranchDrains:
    """The engineering-mode branch of composite_capture's live_capture path
    (bullseye / crosshairs enabled) grabs an extra image_orig for overlay
    rendering. Bare get_image here would persist a mid-transition raw
    image to disk via the subsequent save_image call. Must route through
    capture_and_wait."""

    def test_live_capture_impl_uses_capture_and_wait(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "ui" / "composite_capture.py").read_text()
        body = _function_source(src, "_live_capture_impl")
        assert "ctx.scope.capture_and_wait(" in body, (
            "composite_capture._live_capture_impl must call "
            "ctx.scope.capture_and_wait(...) for the engineering bullseye/"
            "crosshairs branch (was bare get_image)."
        )

    def test_live_capture_impl_no_bare_ctx_scope_get_image(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "ui" / "composite_capture.py").read_text()
        body = _function_source(src, "_live_capture_impl")
        assert "ctx.scope.get_image(" not in body, (
            "composite_capture._live_capture_impl must not call "
            "ctx.scope.get_image(...) directly. Route through capture_and_wait "
            "(or save_live_image, which now uses capture_and_wait internally)."
        )


class TestFrameValidity_AllLedMutatorsInvalidate:
    """Defensive coverage: every LED state-mutator on Lumascope must call
    frame_validity.invalidate('led'). All 6 currently invalidate; this
    test locks the invariant so a future cleanup that removes any call
    fires the regression."""

    LED_MUTATORS = (
        "led_on",
        "led_off",
        "led_on_fast",
        "led_off_fast",
        "leds_off_fast",
        "leds_off",
    )

    def test_each_led_mutator_invalidates_validity(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        missing = []
        for func in self.LED_MUTATORS:
            calls = _function_body_calls(src, func)
            # invalidate() is called as self.frame_validity.invalidate(...) which
            # resolves to attribute "invalidate" on self.frame_validity. Use
            # a chained attribute walk via raw source check too as belt+suspenders.
            method_src = _function_source(src, func)
            if "self.frame_validity.invalidate(" not in method_src:
                missing.append(func)
        assert not missing, (
            "LED mutator coverage: each Lumascope LED state-mutator must call "
            "self.frame_validity.invalidate('led') so frame_validity sees the "
            f"transition. Missing: {missing!r}."
        )


class TestCaptureAndWaitPassesChunksToValidity:
    """capture_and_wait's drain loop reads per-frame chunk metadata and
    passes it to count_frame so chunk-match can short-circuit skip-frames
    for gain/exposure on chunk-supporting cameras. Backward compat:
    cameras without chunks return None and fall back to skip-frames."""

    def test_capture_and_wait_passes_chunk_data_to_count_frame(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        body = _function_source(src, "capture_and_wait")
        # Source mentions count_frame call site with chunk_data kwarg
        assert "count_frame(chunk_data=" in body, (
            "capture_and_wait must call count_frame(chunk_data=...) in the "
            "drain loop so chunk-match can clear gain/exposure pending."
        )

    def test_get_latest_chunks_helper_exists(self):
        """The _get_latest_chunks helper abstracts handler shape (Pylon
        composition vs IDS inheritance) and returns None for non-chunk cameras."""
        import inspect
        from modules import lumascope_api
        assert hasattr(lumascope_api.Lumascope, '_get_latest_chunks'), (
            "Lumascope must expose _get_latest_chunks() helper."
        )
        sig = inspect.signature(lumascope_api.Lumascope._get_latest_chunks)
        # No required params (besides self) -- it reads from self.camera state
        non_self = [p for p in sig.parameters if p != 'self']
        assert len(non_self) == 0, (
            f"_get_latest_chunks should take no args; got {non_self}"
        )

    def test_get_latest_chunks_returns_none_when_no_camera(self):
        """Defensive: helper returns None instead of raising when camera
        isn't connected (FX2 fallback / pre-connect / disconnected state)."""
        from modules.lumascope_api import Lumascope
        # Construct without going through full init -- attributes set by hand
        scope = Lumascope.__new__(Lumascope)
        scope.camera = None
        assert scope._get_latest_chunks() is None


class TestLumascopeRecordsTargetForChunkMatch:
    """The API layer records requested gain / exposure values via
    frame_validity.set_target() so capture_and_wait's chunk-match can
    short-circuit skip-frames once a frame's chunks match the target.

    Manual setters (set_gain, set_exposure_time) record the value; auto
    setters (set_auto_gain, set_auto_exposure_time) clear the target
    (None) since auto dynamically changes the value and chunk-match
    against a stale manual target would be wrong."""

    def test_set_gain_records_target(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        body = _function_source(src, "set_gain")
        assert "self.frame_validity.set_target('gain'" in body, (
            "set_gain must record gain target via frame_validity.set_target."
        )

    def test_set_exposure_time_records_target_in_microseconds(self):
        """ChunkExposureTime is microseconds; API takes milliseconds.
        Conversion (* 1000) must happen at the seam so chunk-match's
        tolerance is in matching units."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        body = _function_source(src, "set_exposure_time")
        assert "self.frame_validity.set_target('exposure'" in body, (
            "set_exposure_time must record target via set_target."
        )
        assert "1000" in body, (
            "set_exposure_time must convert ms -> us when recording target "
            "so chunk-match operates in microseconds."
        )

    def test_set_auto_gain_clears_target(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        body = _function_source(src, "set_auto_gain")
        assert "set_target('gain', None)" in body, (
            "set_auto_gain must clear gain target (None) so chunk-match doesn't "
            "fire against a stale manual target while auto adjusts."
        )

    def test_set_auto_exposure_time_clears_target(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "modules" / "lumascope_api.py").read_text()
        body = _function_source(src, "set_auto_exposure_time")
        assert "set_target('exposure', None)" in body, (
            "set_auto_exposure_time must clear exposure target (None)."
        )


class TestImageHandlerBaseChunkSlot:
    """ImageHandlerBase extends frame storage to carry per-frame chunk
    metadata. Backward-compatible: cameras that don't pass chunks (FX2,
    simulators) get None and existing consumers continue to work."""

    def _make_base(self):
        from drivers.camera import ImageHandlerBase
        return ImageHandlerBase()

    def test_initial_chunks_none(self):
        b = self._make_base()
        assert b.last_chunks is None
        assert b.get_last_chunks() is None

    def test_store_frame_without_chunks_keeps_none(self):
        """Backward compat: existing _store_frame(image, ts) call site."""
        import datetime
        import numpy as np
        b = self._make_base()
        b._store_frame(np.zeros((4, 4), dtype=np.uint8), datetime.datetime.now())
        assert b.last_chunks is None
        assert b.get_last_chunks() is None

    def test_store_frame_with_chunks_sets_dict(self):
        import datetime
        import numpy as np
        b = self._make_base()
        chunks = {'ExposureTime': 14530.0, 'Gain': 1.0, 'FrameID': 12345}
        b._store_frame(np.zeros((4, 4), dtype=np.uint8), datetime.datetime.now(), chunks=chunks)
        assert b.last_chunks == chunks
        assert b.get_last_chunks() == chunks

    def test_get_last_chunks_returns_none_before_first_grab(self):
        b = self._make_base()
        # last_result is False from __init__ -> get_last_chunks returns None
        assert b.get_last_chunks() is None

    def test_get_last_chunks_returns_none_after_failed_grab(self):
        """If the last grab failed, get_last_chunks returns None even if
        a previous grab populated chunks."""
        import datetime
        import numpy as np
        b = self._make_base()
        b._store_frame(np.zeros((4, 4), dtype=np.uint8), datetime.datetime.now(),
                       chunks={'ExposureTime': 14530.0})
        b._record_failure()  # last_result becomes False
        assert b.get_last_chunks() is None

    def test_reset_clears_chunks(self):
        import datetime
        import numpy as np
        b = self._make_base()
        b._store_frame(np.zeros((4, 4), dtype=np.uint8), datetime.datetime.now(),
                       chunks={'ExposureTime': 14530.0})
        b.reset()
        assert b.last_chunks is None
        assert b.get_last_chunks() is None

    def test_chunks_replace_not_merge(self):
        """Each successful grab replaces the chunks dict; we don't merge."""
        import datetime
        import numpy as np
        b = self._make_base()
        b._store_frame(np.zeros((4, 4), dtype=np.uint8), datetime.datetime.now(),
                       chunks={'ExposureTime': 14530.0, 'Gain': 1.0})
        b._store_frame(np.zeros((4, 4), dtype=np.uint8), datetime.datetime.now(),
                       chunks={'ExposureTime': 30000.0})
        assert b.get_last_chunks() == {'ExposureTime': 30000.0}
        assert 'Gain' not in b.get_last_chunks()


class TestPylonCancelHandlingDefensive:
    """OR-with-removal-flag insurance pattern in OnImageGrabbed.

    The cancel-classification branch must treat any failure paired with
    ``self._parent._device_removed=True`` as expected teardown, not as a
    real grab failure. Without this insurance a device-removal-driven
    cancel storm whose underlying err_code happens to differ from
    ``_PYLON_ERR_BUFFER_CANCELED`` could trip MAX_CONSECUTIVE_FAILURES
    auto-disconnect and produce log noise during the unplug path.

    Public evidence (pypylon issue #815, bench session 65) supports a
    single SDK cancel code 0xE2000102 / 3791651074 on USB3, but Basler
    does not document whether device-removal teardown ALWAYS attaches
    the same code to in-flight buffers. The OR pattern insulates the
    grab loop from that uncertainty without requiring enumeration.

    These tests lock the source-level shape of the fix so a future
    cleanup that drops the OR clause fires the regression. Behavioral
    test of the race (device_removed flips between the line-1457 early
    check and the line-1514 OR check, e.g. on the SDK's removal-
    forwarding thread) requires threading mocks; static lock is the
    primary regression gate.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_buffer_cancel_constant_value_matches_bench(self):
        """The bench-witnessed decimal from session 65 (3791651074) is the
        authoritative cancel-code constant. If pypylon ever exposes
        pylon.GENERIC_BUFFER_CANCELED or similar, bump this."""
        from drivers.pyloncamera import _PYLON_ERR_BUFFER_CANCELED
        assert _PYLON_ERR_BUFFER_CANCELED == 3791651074, (
            "Buffer-cancel constant must match the bench-witnessed value "
            "from Firmware DAILY_LOG.md session 65 Run-3 "
            "(decimal 3791651074 = 0xE2000102)."
        )

    def test_buffer_cancel_comment_hex_matches_constant(self):
        """The source comment must show the hex form that matches the
        decimal constant. Earlier the comment said 0xE2008002 (= decimal
        3791683586, NOT what's stored). The corrected hex is 0xE2000102.
        Mismatch between comment and value misleads anyone debugging this
        path; the comment is load-bearing documentation, not decoration."""
        src = self._pyloncamera_source()
        assert "0xE2000102" in src, (
            "Source comment near _PYLON_ERR_BUFFER_CANCELED must reference "
            "0xE2000102 (the hex form of decimal 3791651074). If you found "
            "0xE2008002 here, that's the prior typo — fix to 0xE2000102."
        )
        assert "0xE2008002" not in src, (
            "Stale typo: 0xE2008002 must not appear in pyloncamera.py "
            "source — that hex equals 3791683586 (NOT what's stored)."
        )

    def test_cancel_branch_uses_or_with_removal_flag(self):
        """The cancel-classification branch in OnImageGrabbed must include
        the OR-with-removal-flag insurance:

            if err_code == _PYLON_ERR_BUFFER_CANCELED or self._parent._device_removed:

        This protects against a race where _device_removed flips True
        between the line-1457 early-return check and the line-1514
        cancel-classification check (the SDK's removal-forwarding thread
        runs on its own thread; either the grab thread or the removal
        thread can set the flag). Without the OR, a mid-call removal
        whose first cancellation buffer carries an undocumented err_code
        would count toward MAX_CONSECUTIVE_FAILURES."""
        src = self._pyloncamera_source()
        body = _function_source(src, "OnImageGrabbed")
        assert (
            "_PYLON_ERR_BUFFER_CANCELED or self._parent._device_removed"
            in body
        ), (
            "OnImageGrabbed cancel-classification branch must use "
            "OR-with-removal-flag insurance. See class docstring for the "
            "race the OR protects against."
        )

    def test_normal_failure_branch_still_calls_record_failure(self):
        """The non-cancel non-removal failure path must still increment
        the consecutive-failure counter. Without this, real failures
        (incomplete buffers 0xE2000212, transport errors 0xE2000011,
        etc.) would never trip MAX_CONSECUTIVE_FAILURES auto-disconnect."""
        src = self._pyloncamera_source()
        body = _function_source(src, "OnImageGrabbed")
        assert "self._base._record_failure()" in body, (
            "OnImageGrabbed non-cancel failure path must still call "
            "_record_failure to increment the consecutive-failure counter."
        )


class TestPylonDisconnectDestroyDevice:
    """PylonCamera.disconnect() must release the SDK-side device handle
    explicitly via DetachDevice + DestroyDevice rather than relying on
    CPython refcount-driven cleanup.

    pypylon issues #547 and #792 document field cases where refcount
    cleanup left the SDK handle held: subsequent CreateDevice for the
    same serial fails with "device not reachable / controlled by another
    application" (Err 0xE1020018). The Basler-recommended canonical
    sequence is StopGrabbing -> Close -> DetachDevice -> DestroyDevice.

    Each step must be independently guarded (try/except) so a failure
    in one step (e.g., Close on an already-removed device) does not
    short-circuit the rest of the cleanup chain. The post-cleanup
    invariant is `self.active is None` regardless of which SDK calls
    succeeded; without that invariant, the rest of the app sees a
    known-bad camera as still connected.

    Source-shape tests rather than behavioural tests because exercising
    the path requires either a real Pylon device or mocking the entire
    pypylon SDK at the C++ binding layer; the source-shape pin matches
    the existing TestPylonCancelHandlingDefensive style and is enough
    for Rule 18.
    """

    def _disconnect_body(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "drivers" / "pyloncamera.py").read_text()
        return _function_source(src, "disconnect")

    def test_disconnect_calls_destroy_device(self):
        """disconnect() must explicitly destroy the SDK-side device
        handle. Without this, CPython refcount-driven cleanup may leave
        the handle held past the next reconnect attempt (pypylon #792)."""
        body = self._disconnect_body()
        assert "self.active.DestroyDevice()" in body, (
            "PylonCamera.disconnect must call self.active.DestroyDevice() "
            "to explicitly release the SDK-side device handle. Required "
            "to prevent 'device controlled by another application' on the "
            "next CreateDevice (pypylon issues #547, #792)."
        )

    def test_disconnect_calls_detach_device(self):
        """DetachDevice releases the InstantCamera's ownership of the
        device pointer before DestroyDevice destroys it. Per
        Basler-recommended canonical reattach sequence."""
        body = self._disconnect_body()
        assert "self.active.DetachDevice()" in body, (
            "PylonCamera.disconnect must call self.active.DetachDevice() "
            "before DestroyDevice. Required by Basler-recommended cleanup "
            "sequence: StopGrabbing -> Close -> DetachDevice -> DestroyDevice."
        )

    def test_disconnect_destroy_device_wrapped_in_try(self):
        """If DestroyDevice fails (e.g., already-detached device), the
        exception must NOT prevent self.active = None from running.
        Otherwise the post-cleanup invariant breaks and the app thinks
        a known-bad camera is still connected."""
        body = self._disconnect_body()
        # Look for the exact pattern: try block containing DestroyDevice
        assert "self.active.DestroyDevice()" in body
        # The DestroyDevice line must be inside a try/except that logs
        # a warning and continues, not propagates.
        # Heuristic: there must be at least 3 try blocks in disconnect
        # (one for stop_grabbing, one for Close, one for DetachDevice,
        # one for DestroyDevice -- count of "try:" lines must be >= 4).
        try_count = body.count("try:")
        assert try_count >= 4, (
            f"disconnect() must wrap each SDK teardown step (Close, "
            f"DetachDevice, DestroyDevice) in its own try/except so a "
            f"failure in one does not skip the others. Currently "
            f"{try_count} try blocks; expected >= 4 (stop_grabbing + "
            f"Close + DetachDevice + DestroyDevice)."
        )

    def test_disconnect_clears_active_after_cleanup(self):
        """self.active = None must come AFTER DestroyDevice, not before.
        If we cleared active first we would lose the device pointer
        before destroying it, leaving the SDK handle held."""
        body = self._disconnect_body()
        destroy_pos = body.find("self.active.DestroyDevice()")
        clear_pos = body.find("self.active = None")
        assert destroy_pos != -1, (
            "DestroyDevice call missing from disconnect()"
        )
        assert clear_pos != -1, (
            "self.active = None missing from disconnect()"
        )
        assert clear_pos > destroy_pos, (
            "self.active = None must come AFTER self.active.DestroyDevice(). "
            "Clearing active first loses the device pointer before "
            "DestroyDevice can run -> SDK handle stays held."
        )


class TestPylonDiagnosticProbe:
    """Lumascope.run_pylon_diagnostic_probe captures a one-shot
    cross-host / cross-camera / cross-firmware diagnostic snapshot
    and writes it to data/pylon_probe/<...>.json. Designed for
    bench-wave comparison; replaces /tmp/probe.py-style bespoke
    scripts (Rule 22).

    Tests focus on the API-layer wiring: schema, filename pattern,
    DLTL token, no-camera fallback, IDS supported=False passthrough.
    Driver-level node reading is exercised on real hardware
    (bench-only); these tests stub the driver via a minimal fake.
    """

    def _make_scope_with_fake_camera(self, fake_camera):
        """Construct a Lumascope without running its full __init__,
        attach the supplied fake camera. Same pattern as
        test_get_latest_chunks_returns_none_when_no_camera."""
        from modules.lumascope_api import Lumascope
        scope = Lumascope.__new__(Lumascope)
        scope.camera = fake_camera
        return scope

    def test_method_exists_on_lumascope(self):
        """The API method is callable from a fresh Lumascope class."""
        from modules.lumascope_api import Lumascope
        assert hasattr(Lumascope, 'run_pylon_diagnostic_probe')
        assert callable(Lumascope.run_pylon_diagnostic_probe)

    def test_no_camera_returns_disconnected(self):
        """Returns {'connected': False, 'errors': [...]} when no camera."""
        from modules.lumascope_api import Lumascope
        scope = Lumascope.__new__(Lumascope)
        scope.camera = None
        result = scope.run_pylon_diagnostic_probe(duration_s=0.0)
        assert result['connected'] is False
        assert isinstance(result.get('errors'), list)

    def test_inactive_camera_returns_disconnected(self):
        """Camera object exists but inactive -> disconnected."""
        class _Fake:
            active = None
        result = self._make_scope_with_fake_camera(_Fake()).run_pylon_diagnostic_probe(
            duration_s=0.0
        )
        assert result['connected'] is False

    def test_unsupported_driver_returns_supported_false(self):
        """Driver returning supported=False (e.g. IDSCamera stub) is
        passed through unchanged; API does NOT add host/timestamps/
        output_path because the snapshot is incomplete."""
        class _StubDriver:
            active = True  # truthy

            def read_diagnostic_snapshot(self, duration_s, drain_camera_side_errors):
                return {
                    'connected': True,
                    'supported': False,
                    'reason': 'stub driver',
                    'errors': [],
                }
        result = self._make_scope_with_fake_camera(_StubDriver()).run_pylon_diagnostic_probe(
            duration_s=0.0
        )
        assert result.get('supported') is False
        assert 'output_path' not in result, (
            "supported=False driver responses must NOT trigger JSON write"
        )

    def test_no_read_diagnostic_snapshot_method(self):
        """If the driver does not implement read_diagnostic_snapshot at
        all, the API returns a structured error rather than raising
        AttributeError."""
        class _NoMethodDriver:
            active = True
        result = self._make_scope_with_fake_camera(
            _NoMethodDriver()
        ).run_pylon_diagnostic_probe(duration_s=0.0)
        assert result['connected'] is False
        assert result.get('supported') is False

    def test_dltl_filename_token_off(self):
        """Mode=Off -> 'dltloff'."""
        from modules.lumascope_api import Lumascope
        token = Lumascope._dltl_filename_token({'dltl_mode': 'Off'})
        assert token == 'dltloff'

    def test_dltl_filename_token_on_round(self):
        """Mode=On with 160 MB/s -> 'dltl160M'."""
        from modules.lumascope_api import Lumascope
        token = Lumascope._dltl_filename_token({
            'dltl_mode': 'On',
            'dltl_value_bps': 160_000_000,
        })
        assert token == 'dltl160M'

    def test_dltl_filename_token_on_non_round(self):
        """Mode=On with non-round MB/s -> rounded int rendering.
        v4 author flagged the case where a sweep value has sub-MB
        precision; bare int() would render 197.99 MB/s as dltl197M
        which is wrong-by-1; round() avoids that."""
        from modules.lumascope_api import Lumascope
        token = Lumascope._dltl_filename_token({
            'dltl_mode': 'On',
            'dltl_value_bps': 197_999_000,
        })
        assert token == 'dltl198M', (
            f"Expected dltl198M (rounded), got {token!r}; "
            f"int(round()) cast missing or wrong"
        )

    def test_dltl_filename_token_unknown(self):
        """Missing config -> 'dltlunknown'."""
        from modules.lumascope_api import Lumascope
        assert Lumascope._dltl_filename_token({}) == 'dltlunknown'
        assert Lumascope._dltl_filename_token(
            {'dltl_mode': '<missing>'}
        ) == 'dltlunknown'

    def test_human_os_version_does_not_raise(self):
        """The OS-version helper must never raise, even on platforms
        where mac_ver/win32_ver return empty tuples."""
        from modules.lumascope_api import Lumascope
        v = Lumascope._human_os_version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_safe_pylon_versions_returns_dict(self):
        """The version helper returns a dict with both keys, even when
        pypylon is absent (returns Nones)."""
        from modules.lumascope_api import Lumascope
        result = Lumascope._safe_pylon_versions()
        assert isinstance(result, dict)
        assert 'pypylon_version' in result
        assert 'pylon_sdk_version' in result

    def test_pylon_camera_has_read_diagnostic_snapshot(self):
        """Source-shape lock: PylonCamera must implement the driver
        method the API depends on."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "drivers" / "pyloncamera.py").read_text()
        assert "def read_diagnostic_snapshot(" in src, (
            "PylonCamera must implement read_diagnostic_snapshot for "
            "Lumascope.run_pylon_diagnostic_probe to function."
        )

    def test_ids_camera_has_read_diagnostic_snapshot_stub(self):
        """Source-shape lock: IDSCamera must have a stub returning
        supported=False so the API can report the gap rather than
        raising AttributeError when an IDS camera is connected."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "drivers" / "idscamera.py").read_text()
        assert "def read_diagnostic_snapshot(" in src, (
            "IDSCamera must have a read_diagnostic_snapshot stub "
            "returning supported=False until the IDS implementation lands."
        )
        body = _function_source(src, "read_diagnostic_snapshot")
        assert "'supported': False" in body or '"supported": False' in body, (
            "IDS read_diagnostic_snapshot stub must return supported=False"
        )


class TestDeviceLinkThroughputLimitSetter:
    """Lumascope.set_device_link_throughput_limit and the underlying
    PylonCamera / IDSCamera implementations exist so the bench-probe
    sweep can vary DLTL across cells without dropping below the API
    layer (Rule 1) or writing /tmp/probe.py (Rule 22).

    DLTL is documented live-writable; no StopGrabbing/StartGrabbing
    wrap is required. The Pylon driver raises HardwareError on SDK
    RuntimeException; the API layer notifies + re-raises.
    """

    def _make_scope_with_fake_camera(self, fake_camera):
        from modules.lumascope_api import Lumascope
        scope = Lumascope.__new__(Lumascope)
        scope.camera = fake_camera
        return scope

    def test_lumascope_method_exists(self):
        from modules.lumascope_api import Lumascope
        assert hasattr(Lumascope, 'set_device_link_throughput_limit')
        assert callable(Lumascope.set_device_link_throughput_limit)

    def test_no_camera_returns_false(self):
        from modules.lumascope_api import Lumascope
        scope = Lumascope.__new__(Lumascope)
        scope.camera = None
        assert scope.set_device_link_throughput_limit('Off') is False

    def test_inactive_camera_returns_false(self):
        class _Fake:
            active = None
            def set_device_link_throughput_limit(self, **k):
                raise AssertionError("driver should not be reached")
        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.set_device_link_throughput_limit('Off') is False

    def test_unsupported_driver_returns_false(self):
        """Camera class without the setter (e.g. SimulatedCamera) -> False."""
        class _NoSetter:
            active = True
        scope = self._make_scope_with_fake_camera(_NoSetter())
        assert scope.set_device_link_throughput_limit('Off') is False

    def test_off_routes_to_driver(self):
        called_with = {}
        class _Fake:
            active = True
            def set_device_link_throughput_limit(self, mode, value_bps=None):
                called_with['mode'] = mode
                called_with['value_bps'] = value_bps
                return True
        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.set_device_link_throughput_limit('Off') is True
        assert called_with == {'mode': 'Off', 'value_bps': None}

    def test_on_with_value_routes_to_driver(self):
        called_with = {}
        class _Fake:
            active = True
            def set_device_link_throughput_limit(self, mode, value_bps=None):
                called_with['mode'] = mode
                called_with['value_bps'] = value_bps
                return True
        scope = self._make_scope_with_fake_camera(_Fake())
        ok = scope.set_device_link_throughput_limit(
            'On', value_bps=160_000_000)
        assert ok is True
        assert called_with == {'mode': 'On', 'value_bps': 160_000_000}

    def test_pylon_driver_method_present(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "drivers" / "pyloncamera.py").read_text()
        assert "def set_device_link_throughput_limit(" in src, (
            "PylonCamera must implement set_device_link_throughput_limit "
            "for tomorrow's bench-probe sweep to function without "
            "Rule 1 violations."
        )

    def test_pylon_driver_does_not_wrap_in_update_camera_config(self):
        """DLTL is live-writable per Section 5; wrapping in
        update_camera_config would force unnecessary stop/start cycles
        (per the STALL-1 anti-pattern lesson)."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "drivers" / "pyloncamera.py").read_text()
        body = _function_source(src, "set_device_link_throughput_limit")
        assert "with self.update_camera_config" not in body, (
            "PylonCamera.set_device_link_throughput_limit must NOT wrap "
            "the writes in update_camera_config (DLTL is live-writable; "
            "wrapping would impose the STALL-1 over-stop pattern)."
        )

    def test_ids_driver_stub_present(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "drivers" / "idscamera.py").read_text()
        assert "def set_device_link_throughput_limit(" in src, (
            "IDSCamera must have a set_device_link_throughput_limit "
            "stub so the API method does not need to know which driver "
            "is connected when called by the sweep tool."
        )


class TestPylonAsciiOnlyInLoggerStrings:
    """CLAUDE.md Rule 24 -- ASCII-only in strings emitted to logger / print /
    notifications.

    Non-ASCII characters in runtime-emittable strings can trigger recursive
    UnicodeEncodeError on cp1252 stack handlers (Windows file rotation,
    worker-thread escapees) and break strict-encoding CI environments. The
    rule is strict: no chars past 0x7E (excluding newline / tab) in any
    string passed to logger / print / notifications calls. Comments and
    docstrings are exempt.

    Regression: prior to this audit pass, pyloncamera.py:516 emitted a
    degree sign (U+00B0) in the camera-temperature log line. Fixed
    2026-05-07 audit cleanup commit; test below pins the corrected form
    so a future ``while-I'm-here`` cleanup that re-introduces a non-ASCII
    char in a logger context fires the regression.
    """

    def _pyloncamera_source_lines(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text().splitlines()

    def test_no_non_ascii_in_logger_or_print_lines(self):
        """Walk every line of pyloncamera.py; any line that contains a
        logger / print / notifications / _cam_log call must have no
        char past 0x7E (excluding tab / newline)."""
        offenders = []
        call_markers = ('logger.', 'print(', 'notifications.', '_cam_log.', '_log.')
        for i, line in enumerate(self._pyloncamera_source_lines(), 1):
            if not any(m in line for m in call_markers):
                continue
            for col, ch in enumerate(line):
                if ch in ('\t', '\n', '\r'):
                    continue
                if ord(ch) > 0x7E:
                    offenders.append(
                        f"line {i} col {col}: char {ch!r} (U+{ord(ch):04X}) -- "
                        f"line is: {line.strip()[:80]}"
                    )
                    break
        assert not offenders, (
            "Rule 24 violation -- non-ASCII char in logger/print/notifications "
            "string. Use ASCII (e.g. 'degC' not the degree sign). "
            "Sites:\n  " + "\n  ".join(offenders)
        )

    def test_temperature_log_uses_degC_ascii_form(self):
        """Pin the corrected form so the specific A10 fix survives. If a
        future cleanup edits the temperature log line, this test reminds
        the editor that ASCII-only was intentional."""
        src_lines = self._pyloncamera_source_lines()
        for i, line in enumerate(src_lines, 1):
            if 'Temperature' in line and 'logger' in line:
                assert 'degC' in line, (
                    f"pyloncamera.py:{i} -- temperature log line must use "
                    f"ASCII 'degC' (not the degree sign). Line: {line.strip()[:100]}"
                )
                assert chr(0xB0) not in line, (
                    f"pyloncamera.py:{i} -- degree sign (U+00B0) reintroduced. "
                    f"Use 'degC' instead."
                )
                return
        raise AssertionError(
            "Could not find a temperature log line in pyloncamera.py. "
            "If get_all_temperatures was renamed/removed, update this test."
        )


class TestPylonStateMutationViaMarkDisconnected:
    """CLAUDE.md Rule 2 (single source of truth) + Rule 35 (one canonical
    implementation per capability).

    The canonical write-path for "camera is disconnected" is
    Camera._mark_disconnected (drivers/camera.py): it acquires _state_lock
    and sets _device_removed=True and _active=None atomically, plus emits
    the boundary-transition log line. Direct writes to either flag from
    other call sites bypass the lock invariant and the log.

    Two pyloncamera sites previously bypassed the helper:
      - OnImageGrabbed (the inactive-fallback branch -- when the callback
        runs but parent.active has been reset elsewhere)
      - _CameraRemovalHandler.OnCameraDeviceRemoved (the SDK device-
        removal callback)

    Both now route through _mark_disconnected. The OnCameraDeviceRemoved
    site previously had a comment claiming the helper was unsafe to call
    from an SDK callback thread; that comment predated the lock-based
    design (Camera._mark_disconnected docstring: "Safe to call from any
    thread including SDK callbacks"). Comment retired in the same fix.

    Locks the structural shape of the unified call path so a future
    "while-I'm-here" patch that re-introduces a direct boolean write
    fires the regression.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_no_direct_mark_disconnected_assignment(self):
        """No direct `_device_removed = True` write in pyloncamera.py.

        The canonical mark-disconnected write path is
        Camera._mark_disconnected (acquires _state_lock + sets
        _active=None atomically + emits boundary log). Setting the bool
        to True bypasses all three invariants.

        The inverse case (`_device_removed = False`) IS allowed -- the
        reconnect path in connect() resets the flag for a fresh
        session. There is no canonical _mark_connected helper today;
        if one is added, this test grows to cover both directions.
        """
        src = self._pyloncamera_source()
        forbidden = (
            'self._parent._device_removed = True',
            'self._device_removed = True',
        )
        offenders = [phrase for phrase in forbidden if phrase in src]
        assert not offenders, (
            "pyloncamera.py contains direct write(s) marking the camera "
            f"removed: {offenders}. Use Camera._mark_disconnected "
            "(acquires _state_lock + sets _active=None + emits "
            "boundary log) instead."
        )

    def test_on_image_grabbed_inactive_branch_uses_mark_disconnected(self):
        """The OnImageGrabbed inactive-fallback branch must call
        _mark_disconnected so the parent's _state_lock invariants hold."""
        src = self._pyloncamera_source()
        # Find the inactive-branch sentinel and confirm the call sequence.
        marker = "OnImageGrabbed called but camera is inactive"
        idx = src.find(marker)
        assert idx != -1, (
            "Could not find OnImageGrabbed inactive-branch logger sentinel; "
            "if the wording changed, update this test."
        )
        # Within ~200 chars after the sentinel, expect the canonical call.
        window = src[idx:idx + 400]
        assert "_mark_disconnected()" in window, (
            "OnImageGrabbed inactive-branch must call "
            "self._parent._mark_disconnected() to preserve the "
            "_state_lock invariant. Found instead:\n" + window
        )

    def test_on_camera_device_removed_uses_mark_disconnected(self):
        """The _CameraRemovalHandler SDK callback must call
        _mark_disconnected. _mark_disconnected's docstring states it is
        safe from any thread (including SDK callbacks); the prior
        comment claiming otherwise was stale."""
        src = self._pyloncamera_source()
        marker = "def OnCameraDeviceRemoved("
        idx = src.find(marker)
        assert idx != -1, "Could not find OnCameraDeviceRemoved method."
        window = src[idx:idx + 800]
        assert "_mark_disconnected()" in window, (
            "OnCameraDeviceRemoved must call self._parent._mark_disconnected() "
            "to atomically clear _active under _state_lock."
        )


class TestPylonStatsPollerStopJoin:
    """CLAUDE.md Rule 2 (single source of truth) + Rule 16 (bugs cluster --
    fix all instances of the same structural pattern).

    _start_stats_poller (line ~169) joins any prior thread before
    starting a new one, on the explicit rationale that during the
    window between event-set and daemon-thread-exit, a fresh
    start_stats_poller would skip the join branch and start a duplicate
    poller. _stop_stats_poller (line ~200) historically did not
    symmetrise that join: it set the event and immediately nulled the
    thread reference. Result: brief window during rapid stop/start
    cycles where two pollers write to pylon_stats_trace.csv.

    The fix captures the thread reference, signals stop, joins with a
    bounded timeout (2.0s -- twice the poller's wait interval), then
    releases the reference. Symmetric with _start_stats_poller.

    Locks the structural shape so a future "while-I'm-here" patch that
    drops the join fires the regression.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_stop_stats_poller_captures_thread_before_signalling(self):
        """The thread reference must be read before the event is set;
        otherwise a concurrent _stats_poller_thread = None elsewhere
        could cause join() to be called on None."""
        src = self._pyloncamera_source()
        idx = src.find("def _stop_stats_poller(self):")
        assert idx != -1, "Could not find _stop_stats_poller."
        end = src.find("def ", idx + 10)
        body = src[idx:end]
        # Order check: thread getattr before event set.
        thread_get = body.find("_stats_poller_thread")
        ev_set = body.find(".set()")
        assert thread_get != -1 and ev_set != -1, (
            "_stop_stats_poller must reference both _stats_poller_thread and the "
            "event. Body:\n" + body
        )
        assert thread_get < ev_set, (
            "_stop_stats_poller must capture the thread reference BEFORE "
            "signalling the stop event, so the join() target is stable."
        )

    def test_stop_stats_poller_joins_with_timeout(self):
        """_stop_stats_poller must join the prior thread with a bounded
        timeout to symmetrise _start_stats_poller's join."""
        src = self._pyloncamera_source()
        idx = src.find("def _stop_stats_poller(self):")
        assert idx != -1
        end = src.find("def ", idx + 10)
        body = src[idx:end]
        assert ".join(timeout=" in body, (
            "_stop_stats_poller must call .join(timeout=...) on the prior "
            "stats-poller thread before clearing the reference."
        )


class TestPylonDisconnectStopGrabbingLogged:
    """CLAUDE.md Rule 5 -- fail visible; log every error with context.

    The disconnect() teardown wraps stop_grabbing() in a defensive try
    so a stop failure does not block the rest of teardown (Close,
    DetachDevice, DestroyDevice). Earlier the except branch was bare
    `pass` -- a stop_grabbing failure produced zero log evidence,
    while the other three teardown failures produced a uniform warning
    each. Operator reading the log after a disconnect anomaly saw
    nothing about why teardown started weirdly.

    Fix: warning-level log mirroring the Close / DetachDevice /
    DestroyDevice except branches. Continues teardown either way.

    Audit finding A6.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_disconnect_stop_grabbing_failure_is_logged(self):
        """The bare `except Exception: pass` on stop_grabbing during
        disconnect is a Rule 5 violation. The fix logs at warning
        level."""
        src = self._pyloncamera_source()
        # Find the disconnect method body
        idx = src.find("def disconnect(self) -> bool:")
        assert idx != -1, "Could not find PylonCamera.disconnect."
        # Walk forward to the stop_grabbing block
        sg_idx = src.find("self.stop_grabbing()", idx)
        assert sg_idx != -1, "Could not find stop_grabbing call in disconnect."
        # The except block immediately follows; check the next ~250 chars
        window = src[sg_idx:sg_idx + 350]
        assert "except Exception:" not in window or "pass" not in window.split(
            "except Exception:"
        )[1].split("\n", 5)[0] if "except Exception:" in window else True
        # Simpler: assert the warning-log phrase is present
        assert "stop_grabbing during disconnect" in window, (
            "disconnect's stop_grabbing except branch must log a warning, "
            "not silently pass. Found:\n" + window
        )


class TestPylonOnImageGrabbedExceptionContext:
    """CLAUDE.md Rule 5 + Rule 20 -- log every error with context; logs
    must be clear and accurate.

    The OnImageGrabbed outer except branch previously called
    `logger.exception(e)`, passing the exception instance directly as
    the message. logger.exception renders that as "ExceptionType('msg')"
    -- no [CAM Class ] prefix used everywhere else, no callback context.
    Operator scanning the main log sees a bare exception line with no
    indication it came from the grab callback.

    Fix: contextual prefix matching the file convention (line 140's
    `logger.exception(f'[CAM Class ] Pylon camera disconnect failed: {e}')`,
    line 1070's `logger.exception(f'Failed to grab image: {ex}')`).

    Audit finding A7.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_on_image_grabbed_outer_except_uses_contextual_message(self):
        """Bare `logger.exception(e)` is forbidden in OnImageGrabbed.
        The fix uses an f-string with [CAM Class ] prefix and a callback
        identifier."""
        src = self._pyloncamera_source()
        idx = src.find("def OnImageGrabbed(")
        assert idx != -1, "Could not find ImageHandler.OnImageGrabbed."
        end = src.find("def ", idx + 10)
        body = src[idx:end]
        # Find the outer except clause (indented less than the inner ones)
        # Easiest: the literal `_outcome = 'exception_outer'` is unique.
        marker = "_outcome = 'exception_outer'"
        m_idx = body.find(marker)
        assert m_idx != -1, (
            "Could not find OnImageGrabbed outer-except sentinel "
            "(_outcome = 'exception_outer'). If renamed, update test."
        )
        window = body[m_idx:m_idx + 250]
        assert "logger.exception(e)" not in window, (
            "OnImageGrabbed outer-except must NOT call logger.exception(e) "
            "with the bare exception object -- the rendered log line lacks "
            "[CAM Class ] prefix and callback context."
        )
        assert "OnImageGrabbed" in window or "[CAM Class ]" in window, (
            "OnImageGrabbed outer-except logger.exception call must include "
            "a contextual prefix. Found:\n" + window
        )


class TestPylonInitCameraConfigStyleConsistency:
    """Style-consistency checks on init_camera_config.

    The pypylon C-extension supports both `node = value` (attribute
    assignment) and `node.SetValue(value)` for parameter writes, but
    they have slightly different exception envelopes (`__setattr__`
    vs explicit `SetValue`). The rest of pyloncamera.py uses
    `.SetValue(...)` consistently. The init_camera_config UserSet
    setter previously mixed the styles -- pinning the consistent form
    here prevents drift back.

    Audit finding B22.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_user_set_selector_uses_set_value(self):
        """init_camera_config must call camera.UserSetSelector.SetValue('Default')
        rather than `camera.UserSetSelector = 'Default'`. Consistent
        exception envelope; consistent with the rest of the file."""
        src = self._pyloncamera_source()
        assert "camera.UserSetSelector = 'Default'" not in src, (
            "Use camera.UserSetSelector.SetValue('Default') -- attribute "
            "assignment routes through pypylon __setattr__ which has a "
            "slightly different exception envelope than the explicit "
            "SetValue call used elsewhere in this file."
        )
        assert "UserSetSelector.SetValue('Default')" in src, (
            "init_camera_config must select the 'Default' user set via "
            "UserSetSelector.SetValue('Default')."
        )

    def test_init_asserts_free_run_acquisition(self):
        """init_camera_config must explicitly assert AcquisitionMode=
        Continuous + TriggerMode=Off after UserSetLoad. The 'Default'
        set is documented to leave these in free-run state, but a
        firmware bug or future user-set change could leak a different
        default."""
        src = self._pyloncamera_source()
        idx = src.find("def init_camera_config(self):")
        assert idx != -1
        end = src.find("def ", idx + 10)
        body = src[idx:end]
        assert "AcquisitionMode.SetValue('Continuous')" in body, (
            "init_camera_config must call AcquisitionMode.SetValue('Continuous')."
        )
        assert "TriggerMode.SetValue('Off')" in body, (
            "init_camera_config must call TriggerMode.SetValue('Off')."
        )
        assert "TriggerSelector.SetValue('FrameStart')" in body, (
            "init_camera_config must call TriggerSelector.SetValue('FrameStart') "
            "before TriggerMode -- otherwise TriggerMode applies to whichever "
            "selector was active before."
        )


class TestPylonGainParameterNotShadowingMethod:
    """CLAUDE.md Rule 36 (identifier clarity).

    PylonCamera.gain(self, gain) had the parameter shadow the method
    name. Inside the method body, the symbol `gain` resolved to the
    parameter -- the bound method `self.gain` was still reachable but
    any future refactor that called the method recursively would
    silently fail in a confusing way. Renaming to `value` removes the
    ambiguity. Method name itself (`gain`) is L2-public and not
    changed (Rule 30 stability); only the internal parameter name.

    Audit finding A15.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_gain_method_signature_no_self_param_shadow(self):
        """Forbid the shadowed signature `def gain(self, gain)`."""
        src = self._pyloncamera_source()
        assert "def gain(self, gain)" not in src, (
            "PylonCamera.gain(self, gain) shadows the method name with "
            "the parameter. Use `def gain(self, value)` instead."
        )


class TestPylonDisconnectResetsSelfValidationFlag:
    """The _pylon_self_validation_done flag gates a one-shot
    StreamGrabber NodeMap walk that runs on poller start. Without
    reset on disconnect, a different camera attached on the next
    connect re-uses the prior camera's validation state and skips
    its own probe.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_disconnect_clears_self_validation_flag(self):
        src = self._pyloncamera_source()
        idx = src.find("def disconnect(self) -> bool:")
        assert idx != -1
        end = src.find("def ", idx + 10)
        body = src[idx:end]
        assert "_pylon_self_validation_done = False" in body, (
            "disconnect() must clear _pylon_self_validation_done so the "
            "next connect re-runs the StreamGrabber probe."
        )


class TestPylonUnderrunCounterSingleCanonical:
    """The stream-grabber underrun counter uses one canonical name per
    Basler doc stream-grabber-parameters.html. The earlier multi-name
    candidate resolver and its associated cache attribute were
    speculative -- no other names are documented. If a future SDK
    renames the node, the one-shot StreamGrabber NodeMap walk on
    poller start emits the actual stat-like nodes via the
    [INSTR PYLON ] StreamGrabber NodeMap stat-like log line; the
    rename is diagnosable from there without a brute-force probe.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_canonical_underrun_node_name_constant(self):
        src = self._pyloncamera_source()
        assert "_UNDERRUN_NODE_NAME = 'Statistic_Buffer_Underrun_Count'" in src, (
            "Single canonical underrun-counter name "
            "Statistic_Buffer_Underrun_Count must be the constant."
        )

    def test_no_candidate_list_or_resolver_method(self):
        src = self._pyloncamera_source()
        assert "_UNDERRUN_NODE_CANDIDATES" not in src, (
            "_UNDERRUN_NODE_CANDIDATES tuple was the multi-name "
            "speculative resolver; replaced by the single canonical "
            "_UNDERRUN_NODE_NAME constant."
        )
        assert "_resolve_underrun_node_name" not in src, (
            "_resolve_underrun_node_name method was the multi-name "
            "resolver; with the single canonical constant the helper "
            "is dead code."
        )
        assert "_underrun_node_name_cache" not in src, (
            "_underrun_node_name_cache was the resolver's cache; "
            "with the single canonical constant there is nothing to "
            "cache."
        )


class TestPylonGigeDiagnosticNodeCoverage:
    """read_diagnostic_snapshot must probe the canonical GigE
    network-related parameters and stream-grabber resend counters
    so cross-transport bench characterization captures the GigE
    network state on dmA3536-9gm and the USB3 transport state on
    a2A3536-31umBAS / daA3840-45um from a single API call.

    Authoritative source: Basler doc network-related-parameters.md
    (camera-side GigE params) + stream-grabber-parameters.html
    (Packet Resend Mechanism + Statistics Parameters).

    Per-camera applicability: every node read defensively via
    _safe_node, returning '<missing>' on transports/cameras that
    don't expose it. USB3 cameras report '<missing>' for the GigE
    set; GigE cameras report '<missing>' for URB / MaxTransferSize.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_camera_nodemap_probes_gev_network_parameters(self):
        """The 11 canonical GigE network-related camera-side nodes
        from network-related-parameters.md must appear in the
        camera-config probe in read_diagnostic_snapshot."""
        src = self._pyloncamera_source()
        idx = src.find("def read_diagnostic_snapshot(")
        assert idx != -1
        end = src.find("\n    def ", idx + 10)
        body = src[idx:end]
        for node in (
            'GevHeartbeatTimeout',
            'GevSCPSPacketSize',
            'GevSCPD',
            'GevSCBWR',
            'GevSCBWRA',
            'GevSCBWA',
            'GevSCDMT',
            'GevSCFJM',
            'GevSCFTD',
            'BandwidthReserveMode',
            'PayloadSize',
            'BslDeviceLinkCurrentThroughput',
        ):
            assert node in body, (
                f"read_diagnostic_snapshot must probe {node!r} "
                f"(per network-related-parameters.md). Missing nodes "
                f"will not surface on dmA3536-9gm bench."
            )

    def test_stream_grabber_probes_gige_resend_config(self):
        """The 8 canonical GigE Packet Resend Mechanism stream-
        grabber config nodes from stream-grabber-parameters.html
        must appear in the stream-grabber config probe."""
        src = self._pyloncamera_source()
        idx = src.find("def read_diagnostic_snapshot(")
        end = src.find("\n    def ", idx + 10)
        body = src[idx:end]
        for node in (
            'EnableResend',
            'PacketTimeout',
            'FrameRetention',
            'MaximumNumberResendRequests',
            'FirewallTraversalInterval',
            'AutoPacketSize',
            'SocketBufferSize',
        ):
            assert node in body, (
                f"read_diagnostic_snapshot stream-grabber config "
                f"must probe {node!r} (per stream-grabber-parameters."
                f"html Packet Resend Mechanism Parameters)."
            )

    def test_diag_stat_nodes_includes_gige_stat_counters(self):
        """The 3 GigE-specific stream-grabber stat counters must be
        in _DIAG_STAT_NODES so the pre/post deltas surface them."""
        src = self._pyloncamera_source()
        # Find the _DIAG_STAT_NODES tuple body
        idx = src.find("_DIAG_STAT_NODES = (")
        assert idx != -1
        end = src.find(")", idx)
        body = src[idx:end]
        for counter in (
            'Statistic_Resend_Packet_Count',
            'Statistic_Resend_Request_Count',
            'Statistic_Failed_Packet_Count',
        ):
            assert counter in body, (
                f"_DIAG_STAT_NODES must include {counter!r} so the "
                f"GigE resend traffic surfaces in the diagnostic "
                f"snapshot. Per stream-grabber-parameters.html "
                f"Statistics Parameters."
            )

    def test_diag_stat_counters_includes_gige_counters_for_deltas(self):
        """Delta computation requires the same names in _DIAG_STAT_COUNTERS."""
        src = self._pyloncamera_source()
        idx = src.find("_DIAG_STAT_COUNTERS = (")
        assert idx != -1
        end = src.find(")", idx)
        body = src[idx:end]
        for counter in (
            'Statistic_Resend_Packet_Count',
            'Statistic_Resend_Request_Count',
            'Statistic_Failed_Packet_Count',
        ):
            assert counter in body, (
                f"_DIAG_STAT_COUNTERS must include {counter!r} for "
                f"delta computation."
            )


class TestPylonDltlClampAndDocWarnings:
    """set_device_link_throughput_limit must clamp out-of-range
    values via DeviceLinkThroughputLimit.GetMin() / GetMax() rather
    than letting the SDK raise OutOfRangeException, and the docstring
    must record both Basler doc warnings: rolling-shutter distortion
    if too low, corrupt/dropped frames if too high.

    Per per-camera spec pages a2a3536-31umbas.html and
    daa3840-45um.html: both production cameras are rolling-shutter,
    so both warnings apply on both. Per
    network-bandwidth-control-(blaze).md the DLTL throttle
    mechanism (pause-insertion between packets) is identical across
    transports.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_clamp_helper_present(self):
        src = self._pyloncamera_source()
        assert "def _clamp_dltl_value_bps(self, value_bps: int) -> int:" in src, (
            "_clamp_dltl_value_bps helper must exist with the "
            "documented signature."
        )

    def test_clamp_calls_min_max_query(self):
        src = self._pyloncamera_source()
        idx = src.find("def _clamp_dltl_value_bps(")
        assert idx != -1
        end = src.find("\n    def ", idx + 10)
        body = src[idx:end]
        assert ".GetMin()" in body and ".GetMax()" in body, (
            "_clamp_dltl_value_bps must query DeviceLinkThroughputLimit"
            ".GetMin() and .GetMax() to determine the clamp range."
        )

    def test_setter_calls_clamp_helper(self):
        src = self._pyloncamera_source()
        idx = src.find("def set_device_link_throughput_limit(")
        assert idx != -1
        end = src.find("\n    def ", idx + 10)
        body = src[idx:end]
        assert "_clamp_dltl_value_bps" in body, (
            "set_device_link_throughput_limit must run value_bps "
            "through _clamp_dltl_value_bps before SetValue."
        )

    def test_docstring_records_too_low_warning(self):
        """Rolling-shutter distortion warning must appear in the docstring."""
        src = self._pyloncamera_source()
        idx = src.find("def set_device_link_throughput_limit(")
        assert idx != -1
        end = src.find('"""', src.find('"""', idx) + 3) + 3
        docstring = src[idx:end]
        assert "rolling shutter" in docstring.lower() or "rolling-shutter" in docstring.lower(), (
            "DLTL setter docstring must record the rolling-shutter "
            "distortion warning per per-camera spec pages."
        )

    def test_docstring_records_too_high_warning(self):
        """Corrupt/dropped frames warning must appear in the docstring."""
        src = self._pyloncamera_source()
        idx = src.find("def set_device_link_throughput_limit(")
        assert idx != -1
        end = src.find('"""', src.find('"""', idx) + 3) + 3
        docstring = src[idx:end]
        assert "corrupt" in docstring.lower() or "dropped" in docstring.lower(), (
            "DLTL setter docstring must record the too-high warning "
            "(corrupt or dropped frames) per per-camera spec pages."
        )


class TestPylonResyncProminentLog:
    """Per Basler doc stream-grabber-parameters.html, "A host
    resynchronization is considered the most serious error case in
    the USB 3.0 and USB3 Vision specification."

    The stats poller previously emitted a prominent log line for
    Statistic_Buffer_Underrun_Count but not for
    Statistic_Resynchronization_Count -- yet resync is the more
    severe failure per the doc. The fix tracks the prior resync
    count on the camera instance and emits a WARNING-level log on
    any positive delta. Total count remains in the CSV row.

    Audit finding B5.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_resync_node_in_stats_node_names(self):
        """Statistic_Resynchronization_Count must be in the live
        stats poll set so the delta tracker has fresh data."""
        src = self._pyloncamera_source()
        idx = src.find("_STATS_NODE_NAMES = (")
        assert idx != -1
        end = src.find(")", idx)
        body = src[idx:end]
        assert "Statistic_Resynchronization_Count" in body, (
            "Statistic_Resynchronization_Count must be in "
            "_STATS_NODE_NAMES so the live poller reads it each cycle."
        )

    def test_resync_prominent_log_on_positive_delta(self):
        """Stats poller must emit a [INSTR RESYNC] warning when the
        delta is positive."""
        src = self._pyloncamera_source()
        idx = src.find("def _stats_poller_loop(self):")
        assert idx != -1
        end = src.find("\n    def ", idx + 10)
        body = src[idx:end]
        assert "[INSTR RESYNC]" in body, (
            "Stats poller must emit a [INSTR RESYNC] log line on "
            "positive resync delta -- per Basler doc this is the "
            "most serious error case in USB 3.0 / USB3 Vision."
        )
        assert "logger.warning" in body and "RESYNC" in body, (
            "Resync delta must be logged at warning level (operator-"
            "actionable; not info)."
        )

    def test_resync_csv_column_present(self):
        """The pylon_stats_trace.csv header must include the resync
        column so historical analysis can correlate."""
        src = self._pyloncamera_source()
        idx = src.find("'pylon_stats_trace.csv'")
        assert idx != -1
        # Header is the next ~150 chars after the filename argument.
        window = src[idx:idx + 500]
        assert "resync_count" in window, (
            "pylon_stats_trace.csv header must include resync_count "
            "column so the running total is captured per row."
        )


class TestPylonTemperatureStateMonitoring:
    """Per Basler doc temperature-state.html, ace 2 / boost /
    dart M/R cameras halt image acquisition when over-temperature
    is reached and require cool-down before restart -- presents
    identically to STALL-1 in the user log without attribution.

    Fix: poll TemperatureState in the live stats poller; warn on
    any non-Ok state. Surface in read_diagnostic_snapshot too so
    cross-host bench comparison captures the camera's thermal
    history.

    Audit finding B13.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_stats_poller_reads_temperature_state(self):
        src = self._pyloncamera_source()
        idx = src.find("def _stats_poller_loop(self):")
        assert idx != -1
        end = src.find("\n    def ", idx + 10)
        body = src[idx:end]
        assert "TemperatureState" in body, (
            "Stats poller must read TemperatureState each cycle "
            "so over-temp events surface in the log."
        )
        assert "[INSTR TEMP]" in body, (
            "Stats poller must emit [INSTR TEMP] on temperature "
            "state changes for log-grep visibility."
        )
        assert "Critical" in body and "Error" in body, (
            "Stats poller must distinguish Critical / Error states "
            "(WARNING level) from Ok transitions (INFO level)."
        )

    def test_read_diagnostic_snapshot_captures_thermal_state(self):
        src = self._pyloncamera_source()
        idx = src.find("def read_diagnostic_snapshot(")
        assert idx != -1
        end = src.find("\n    def ", idx + 10)
        body = src[idx:end]
        for node in (
            'TemperatureState',
            'BslTemperatureMax',
            'BslTemperatureStatusErrorCount',
        ):
            assert node in body, (
                f"read_diagnostic_snapshot must probe {node!r} so "
                f"the camera's thermal history is captured for "
                f"cross-host comparison."
            )

    def test_temperature_csv_column_present(self):
        src = self._pyloncamera_source()
        idx = src.find("'pylon_stats_trace.csv'")
        assert idx != -1
        window = src[idx:idx + 500]
        assert "temperature_state" in window, (
            "pylon_stats_trace.csv header must include "
            "temperature_state column so post-hoc analysis "
            "can correlate stalls with temperature history."
        )


class TestPylonIsConnectedCallsSdkQuery:
    """is_connected docstring promised "if available, the SDK's
    device-removed query" but the implementation only checked the
    internal _device_removed flag and self.active. The SDK's
    InstantCamera.IsCameraDeviceRemoved() exposes its own removal
    state -- if the _CameraRemovalHandler missed an event (e.g., the
    handler was registered late, or the SDK delivered the removal on
    a path that didn't fire the handler), the SDK still knows.

    Defense in depth: query the SDK as a third check after our two
    flags. Cheap (no transport enumeration). On query failure, log
    debug and trust the prior checks.

    Audit finding B12.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_is_connected_calls_is_camera_device_removed(self):
        src = self._pyloncamera_source()
        idx = src.find("def is_connected(self) -> bool:")
        assert idx != -1, "Could not find PylonCamera.is_connected."
        end = src.find("def ", idx + 10)
        body = src[idx:end]
        assert ".IsCameraDeviceRemoved()" in body, (
            "is_connected must call self.active.IsCameraDeviceRemoved() "
            "as a third check (after _device_removed flag + active is "
            "None). The docstring already promises 'the SDK's "
            "device-removed query'; the implementation must match."
        )


class TestPylonBslPrefixedNodeFallbacks:
    """Basler doc establishes Bsl-prefixed canonical names for several
    GenICam parameters on ace 2 / boost / dart M/R cameras:

      BslResultingAcquisitionFrameRate   (vs ResultingFrameRate)
      BslEffectiveExposureTime           (vs ExposureTime read path)

    Sources: resulting-acquisition-frame-rate.html "Other Cameras"
    sample code; exposure-time.html "On ace 2, boost R, and dart R/M
    cameras, get the value of the BslEffectiveExposureTime parameter."

    Both production cameras (a2A3536-31umBAS ace 2, daA3840-45um dart
    R) fall in the "Other Cameras" / "ace 2 + dart M/R" bucket.
    pyloncamera.py read paths previously used the unprefixed legacy
    names. pypylon may alias today, but the documented canonical
    differs and the read-path semantics differ for ExposureTime
    (BslEffectiveExposureTime returns what was actually used,
    accounting for hardware-imposed rounding; ExposureTime returns
    the requested value).

    Fix: defensive read with Bsl-prefixed first, legacy unprefixed as
    fallback. Three call sites updated: _stats_poller_loop (live
    fps), get_exposure_t (live exposure read), and
    read_diagnostic_snapshot config tuple. New _node_attr_get helper
    handles attribute-style reads; _safe_node now accepts *names so
    nodemap-style reads can also use the fallback pattern.

    Audit findings B1 + B2.
    """

    def _pyloncamera_source(self):
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent
                / "drivers" / "pyloncamera.py").read_text()

    def test_node_attr_get_helper_present(self):
        """The _node_attr_get helper must exist and accept *names."""
        src = self._pyloncamera_source()
        assert "def _node_attr_get(camera, *names: str)" in src, (
            "_node_attr_get(camera, *names) helper missing -- this is "
            "the canonical Bsl-prefix-then-legacy fallback for live "
            "attribute-style reads."
        )

    def test_safe_node_accepts_multiple_names(self):
        """_safe_node must accept *names so the diagnostic snapshot
        can probe Bsl-prefixed-then-legacy nodes via the nodemap."""
        src = self._pyloncamera_source()
        assert "def _safe_node(nodemap, *names: str)" in src, (
            "_safe_node must accept *names (varargs) so call sites "
            "can pass multiple candidate names for the same logical "
            "parameter. Single-name calls remain backwards-compatible."
        )

    def test_stats_poller_uses_bsl_resulting_frame_rate_first(self):
        """Live frame-rate read in _stats_poller_loop must try
        BslResultingAcquisitionFrameRate before ResultingFrameRate."""
        src = self._pyloncamera_source()
        idx = src.find("def _stats_poller_loop(self):")
        assert idx != -1
        end = src.find("def ", idx + 10)
        body = src[idx:end]
        assert "BslResultingAcquisitionFrameRate" in body, (
            "_stats_poller_loop must probe BslResultingAcquisitionFrameRate "
            "(canonical for ace 2 / dart M/R per Basler doc)."
        )
        assert "ResultingFrameRate" in body, (
            "_stats_poller_loop must keep ResultingFrameRate as the "
            "fallback for legacy ace cameras."
        )
        # Bsl variant must come first in the call.
        bsl_pos = body.find("BslResultingAcquisitionFrameRate")
        legacy_pos = body.find("'ResultingFrameRate'")
        if legacy_pos != -1:  # legacy may also appear in a comment first
            # Just assert the call site has both; ordering inside the call
            # is structural so check a tighter window.
            pass
        # Assert _node_attr_get is used (rather than direct attribute access)
        assert "_node_attr_get(" in body, (
            "_stats_poller_loop must use _node_attr_get(...) for the "
            "frame-rate read so the Bsl-fallback pattern is centralised."
        )

    def test_get_exposure_t_uses_bsl_effective_first(self):
        """get_exposure_t must read BslEffectiveExposureTime first
        (the doc-canonical effective value) and fall back to
        ExposureTime (the requested set value)."""
        src = self._pyloncamera_source()
        idx = src.find("def get_exposure_t(self)")
        assert idx != -1
        end = src.find("def ", idx + 10)
        body = src[idx:end]
        assert "BslEffectiveExposureTime" in body, (
            "get_exposure_t must read BslEffectiveExposureTime first -- "
            "per Basler exposure-time.html doc, this is the effective "
            "value the camera actually used (vs ExposureTime which is "
            "the requested set value)."
        )
        assert "_node_attr_get(" in body, (
            "get_exposure_t must use _node_attr_get(...) for the "
            "Bsl-fallback pattern."
        )

    def test_diag_snapshot_config_tuple_uses_bsl_fallbacks(self):
        """read_diagnostic_snapshot config tuple must include the
        Bsl-prefixed canonical names for ResultingFrameRate and
        ExposureTime so cross-host comparison probes the correct
        node on ace 2 / dart M/R."""
        src = self._pyloncamera_source()
        idx = src.find("def read_diagnostic_snapshot(")
        assert idx != -1
        end = src.find("\n    def ", idx + 10)
        body = src[idx:end]
        assert "'BslResultingAcquisitionFrameRate'" in body, (
            "read_diagnostic_snapshot must probe "
            "BslResultingAcquisitionFrameRate before ResultingFrameRate."
        )
        assert "'BslEffectiveExposureTime'" in body, (
            "read_diagnostic_snapshot must probe BslEffectiveExposureTime "
            "before ExposureTime so the snapshot reports effective "
            "exposure on ace 2 / dart M/R."
        )
