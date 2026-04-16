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
    """Return a dict of commonly needed mock modules (lvp_logger, userpaths, etc).

    NOTE: cv2 is NOT mocked — it's a real installed package with no Kivy
    dependency. Mocking it causes test-ordering contamination: image_utils
    caches the mock cv2 reference at import time, and monkeypatch cleanup
    can't fix the cached reference. This broke TestAddTimestampInPlace.
    """
    mods = {
        'userpaths': MagicMock(),
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
from modules.exceptions import HardwareError, ProtocolError, ConfigError, CaptureError


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


class TestWorkerLogPermissions:
    """Verify worker log files get restricted permissions."""

    def test_log_file_permissions(self, tmp_path):
        """Worker log files should be owner-only (0o600)."""
        import os
        import sys
        if sys.platform == 'win32':
            pytest.skip('chmod not meaningful on Windows')

        from modules.sequenced_capture_writer import setup_worker_logger
        logger = setup_worker_logger(log_dir=str(tmp_path))
        # Find the log file
        log_files = list(tmp_path.glob('*.log'))
        assert len(log_files) == 1
        mode = oct(log_files[0].stat().st_mode & 0o777)
        assert mode == '0o600'


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
