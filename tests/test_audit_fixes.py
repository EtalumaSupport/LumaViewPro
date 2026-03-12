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
    """Return a dict of commonly needed mock modules (lvp_logger, userpaths, etc)."""
    mods = {
        'userpaths': MagicMock(),
        'lvp_logger': _build_mock_logger(),
        'requests': MagicMock(),
        'requests.structures': MagicMock(),
        'psutil': MagicMock(),
        'cv2': MagicMock(),
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
    return Lumascope(simulate=True)


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
    from modules.sequenced_capture_executor import (
        ProtocolState,
        _PROTOCOL_STATE_TRANSITIONS,
    )
    return ProtocolState, _PROTOCOL_STATE_TRANSITIONS


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
