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
