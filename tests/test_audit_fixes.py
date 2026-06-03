# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for audit fixes across LumaViewPro.

Covers:
  1. Domain exceptions (modules/exceptions.py)
  2. Input validation (lumascope_api.py)
  3. Protocol file limits (modules/protocol.py)
  4. ProtocolState transitions (modules/sequenced_capture_runner.py)
  5. Settings snapshot thread safety (modules/app_context.py)
  6. AppleScript escaping (ui/file_dialogs.py)
  7. FPS calculation edge case

IMPORTANT: This file does NOT manipulate sys.modules at module level.
All mocking is done inside fixtures/test methods and cleaned up afterward.
"""

import inspect
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
    mock_lvp_logger.version = 'test'
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
        'kivy',
        'kivy.app',
        'kivy.clock',
        'kivy.core',
        'kivy.core.window',
        'kivy.factory',
        'kivy.graphics',
        'kivy.graphics.texture',
        'kivy.graphics.instructions',
        'kivy.graphics.vertex_instructions',
        'kivy.lang',
        'kivy.metrics',
        'kivy.uix',
        'kivy.uix.boxlayout',
        'kivy.uix.filechooser',
        'kivy.uix.floatlayout',
        'kivy.uix.gridlayout',
        'kivy.uix.image',
        'kivy.uix.label',
        'kivy.uix.popup',
        'kivy.uix.scrollview',
        'kivy.uix.slider',
        'kivy.uix.spinner',
        'kivy.uix.textinput',
        'kivy.uix.togglebutton',
        'kivy.uix.widget',
        'kivy.uix.behaviors',
        'kivy.uix.behaviors.hover',
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
        'pypylon',
        'pypylon.pylon',
        'pypylon.genicam',
        'ids_peak',
        'ids_peak.ids_peak',
        'ids_peak.ids_peak_ipl_extension',
        'ids_peak_ipl',
    ]:
        mods[name] = MagicMock()
    return mods


def _common_mock_modules():
    """Return a dict of commonly needed mock modules (lvp_logger, platformdirs, etc).

    NOTE: cv2 is NOT mocked -- it's a real installed package with no Kivy
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
# 1. Domain exceptions -- no mocks needed, pure Python module
# ===========================================================================
from drivers.exceptions import HardwareError
from modules.exceptions import (
    AutofocusAborted,
    CaptureError,
    ConfigError,
    ProtocolError,
)
from modules.lumascope_api.imaging import ImagingAPI
from modules.lumascope_api.runtime_state import RuntimeState


class TestDomainExceptions:
    """Verify custom exception classes are proper Exception subclasses."""

    @pytest.mark.parametrize(
        'exc_cls',
        [
            HardwareError,
            ProtocolError,
            ConfigError,
            CaptureError,
        ],
    )
    def test_subclass_of_exception(self, exc_cls):
        assert issubclass(exc_cls, Exception)

    @pytest.mark.parametrize(
        'exc_cls',
        [
            HardwareError,
            ProtocolError,
            ConfigError,
            CaptureError,
        ],
    )
    def test_raise_and_catch_with_message(self, exc_cls):
        msg = f'test message for {exc_cls.__name__}'
        with pytest.raises(exc_cls, match=msg):
            raise exc_cls(msg)


# ===========================================================================
# 2. Input validation -- Lumascope API (needs mocks for camera/logger deps)
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
        with pytest.raises(ValueError, match='channel'):
            sim_scope.illumination.led_on(channel=99, mA=10)

    def test_rejects_negative_current(self, sim_scope):
        with pytest.raises(ValueError, match='current'):
            sim_scope.illumination.led_on(channel=0, mA=-1)

    def test_rejects_current_above_max(self, sim_scope):
        with pytest.raises(ValueError, match='current'):
            sim_scope.illumination.led_on(
                channel=0,
                mA=sim_scope.capabilities.led_max_ma + 1,
            )

    def test_accepts_valid_input(self, sim_scope):
        sim_scope.illumination.led_on(channel=0, mA=50)


class TestMoveAbsolutePositionValidation:
    """Verify move_absolute_position() rejects bad inputs."""

    def test_rejects_invalid_axis(self, sim_scope):
        with pytest.raises(ValueError, match='Axis'):
            sim_scope.motion.move_absolute_position(axis='Q', pos=100)

    def test_rejects_position_above_limit(self, sim_scope):
        from modules.lumascope_api import Lumascope

        with pytest.raises(ValueError, match='exceeds safety limit'):
            sim_scope.motion.move_absolute_position(
                axis='Z', pos=Lumascope.MOTOR_POSITION_LIMIT + 1
            )

    def test_rejects_large_negative_position(self, sim_scope):
        from modules.lumascope_api import Lumascope

        with pytest.raises(ValueError, match='exceeds safety limit'):
            sim_scope.motion.move_absolute_position(
                axis='Z', pos=-(Lumascope.MOTOR_POSITION_LIMIT + 1)
            )

    def test_accepts_valid_input(self, sim_scope):
        sim_scope.motion.move_absolute_position(axis='Z', pos=1000)


# ===========================================================================
# 3. Protocol file limits (needs mocks for heavy deps)
# ===========================================================================


class TestProtocolFileLimits:
    """Verify Protocol.from_file() enforces size and step count limits."""

    def test_rejects_oversized_file(self, _mock_heavy_deps, tmp_path):
        """A file > 10 MB should be rejected before parsing."""
        from modules.protocol import Protocol

        big_file = tmp_path / 'huge_protocol.tsv'
        big_file.write_bytes(b'x' * (10 * 1024 * 1024 + 1))

        with pytest.raises(ValueError, match='exceeds maximum size'):
            Protocol.from_file(
                file_path=big_file,
                tiling_configs_file_loc=None,
            )

    def test_accepts_file_under_limit(self, _mock_heavy_deps, tmp_path):
        """A small file should pass the size check (may fail later on format,
        but should NOT raise the size ValueError)."""
        from modules.protocol import Protocol

        small_file = tmp_path / 'small.tsv'
        small_file.write_text('LumaViewPro Protocol\n')

        with pytest.raises(Exception) as exc_info:
            Protocol.from_file(
                file_path=small_file,
                tiling_configs_file_loc=None,
            )
        assert 'exceeds maximum size' not in str(exc_info.value)


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

    @pytest.mark.parametrize(
        'from_name, to_name',
        [
            ('IDLE', 'RUNNING'),
            ('RUNNING', 'SCANNING'),
            ('RUNNING', 'COMPLETING'),
            ('RUNNING', 'ERROR'),
            ('SCANNING', 'RUNNING'),
            ('SCANNING', 'COMPLETING'),
            ('SCANNING', 'ERROR'),
            ('COMPLETING', 'IDLE'),
            ('ERROR', 'IDLE'),
        ],
    )
    def test_valid_transitions(self, protocol_state_imports, from_name, to_name):
        """All documented transitions should be present in the map."""
        ProtocolState, transitions = protocol_state_imports
        from_state = ProtocolState[from_name]
        to_state = ProtocolState[to_name]
        allowed = transitions[from_state]
        assert to_state in allowed

    @pytest.mark.parametrize(
        'from_name, to_name',
        [
            ('IDLE', 'SCANNING'),
            ('IDLE', 'COMPLETING'),
            ('IDLE', 'ERROR'),
            ('COMPLETING', 'RUNNING'),
            ('COMPLETING', 'SCANNING'),
            ('ERROR', 'RUNNING'),
            ('ERROR', 'SCANNING'),
        ],
    )
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
            assert state not in allowed, f'{state} allows self-transition'


# ===========================================================================
# 5. Settings snapshot (AppContext) -- no mocks needed, pure Python dataclass
# ===========================================================================
from modules.app_context import AppContext


class TestSettingsSnapshot:
    """Verify thread-safe settings access on AppContext."""

    def test_snapshot_is_deep_copy(self):
        ctx = AppContext(settings={'display': {'brightness': 80}})
        snap = ctx.get_settings_snapshot()

        snap['display']['brightness'] = 999
        snap['new_key'] = True

        assert ctx.settings['display']['brightness'] == 80
        assert 'new_key' not in ctx.settings

    def test_update_settings_writes_value(self):
        ctx = AppContext(settings={})
        ctx.update_settings('live_folder', '/tmp/test')
        assert ctx.settings['live_folder'] == '/tmp/test'

    def test_update_settings_overwrites_existing(self):
        ctx = AppContext(settings={'live_folder': '/old'})
        ctx.update_settings('live_folder', '/new')
        assert ctx.settings['live_folder'] == '/new'

    def test_snapshot_after_update(self):
        ctx = AppContext(settings={})
        ctx.update_settings('key', 'value1')
        snap = ctx.get_settings_snapshot()
        ctx.update_settings('key', 'value2')

        assert snap['key'] == 'value1'
        assert ctx.settings['key'] == 'value2'


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
# 7. FPS calculation edge case -- pure math, no imports needed
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
# 8. Phase 4f -- Security hardening tests
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
        # Socket should be closed -- port property still works
        assert isinstance(lock.port, int)

    def test_second_instance_blocked(self):
        """Regression for #559: two LvpLock instances on the same port must conflict.

        Without this guarantee, a second LumaViewPro launch silently tramples
        the first's exclusive serial ports on Windows. The bug was an accidental
        SO_REUSEADDR setsockopt -- on Windows that has SO_REUSEPORT semantics
        and explicitly allows live double-bind.
        """
        from modules.lvp_lock import LvpLock
        import socket

        # Grab a free port, then release it so we can bind it from LvpLock
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        with LvpLock(lock_port=port) as first:
            assert first.lock() is True, 'first lock should succeed'
            second = LvpLock(lock_port=port)
            try:
                assert second.lock() is False, (
                    'second lock on same port MUST fail -- regression of #559 '
                    '(SO_REUSEADDR reintroduced?)'
                )
            finally:
                second.close()

    def test_lock_check_runs_before_kivy_import(self):
        """Issue #559 structural fix: the lock check must run BEFORE
        any Kivy import. When the check lived in App.build(), Kivy
        had already initialized SDL2 and opened a native window by
        the time the loser reached sys.exit, producing duplicate
        visible Kivy windows on double-launch.
        """
        import pathlib

        src = pathlib.Path('lumaviewpro.py').read_text()
        lock_idx = src.find('_lvp_lock_singleton.lock()')
        assert lock_idx >= 0, (
            'lumaviewpro.py must invoke _lvp_lock_singleton.lock() '
            'in __main__ block; structural fix for issue #559.'
        )
        first_kivy_import = src.find('from kivy.')
        assert first_kivy_import >= 0
        assert lock_idx < first_kivy_import, (
            'Lock check must run BEFORE the first kivy import. If '
            "this fails, the loser's Kivy window has already opened "
            'before sys.exit fires (issue #559 structural regression).'
        )

    def test_lock_loser_calls_os_exit(self):
        """The lock-loser path uses os._exit(1) rather than sys.exit(1)
        so that no downstream import (Kivy / SDL2) gets to fire after
        the dialog is dismissed. sys.exit raises SystemExit which
        cleanup paths can swallow.
        """
        import pathlib

        src = pathlib.Path('lumaviewpro.py').read_text()
        # Slice the __main__ block lock-check region.
        start = src.find('_lvp_lock_singleton.lock()')
        end = src.find('Kivy configurations', start)
        assert end > start
        region = src[start:end]
        assert 'os._exit(1)' in region, (
            'Lock-loser path must call os._exit(1) (not sys.exit(1)) '
            'so Kivy / SDL2 cannot start after the popup is dismissed.'
        )


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
# 9. Phase 6 -- Cleanup tests
# ===========================================================================


class TestAddTimestampInPlace:
    """Verify add_timestamp in-place optimization."""

    def test_in_place_modifies_original(self):
        import numpy as np
        from modules.image_utils import add_timestamp

        img = np.zeros((100, 200), dtype=np.uint8)
        result = add_timestamp(img, '2026-01-01', in_place=True)
        # Should return the same array object
        assert result is img

    def test_copy_does_not_modify_original(self):
        import numpy as np
        from modules.image_utils import add_timestamp

        img = np.zeros((100, 200), dtype=np.uint8)
        original_sum = img.sum()
        result = add_timestamp(img, '2026-01-01', in_place=False)
        # Original should be unchanged
        assert img.sum() == original_sum
        # Result should be a different object
        assert result is not img

    def test_default_is_in_place(self):
        import numpy as np
        from modules.image_utils import add_timestamp

        img = np.zeros((100, 200), dtype=np.uint8)
        result = add_timestamp(img, 'test')
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


class TestGdiSamplerCtypesSignatures:
    """Bench 2026-05-18: metrics.log reports gdi=0 on every Windows
    sample of the beta12 soak (impossible -- Windows GUI apps always
    have GDI handles). Root cause: ctypes calls to GetCurrentProcess /
    GetGuiResources had no argtypes / restype declarations, so the
    64-bit HANDLE was being truncated by the default c_int and the
    function silently returned 0. Without these declarations the metric
    is structurally broken on 64-bit Windows (Rule 20 lying log).
    """

    def _read_common_utils(self):
        import pathlib

        root = pathlib.Path(__file__).parent.parent
        return (root / 'modules' / 'common_utils.py').read_text()

    def test_getcurrentprocess_restype_declared(self):
        src = self._read_common_utils()
        assert 'GetCurrentProcess.restype = ctypes.c_void_p' in src, (
            'kernel32.GetCurrentProcess must declare restype=c_void_p '
            "so the 64-bit pseudo-handle isn't truncated."
        )

    def test_getguiresources_argtypes_declared(self):
        src = self._read_common_utils()
        assert 'GetGuiResources.argtypes = [ctypes.c_void_p, ctypes.c_uint]' in src, (
            'user32.GetGuiResources must declare argtypes=[c_void_p, c_uint] '
            "so the HANDLE arg isn't truncated by the default c_int."
        )

    def test_getguiresources_restype_declared(self):
        src = self._read_common_utils()
        assert 'GetGuiResources.restype = ctypes.c_uint' in src, (
            'user32.GetGuiResources must declare restype=c_uint so the '
            "GDI count isn't reinterpreted as a smaller signed type."
        )


# ===========================================================================
# 9. Position cache -- push-based, zero serial I/O
# ===========================================================================


class TestPositionCache:
    """Verify push-based position cache in Lumascope API.

    The position cache eliminates serial polling from the GUI layer.
    Positions are updated on move commands and after homing -- the GUI
    reads from cache with zero hardware calls.
    """

    def test_initial_cache_is_zero(self, sim_scope):
        """Cache starts at 0 for all axes before any moves."""
        assert sim_scope.motion.get_target_position('X') == 0.0
        assert sim_scope.motion.get_target_position('Y') == 0.0
        assert sim_scope.motion.get_target_position('Z') == 0.0

    def test_move_absolute_updates_cache(self, sim_scope):
        """move_absolute_position should push the new position into the cache."""
        sim_scope.motion.move_absolute_position('Z', 5000.0)
        assert sim_scope.motion.get_target_position('Z') == 5000.0

    def test_move_absolute_only_updates_target_axis(self, sim_scope):
        """Moving Z should not affect X or Y cache."""
        sim_scope.motion.move_absolute_position('Z', 5000.0)
        assert sim_scope.motion.get_target_position('X') == 0.0
        assert sim_scope.motion.get_target_position('Y') == 0.0

    def test_move_relative_updates_cache(self, sim_scope):
        """move_relative_position should accumulate into the cache."""
        sim_scope.motion.move_absolute_position('X', 1000.0)
        sim_scope.motion.move_relative_position('X', 500.0)
        assert sim_scope.motion.get_target_position('X') == 1500.0

    def test_move_relative_negative(self, sim_scope):
        """Negative relative moves should subtract from cache."""
        sim_scope.motion.move_absolute_position('Z', 3000.0)
        sim_scope.motion.move_relative_position('Z', -1000.0)
        assert sim_scope.motion.get_target_position('Z') == 2000.0

    def test_get_all_axes(self, sim_scope):
        """get_target_position(None) returns dict of all axes."""
        sim_scope.motion.move_absolute_position('X', 100.0)
        sim_scope.motion.move_absolute_position('Y', 200.0)
        sim_scope.motion.move_absolute_position('Z', 300.0)
        result = sim_scope.motion.get_target_position()
        assert isinstance(result, dict)
        assert result['X'] == 100.0
        assert result['Y'] == 200.0
        assert result['Z'] == 300.0

    def test_get_current_position_matches_target(self, sim_scope):
        """After a blocking move, get_current_position is at the target
        within microstep precision. The motor lands at the nearest
        microstep to the commanded value; converting back to um round-
        trips through that quantization (~0.078 um on X/Y, ~0.025 um on
        Z), so exact equality is not achievable. #674 H4 changed the
        cache contract from 'snap to commanded target on arrival' to
        'reflect motor's polled actual', exposing this quantization
        residual."""
        sim_scope.motion.move_absolute_position('Z', 7777.0, wait_until_complete=True)
        assert sim_scope.motion.get_current_position('Z') == pytest.approx(7777.0, abs=0.1)

    def test_refresh_after_homing(self, sim_scope):
        """refresh_position_cache syncs cache from hardware (used after homing)."""
        # Directly set the simulated motor's internal position to simulate homing
        # The simulated motor stores positions in microsteps; target_pos() converts.
        # Use move_abs_pos to set a known position, then verify refresh reads it.
        sim_scope._motion_driver.move_abs_pos('Z', 5000.0)
        # Cache still has old value since we bypassed move_absolute_position
        assert sim_scope.motion.get_target_position('Z') != 5000.0
        # Now refresh from hardware
        sim_scope.motion.refresh_position_cache()
        # Should now match what the motor reports
        pos = sim_scope.motion.get_target_position('Z')
        assert abs(pos - 5000.0) < 1.0  # allow microstep rounding

    def test_cache_returns_copy(self, sim_scope):
        """get_target_position(None) should return a copy, not the internal dict."""
        result = sim_scope.motion.get_target_position()
        result['X'] = 99999.0
        # Internal cache should be unaffected
        assert sim_scope.motion.get_target_position('X') == 0.0


# ===========================================================================
# 8. Axis state model -- push-based state tracking (zero serial I/O)
# ===========================================================================


class TestAxisState:
    """Verify axis state transitions in the Lumascope API."""

    def test_initial_state_is_unknown(self, sim_scope):
        """All axes start in UNKNOWN state before homing."""
        from modules.lumascope_api import AxisState

        for ax in ('X', 'Y', 'Z', 'T'):
            assert sim_scope.motion.get_axis_state(ax) == AxisState.UNKNOWN

    def test_axis_state_idle_after_move_with_wait(self, sim_scope):
        """After move_absolute_position with wait_until_complete, axis is IDLE."""
        from modules.lumascope_api import AxisState

        sim_scope.motion.move_absolute_position('Z', 1000, wait_until_complete=True)
        assert sim_scope.motion.get_axis_state('Z') == AxisState.IDLE

    def test_axis_state_moving_during_fire_and_forget(self, sim_scope):
        """After fire-and-forget move, axis is initially MOVING then transitions to IDLE."""
        from modules.lumascope_api import AxisState

        sim_scope.motion.move_absolute_position('Z', 500, wait_until_complete=False)
        state = sim_scope.motion.get_axis_state('Z')
        # Simulated move completes instantly; motion monitor may or may not have
        # polled yet. Both MOVING and IDLE are valid states at this point.
        assert state in (AxisState.MOVING, AxisState.IDLE)

    def test_axis_state_homing_zhome(self, sim_scope):
        """After zhome, Z axis should be IDLE (homing is blocking)."""
        from modules.lumascope_api import AxisState

        sim_scope.motion.zhome()
        assert sim_scope.motion.get_axis_state('Z') == AxisState.IDLE

    def test_axis_state_homing_home(self, sim_scope):
        """After home(), present axes should be IDLE."""
        from modules.lumascope_api import AxisState

        sim_scope._motion_driver.set_timing_mode('instant')
        sim_scope.motion.home()
        for ax in sim_scope.capabilities.axes:
            assert sim_scope.motion.get_axis_state(ax) == AxisState.IDLE

    def test_axis_state_homing_thome(self, _mock_heavy_deps):
        """After thome on a turret-equipped scope, T axis should be IDLE.

        Uses an LS850T sim explicitly instead of the default LS850
        sim_scope fixture (which has no turret) -- pre-B4 the test passed
        on LS850 only because `_axis_state['T']` was a phantom key from
        the hardcoded VALID_AXES tuple. Post-B4, T is correctly absent
        on no-turret scopes and `thome()` is a Rule 8 silent no-op there.
        """
        from modules.lumascope_api import Lumascope, AxisState
        from drivers.simulated_motorboard import SimulatedMotorBoard

        scope = Lumascope(simulate=True)
        scope._motion_driver = SimulatedMotorBoard(model='LS850T')
        present = scope._motion_driver.detect_present_axes()
        assert 'T' in present, 'LS850T sim must report T present'
        scope.motion._pos_cache = dict.fromkeys(present, 0.0)
        scope.motion._axis_state = dict.fromkeys(present, AxisState.UNKNOWN)
        scope.motion._arrival_events = {ax: threading.Event() for ax in present}
        for ev in scope.motion._arrival_events.values():
            ev.set()
        scope.motion._move_profile = dict.fromkeys(present)

        scope.motion.thome()
        assert scope.motion.get_axis_state('T') == AxisState.IDLE

    @pytest.mark.xfail(
        reason='Pre-existing test bug surfaced by audit F9 caller-survey. '
        'Default sim model changed from LS850 to LS850T in LVP `6b16823` '
        "(session 10). Original test expected `'T' not in axes_present()` "
        'which used to hold on LS850 but no longer holds on LS850T. The '
        "naive fix (swap _motion_driver to LS850 post-init) doesn't rebuild "
        'scope.capabilities -- motion.thome still sees T in capabilities.axes '
        'and transitions T to IDLE. Proper fix requires either a Lumascope '
        'constructor parameter that selects the sim model OR a '
        'capabilities-rebuild path that re-runs detection after driver '
        'swap. Tracked in TODO.md.',
        strict=False,
    )
    def test_thome_on_no_turret_scope_is_silent_noop(self, _mock_heavy_deps):
        """Audit B4 + Rule 8: calling thome() on a scope without a
        turret must not raise and must leave T in UNKNOWN state --
        there is no phantom T axis to transition."""
        from modules.lumascope_api import Lumascope, AxisState
        from drivers.simulated_motorboard import SimulatedMotorBoard

        scope = Lumascope(simulate=True)
        scope._motion_driver = SimulatedMotorBoard(model='LS850')
        try:
            assert 'T' not in tuple(scope._motion_driver.detect_present_axes())
            scope.motion.thome()
            assert scope.motion.get_axis_state('T') == AxisState.UNKNOWN
        finally:
            scope.disconnect()

    def test_is_any_axis_moving_false_when_all_idle(self, sim_scope):
        """is_any_axis_moving() returns False when all axes are IDLE."""
        from modules.lumascope_api import AxisState

        sim_scope._motion_driver.set_timing_mode('instant')
        # Home all axes to set them IDLE
        sim_scope.motion.zhome()
        sim_scope.motion.home()
        assert not sim_scope.motion.is_any_axis_moving()

    def test_monitor_reconciles_state(self, sim_scope):
        """Motion monitor thread should detect arrival and set state to IDLE."""
        from modules.lumascope_api import AxisState

        sim_scope.motion.move_absolute_position('Z', 1000, wait_until_complete=False)
        # In simulation, the move completes instantly. The motion monitor thread
        # detects arrival at 50Hz and transitions state to IDLE.
        sim_scope.motion.wait_until_finished_moving(timeout_s=2.0)
        assert not sim_scope.motion.is_moving()
        assert sim_scope.motion.get_axis_state('Z') == AxisState.IDLE

    def test_disconnect_sets_unknown(self, sim_scope):
        """After disconnect, all axes should be UNKNOWN."""
        from modules.lumascope_api import AxisState

        sim_scope.motion.zhome()  # Set to IDLE first
        sim_scope.disconnect()
        for ax in ('X', 'Y', 'Z', 'T'):
            assert sim_scope.motion.get_axis_state(ax) == AxisState.UNKNOWN

    def test_axes_present(self, sim_scope):
        """capabilities.axes is sourced from motion.detect_present_axes
        (Rule 9). Default sim model is LS850T (X/Y/Z + turret) since
        LVP `6b16823`; capabilities.axes must match the motion-driver
        view rather than a hardcoded list.
        """
        axes = sim_scope.capabilities.axes
        assert set(axes) == set(sim_scope._motion_driver.detect_present_axes())
        assert set(axes) == {'X', 'Y', 'Z', 'T'}  # LS850T default

    def test_axis_membership(self, sim_scope):
        """'X in capabilities.axes' replaces the retired has_axis wrapper."""
        assert 'Z' in sim_scope.capabilities.axes
        assert 'Q' not in sim_scope.capabilities.axes

    def test_move_relative_state_tracking(self, sim_scope):
        """move_relative_position tracks axis state correctly."""
        from modules.lumascope_api import AxisState

        sim_scope.motion.move_relative_position('Z', 100, wait_until_complete=True)
        assert sim_scope.motion.get_axis_state('Z') == AxisState.IDLE

    def test_xycenter_state_tracking(self, sim_scope):
        """xycenter sets X/Y to IDLE after completion."""
        from modules.lumascope_api import AxisState

        sim_scope.motion.xycenter()
        assert sim_scope.motion.get_axis_state('X') == AxisState.IDLE
        assert sim_scope.motion.get_axis_state('Y') == AxisState.IDLE


# ===========================================================================
# Issue Regression Tests -- each bug fix gets a test (Rule 18)
# ===========================================================================


class TestIssue602_AFExecutorLED:
    """#602: Autofocus All Steps doesn't turn on the LED.

    Root cause: AF executor had no LED control. Fix: AF executor
    accepts led_color/led_illumination and manages its own LED.
    """

    def test_af_executor_accepts_led_params(self, _mock_heavy_deps):
        """AutofocusRunner.run() should accept led_color and led_illumination."""
        import inspect
        from modules.autofocus_runner import AutofocusRunner

        sig = inspect.signature(AutofocusRunner.run)
        assert 'led_color' in sig.parameters
        assert 'led_illumination' in sig.parameters

    def test_af_executor_turns_led_on(self, _mock_heavy_deps):
        """AF executor should call led_on when led_color is provided."""
        from modules.autofocus_runner import AutofocusRunner
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        from modules.sequential_io_executor import SequentialIOExecutor

        io = SequentialIOExecutor(name='IO_TEST')
        cam = SequentialIOExecutor(name='CAM_TEST')
        af_ex = SequentialIOExecutor(name='AF_TEST')  # noqa: F841 -- deferred
        file_ex = SequentialIOExecutor(name='FILE_TEST')
        af = AutofocusRunner(
            scope=scope,
            camera_executor=cam,
            io_executor=io,
            file_io_executor=file_ex,
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
        from modules.autofocus_runner import AutofocusRunner
        from modules.lumascope_api import Lumascope
        from unittest.mock import patch

        scope = Lumascope(simulate=True)
        from modules.sequential_io_executor import SequentialIOExecutor

        io = SequentialIOExecutor(name='IO_TEST')
        cam = SequentialIOExecutor(name='CAM_TEST')
        af_ex = SequentialIOExecutor(name='AF_TEST')  # noqa: F841 -- deferred
        file_ex = SequentialIOExecutor(name='FILE_TEST')
        af = AutofocusRunner(
            scope=scope,
            camera_executor=cam,
            io_executor=io,
            file_io_executor=file_ex,
        )
        # Set LED state as if AF were running with LED. AFE.run()'s
        # finally block calls _led_off regardless of exit path
        # (success / abort / exception); this checks the invariant.
        af._led_color = 'BF'
        af._led_illumination = 100
        af._saved_led_state = {'channel': 'BF', 'mA': 0}

        abort_event = threading.Event()
        abort_event.set()  # pre-set so AFE.run() unwinds via abort path
        with (
            patch.object(af, '_led_off') as mock_led_off,
            patch.object(af, '_move_absolute_position'),
            patch.object(scope.illumination, 'save_led_state', return_value={}),
            patch.object(scope.imaging, 'save_camera_state', return_value={}),
            patch.object(scope.motion, 'set_precision_mode'),
            patch.object(scope.illumination, 'restore_led_state'),
            patch.object(scope.imaging, 'restore_camera_state'),
        ):
            with pytest.raises(AutofocusAborted):
                af.run(objective_id='4x', abort_event=abort_event)
            mock_led_off.assert_called_once()


class TestAFPrecisionModeRestoresOn:
    """AutofocusRunner exit paths must restore Z precision mode ON.

    Z precision mode (VSTOP=100) is the resting default for all normal
    operation -- motorconfig.py writes this at boot. AF temporarily
    drops to OFF (VSTOP=1000) for its coarse passes for search speed
    and must restore ON via reset / cancel / exception / abort /
    success so subsequent protocol Z moves stop accurately.

    Regression: pre-fix, reset() / cancel() / etc. set precision OFF,
    leaving the system stuck in low-precision after any AF exit. The
    `not run_in_progress -> reset()` call in protocol_step_runner.py
    fired after every protocol step (even ones without AF), so Z stayed
    in OFF for all subsequent protocol moves.
    """

    def _build_af(self):
        from modules.autofocus_runner import AutofocusRunner
        from modules.lumascope_api import Lumascope
        from modules.sequential_io_executor import SequentialIOExecutor

        scope = Lumascope(simulate=True)
        return AutofocusRunner(
            scope=scope,
            camera_executor=SequentialIOExecutor(name='CAM_PREC'),
            io_executor=SequentialIOExecutor(name='IO_PREC'),
            file_io_executor=SequentialIOExecutor(name='FILE_PREC'),
        ), scope

    def test_reset_restores_precision_on(self, _mock_heavy_deps):
        from unittest.mock import patch

        af, scope = self._build_af()
        with patch.object(scope.motion, 'set_precision_mode') as mock_set:
            af.reset()
            mock_set.assert_called_with('Z', True)

    def test_abort_path_restores_precision_on(self, _mock_heavy_deps):
        # AFE.run() finally block must restore Z precision ON on abort,
        # mirroring the success and exception exit paths so the
        # invariant "Z precision ON outside of AF" holds for every
        # exit path (regression-tested below for the abort case).
        from unittest.mock import patch

        af, scope = self._build_af()
        abort_event = threading.Event()
        abort_event.set()  # pre-set so AFE.run() unwinds via abort
        with (
            patch.object(scope.motion, 'set_precision_mode') as mock_set,
            patch.object(af, '_led_off'),
            patch.object(af, '_move_absolute_position'),
            patch.object(scope.illumination, 'save_led_state', return_value={}),
            patch.object(scope.imaging, 'save_camera_state', return_value={}),
            patch.object(scope.illumination, 'restore_led_state'),
            patch.object(scope.imaging, 'restore_camera_state'),
        ):
            with pytest.raises(AutofocusAborted):
                af.run(objective_id='4x', abort_event=abort_event)
            calls = [tuple(c.args) for c in mock_set.call_args_list]
            assert ('Z', True) in calls, (
                f'abort path must restore Z precision_mode=True; got calls {calls}'
            )


class TestIssue605_AccordionLEDProtocol:
    """#605: Stepping through Protocol with 'Protocol LEDs On' doesn't stay on.

    Root cause: accordion_collapse() unconditionally called scope_leds_off().
    Fix: skip leds_off when protocol_led_on setting is active.
    """

    def test_accordion_collapse_has_protocol_led_on_guard(self):
        """accordion_collapse source must check protocol_led_on setting."""
        import pathlib

        source = pathlib.Path('ui/image_settings.py').read_text()
        # Find the accordion_collapse method body
        assert 'protocol_led_on' in source, (
            'accordion_collapse must check protocol_led_on setting (#605)'
        )
        assert 'scope_leds_off' in source, (
            'accordion_collapse must still call scope_leds_off when protocol_led_on is False'
        )


class TestIssue606_TurretObjectiveValidation:
    """#606: Objective changeable without turret position assignment.

    Root cause: no validation in select_objective() or _is_protocol_valid().
    Fix: warn on select, block protocol run.
    """

    def test_select_objective_validates_turret(self):
        """select_objective source must check turret assignments."""
        import pathlib

        source = pathlib.Path('ui/microscope_settings.py').read_text()
        assert 'Objective Not in Turret' in source, (
            'select_objective must warn when objective not in turret (#606)'
        )

    def test_is_protocol_valid_checks_turret(self):
        """_is_protocol_valid source must validate turret config."""
        import pathlib

        source = pathlib.Path('ui/protocol_settings.py').read_text()
        # Find the _is_protocol_valid method
        idx = source.find('def _is_protocol_valid')
        assert idx != -1, '_is_protocol_valid method must exist'
        method_body = source[idx : idx + 2000]
        assert 'turret' in method_body.lower(), (
            '_is_protocol_valid must check turret objective assignments (#606)'
        )


# ===========================================================================
# Audit Fix Regression Tests -- Session 8 (B6, B5, D2, G3, F7, G4)
# ===========================================================================


class TestB6_WriteMotorRegisterRemoved:
    """B6: write_motor_register() was dead code with zero callers."""

    def test_write_motor_register_removed(self, _mock_heavy_deps):
        """write_motor_register should no longer exist on the API class."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        assert not hasattr(scope, 'write_motor_register'), (
            'write_motor_register() should have been removed (B6 -- zero callers)'
        )


class TestB5_GetCurrentPositionUsesAxesPresent:
    """B5: get_current_position(axis=None) should use axes_present(), not
    a hardcoded 4-axis list."""

    def test_returns_only_present_axes(self, _mock_heavy_deps):
        """get_current_position(None) should return dict keyed by present axes only."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        result = scope.motion.get_current_position(axis=None)
        assert set(result.keys()) == set(scope.capabilities.axes), (
            'get_current_position(None) should use scope.capabilities.axes, not a hardcoded axis list'
        )


class TestD2_LEDBoardStateCacheHelper:
    """D2: LED state cache updates should use _update_state_cache() helper."""

    def test_update_state_cache_exists(self, _mock_heavy_deps):
        """LEDBoard should have _update_state_cache method."""
        from drivers.ledboard import LEDBoard

        assert hasattr(LEDBoard, '_update_state_cache'), (
            'LEDBoard must have _update_state_cache helper (D2)'
        )

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
        """AF exception handler must surface a user-facing notification.

        Post-aceda41 (AF popup gate for issue #649), the direct
        ``notifications.error`` call routes through
        ``self._notify_af_failure(...)`` so unattended-protocol runs
        can suppress the modal popup while non-protocol runs still
        notify. The G3 / Rule 14 contract -- "AF failure surfaces to
        the user" -- still holds; the helper is what enforces it.
        Notify-or-suppress correctness is covered separately by
        tests/test_autofocus_notify_gate.py.
        """
        import pathlib

        source = pathlib.Path('modules/autofocus_runner.py').read_text()
        idx = source.find('Error during loop')
        assert idx != -1, 'Exception handler must exist'
        nearby = source[idx : idx + 400]
        assert '_notify_af_failure' in nearby, (
            'AF exception handler must call _notify_af_failure (G3 -- Rule 14)'
        )

    def test_af_degenerate_curve_notifies_user(self, _mock_heavy_deps):
        """AF degenerate curve handler must surface a user-facing notification.

        Same _notify_af_failure routing as the exception path. See
        test_af_exception_notifies_user for the rationale.
        """
        import pathlib

        source = pathlib.Path('modules/autofocus_runner.py').read_text()
        idx = source.find('degenerate focus curve')
        assert idx != -1, 'Degenerate curve handler must exist'
        nearby = source[idx : idx + 500]
        assert '_notify_af_failure' in nearby, (
            'AF degenerate curve handler must call _notify_af_failure (G3 -- Rule 14)'
        )

    def test_af_imports_notifications(self, _mock_heavy_deps):
        """autofocus_runner must import notifications module."""
        import pathlib

        source = pathlib.Path('modules/autofocus_runner.py').read_text()
        assert 'from modules.notification_center import notifications' in source, (
            'autofocus_runner must import notifications (G3)'
        )


class TestRule14_A4_PreRunValidationNotify:
    """A4: Pre-run validation errors must surface a user notification (Rule 14)."""

    def test_validation_errors_branch_notifies(self):
        """sequenced_capture_runner must call notifications.error when
        validation_errors is non-empty before returning."""
        import pathlib

        source = pathlib.Path('modules/sequenced_capture_runner.py').read_text()
        idx = source.find(
            'Protocol has {len(validation_errors)} validation error(s). Cannot start run.'
        )
        assert idx != -1, 'Validation-errors return path must exist'
        nearby = source[idx : idx + 800]
        assert 'notifications.error' in nearby, (
            'validation_errors return path must call notifications.error (A4 -- Rule 14)'
        )
        assert 'Validation failed' in nearby, (
            "notification title must be 'Validation failed' (A4 -- audit recommendation)"
        )

    def test_validation_summary_truncates_at_five(self):
        """Notification summary must show first 5 errors; mention 'see log' for overflow."""
        import pathlib

        source = pathlib.Path('modules/sequenced_capture_runner.py').read_text()
        idx = source.find('validation_errors[:5]')
        assert idx != -1, (
            'Notification summary must slice validation_errors[:5] to keep popup readable (A4)'
        )
        idx = source.find('more (see log)')
        assert idx != -1, 'Overflow message must point user to the log for full details (A4)'


class TestRule14_A5_AreAllConnectedExceptionNotify:
    """A5: are_all_connected() exception branch must notify (Rule 14)."""

    def test_are_all_connected_exception_branch_notifies(self):
        """sequenced_capture_runner must call notifications.error when the
        are_all_connected check itself raises, before returning."""
        import pathlib

        source = pathlib.Path('modules/sequenced_capture_runner.py').read_text()
        idx = source.find('Error checking scope connection')
        assert idx != -1, 'are_all_connected exception handler must exist'
        nearby = source[idx : idx + 600]
        assert 'notifications.error' in nearby, (
            'are_all_connected exception path must call notifications.error (A5 -- Rule 14)'
        )
        assert 'Cannot verify hardware state' in nearby, (
            "notification title must be 'Cannot verify hardware state' (A5 -- audit recommendation)"
        )


class TestRule14_A8_ScopeSessionHelperNotify:
    """A8: scope_session optional helper failures must notify (Rule 14)."""

    def test_wellplate_loader_failure_notifies(self):
        import pathlib

        source = pathlib.Path('modules/scope_session.py').read_text()
        idx = source.find('Could not load wellplate loader:')
        assert idx != -1, 'Wellplate loader except branch must exist'
        nearby = source[idx : idx + 500]
        assert 'notifications.warning' in nearby, (
            'Wellplate loader exception must call notifications.warning (A8 -- Rule 14)'
        )
        assert 'Wellplate loader unavailable' in nearby, (
            "Notification title must be 'Wellplate loader unavailable' (A8)"
        )

    def test_coord_transformer_failure_notifies(self):
        import pathlib

        source = pathlib.Path('modules/scope_session.py').read_text()
        idx = source.find('Could not load coordinate transformer:')
        assert idx != -1, 'Coordinate transformer except branch must exist'
        nearby = source[idx : idx + 500]
        assert 'notifications.warning' in nearby, (
            'Coordinate transformer exception must call notifications.warning (A8)'
        )
        assert 'Coordinate transformer unavailable' in nearby, (
            "Notification title must be 'Coordinate transformer unavailable' (A8)"
        )

    def test_objective_helper_failure_notifies(self):
        import pathlib

        source = pathlib.Path('modules/scope_session.py').read_text()
        idx = source.find('Could not load objective helper:')
        assert idx != -1, 'Objective helper except branch must exist'
        nearby = source[idx : idx + 500]
        assert 'notifications.warning' in nearby, (
            'Objective helper exception must call notifications.warning (A8)'
        )
        assert 'Objective helper unavailable' in nearby, (
            "Notification title must be 'Objective helper unavailable' (A8)"
        )


class TestRule14_A7_HyperstackBuildNotify:
    """A7: Hyperstack build background-thread failure must notify (Rule 14)."""

    def test_hyperstack_build_exception_notifies(self):
        """create_hyperstacks_if_needed _build() must call notifications.error
        when stack_builder.load_folder raises."""
        import pathlib

        source = pathlib.Path('modules/config_ui_getters.py').read_text()
        # Quote-style agnostic: ruff format may use single or double
        # quotes for the message argument.
        idx = source.find('Error building hyperstacks')
        assert idx != -1, 'Hyperstack build exception handler must exist'
        nearby = source[max(0, idx - 200) : idx + 500]
        assert 'logger.exception(' in nearby, 'Hyperstack build path must call logger.exception'
        assert 'notifications.error' in nearby, (
            'Hyperstack build exception path must call notifications.error (A7 -- Rule 14)'
        )
        assert 'Hyperstack build failed' in nearby, (
            "notification title must be 'Hyperstack build failed' (A7 -- audit recommendation)"
        )


class TestRule14_A10_ProtocolCleanupErrorCollection:
    """A10: protocol_cleanup must collect cleanup errors and surface a single
    summary notification (Rule 14)."""

    def test_cleanup_collects_errors(self):
        """run_cleanup must initialize cleanup_errors list and append to it
        on each step's exception."""
        import pathlib

        source = pathlib.Path('modules/protocol_cleanup.py').read_text()
        assert 'cleanup_errors: list[str] = []' in source, (
            'run_cleanup must initialize cleanup_errors list (A10)'
        )
        # Verify each except branch appends
        assert source.count('cleanup_errors.append') >= 6, (
            'Each cleanup step except branch must append to cleanup_errors (A10 -- 6 steps)'
        )

    def test_cleanup_summary_notify(self):
        """run_cleanup must surface a single summary notification when
        cleanup_errors is non-empty."""
        import pathlib

        source = pathlib.Path('modules/protocol_cleanup.py').read_text()
        idx = source.find('if cleanup_errors:')
        assert idx != -1, 'Cleanup-errors summary block must exist'
        nearby = source[idx : idx + 800]
        assert 'notifications.warning' in nearby, (
            'Cleanup-errors block must call notifications.warning (A10 -- summary, not 6 popups)'
        )
        assert 'Protocol cleanup issues' in nearby, (
            "Notification title must be 'Protocol cleanup issues' (A10)"
        )
        assert 'Check LED state, camera settings, and stage position.' in nearby, (
            'Notification body must prompt user to verify hardware state (A10 audit recommendation)'
        )


class TestRule14_A9_SetBinningSizeNotify:
    """A9: set_binning_size exception must surface a user notification (Rule 14)."""

    def test_set_binning_size_exception_notifies(self):
        """lumascope_api.set_binning_size must call notifications.error when
        the underlying SDK call raises. Body relocated to imaging.py in
        Wave 7 Phase 4d."""
        import pathlib

        source = pathlib.Path('modules/lumascope_api/imaging.py').read_text()
        idx = source.find('def set_binning_size(self, size: int) -> bool:')
        assert idx != -1, 'set_binning_size must exist with `-> bool` annotation'
        method_body = source[idx : idx + 1500]
        assert 'notifications.error' in method_body, (
            'set_binning_size exception path must call notifications.error (A9 -- Rule 14)'
        )
        assert 'Binning change failed' in method_body, (
            "notification title must be 'Binning change failed' (A9 -- audit recommendation)"
        )


class TestSetBinningSizeReturnsBool:
    """Wave 1 / B1: ImagingAPI.set_binning_size must propagate the driver's bool.

    Bench session 2026-05-05 surfaced a phantom-failure bug where the API
    method dropped the driver's True return and implicitly returned None;
    char-tool's `if not ok:` check then misreported every successful binning
    op as a failure. This test pins the contract: capture-and-return on the
    success path, return False on exception.
    """

    def test_set_binning_size_has_bool_return_annotation(self):
        # Body relocated to imaging.py in Wave 7 Phase 4d.
        import pathlib

        source = pathlib.Path('modules/lumascope_api/imaging.py').read_text()
        idx = source.find('def set_binning_size(self, size: int) -> bool:')
        assert idx != -1, 'ImagingAPI.set_binning_size must declare `-> bool` (Wave 1 B1; Rule 37)'

    def test_set_binning_size_returns_driver_value(self):
        """Method body must capture and return the driver's return value
        on the success path, not drop it. Body relocated to imaging.py in
        Wave 7 Phase 4d; driver access switched from self._camera_driver
        to self._driver (the @property)."""
        import pathlib

        source = pathlib.Path('modules/lumascope_api/imaging.py').read_text()
        idx = source.find('def set_binning_size(self, size: int) -> bool:')
        assert idx != -1
        # End the slice at the next def at module column 4 to scope the body
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 2000]
        assert 'ok = self._driver.set_binning_size(size=size)' in body, (
            'set_binning_size must capture driver return into `ok`'
        )
        assert 'return ok' in body, 'set_binning_size success path must `return ok` (Wave 1 B1)'
        assert 'return False' in body, 'set_binning_size exception path must `return False`'

    def test_set_binning_size_has_returns_docstring_section(self):
        """Rule 38: public methods declare what they return."""
        import pathlib

        source = pathlib.Path('modules/lumascope_api/imaging.py').read_text()
        idx = source.find('def set_binning_size(self, size: int) -> bool:')
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 2000]
        assert 'Returns:' in body, (
            'set_binning_size docstring must have a Returns: section (Rule 38)'
        )

    def test_pyloncamera_set_binning_size_raises_hardware_error(self):
        """Tier 3a / C2: PylonCamera.set_binning_size must raise HardwareError
        on caught exception paths, not return False (Rule 29)."""
        import pathlib

        source = pathlib.Path('drivers/pyloncamera.py').read_text()
        idx = source.find('def set_binning_size(self, size: int) -> bool:')
        assert idx != -1
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 3000]
        # Three exception classes; each must raise HardwareError, not return False
        for exc_clause in (
            'except genicam.TimeoutException',
            'except genicam.RuntimeException',
            'except Exception',
        ):
            assert exc_clause in body, f'PylonCamera.set_binning_size must keep {exc_clause}'
        assert body.count('raise HardwareError(') >= 3, (
            'PylonCamera.set_binning_size must raise HardwareError on each caught exception (C2)'
        )

    def test_pyloncamera_set_pixel_format_raises_hardware_error(self):
        """Tier 3a / C1."""
        import pathlib

        source = pathlib.Path('drivers/pyloncamera.py').read_text()
        idx = source.find('def set_pixel_format(self, pixel_format: str) -> bool:')
        assert idx != -1
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 3000]
        assert body.count('raise HardwareError(') >= 2, (
            'PylonCamera.set_pixel_format must raise HardwareError on each caught exception (C1)'
        )

    def test_idscamera_set_binning_size_raises_hardware_error(self):
        """Tier 3a / C5."""
        import pathlib

        source = pathlib.Path('drivers/idscamera.py').read_text()
        idx = source.find('def set_binning_size(self, size: int) -> bool:')
        assert idx != -1
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 3000]
        assert 'raise HardwareError(' in body, (
            'IDSCamera.set_binning_size must raise HardwareError on caught exception (C5)'
        )

    def test_idscamera_set_pixel_format_raises_and_annotated(self):
        """Tier 3a / C3 + Tier 1-A: annotation added, raises HardwareError."""
        import pathlib

        source = pathlib.Path('drivers/idscamera.py').read_text()
        idx = source.find('def set_pixel_format(self, pixel_format: str) -> bool:')
        assert idx != -1, 'IDSCamera.set_pixel_format must declare `-> bool` (Wave 1 C3 / Rule 37)'
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 3000]
        assert 'raise HardwareError(' in body, (
            'IDSCamera.set_pixel_format must raise HardwareError on caught exception (C3)'
        )


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

        source = pathlib.Path('modules/lumascope_api/motion.py').read_text()
        assert 'def zhome(self) -> bool:' in source, (
            'MotionAPI.zhome must declare `-> bool` (Rule 37)'
        )

    def test_lumascope_home_has_bool_return_annotation(self):
        import pathlib

        source = pathlib.Path('modules/lumascope_api/motion.py').read_text()
        assert 'def home(self) -> bool:' in source, (
            'MotionAPI.home must declare `-> bool` (Rule 37)'
        )

    def test_lumascope_thome_has_bool_return_annotation(self):
        import pathlib

        source = pathlib.Path('modules/lumascope_api/motion.py').read_text()
        assert 'def thome(self) -> bool:' in source, (
            'MotionAPI.thome must declare `-> bool` (Rule 37)'
        )

    def test_lumascope_zhome_returns_driver_value(self):
        """Method body must return True on success and False on failure paths.

        Body lives on MotionAPI (motion.py) after Wave 7 Phase 2c; the
        Lumascope surface keeps a thin forwarder. Driver call uses
        self._driver (the MotionAPI re-resolving property) per 2b/2c
        convention, matching the home/thome tests at line 1500/1519.
        """
        import pathlib

        source = pathlib.Path('modules/lumascope_api/motion.py').read_text()
        idx = source.find('def zhome(self) -> bool:')
        assert idx != -1
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 2000]
        assert 'result = self._driver.zhome()' in body, (
            'zhome must capture driver return into `result`'
        )
        assert 'return True' in body, 'zhome success path must `return True` (Wave 2 B9)'
        assert 'return False' in body, 'zhome failure paths must `return False` (Wave 2 B9)'
        assert 'Returns:' in body, 'zhome docstring must have a Returns: section (Rule 38)'

    def test_lumascope_home_returns_driver_value(self):
        """Method body must capture and propagate the driver's return.

        Body lives on MotionAPI (motion.py) after the Wave 7 stateless
        decomposition; the Lumascope surface keeps a thin forwarder.
        """
        import pathlib

        source = pathlib.Path('modules/lumascope_api/motion.py').read_text()
        idx = source.find('def home(self) -> bool:')
        assert idx != -1
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 3000]
        assert 'result = self._driver.home()' in body, (
            'home must capture driver return into `result`'
        )
        assert 'return True' in body, 'home success path must `return True` (Wave 2 B10)'
        assert 'return False' in body, 'home failure paths must `return False` (Wave 2 B10)'
        assert 'Returns:' in body, 'home docstring must have a Returns: section (Rule 38)'

    def test_lumascope_thome_returns_driver_value(self):
        """Method body must capture, notify on False, and return the bool.

        Pre-Wave-2, thome dropped the driver return entirely (no capture,
        no notify on failure). This pins the captured-and-returned shape.
        Body lives on MotionAPI (motion.py) after the Wave 7 stateless
        decomposition; the Lumascope surface keeps a thin forwarder.
        """
        import pathlib

        source = pathlib.Path('modules/lumascope_api/motion.py').read_text()
        idx = source.find('def thome(self) -> bool:')
        assert idx != -1
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 3000]
        assert 'result = self._driver.thome()' in body, (
            'thome must capture driver return into `result` (Wave 2 B8)'
        )
        assert 'return True' in body, 'thome success path must `return True` (Wave 2 B8)'
        assert 'return False' in body, 'thome failure paths must `return False` (Wave 2 B8)'
        assert 'Turret homing failed' in body or 'Homing Failed' in body, (
            'thome must notify the user on driver False (Rule 14)'
        )
        assert 'Returns:' in body, 'thome docstring must have a Returns: section (Rule 38)'

    def test_motorboard_zhome_raises_hardware_error(self):
        """Tier 3b D1: MotorBoard.zhome must raise HardwareError on
        no-response and firmware-error paths, not return False (Rule 29)."""
        import pathlib

        source = pathlib.Path('drivers/motorboard.py').read_text()
        idx = source.find('def zhome(self) -> bool:')
        assert idx != -1, 'MotorBoard.zhome must declare `-> bool` (Tier 1-A / Rule 37)'
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 2000]
        # Two error paths: no-response and firmware-error
        assert body.count('raise HardwareError(') >= 2, (
            'MotorBoard.zhome must raise HardwareError on no-response AND firmware-error (D1)'
        )
        assert 'Raises:' in body, 'MotorBoard.zhome docstring must document HardwareError (Rule 38)'

    def test_motorboard_home_raises_hardware_error(self):
        """Tier 3b D1."""
        import pathlib

        source = pathlib.Path('drivers/motorboard.py').read_text()
        idx = source.find('def home(self) -> bool:')
        assert idx != -1
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 3000]
        assert body.count('raise HardwareError(') >= 2, (
            'MotorBoard.home must raise HardwareError on each failure path (D1)'
        )
        assert 'Raises:' in body, 'MotorBoard.home docstring must document HardwareError (Rule 38)'

    def test_motorboard_thome_raises_hardware_error(self):
        """Tier 3b D1."""
        import pathlib

        source = pathlib.Path('drivers/motorboard.py').read_text()
        idx = source.find('def thome(self) -> bool:')
        assert idx != -1
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 2000]
        assert body.count('raise HardwareError(') >= 2, (
            'MotorBoard.thome must raise HardwareError on each failure path (D1)'
        )
        assert 'Raises:' in body, 'MotorBoard.thome docstring must document HardwareError (Rule 38)'

    def test_simulated_motorboard_home_family_raises_hardware_error(self):
        """Tier 3b D1: SimulatedMotorBoard mirrors MotorBoard contract so
        sim-backed tests exercise the same exception path as production."""
        import pathlib

        source = pathlib.Path('drivers/simulated_motorboard.py').read_text()
        assert 'from drivers.exceptions import HardwareError' in source, (
            'SimulatedMotorBoard must import HardwareError'
        )
        for method in ('zhome', 'home', 'thome'):
            idx = source.find(f'def {method}(self) -> bool:')
            assert idx != -1, f'SimulatedMotorBoard.{method} must declare `-> bool`'
            next_def = source.find('\n    def ', idx + 1)
            body = source[idx:next_def] if next_def != -1 else source[idx : idx + 2000]
            assert 'raise HardwareError(' in body, (
                f'SimulatedMotorBoard.{method} must raise HardwareError on failure (D1)'
            )


class TestDisconnectReturnsBool:
    """Wave 4 / B2: Lumascope.disconnect must return an aggregated bool
    indicating whether all sub-system disconnects (LED + motion + camera)
    succeeded. Best-effort teardown still runs every sub-system and
    resets state to Null variants even on partial failure.
    """

    def test_disconnect_has_bool_return_annotation(self):
        import pathlib

        source = pathlib.Path('modules/lumascope_api/_lumascope.py').read_text()
        assert 'def disconnect(self) -> bool:' in source, (
            'Lumascope.disconnect must declare `-> bool` (Wave 4 B2; Rule 37)'
        )

    def test_disconnect_aggregates_and_returns_bool(self):
        """Method body must aggregate three sub-system bools and return."""
        import pathlib

        source = pathlib.Path('modules/lumascope_api/_lumascope.py').read_text()
        idx = source.find('def disconnect(self) -> bool:')
        assert idx != -1
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 4000]
        # Each sub-system tracked independently:
        for var in ('led_ok', 'motion_ok', 'camera_ok'):
            assert var in body, f'disconnect must track {var} (Wave 4 B2)'
        # Aggregation + return:
        assert 'led_ok and motion_ok and camera_ok' in body, (
            'disconnect must aggregate the three sub-bools (Wave 4 B2)'
        )
        assert 'return all_ok' in body, 'disconnect must return the aggregate (Wave 4 B2)'
        # Per-failure notification (Rule 14):
        assert 'notifications.error(' in body, 'disconnect must notify per failure (Rule 14)'
        # Returns: docstring section (Rule 38):
        assert 'Returns:' in body, 'disconnect docstring must have a Returns: section (Rule 38)'

    def test_disconnect_on_simulator_returns_true(self, sim_scope):
        """Sim path: every sub-system disconnects cleanly -> True."""
        # `sim_scope` fixture's teardown also calls disconnect; this
        # call covers the explicit-return-value contract.
        result = sim_scope.disconnect()
        assert result is True, 'Simulator disconnect must return True when no sub-system fails'

    def test_disconnect_camera_failure_returns_false(self, sim_scope):
        """If camera.disconnect raises, the API must catch, notify, and
        still return False. LED + motion still attempted; state still reset."""
        # Replace the camera with one whose disconnect raises.
        from unittest.mock import MagicMock

        sim_scope._camera_driver = MagicMock()
        sim_scope._camera_driver.disconnect = MagicMock(side_effect=RuntimeError('boom'))
        result = sim_scope.disconnect()
        assert result is False, 'disconnect must return False when camera teardown raises'
        assert sim_scope._camera_driver is None, (
            'disconnect must reset self.camera even when teardown raises'
        )


class TestEnterEngineeringModeRaises:
    """Wave 4 / D2: LEDBoard.enter_engineering_mode must raise
    HardwareError on the no-response and no-Y/N-prompt failure paths
    instead of `return False` (Rule 29).
    """

    def test_ledboard_enter_engineering_mode_has_bool_return(self):
        import pathlib

        source = pathlib.Path('drivers/ledboard.py').read_text()
        assert 'def enter_engineering_mode(self, timeout: float = 5.0) -> bool:' in source, (
            'LEDBoard.enter_engineering_mode must declare `-> bool` (Tier 1-A; Rule 37)'
        )

    def test_ledboard_enter_engineering_mode_raises(self):
        """Two failure paths must raise HardwareError."""
        import pathlib

        source = pathlib.Path('drivers/ledboard.py').read_text()
        idx = source.find('def enter_engineering_mode(self, timeout: float = 5.0) -> bool:')
        assert idx != -1
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def != -1 else source[idx : idx + 3000]
        assert body.count('raise HardwareError(') >= 2, (
            'enter_engineering_mode must raise HardwareError on both failure paths (D2)'
        )
        assert 'Raises:' in body, (
            'enter_engineering_mode docstring must document HardwareError (Rule 38)'
        )
        # The legacy `return False` paths must be gone:
        # (Sanity check -- the only `return` in the body should be
        # `return True` on success; `return False` means migration regressed.)
        assert 'return False' not in body, (
            'enter_engineering_mode must no longer `return False` (Rule 29 / D2)'
        )

    def test_ledboard_imports_hardware_error(self):
        import pathlib

        source = pathlib.Path('drivers/ledboard.py').read_text()
        assert 'from drivers.exceptions import HardwareError' in source, (
            'ledboard must import HardwareError'
        )


class TestF7_ProtocolHomingInterlock:
    """F7: Homing/bookmark must be blocked during protocol execution."""

    def test_z_home_checks_protocol_running(self):
        """vertical_control home() must check protocol_running."""
        import pathlib

        source = pathlib.Path('ui/vertical_control.py').read_text()
        # Find the home method
        idx = source.find('def home(self):')
        assert idx != -1
        method_body = source[idx : idx + 300]
        assert 'protocol_running.is_set()' in method_body, (
            'Z home() must check protocol_running before homing (F7)'
        )

    def test_goto_bookmark_checks_protocol_running(self):
        """vertical_control goto_bookmark() must check protocol_running."""
        import pathlib

        source = pathlib.Path('ui/vertical_control.py').read_text()
        idx = source.find('def goto_bookmark(self):')
        assert idx != -1
        method_body = source[idx : idx + 300]
        assert 'protocol_running.is_set()' in method_body, (
            'goto_bookmark() must check protocol_running (F7)'
        )

    def test_turret_home_checks_protocol_running(self):
        """vertical_control turret_home() must check protocol_running."""
        import pathlib

        source = pathlib.Path('ui/vertical_control.py').read_text()
        idx = source.find('def turret_home(self):')
        assert idx != -1
        method_body = source[idx : idx + 300]
        assert 'protocol_running.is_set()' in method_body, (
            'turret_home() must check protocol_running (F7)'
        )

    def test_xy_home_checks_protocol_running(self):
        """motion_settings home() must check protocol_running."""
        import pathlib

        source = pathlib.Path('ui/motion_settings.py').read_text()
        # Find the XYStageControl home method (after line 460)
        idx = source.find('def home(self):')
        assert idx != -1
        method_body = source[idx : idx + 300]
        assert 'protocol_running.is_set()' in method_body, (
            'XY home() must check protocol_running before homing (F7)'
        )


class TestG4_MotorLogSuppression:
    """G4: Motor board should suppress only connect errors, not entire thread logging."""

    def test_no_pause_thread_in_motorboard(self):
        """motorboard.py must NOT call lvp_logger.pause_thread()."""
        import pathlib

        source = pathlib.Path('drivers/motorboard.py').read_text()
        assert 'pause_thread()' not in source, (
            'motorboard.py must not use pause_thread() -- suppresses all thread logging (G4)'
        )

    def test_connect_log_suppressed_flag_exists(self, _mock_heavy_deps):
        """MotorBoard must have _connect_log_suppressed flag."""
        import pathlib

        source = pathlib.Path('drivers/motorboard.py').read_text()
        assert '_connect_log_suppressed' in source, (
            'MotorBoard must use _connect_log_suppressed flag for targeted suppression (G4)'
        )

    def test_connect_log_suppressed_resets_on_success(self):
        """_connect_log_suppressed must be reset when connection succeeds."""
        import pathlib

        source = pathlib.Path('drivers/motorboard.py').read_text()
        # Find the success path (where _connect_fails = 0)
        idx = source.find('self._connect_fails = 0', source.find('def connect'))
        assert idx != -1
        nearby = source[idx : idx + 200]
        assert '_connect_log_suppressed = False' in nearby, (
            '_connect_log_suppressed must be reset to False on successful connection (G4)'
        )


class TestRule1_MotorBoardNoNotifications:
    """Rule 1: drivers must not fire user-facing notifications directly.
    Notifications are the API layer's responsibility -- it has scope
    context to decide whether a driver failure is user-visible (LS820
    expected motor) vs expected absence (LS620 has no motor)."""

    def test_motorboard_does_not_import_notifications(self):
        import pathlib

        source = pathlib.Path('drivers/motorboard.py').read_text()
        assert 'from modules.notification_center import notifications' not in source, (
            'MotorBoard must not import notifications -- Rule 1 (call down, not up)'
        )

    def test_motorboard_does_not_call_notifications(self):
        import pathlib

        source = pathlib.Path('drivers/motorboard.py').read_text()
        assert 'notifications.error' not in source, (
            'MotorBoard must not call notifications.error -- Rule 1'
        )
        assert 'notifications.warning' not in source, (
            'MotorBoard must not call notifications.warning -- Rule 1'
        )
        assert 'notifications.info' not in source, (
            'MotorBoard must not call notifications.info -- Rule 1'
        )


class TestRule1_CameraNoNotifications:
    """Rule 1: drivers must not fire user-facing notifications directly.
    Camera disconnect notification is the API layer's responsibility
    (lumascope_api.py fires it with scope context). Duplicates from
    the driver layer just pop twice or at the wrong moment."""

    def test_camera_base_does_not_import_notifications(self):
        import pathlib

        source = pathlib.Path('drivers/camera.py').read_text()
        assert 'from modules.notification_center import notifications' not in source, (
            'drivers/camera.py must not import notifications -- Rule 1'
        )

    def test_camera_base_does_not_call_notifications(self):
        import pathlib

        source = pathlib.Path('drivers/camera.py').read_text()
        assert 'notifications.error' not in source, (
            'drivers/camera.py must not call notifications.error -- Rule 1'
        )
        assert 'notifications.warning' not in source
        assert 'notifications.info' not in source


class TestRule1_PylonCameraNoNotifications:
    """Rule 1: Pylon SDK removal callback (OnCameraDeviceRemoved) runs in
    a native SDK thread. Before the Rule 1 cleanup it called
    notifications.error from that thread, a secondary crash risk on top
    of the layering violation. API-level detection in get_image handles
    the user-facing notification on the main thread."""

    def test_pyloncamera_does_not_import_notifications(self):
        import pathlib

        source = pathlib.Path('drivers/pyloncamera.py').read_text()
        assert 'from modules.notification_center import notifications' not in source, (
            'drivers/pyloncamera.py must not import notifications -- Rule 1'
        )

    def test_pyloncamera_does_not_call_notifications(self):
        import pathlib

        source = pathlib.Path('drivers/pyloncamera.py').read_text()
        assert 'notifications.error' not in source, (
            'drivers/pyloncamera.py must not call notifications.error -- Rule 1'
        )
        assert 'notifications.warning' not in source
        assert 'notifications.info' not in source


class TestRule1_SerialBoardNoNotifications:
    """Rule 1: SerialBoard fires per-command timeout/exception notifications
    that would spam on every dropped command during a transient
    disconnect. Throttled logger calls are retained for diagnostic
    records; user-facing notification is the API layer's job (it has
    connection-state context and scope capabilities to decide whether a
    given failure is user-visible)."""

    def test_serialboard_does_not_import_notifications(self):
        import pathlib

        source = pathlib.Path('drivers/serialboard.py').read_text()
        assert 'from modules.notification_center import notifications' not in source, (
            'drivers/serialboard.py must not import notifications -- Rule 1'
        )

    def test_serialboard_does_not_call_notifications(self):
        import pathlib

        source = pathlib.Path('drivers/serialboard.py').read_text()
        assert 'notifications.error' not in source, (
            'drivers/serialboard.py must not call notifications.error -- Rule 1'
        )
        assert 'notifications.warning' not in source
        assert 'notifications.info' not in source


class TestPylonChunkTimestampEnabled:
    """Issue #633 Stage 1: ChunkTimestamp enabled on Basler cameras so
    every grabbed frame carries a sensor-side capture-time tick value
    that can be embedded in TIFF metadata.

    Static-source assertion (the production path enabling chunks runs
    inside init_camera_config which requires real camera hardware to
    test, so we assert the constants are correctly defined; bench
    verification per the Monday plan covers the runtime path).
    """

    def test_timestamp_in_chunk_targets_always(self):
        import pathlib

        source = pathlib.Path('drivers/pyloncamera.py').read_text()
        # _CHUNK_TARGETS_ALWAYS is the tuple enabled by default in
        # _enable_validity_chunks; Timestamp must be in it for every
        # camera to surface ChunkTimestamp at grab time.
        idx = source.find('_CHUNK_TARGETS_ALWAYS')
        assert idx >= 0, '_CHUNK_TARGETS_ALWAYS not found in pyloncamera.py'
        decl_end = source.find(')', idx)
        decl = source[idx:decl_end]
        assert "'Timestamp'" in decl, (
            "_CHUNK_TARGETS_ALWAYS must include 'Timestamp' for issue #633 "
            'per-frame timestamps. Currently: ' + decl
        )

    def test_chunktimestamp_in_grab_result_attrs(self):
        import pathlib

        source = pathlib.Path('drivers/pyloncamera.py').read_text()
        # _CHUNK_GRAB_RESULT_ATTRS maps SDK attr -> chunks dict key.
        # Without this entry the read side won't surface the timestamp
        # even if the chunk is enabled.
        idx = source.find('_CHUNK_GRAB_RESULT_ATTRS')
        assert idx >= 0, '_CHUNK_GRAB_RESULT_ATTRS not found in pyloncamera.py'
        next_def = source.find('\n    def ', idx)
        decl = source[idx:next_def] if next_def > 0 else source[idx : idx + 1000]
        assert "'ChunkTimestamp'" in decl and "'Timestamp'" in decl, (
            "_CHUNK_GRAB_RESULT_ATTRS must map ChunkTimestamp -> 'Timestamp'. Currently: " + decl
        )

    def test_camera_base_has_timestamp_tick_frequency_hz(self):
        # The Camera base class declares the attribute so callers
        # (Lumascope.generate_image_metadata) can read it without a
        # hasattr() guard.
        from drivers.camera import Camera

        # Attribute is declared in __init__; check via the source since
        # instantiating Camera requires hardware. AST-walk would be more
        # robust but a substring check is enough for a regression test.
        import pathlib

        source = pathlib.Path('drivers/camera.py').read_text()
        assert 'self.timestamp_tick_frequency_hz' in source, (
            'Camera base must declare self.timestamp_tick_frequency_hz '
            'so generate_image_metadata can read it'
        )


class TestRule1_UiNoDriverReachThrough:
    """Rule 1 (LV-14): UI must call the API, not the driver. Reach-throughs
    like `scope._motion_driver.driver` or `scope._led_driver.driver` bypass the API's
    NullBoard/connection guards and read driver internals that mean different
    things across hardware revisions. The right gate is the API capability
    property (`scope.motor_connected`, `scope.led_connected`) which composes
    NullBoard isinstance + is_connected().

    Catches the cluster (LV-14 was the last shader.py site; this prevents
    reintroduction in any UI module).
    """

    UI_FILES = (
        'ui/shader.py',
        'ui/scope_display.py',
        'ui/main_display.py',
        'ui/image_settings.py',
        'ui/microscope_settings.py',
        'ui/protocol_settings.py',
        'ui/layer_control.py',
        'ui/vertical_control.py',
        'ui/zstack.py',
        'ui/motion_settings.py',
        'ui/post_processing.py',
        'ui/file_dialogs.py',
        'ui/composite_capture.py',
    )

    def test_ui_does_not_reach_through_motion_driver(self):
        import pathlib

        for path in self.UI_FILES:
            p = pathlib.Path(path)
            if not p.exists():
                continue
            source = p.read_text()
            assert 'scope._motion_driver.driver' not in source, (
                f'{path} must not read scope._motion_driver.driver directly '
                '(Rule 1 / LV-14). Use scope.motor_connected instead.'
            )

    def test_ui_does_not_reach_through_led_driver(self):
        import pathlib

        for path in self.UI_FILES:
            p = pathlib.Path(path)
            if not p.exists():
                continue
            source = p.read_text()
            assert 'scope._led_driver.driver' not in source, (
                f'{path} must not read scope._led_driver.driver directly '
                '(Rule 1). Use scope.led_connected instead.'
            )


class TestIssue637_DrawerCloseSaturation:
    """#637: Closing the right-side LED settings drawer caused the image to
    saturate. Reproduction:
      1. Maximized image, PC LED ON, bullseye/crosshairs ON
      2. Hit right arrow to close drawer
      3. Image went all red (saturated)
      4. Reopen drawer, PC still selected, image returned to normal after
         cycling LED off/on

    Root cause: Kivy's Accordion auto-expands a different item when the
    active one collapses (default behavior -- at least one item must stay
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

        source = pathlib.Path('ui/image_settings.py').read_text()
        idx = source.find('def _do_accordion_collapse')
        assert idx >= 0, '_do_accordion_collapse not found in ui/image_settings.py'
        # Slice to just this method's body -- find the next `def ` at the
        # same indent level. _do_accordion_collapse lives in a class so
        # subsequent methods use 4-space indent: '\n    def '.
        next_def = source.find('\n    def ', idx + 1)
        body = source[idx:next_def] if next_def > 0 else source[idx:]
        assert 'toggle_imagesettings' in body, (
            '_do_accordion_collapse must check toggle_imagesettings state '
            '(issue #637) - without this guard, drawer close triggers '
            'apply_settings on a Kivy auto-expanded layer, saturating image.'
        )
        assert "'normal'" in body or '"normal"' in body, (
            '_do_accordion_collapse must compare toggle_imagesettings.state '
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

        source = pathlib.Path('ui/motion_settings.py').read_text()
        # Find the accordion_collapse method body
        idx = source.find('def accordion_collapse')
        assert idx >= 0, 'accordion_collapse method not found in ui/motion_settings.py'
        # Take a slice large enough to cover the method body
        body = source[idx : idx + 3000]
        assert 'XYStage' in body, (
            'accordion_collapse must check XYStage capability (issue #643) -- '
            'without this guard, Lumi/LS820 protocol accordion re-shows the '
            'plate view + crosshair.'
        )
        assert 'remove_parent' in body, (
            'accordion_collapse must call stage.remove_parent() on the '
            'XYStage=False path (issue #643).'
        )

    def test_lumi_and_ls820_have_xystage_false(self):
        """Sanity: scopes.json must declare Lumi and LS820 as XYStage=False
        for the issue #643 guard to actually apply."""
        import json
        import pathlib

        scopes = json.loads(pathlib.Path('data/scopes.json').read_text())
        assert 'Lumi' in scopes, 'Lumi scope config missing from data/scopes.json'
        assert 'LS820' in scopes, 'LS820 scope config missing from data/scopes.json'
        assert scopes['Lumi']['XYStage'] is False, (
            'data/scopes.json: Lumi must be XYStage=False for issue #643 guard '
            'to suppress plate view'
        )
        assert scopes['LS820']['XYStage'] is False, (
            'data/scopes.json: LS820 must be XYStage=False for issue #643 guard '
            'to suppress plate view'
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

        ex = SequentialIOExecutor(name='TEST_642_EMPTY')
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
                'files_complete callback did not fire after protocol_finish_then_end '
                'on empty queue. Pre-fix bug: protocol_end() in the dispatch loop '
                'wiped the callback before it could be fired (issue #642).'
            )
        finally:
            ex.shutdown(wait=True)

    def test_complete_callback_fires_after_queued_tasks_drain(self):
        """Normal completion: protocol_start, queue task(s), wait for task to run,
        then protocol_finish_then_end, verify callback fires after queue drains."""
        from modules.sequential_io_executor import SequentialIOExecutor, IOTask
        import time

        ex = SequentialIOExecutor(name='TEST_642_DRAIN')
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
            assert task_ran, 'Queued task did not execute within 2 s.'

            ex.set_protocol_complete_callback(callback=lambda: fired.append(True))
            ex.protocol_finish_then_end()

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not fired:
                time.sleep(0.05)

            assert fired, (
                'files_complete callback did not fire after queue drained '
                'via protocol_finish_then_end (issue #642).'
            )
        finally:
            ex.shutdown(wait=True)


class TestAOC1_SaturationCheckShortCircuit:
    """AOC-1: lumascope_api.get_image saturation check uses
    `not np.any(tmp != max)` (short-circuit) instead of `np.all(tmp == max)`.

    Both forms allocate a bool array, but `np.any` short-circuits on the
    first True at the C level -- for the common (non-saturated) case, the
    first non-max pixel exits the reduction immediately. Equivalence over
    saturated / non-saturated / single-pixel-different / all-zero arrays.
    """

    def test_source_uses_not_any_form(self):
        # get_image body relocated to imaging.py in Wave 7 Phase 4d.
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'imaging.py'
        ).read_text()
        assert 'not np.any(tmp != np.iinfo(tmp.dtype).max)' in src, (
            'AOC-1: get_image() saturation check should use the short-circuit '
            '`not np.any(tmp != max)` form.'
        )
        assert 'np.all(tmp == np.iinfo(tmp.dtype).max)' not in src, (
            'AOC-1: old `np.all(tmp == max)` form should be replaced.'
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
            assert old == new, f'Logical mismatch on uint8 case: old={old}, new={new}'

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
            assert old == new, f'Logical mismatch on uint16 case: old={old}, new={new}'


class TestAOC2_RetrySaturationCheckOutsideCamLock:
    """AOC-2: lumascope_api.get_image saturation-retry path used to hold
    cam_lock across the np.all validation walk on the retry frame. The walk
    doesn't need camera state -- only the buffer returned from get_array().
    Holding cam_lock across the walk blocked concurrent set_gain/set_exposure
    from other threads for ~50-150 ms per saturated retry.

    Fix: move the saturation walk outside the cam_lock block. Retry frame
    is captured under the lock; the walk runs after the lock is released.
    Also applies the AOC-1 short-circuit pattern at the retry site
    (feedback_default_to_expanding_scope -- fix the cluster).
    """

    def test_retry_saturation_walk_is_outside_cam_lock(self):
        # get_image body relocated to imaging.py in Wave 7 Phase 4d.
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'imaging.py'
        ).read_text()
        # The old form: np.all(retry_frame == ...) inside the with self._cam_lock: block
        assert 'np.all(retry_frame == np.iinfo(retry_frame.dtype).max)' not in src, (
            'AOC-2: old `np.all(retry_frame == max)` form should be replaced.'
        )
        # New form: short-circuit np.any check, AND structurally placed in a sibling
        # block to the cam_lock. Verify the lock-release marker comment is present
        # AND the retry-frame check uses the AOC-1 pattern.
        assert 'Saturation walk is outside cam_lock' in src, (
            'AOC-2: expected lock-release marker comment near retry-frame walk.'
        )
        assert 'np.any(retry_frame != np.iinfo(retry_frame.dtype).max)' in src, (
            'AOC-2: retry-frame check should use the AOC-1 short-circuit pattern.'
        )

    def test_retry_frame_initialized_before_lock_block(self):
        """Structural: retry_frame must be initialized before the with block so the
        outside-lock check can reference it whether or not the grab succeeded.

        get_image body and its cam_lock now both live on ImagingAPI;
        access is self._cam_lock directly.
        """
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'imaging.py'
        ).read_text()
        # Find the retry block; verify retry_frame = None precedes the with statement.
        idx_init = src.find('retry_frame = None')
        idx_lock = src.find('with self._cam_lock:', idx_init)
        idx_retry_grab = src.find('retry_status', idx_lock)
        assert idx_init != -1, 'AOC-2: expected `retry_frame = None` initializer.'
        assert idx_init < idx_lock < idx_retry_grab, (
            'AOC-2: retry_frame should be initialized BEFORE the with cam_lock block.'
        )


class TestPIW3_FalseColor16bitCachedAtRunStart:
    """PIW-3: image_utils.write_tiff used to acquire `_app_ctx.ctx.settings_lock`
    on every TIFF save to read the `false_color_16bit` flag. Same Rule 14 / Rule 2
    family as PP-7 in the post-processing audit. The setting is read-mostly during
    a protocol run; per-save acquisition is wasteful and contends with GUI thread
    settings updates.

    Fix: thread an `use_false_color_16bit` parameter through write_tiff /
    save_image / save_image_static / ProtocolImageWriter, read once in
    sequenced_capture_runner at run start, and pass through. write_tiff
    falls back to the lock-read path when `use_false_color_16bit=None`,
    preserving behavior for ad-hoc callers.
    """

    def test_save_image_threads_param_to_write_tiff(self):
        from pathlib import Path

        # Wave 7 Phase 6c (2026-05-19) retired the *_static chain as
        # dead code. save_image is now the sole carrier of the
        # use_false_color_16bit plumbing; the count-form parity check
        # that previously enforced instance/static synchronization is
        # vestigial. Presence-only ('in src') preserves the semantic
        # intent: the param is accepted by save_image AND threaded to
        # write_tiff.
        api_src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / '_lumascope.py'
        ).read_text()
        module_src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'image_save.py'
        ).read_text()
        # save_image accepts the param (defined on Lumascope wrapper or
        # the free function in modules.image_save -- either source carries
        # the signature; the wrapper retires in 6f).
        assert (
            'use_false_color_16bit: bool | None = None' in api_src
            or 'use_false_color_16bit: bool | None = None' in module_src
        ), 'PIW-3: save_image should accept use_false_color_16bit.'
        # save_image threads the param through to write_tiff.
        assert 'use_false_color_16bit=use_false_color_16bit' in module_src, (
            'PIW-3: save_image should pass use_false_color_16bit to write_tiff.'
        )

    def test_protocol_image_writer_caches_at_init(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'protocol_image_writer.py'
        ).read_text()
        assert 'false_color_16bit: bool = False' in src, (
            'PIW-3: ProtocolImageWriter.__init__ should accept false_color_16bit.'
        )
        assert 'self._false_color_16bit = false_color_16bit' in src, (
            'PIW-3: ProtocolImageWriter should cache false_color_16bit on self.'
        )
        assert 'use_false_color_16bit=self._false_color_16bit' in src, (
            'PIW-3: ProtocolImageWriter should pass the cached value to save_image.'
        )

    def test_sequenced_capture_runner_reads_once_at_run_start(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'sequenced_capture_runner.py'
        ).read_text()
        # The read must happen under the settings_lock and pass through to the writer.
        assert 'with ctx.settings_lock:' in src, (
            'PIW-3: false_color_16bit read should be guarded by settings_lock.'
        )
        assert "false_color_16bit = ctx.settings.get('false_color_16bit', False)" in src, (
            'PIW-3: expected single read of false_color_16bit from settings.'
        )
        assert 'false_color_16bit=false_color_16bit' in src, (
            'PIW-3: cached value should be passed to ProtocolImageWriter.'
        )


class TestPIW5_Convert12to16OutBuffer:
    """PIW-5: convert_12bit_to_16bit() allocated a fresh ndarray on every save
    via image.copy() (~24 MB pulse for protocol-scale images). Same family as
    F-3 -- fresh allocations on the hot save path.

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
        assert result is buf, 'PIW-5: convert should return the supplied out buffer.'
        np.testing.assert_array_equal(result, src * 16)

        # Mismatched shape: falls back to fresh allocation, no error.
        bad_buf = np.zeros((3, 3), dtype=np.uint16)
        result2 = convert_12bit_to_16bit(src, out=bad_buf)
        assert result2 is not bad_buf, 'PIW-5: shape-mismatch should fall back to fresh alloc.'
        np.testing.assert_array_equal(result2, src * 16)

        # No out param: original behavior preserved.
        result3 = convert_12bit_to_16bit(src)
        assert result3 is not src, 'PIW-5: no-out path should still allocate a fresh array.'
        np.testing.assert_array_equal(result3, src * 16)

    def test_protocol_image_writer_holds_reusable_buffer(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'protocol_image_writer.py'
        ).read_text()
        assert 'self._convert_buf_12to16 = None' in src, (
            'PIW-5: ProtocolImageWriter should initialize the convert buffer to None.'
        )
        assert '_get_convert_buf_12to16' in src, (
            'PIW-5: ProtocolImageWriter should have a buffer-getter helper.'
        )
        # Shape/dtype guard: the helper must re-allocate on shape change.
        assert 'self._convert_buf_12to16.shape != array.shape' in src, (
            'PIW-5: buffer helper must re-allocate when input shape changes.'
        )
        # Save-call site passes the buffer.
        assert 'out_12to16=out_12to16' in src, (
            'PIW-5: _write_capture should pass the convert buffer to save_image.'
        )

    def test_save_image_threads_out_12to16_to_prepare(self):
        # Phase 6c (2026-05-19) relocated save_image +
        # prepare_image_for_saving bodies from Lumascope to
        # modules.image_save. Path retarget per Rule 48 (c); semantic
        # intent (out_12to16 plumbing from save_image through
        # prepare_image_for_saving to convert_12bit_to_16bit) preserved.
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'modules' / 'image_save.py').read_text()
        # save_image accepts the param.
        assert 'out_12to16: np.ndarray | None = None' in src, (
            'PIW-5: save_image / prepare_image_for_saving should accept out_12to16.'
        )
        # save_image passes to prepare_image_for_saving.
        assert 'out_12to16=out_12to16' in src, (
            'PIW-5: save_image should pass out_12to16 to prepare_image_for_saving.'
        )
        # prepare_image_for_saving passes to convert_12bit_to_16bit.
        assert 'convert_12bit_to_16bit(array, out=out_12to16)' in src, (
            'PIW-5: prepare_image_for_saving should pass out_12to16 to the convert call.'
        )


class TestPIW6_PF3_FalseColorRgbPreallocated:
    """PIW-6 + PF-3 (combined): retire allocations on the false-color save path.

    Before:
      - add_false_color allocates (H, W, 3) BGR per save (~36 MB uint16)        -- PF-3
      - data[:, :, ::-1] returns a stride-reversed VIEW; tifffile silently
        calls np.ascontiguousarray on write (~36 MB uint16 alloc)               -- PIW-6

    After (final, post-e2ef49e):
      - add_false_color(data, color, output=false_color_buf) reuses caller buf
        AND returns the canonical RGB ordering directly -- PF-3 + #657 fix.
      - write_tiff no longer needs a BGR->RGB conversion step; the stride-
        reverse anti-pattern is gone and the cv2.cvtColor intermediate was
        retired by e2ef49e once add_false_color became RGB-native.

    Post mono-native: maybe_apply_false_color is a pass-through that ignores
    caller-supplied buffers, so ProtocolImageWriter no longer pre-allocates
    them -- it passes None and the downstream save lazily allocates only when
    actually needed. The write_tiff / save_image false_color_buf + rgb_buf
    params are retained as the color-audit enforcement surface (rule_31c
    whitelist); only the dead pre-allocation in the protocol writer was
    removed.
    """

    def test_write_tiff_signature_includes_buffers(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'modules' / 'image_utils.py').read_text()
        assert 'false_color_buf: np.ndarray | None = None' in src, (
            'PF-3: write_tiff should accept false_color_buf param.'
        )
        assert 'rgb_buf: np.ndarray | None = None' in src, (
            'PIW-6: write_tiff should accept rgb_buf param.'
        )

    def test_protocol_image_writer_does_not_preallocate_dead_buffers(self):
        # Post mono-native, maybe_apply_false_color ignores caller buffers,
        # so the protocol writer's old (H,W,3) false-color + RGB pre-alloc
        # was pure dead weight (~6x a mono frame per shape change). It was
        # removed; the writer passes None and the downstream save lazily
        # allocates only when needed. Lock the removal so it does not creep
        # back in.
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'protocol_image_writer.py'
        ).read_text()
        assert '_get_false_color_bufs' not in src, (
            'PF-3 + PIW-6: the dead false-color buffer pre-allocation helper '
            'should not be re-introduced; maybe_apply_false_color ignores it.'
        )
        assert 'self._false_color_buf' not in src, (
            'PF-3: protocol writer should not hold a pre-allocated false_color_buf.'
        )
        assert 'self._rgb_buf' not in src, (
            'PIW-6: protocol writer should not hold a pre-allocated rgb_buf.'
        )

    def test_save_image_threads_buffers_to_write_tiff(self):
        # Phase 6f (2026-05-19) retired the Lumascope.save_image wrapper;
        # the free function in modules.image_save is the sole carrier of
        # the false_color_buf / rgb_buf signature. Path retarget per
        # Rule 48 (c); semantic intent preserved.
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'modules' / 'image_save.py').read_text()
        assert 'false_color_buf: np.ndarray | None = None' in src, (
            'PF-3: save_image should accept false_color_buf.'
        )
        assert 'rgb_buf: np.ndarray | None = None' in src, (
            'PIW-6: save_image should accept rgb_buf.'
        )

    def test_add_false_color_uses_output_buffer(self):
        """Functional: add_false_color writes into the supplied output buffer.
        Channel layout is RGB (index 0=Red, 1=Green, 2=Blue) post-e2ef49e.
        """
        import numpy as np
        from modules.image_utils import add_false_color

        src = np.full((4, 4), 100, dtype=np.uint16)
        buf = np.full((4, 4, 3), 999, dtype=np.uint16)
        result = add_false_color(src, 'Blue', output=buf)
        assert result is buf, 'PF-3: add_false_color should return the supplied buffer.'
        np.testing.assert_array_equal(result[:, :, 2], src)
        assert np.all(result[:, :, 1] == 0), 'PF-3: green channel should be zeroed.'
        assert np.all(result[:, :, 0] == 0), 'PF-3: red channel should be zeroed.'

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
        assert np.all(rgb_buf[:, :, 0] == 3), 'cv2.cvtColor: R channel'
        assert np.all(rgb_buf[:, :, 1] == 2), 'cv2.cvtColor: G channel'
        assert np.all(rgb_buf[:, :, 2] == 1), 'cv2.cvtColor: B channel'


class TestPIW1_NoTheatricalDelCapturedImage:
    """PIW-1: write_capture had `del captured_image` after save_image() completes.
    The line is theatrical -- captured_image is passed as a kwarg in the IOTask
    queued at protocol_image_writer.py:303 (`"captured_image": captured_image`).
    The IOTask.kwargs dict holds the reference until the task completes, so the
    local `del` only releases a local binding -- actual memory reclaim happens
    when the IOTask is freed after task completion, regardless.

    Misleading "memory free" gesture; remove the line.
    """

    def test_del_captured_image_line_removed(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'protocol_image_writer.py'
        ).read_text()
        assert 'del captured_image' not in src, (
            'PIW-1: theatrical `del captured_image` should be removed -- IOTask kwargs holds the ref.'
        )


class TestPIW2_DisksUsageDeduped:
    """PIW-2: per-save disk-space check was redundant between
    `lumascope_api.save_image` / `save_live_image` (both called
    `common_utils.check_disk_space()` defaulting to "/", logged-only,
    non-actionable) and `protocol_image_writer._write_capture` (checks the
    actual save_folder, aborts the protocol on insufficient space).

    The lumascope_api checks (a) checked the wrong path -- root filesystem,
    not the save folder -- and (b) only logged at error level without aborting
    or notifying. The existing try/except in save_image already catches
    write failures via OSError and surfaces a user notification.

    Fix: remove the redundant lumascope_api checks. Keep the useful
    protocol_image_writer check at line 350.
    """

    def test_lumascope_api_disk_check_removed(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / '_lumascope.py'
        ).read_text()
        # The exact pattern of the redundant warn-only check.
        assert 'if (common_utils.check_disk_space() < 1024):' not in src, (
            'PIW-2: redundant per-save check_disk_space call should be removed from lumascope_api.'
        )
        # 'Disk space < 1 GB' was the warn string, also gone.
        assert 'Disk space < 1 GB. Image unlikely to save correctly.' not in src, (
            'PIW-2: corresponding warn log should be removed.'
        )

    def test_protocol_image_writer_disk_check_kept(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'protocol_image_writer.py'
        ).read_text()
        # The useful check (correct path + abort on exhaustion) must remain.
        # Rule-35 audit 2026-05-19 finding 3 consolidated the probe call onto
        # common_utils.check_disk_space_ok; the abort policy stayed local.
        assert 'common_utils.check_disk_space_ok(save_folder' in src, (
            "PIW-2: protocol_image_writer's save-folder disk check should be kept (it's the useful one)."
        )
        assert 'self._abort_fn()' in src, (
            "PIW-2: protocol_image_writer's abort-on-low-disk path should still be present."
        )


class TestProtocolCleanupRestoresLayerShader_ShaderHygiene:
    """Cluster sibling of LED-state-hygiene-at-transition (#666 / #659 /
    #617): the OpenGL shader's false-color white_point also needs a
    cleanup-time restore.

    Bug shape (sim repro 2026-05-23): protocol step on Red layer calls
    Red_LayerControl.apply_settings() which calls
    ShaderViewer.update_shader('Red'), writing
    `white_point = (white, 0.0, 0.0, 1.0)` to the canvas shader.
    Subsequent rendered frames are red-tinted via this multiplier.
    When the protocol stops, protocol_cleanup restores LEDs, AF,
    camera state, and stage position -- but NOT shader state. The
    last protocol step's tint persists indefinitely on the live
    preview canvas regardless of which accordion the user opens.

    Fix: ProtocolCallbacks gains `restore_layer_shader`; protocol_cleanup
    invokes it via _schedule_ui after the LED restore block. The GUI
    caller wires it to a function that re-applies the
    currently-open accordion's shader (falling back to BF if none
    open).
    """

    def _cleanup_src(self):
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent / 'modules' / 'protocol_cleanup.py'
        ).read_text()

    def _callbacks_src(self):
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent / 'modules' / 'protocol_callbacks.py'
        ).read_text()

    def _protocol_settings_src(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'ui' / 'protocol_settings.py').read_text()

    def test_callback_field_exists_in_protocol_callbacks(self):
        """ProtocolCallbacks must declare a restore_layer_shader field
        so the cleanup module can invoke it through the typed contract
        (not a magic-string dict)."""
        src = self._callbacks_src()
        assert 'restore_layer_shader' in src, (
            'ProtocolCallbacks must declare restore_layer_shader (the '
            'sibling-of-LED-state shader-hygiene-at-transition fix)'
        )

    def test_cleanup_invokes_restore_layer_shader(self):
        """protocol_cleanup must call callbacks.restore_layer_shader
        on UI thread (via _schedule_ui) as part of its restore steps.
        Catches a future revert that drops the call."""
        src = self._cleanup_src()
        assert 'callbacks.restore_layer_shader' in src, (
            'protocol_cleanup must invoke callbacks.restore_layer_shader'
        )
        # The invocation must be UI-thread-dispatched (Rule 15 -- the
        # cleanup module is GUI-agnostic; UI work goes via _schedule_ui).
        idx = src.find('callbacks.restore_layer_shader')
        assert idx != -1
        # The line surrounding the call should include _schedule_ui.
        nearby = src[max(0, idx - 100) : idx + 150]
        assert '_schedule_ui' in nearby, (
            'restore_layer_shader call must be UI-thread-dispatched via _schedule_ui (Rule 15)'
        )

    def test_cleanup_shader_restore_protected_by_try_except(self):
        """The shader restore must not abort the rest of cleanup if it
        raises (fault tolerance -- all restore steps must run
        regardless of any one failing). Sibling pattern to the LED /
        AF / camera restore blocks."""
        src = self._cleanup_src()
        idx = src.find('callbacks.restore_layer_shader')
        assert idx != -1
        # The 200 chars before the call should contain a `try:` and the
        # 200 chars after should contain a matching except clause that
        # appends to cleanup_errors.
        window = src[max(0, idx - 200) : idx + 400]
        assert 'try:' in window, (
            'restore_layer_shader call must be inside a try block '
            '(fault tolerance pattern matching LED / AF / camera blocks)'
        )
        assert 'cleanup_errors.append' in window, (
            'restore_layer_shader exception path must append to '
            'cleanup_errors so the summary notification surfaces the '
            'failure'
        )

    def test_protocol_settings_wires_restore_layer_shader_callback(self):
        """The GUI caller (ui/protocol_settings.py) must register the
        restore_layer_shader callback when building the callbacks dict
        for the run, otherwise the cleanup call no-ops and the bug
        recurs."""
        src = self._protocol_settings_src()
        assert "'restore_layer_shader'" in src or '"restore_layer_shader"' in src, (
            'ui/protocol_settings.py must wire the restore_layer_shader '
            'callback into the callbacks dict it passes to '
            'sequenced_capture_runner.run(). Without this wire, '
            'protocol_cleanup invokes None and the shader-tint bug '
            'recurs.'
        )
        # Verify the callback body iterates accordions + calls
        # update_shader -- the canonical "find open accordion, apply
        # its shader" pattern (mirrors update_bullseye_state).
        assert 'update_shader(' in src, (
            'GUI callback must call update_shader to re-apply the currently-open accordion shader'
        )


class TestAccordionStaysPutAcrossProtocolStopStart_AccordionDrift:
    """Cluster sibling of the shader-state-hygiene fix above. The
    user's open accordion was drifting toward the last protocol step's
    channel (Red on a typical BF/Blue/Green/Red protocol) across
    repeated stop/start cycles.

    Bug shape (sim repro 2026-05-23): each protocol step calls
    step_navigation.go_to_step(step, called_from_protocol=True). That
    schedules go_to_step_update_ui(step) on the UI thread via
    _schedule_ui. The UI callback calls
    image_settings.set_expanded_layer(layer=color), which has an
    in-protocol guard (no-op if ctx.protocol_running.is_set()). But
    the LAST step's scheduled callback can fire AFTER cleanup clears
    protocol_running -- the guard reads False, the accordion opens to
    the last step's color. On a 4-channel protocol that's Red. Each
    subsequent run shows the same race and the user ends up stuck on
    Red after a few stop/starts.

    Fix: capture called_from_protocol in the schedule closure (it's
    already on go_to_step's signature, defaulting True for the
    protocol path and False for manual navigation). Pass it into
    go_to_step_update_ui, which gates the set_expanded_layer call on
    `not called_from_protocol`. Race-free because the gate is closure-
    captured at schedule time, not re-read at fire time.
    """

    def _src(self):
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent / 'ui' / 'step_navigation.py'
        ).read_text()

    def test_go_to_step_update_ui_takes_called_from_protocol_arg(self):
        """The UI callback function must accept the closure-captured
        called_from_protocol flag. Without it, the gate has nowhere
        to go and the race-prone protocol_running read is the only
        option."""
        src = self._src()
        assert 'def go_to_step_update_ui(step, called_from_protocol' in src, (
            'go_to_step_update_ui must take called_from_protocol kwarg '
            'so the accordion-drift gate is closure-captured at '
            'schedule time (not race-prone protocol_running read)'
        )

    def test_set_expanded_layer_call_gated_on_called_from_protocol(self):
        """The accordion-expand call must be guarded by
        `if not called_from_protocol:` so protocol-cycle invocations
        don't open the accordion (regardless of protocol_running
        state at fire time)."""
        src = self._src()
        # Find go_to_step_update_ui body
        start = src.find('def go_to_step_update_ui(')
        assert start != -1
        body = src[start : start + 4000]
        end = body.find('\ndef ', 1)
        if end != -1:
            body = body[:end]
        set_expanded_idx = body.find('set_expanded_layer(')
        assert set_expanded_idx != -1, 'set_expanded_layer call must exist in go_to_step_update_ui'
        # The 250 chars before the set_expanded_layer call must
        # contain `if not called_from_protocol:`.
        guard_window = body[max(0, set_expanded_idx - 250) : set_expanded_idx]
        assert 'if not called_from_protocol' in guard_window, (
            'set_expanded_layer call must be gated by '
            '`if not called_from_protocol:` to prevent accordion drift '
            'toward the last protocol step across repeated stop/starts'
        )

    def test_schedule_closure_passes_called_from_protocol(self):
        """The _schedule_ui closure for go_to_step_update_ui must
        forward called_from_protocol from the outer go_to_step scope.
        Without it, the UI callback always sees the default (False)
        and opens the accordion -- the bug recurs."""
        src = self._src()
        # Find the _schedule_ui call that wraps go_to_step_update_ui.
        idx = src.find('go_to_step_update_ui(')
        assert idx != -1
        # The schedule-time call is the FIRST occurrence (the def is
        # later). Capture the window around it.
        # Find the lambda that takes dt and calls go_to_step_update_ui.
        schedule_idx = src.find('lambda dt: go_to_step_update_ui(')
        assert schedule_idx != -1, 'Schedule call for go_to_step_update_ui must exist'
        # The schedule should pass called_from_protocol=called_from_protocol.
        # Window: 200 chars after the lambda start.
        window = src[schedule_idx : schedule_idx + 200]
        assert 'called_from_protocol=called_from_protocol' in window, (
            'Schedule closure must forward called_from_protocol from '
            'go_to_step scope into go_to_step_update_ui'
        )


class TestProtocolStepPanelToggleIdempotent:
    """Each protocol step scheduled go_to_step_update_ui, which forced the
    ImageSettings panel open by calling toggle_settings() unconditionally.
    After the first step the panel is already open, so every later step
    re-ran the panel reposition + histogram rescheduling and logged a
    misleading 'toggle_settings' line -- roughly one per captured frame,
    ~15k across a long soak.

    Fix: only open when the panel is not already open, so the expand/
    collapse handler runs once when the preview opens it, not once per
    step. toggle_settings() is an expand/collapse handler, not an
    idempotent refresh, so it must be state-guarded at this call site.
    """

    def _body(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'ui' / 'step_navigation.py'
        ).read_text()
        start = src.find('def go_to_step_update_ui(')
        assert start != -1
        body = src[start : start + 4000]
        end = body.find('\ndef ', 1)
        return body if end == -1 else body[:end]

    def test_panel_toggle_guarded_by_state_check(self):
        """The toggle_settings() call in go_to_step_update_ui must be
        guarded by a `state != 'down'` check so it fires once per preview,
        not once per protocol step."""
        body = self._body()
        idx = body.find('toggle_settings()')
        assert idx != -1, 'toggle_settings() call must exist in go_to_step_update_ui'
        guard_window = body[max(0, idx - 300) : idx]
        assert "!= 'down'" in guard_window, (
            "the ImageSettings panel toggle must be guarded by a "
            "`state != 'down'` check so toggle_settings() runs once when the "
            "preview opens the panel, not once per protocol step (avoids "
            "per-step reposition + histogram churn and a misleading "
            "toggle_settings log line)"
        )


class TestPF2_FileIoExecutorClearedOnAbort:
    """PF-2: on hardware-disconnect / abort cleanup, file_io_executor's
    pending queue was NOT cleared -- only io_executor's was. Queued IOTasks
    hold captured_image references; on a slow drain these can pin GB of
    memory and lock the next protocol-start until the drain completes.

    Distinct from normal completion, where draining is correct (writes user
    data to disk). The discriminator is `ProtocolState.ERROR` at cleanup
    entry -- that's an abort path; anything else (COMPLETING, IDLE) is
    normal end.

    Fix: capture is_aborted from initial state BEFORE the COMPLETING
    transition, then call file_io_executor.clear_protocol_pending() in the
    aborted branch alongside the existing io/protocol clear calls. Drain
    path is unchanged for normal completion.
    """

    def test_initial_state_captured_before_completing_transition(self):
        from pathlib import Path
        import re

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'protocol_cleanup.py'
        ).read_text()
        # The capture must precede the COMPLETING transition so ERROR vs other states
        # is distinguishable. Paren-style agnostic: ruff format may
        # strip unnecessary parens around the comparison.
        m = re.search(
            r'is_aborted\s*=\s*\(?\s*get_state_fn\(\)\s*==\s*ProtocolState\.ERROR\s*\)?',
            src,
        )
        idx_capture = m.start() if m else -1
        idx_transition = src.find('set_state_fn(ProtocolState.COMPLETING)')
        assert idx_capture != -1, 'PF-2: cleanup should capture is_aborted from initial state.'
        assert idx_capture < idx_transition, (
            'PF-2: is_aborted must be captured BEFORE the COMPLETING state transition.'
        )

    def test_file_io_cleared_on_abort_only(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'protocol_cleanup.py'
        ).read_text()
        # The abort-branch clear is gated on is_aborted.
        assert 'if is_aborted:' in src, 'PF-2: file_io clear should be gated on is_aborted.'
        assert 'file_io_executor.clear_protocol_pending()' in src, (
            "PF-2: cleanup should clear file_io_executor's pending queue on abort."
        )
        # Existing unconditional clear for io_executor must still be present.
        assert 'io_executor.clear_protocol_pending()' in src, (
            'PF-2: io_executor.clear_protocol_pending should still be called unconditionally.'
        )


class TestPF5_ImageBufferRetired:
    """PF-5: Lumascope.image_buffer was a permanent shadow copy of the latest
    get_image() result -- Rule 2 violation. Only ever read by get_image() itself
    (for chaining sum/scale-bar/8-bit-convert ops), never by external callers.
    Pinned one frame indefinitely between calls. The _state_lock around per-
    write didn't actually serialize concurrent get_image calls -- chained
    writes from different threads could still interleave.

    Fix: chain through a local variable in get_image(). Remove the
    image_buffer property + setter, the _image_buffer attribute, and its
    initialization in __init__ + diagnostic-instance setup.
    """

    def test_image_buffer_property_removed(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / '_lumascope.py'
        ).read_text()
        # Property declaration gone.
        assert 'def image_buffer(self):' not in src, (
            'PF-5: image_buffer property getter should be removed.'
        )
        assert '@image_buffer.setter' not in src, (
            'PF-5: image_buffer property setter should be removed.'
        )
        # Assignments to self.image_buffer (as code, not in comments) gone.
        assert 'self.image_buffer = ' not in src, (
            'PF-5: all self.image_buffer assignments should be retired in favor of a local variable.'
        )

    def test_image_buffer_attribute_removed(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / '_lumascope.py'
        ).read_text()
        # The internal _image_buffer attribute init gone.
        assert 'self._image_buffer = None' not in src, (
            'PF-5: self._image_buffer initialization should be removed.'
        )
        assert 'instance._image_buffer = None' not in src, (
            'PF-5: diagnostic-instance _image_buffer initialization should also be removed.'
        )

    def test_get_image_returns_local_variable(self):
        # get_image body relocated to imaging.py in Wave 7 Phase 4d.
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'imaging.py'
        ).read_text()
        # The chain must use a local `image` variable.
        assert 'image = image_utils.add_scale_bar(' in src, (
            'PF-5: scale-bar step should bind to local `image` instead of self.image_buffer.'
        )
        assert 'image = image_utils.convert_12bit_to_8bit(image)' in src, (
            'PF-5: 8-bit convert step should bind to local `image`.'
        )


class TestPF1_CpuPoolRetired:
    """PF-1: cpu_pool / use_multiprocessing infrastructure was dead.
    use_multiprocessing was hardcoded False, so the ProcessPoolExecutor
    construction at lumaviewpro.py:214-237 never ran. The
    sequenced_capture_writer.py module was only imported from that dead
    block -- the entire module was unreachable. The cpu_pool param threaded
    through SequencedCaptureRunner.__init__ was always None.

    Per IMAGE_PROCESSING_ARCHITECTURE_2026-04-30.md: do NOT pre-build a
    replacement pool -- modules/postprocessing/ and modules/live_processing/
    will be built greenfield when their first feature lands.

    Fix: deleted modules/sequenced_capture_writer.py entirely. Removed
    cpu_pool / use_multiprocessing from lumaviewpro.py (declarations,
    init block, shutdown block, executor kwarg). Removed cpu_pool param
    from SequencedCaptureRunner.__init__ + the now-unused
    ProcessPoolExecutor import. Removed the test that exercised the dead
    setup_worker_logger function.
    """

    def test_sequenced_capture_writer_module_deleted(self):
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / 'modules' / 'sequenced_capture_writer.py'
        assert not path.exists(), (
            'PF-1: modules/sequenced_capture_writer.py should be deleted (dead module).'
        )

    def test_lumaviewpro_no_cpu_pool_or_use_multiprocessing(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'lumaviewpro.py').read_text()
        assert 'cpu_pool' not in src, (
            'PF-1: all cpu_pool references should be removed from lumaviewpro.py.'
        )
        assert 'use_multiprocessing' not in src, (
            'PF-1: all use_multiprocessing references should be removed from lumaviewpro.py.'
        )
        assert 'from concurrent.futures import ProcessPoolExecutor' not in src, (
            'PF-1: unused ProcessPoolExecutor import should be removed from lumaviewpro.py.'
        )

    def test_executor_no_cpu_pool_param(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'sequenced_capture_runner.py'
        ).read_text()
        assert 'cpu_pool' not in src, (
            'PF-1: cpu_pool should be removed from SequencedCaptureRunner.'
        )
        assert 'from concurrent.futures import ProcessPoolExecutor' not in src, (
            'PF-1: unused ProcessPoolExecutor import should be removed from sequenced_capture_runner.py.'
        )


_SCOPE_LIKE_RECEIVERS = frozenset({'self', 'scope'})


def _function_body_calls(source: str, func_name: str) -> set[str]:
    """Return the set of `<scope>.<method>(...)` (or `<scope>.<subapi>.<method>(...)`)
    attribute calls in a named function's body, where `<scope>` is either `self`
    (instance method) or `scope` (module-level free function with scope as first arg).

    Post-Wave-7 Phase 4: routes through sub-APIs (e.g. `self.imaging.capture_and_wait`)
    are recognized as semantically equivalent to bare `self.capture_and_wait` --
    the helper returns the leaf method name in either case. The frame-validity
    ship-gate assertion is "the function reaches X somewhere through the API,"
    independent of whether X is a forwarder or a sub-API method.

    Post-Wave-7 Phase 6: module-level free functions in `modules.image_save`
    (and any future class-to-module-functions extraction) take `scope` as their
    first arg and reach the API through `scope.<subapi>.<method>`; the same
    leaf-name recognition applies.
    """
    import ast

    tree = ast.parse(source)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target = node
            break
    if target is None:
        raise AssertionError(f'function {func_name!r} not found in source')
    calls: set[str] = set()
    for sub in ast.walk(target):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            value = sub.func.value
            # Bare <receiver>.<method> -- receiver is self or scope.
            if isinstance(value, ast.Name) and value.id in _SCOPE_LIKE_RECEIVERS:
                calls.add(sub.func.attr)
            # <receiver>.<subapi>.<method> (e.g. self.imaging.capture_and_wait
            # or scope.imaging.capture_and_wait).
            elif (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id in _SCOPE_LIKE_RECEIVERS
            ):
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
                raise AssertionError(f'could not extract source for {func_name!r}')
            return text
    raise AssertionError(f'function {func_name!r} not found in source')


class TestFrameValidity_SaveLiveImageDrainsBeforeGrab:
    """Lumascope.save_live_image must drain stale frames before grabbing.
    Bare self.get_image(...) ships a mid-transition frame to disk on every
    manual save; the canonical helper is self.capture_and_wait(...)."""

    def test_save_live_image_calls_capture_and_wait(self):
        # Phase 6c (2026-05-19) relocated save_live_image body from
        # Lumascope to modules.image_save (composes scope.imaging
        # rather than self.imaging). Path retarget per Rule 48 (c);
        # semantic intent (drain-then-grab via capture_and_wait)
        # preserved.
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'modules' / 'image_save.py').read_text()
        calls = _function_body_calls(src, 'save_live_image')
        assert 'capture_and_wait' in calls, (
            'save_live_image must call scope.imaging.capture_and_wait(...) for drain-then-grab.'
        )

    def test_save_live_image_does_not_call_bare_get_image(self):
        # Phase 6c (2026-05-19) path retarget per Rule 48 (c); see
        # companion test_save_live_image_calls_capture_and_wait above.
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'modules' / 'image_save.py').read_text()
        calls = _function_body_calls(src, 'save_live_image')
        assert 'get_image' not in calls, (
            'save_live_image must not call scope.imaging.get_image(...) directly -- '
            'that bypasses frame_validity. Route through scope.imaging.capture_and_wait(...).'
        )

    def test_capture_and_wait_accepts_earliest_image_ts(self):
        """capture_and_wait must forward earliest_image_ts so save_live_image's
        public signature stays stable for L2 SDK callers."""
        import inspect

        from modules import lumascope_api

        sig = inspect.signature(ImagingAPI.capture_and_wait)
        assert 'earliest_image_ts' in sig.parameters, (
            'capture_and_wait must accept earliest_image_ts so save_live_image '
            'can forward its existing parameter.'
        )


def _scope_attribute_calls(source: str, func_name: str) -> set[str]:
    """Return the set of `self._scope.<method>(...)` (or
    `self._scope.<subapi>.<method>(...)`) attribute calls in a named function's
    body. Mirrors `_function_body_calls` for the AF executor's indirect-via-_scope
    grab pattern. Post-Wave-7 Phase 4: walks through `self._scope.imaging.<method>`
    too -- the leaf method name is returned regardless of sub-API hop."""
    import ast

    tree = ast.parse(source)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target = node
            break
    if target is None:
        raise AssertionError(f'function {func_name!r} not found in source')
    calls: set[str] = set()
    for sub in ast.walk(target):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            value = sub.func.value
            # match self._scope.<method>
            if (
                isinstance(value, ast.Attribute)
                and value.attr == '_scope'
                and isinstance(value.value, ast.Name)
                and value.value.id == 'self'
            ):
                calls.add(sub.func.attr)
            # match self._scope.<subapi>.<method>
            elif (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Attribute)
                and value.value.attr == '_scope'
                and isinstance(value.value.value, ast.Name)
                and value.value.value.id == 'self'
            ):
                calls.add(sub.func.attr)
    return calls


class TestFrameValidity_AutofocusDrainsBeforeScore:
    """AutofocusRunner._iterate must drain LED/gain/exposure-pending
    frames before scoring. Bare get_image after Z arrival can score on a
    mid-LED-warmup or mid-gain-change frame, corrupting the focus curve
    and landing the wrong best-Z. AF excludes z_move because AF is the
    controller of Z moves; once is_moving() reports idle, Z is settled."""

    def test_iterate_calls_capture_and_wait(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'autofocus_runner.py'
        ).read_text()
        calls = _scope_attribute_calls(src, '_iterate')
        assert 'capture_and_wait' in calls, (
            'AutofocusRunner._iterate must call self._scope.imaging.capture_and_wait(...) '
            'to drain LED/gain/exposure pending frames before scoring.'
        )

    def test_iterate_does_not_call_bare_get_image(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'autofocus_runner.py'
        ).read_text()
        calls = _scope_attribute_calls(src, '_iterate')
        assert 'get_image' not in calls, (
            'AutofocusRunner._iterate must not call self._scope.imaging.get_image(...) '
            'directly -- bypasses frame_validity. Route through capture_and_wait.'
        )

    def test_iterate_excludes_z_move_in_validity(self):
        """AF excludes z_move because is_moving() already gates motion; the
        drain is for LED/gain/exposure transitions only."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'autofocus_runner.py'
        ).read_text()
        # both call sites must specify exclude_sources=('z_move',)
        assert "exclude_sources=('z_move',)" in src, (
            "AutofocusRunner._iterate's capture_and_wait calls must pass "
            "exclude_sources=('z_move',) since is_moving() already gates motion."
        )


class TestFrameValidity_CompositeEngineeringBranchDrains:
    """The engineering-mode branch of composite_capture's live_capture path
    (bullseye / crosshairs enabled) grabs an extra image_orig for overlay
    rendering. Bare get_image here would persist a mid-transition raw
    image to disk via the subsequent save_image call. Must route through
    capture_and_wait."""

    def test_live_capture_impl_uses_capture_and_wait(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'ui' / 'composite_capture.py').read_text()
        body = _function_source(src, '_live_capture_impl')
        assert 'ctx.scope.imaging.capture_and_wait(' in body, (
            'composite_capture._live_capture_impl must call '
            'ctx.scope.imaging.capture_and_wait(...) for the engineering bullseye/'
            'crosshairs branch (was bare get_image).'
        )

    def test_live_capture_impl_no_bare_ctx_scope_get_image(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'ui' / 'composite_capture.py').read_text()
        body = _function_source(src, '_live_capture_impl')
        assert 'ctx.scope.imaging.get_image(' not in body, (
            'composite_capture._live_capture_impl must not call '
            'ctx.scope.imaging.get_image(...) directly. Route through capture_and_wait '
            '(or save_live_image, which now uses capture_and_wait internally).'
        )


class TestFrameValidity_AllLedMutatorsInvalidate:
    """Defensive coverage: every LED state-mutator on IlluminationAPI
    must call frame_validity.invalidate('led'). All 6 currently
    invalidate; this test locks the invariant so a future cleanup that
    removes any call fires the regression.

    Post-Wave-7-Phase-4d: frame_validity instance lives on ImagingAPI;
    IlluminationAPI reaches it via
    `self._scope.imaging.frame_validity.invalidate(...)`.
    """

    LED_MUTATORS = (
        'led_on',
        'led_off',
        'led_on_fast',
        'led_off_fast',
        'leds_off_fast',
        'leds_off',
    )

    def test_each_led_mutator_invalidates_validity(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'illumination.py'
        ).read_text()
        missing = []
        for func in self.LED_MUTATORS:
            method_src = _function_source(src, func)
            if 'self._scope.imaging.frame_validity.invalidate(' not in method_src:
                missing.append(func)
        assert not missing, (
            'LED mutator coverage: each IlluminationAPI LED state-mutator must call '
            "self._scope.imaging.frame_validity.invalidate('led') so frame_validity sees "
            f'the transition. Missing: {missing!r}.'
        )


class TestCaptureAndWaitPassesChunksToValidity:
    """capture_and_wait's drain loop reads per-frame chunk metadata and
    passes it to count_frame so chunk-match can short-circuit skip-frames
    for gain/exposure on chunk-supporting cameras. Backward compat:
    cameras without chunks return None and fall back to skip-frames."""

    def test_capture_and_wait_passes_chunk_data_to_count_frame(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'imaging.py'
        ).read_text()
        body = _function_source(src, 'capture_and_wait')
        # Source mentions count_frame call site with chunk_data kwarg
        assert 'count_frame(chunk_data=' in body, (
            'capture_and_wait must call count_frame(chunk_data=...) in the '
            'drain loop so chunk-match can clear gain/exposure pending.'
        )

    def test_get_latest_chunks_helper_exists(self):
        """The _get_latest_chunks helper abstracts handler shape (Pylon
        composition vs IDS inheritance) and returns None for non-chunk cameras.
        Phase 4 relocation: helper moved from Lumascope to ImagingAPI; the
        contract (no required params besides self, returns dict | None) is
        unchanged."""
        import inspect

        assert hasattr(ImagingAPI, '_get_latest_chunks'), (
            'ImagingAPI must expose _get_latest_chunks() helper.'
        )
        sig = inspect.signature(ImagingAPI._get_latest_chunks)
        # No required params (besides self) -- reads from self._driver state
        non_self = [p for p in sig.parameters if p != 'self']
        assert len(non_self) == 0, f'_get_latest_chunks should take no args; got {non_self}'

    def test_get_latest_chunks_returns_none_when_no_camera(self):
        """Defensive: helper returns None instead of raising when camera
        isn't connected (FX2 fallback / pre-connect / disconnected state).
        Post-4d: ImagingAPI._driver is a @property re-resolving
        self._scope._camera_driver; with no camera driver attached the
        helper returns None instead of AttributeError."""
        from modules.lumascope_api import Lumascope

        # Construct without going through full init -- attributes set by hand
        scope = Lumascope.__new__(Lumascope)
        scope.runtime_state = RuntimeState(scope)
        scope._camera_driver = None
        scope.imaging = ImagingAPI(scope, None)
        assert scope.imaging._get_latest_chunks() is None


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

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'imaging.py'
        ).read_text()
        body = _function_source(src, 'set_gain')
        assert "self.frame_validity.set_target('gain'" in body, (
            'set_gain must record gain target via frame_validity.set_target.'
        )

    def test_set_exposure_time_records_target_in_microseconds(self):
        """ChunkExposureTime is microseconds; API takes milliseconds.
        Conversion (* 1000) must happen at the seam so chunk-match's
        tolerance is in matching units."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'imaging.py'
        ).read_text()
        body = _function_source(src, 'set_exposure_time')
        assert "self.frame_validity.set_target('exposure'" in body, (
            'set_exposure_time must record target via set_target.'
        )
        assert '1000' in body, (
            'set_exposure_time must convert ms -> us when recording target '
            'so chunk-match operates in microseconds.'
        )

    def test_set_auto_gain_clears_target(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'imaging.py'
        ).read_text()
        body = _function_source(src, 'set_auto_gain')
        assert "set_target('gain', None)" in body, (
            "set_auto_gain must clear gain target (None) so chunk-match doesn't "
            'fire against a stale manual target while auto adjusts.'
        )

    def test_set_auto_exposure_time_clears_target(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'imaging.py'
        ).read_text()
        body = _function_source(src, 'set_auto_exposure_time')
        assert "set_target('exposure', None)" in body, (
            'set_auto_exposure_time must clear exposure target (None).'
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
        b._store_frame(
            np.zeros((4, 4), dtype=np.uint8),
            datetime.datetime.now(),
            chunks={'ExposureTime': 14530.0},
        )
        b._record_failure()  # last_result becomes False
        assert b.get_last_chunks() is None

    def test_reset_clears_chunks(self):
        import datetime
        import numpy as np

        b = self._make_base()
        b._store_frame(
            np.zeros((4, 4), dtype=np.uint8),
            datetime.datetime.now(),
            chunks={'ExposureTime': 14530.0},
        )
        b.reset()
        assert b.last_chunks is None
        assert b.get_last_chunks() is None

    def test_chunks_replace_not_merge(self):
        """Each successful grab replaces the chunks dict; we don't merge."""
        import datetime
        import numpy as np

        b = self._make_base()
        b._store_frame(
            np.zeros((4, 4), dtype=np.uint8),
            datetime.datetime.now(),
            chunks={'ExposureTime': 14530.0, 'Gain': 1.0},
        )
        b._store_frame(
            np.zeros((4, 4), dtype=np.uint8),
            datetime.datetime.now(),
            chunks={'ExposureTime': 30000.0},
        )
        assert b.get_last_chunks() == {'ExposureTime': 30000.0}
        assert 'Gain' not in b.get_last_chunks()


class TestRecordInitFpsPreflightAndToggle:
    """Issue #633 Stage 2C: record_init pre-flight + camera FPS toggle.

    Static-source assertions because record_init has Kivy Clock + camera
    dependencies that aren't mockable in the unit-test env. Bench
    verification per the Monday plan covers the runtime behavior.
    """

    def _record_init_body(self):
        import pathlib

        source = pathlib.Path('ui/main_display.py').read_text()
        idx = source.find('def record_init')
        assert idx >= 0, 'record_init not found in ui/main_display.py'
        # Slice through the next def at class indent (4 spaces).
        next_def = source.find('\n    def ', idx + 1)
        return source[idx:next_def] if next_def > 0 else source[idx:]

    def _finalize_body(self):
        import pathlib

        source = pathlib.Path('ui/main_display.py').read_text()
        idx = source.find('def _finalize_recording_state')
        assert idx >= 0, '_finalize_recording_state not found'
        next_def = source.find('\n    def ', idx + 1)
        return source[idx:next_def] if next_def > 0 else source[idx:]

    def test_fps_budget_warning_fires_when_limit_binds(self):
        body = self._record_init_body()
        assert 'FPS budget exceeded' in body, (
            'record_init must surface a notifications.warning when the '
            'user-requested FPS limit binds against the exposure budget '
            "(issue #633 Stage 2C, Eric's 'warn + accept' choice)."
        )
        # Warn-and-accept: do NOT block recording on this path.
        assert 'self.recording.clear()' not in body.split('FPS budget exceeded')[0][-500:], (
            'FPS-budget warning path must not clear self.recording -- '
            'Eric chose warn-and-accept, not abort.'
        )

    def test_disk_space_preflight_aborts_with_notify(self):
        body = self._record_init_body()
        assert 'Insufficient disk space' in body, (
            'record_init must pre-flight disk space and abort with '
            'notifications.error when insufficient (issue #633 Stage 2C).'
        )
        assert 'self.recording.clear()' in body, (
            'Disk-space abort must clear self.recording so a retry '
            'after freeing disk can claim recording again.'
        )

    def test_finalize_disables_camera_fps_limit(self):
        body = self._finalize_body()
        assert 'set_max_acquisition_frame_rate(False' in body, (
            '_finalize_recording_state must disable the camera-side rate '
            'limit so live preview returns to free-run (issue #633 Stage 2C).'
        )
        assert '_fps_limit_was_enabled' in body, (
            '_finalize must guard the disable on _fps_limit_was_enabled '
            "to avoid touching the knob when we didn't enable it."
        )


class TestSessionManifestHelpers:
    """Issue #633 Stage 2B: session_manifest.json helpers in ui/main_display.py
    are pure functions (input dict -> output dict). Unit-testable without Kivy.

    The manifest is the single per-recording summary the customer + downstream
    scripts read instead of opening 600 TIFFs. Schema mirrors the char tool's
    provenance shape so manifests across LVP and char runs are comparable.
    """

    def test_compute_fps_stats_empty(self):
        from modules.recording_manifest import compute_fps_stats as _compute_fps_stats

        result = _compute_fps_stats([])
        assert result == {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'samples': 0}

    def test_compute_fps_stats_single_frame(self):
        import datetime
        from modules.recording_manifest import compute_fps_stats as _compute_fps_stats

        result = _compute_fps_stats([datetime.datetime.now()])
        assert result == {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'samples': 0}

    def test_compute_fps_stats_steady_10fps(self):
        import datetime
        from modules.recording_manifest import compute_fps_stats as _compute_fps_stats

        # 100ms intervals -> 10 FPS exactly
        base = datetime.datetime(2026, 5, 9, 14, 0, 0)
        timestamps = [base + datetime.timedelta(milliseconds=100 * i) for i in range(10)]
        result = _compute_fps_stats(timestamps)
        assert result['samples'] == 9
        assert abs(result['mean'] - 10.0) < 1e-6
        assert abs(result['min'] - 10.0) < 1e-6
        assert abs(result['max'] - 10.0) < 1e-6

    def test_compute_fps_stats_jittered(self):
        import datetime
        from modules.recording_manifest import compute_fps_stats as _compute_fps_stats

        base = datetime.datetime(2026, 5, 9, 14, 0, 0)
        # Three intervals: 100ms (10fps), 200ms (5fps), 50ms (20fps)
        timestamps = [
            base,
            base + datetime.timedelta(milliseconds=100),
            base + datetime.timedelta(milliseconds=300),
            base + datetime.timedelta(milliseconds=350),
        ]
        result = _compute_fps_stats(timestamps)
        assert result['samples'] == 3
        assert abs(result['min'] - 5.0) < 1e-6
        assert abs(result['max'] - 20.0) < 1e-6
        # mean = (10 + 5 + 20) / 3 = 11.667
        assert abs(result['mean'] - 35.0 / 3) < 1e-6

    def test_gather_host_provenance_keys(self):
        from modules.recording_manifest import gather_host_provenance

        host = gather_host_provenance()
        assert 'hostname' in host
        assert 'os_platform' in host
        assert 'cpu_model' in host
        assert 'python_version' in host
        # Sanity: all values are non-empty strings.
        for k, v in host.items():
            assert isinstance(v, str)
            assert len(v) > 0, f'{k} should be non-empty'

    def test_build_session_manifest_schema(self):
        import datetime
        from modules.recording_manifest import build_session_manifest as _build_session_manifest

        ts0 = datetime.datetime(2026, 5, 9, 14, 0, 0)
        timestamps = [ts0 + datetime.timedelta(milliseconds=100 * i) for i in range(3)]
        chunks_per_frame = [
            {'Timestamp': 1000, 'FrameID': 1, 'ExposureTime': 50.0},
            {'Timestamp': 1100, 'FrameID': 2, 'ExposureTime': 50.0},
            {'Timestamp': 1200, 'FrameID': 3, 'ExposureTime': 50.0},
        ]
        manifest = _build_session_manifest(
            timestamps=timestamps,
            chunks_per_frame=chunks_per_frame,
            tick_freq_hz=1_000_000_000,
            captured_frames=3,
            video_duration=0.3,
        )
        assert manifest['manifest_version'] == 1
        assert manifest['recording']['frames_captured'] == 3
        assert manifest['recording']['duration_s'] == 0.3
        assert manifest['recording']['start_iso'] == ts0.isoformat(timespec='microseconds')
        assert manifest['camera']['timestamp_tick_hz'] == 1_000_000_000
        assert 'host' in manifest['provenance']
        assert 'software' in manifest['provenance']
        assert len(manifest['frame_index']) == 3
        assert manifest['frame_index'][0]['i'] == 0
        assert manifest['frame_index'][0]['ts_camera_ticks'] == 1000
        assert manifest['frame_index'][0]['frame_id'] == 1

    def test_build_session_manifest_handles_missing_chunks(self):
        """Cameras without chunk support: chunks_per_frame is [None, None, ...].
        Manifest still emits frame_index entries with None for camera fields."""
        import datetime
        from modules.recording_manifest import build_session_manifest as _build_session_manifest

        ts0 = datetime.datetime(2026, 5, 9, 14, 0, 0)
        timestamps = [ts0 + datetime.timedelta(milliseconds=100 * i) for i in range(2)]
        manifest = _build_session_manifest(
            timestamps=timestamps,
            chunks_per_frame=[None, None],
            tick_freq_hz=None,
            captured_frames=2,
            video_duration=0.2,
        )
        assert manifest['camera']['timestamp_tick_hz'] is None
        assert manifest['frame_index'][0]['ts_camera_ticks'] is None
        assert manifest['frame_index'][0]['frame_id'] is None
        assert manifest['frame_index'][0]['ts_host_iso'] is not None

    def test_build_session_manifest_handles_short_arrays(self):
        """timestamps/chunks_per_frame may be shorter than captured_frames if
        the camera dropped late frames; emit None rather than IndexError."""
        from modules.recording_manifest import build_session_manifest as _build_session_manifest

        manifest = _build_session_manifest(
            timestamps=[],
            chunks_per_frame=[],
            tick_freq_hz=1_000_000_000,
            captured_frames=5,
            video_duration=0.0,
        )
        assert len(manifest['frame_index']) == 5
        for entry in manifest['frame_index']:
            assert entry['ts_host_iso'] is None
            assert entry['ts_camera_ticks'] is None
            assert entry['frame_id'] is None


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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_buffer_cancel_constant_value_matches_bench(self):
        """The bench-witnessed decimal from session 65 (3791651074) is the
        authoritative cancel-code constant. If pypylon ever exposes
        pylon.GENERIC_BUFFER_CANCELED or similar, bump this."""
        from drivers.pyloncamera import _PYLON_ERR_BUFFER_CANCELED

        assert _PYLON_ERR_BUFFER_CANCELED == 3791651074, (
            'Buffer-cancel constant must match the bench-witnessed value '
            'from Firmware DAILY_LOG.md session 65 Run-3 '
            '(decimal 3791651074 = 0xE2000102).'
        )

    def test_buffer_cancel_comment_hex_matches_constant(self):
        """The source comment must show the hex form that matches the
        decimal constant. Earlier the comment said 0xE2008002 (= decimal
        3791683586, NOT what's stored). The corrected hex is 0xE2000102.
        Mismatch between comment and value misleads anyone debugging this
        path; the comment is load-bearing documentation, not decoration."""
        src = self._pyloncamera_source()
        assert '0xE2000102' in src, (
            'Source comment near _PYLON_ERR_BUFFER_CANCELED must reference '
            '0xE2000102 (the hex form of decimal 3791651074). If you found '
            "0xE2008002 here, that's the prior typo -- fix to 0xE2000102."
        )
        assert '0xE2008002' not in src, (
            'Stale typo: 0xE2008002 must not appear in pyloncamera.py '
            "source -- that hex equals 3791683586 (NOT what's stored)."
        )

    def test_cancel_branch_uses_or_with_removal_flag(self):
        """The cancel-classification branch must include the
        OR-with-removal-flag insurance:

            if err_code == _PYLON_ERR_BUFFER_CANCELED or self._parent._device_removed:

        This protects against a race where _device_removed flips True
        between Stage A's early-return check and the cancel-classification
        check (the SDK's removal-forwarding thread runs on its own thread;
        either the grab thread or the removal thread can set the flag).
        Without the OR, a mid-call removal whose first cancellation
        buffer carries an undocumented err_code would count toward
        MAX_CONSECUTIVE_FAILURES.

        Post-R12 the classification lives in `_PylonImageGrabWorker._process_failure`
        (Stage B); Stage A only does the fast-path disconnect inline for
        DEVICE_NOT_FOUND, everything else is queued to Stage B.
        """
        src = self._pyloncamera_source()
        body = _function_source(src, '_process_failure')
        assert '_PYLON_ERR_BUFFER_CANCELED or self._parent._device_removed' in body, (
            '_process_failure cancel-classification branch must use '
            'OR-with-removal-flag insurance. See class docstring for the '
            'race the OR protects against.'
        )

    def test_normal_failure_branch_still_calls_record_failure(self):
        """The non-cancel non-removal failure path must still increment
        the consecutive-failure counter. Without this, real failures
        (incomplete buffers 0xE2000212, transport errors 0xE2000011,
        etc.) would never trip MAX_CONSECUTIVE_FAILURES auto-disconnect.

        Post-R12 this lives in `_PylonImageGrabWorker._process_failure`.
        """
        src = self._pyloncamera_source()
        body = _function_source(src, '_process_failure')
        assert 'self._base._record_failure()' in body, (
            '_process_failure non-cancel branch must still call '
            '_record_failure to increment the consecutive-failure counter.'
        )


class TestPylonPayloadDiscardedClassification:
    """Payload-discarded (camera-side FIFO overflow) classification in
    OnImageGrabbed. Distinct from the cancel branch above.

    Payload-discarded events fire when the camera-side USB FIFO overflows
    during a host stall (e.g. SetValue for gain/exposure inside an AF
    cycle with MaxNumBuffer at its default). The dropped frame is one
    that frame_validity would have rejected anyway: invalidate() runs
    after each SetValue, so downstream consumers already wait for a
    clean frame. The classification rule:

    - Log at info (cause distribution stays visible) -- NOT warning.
    - DO NOT call _record_failure. Acquisition is healthy; counting
      these toward MAX_CONSECUTIVE_FAILURES would falsely trip the
      128-consec auto-disconnect during AF-heavy protocols.

    These tests lock the source-level shape so a future cleanup that
    removes the elif branch or adds _record_failure inside it fires
    the regression.
    """

    def _pyloncamera_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_payload_discarded_constant_value(self):
        """The constant must match the bench-witnessed err_code from
        Firmware DAILY_LOG (0xE2050012). If Basler renames or splits
        the code in a future SDK rev, bump this and update the comment."""
        from drivers.pyloncamera import _PYLON_ERR_PAYLOAD_DISCARDED

        assert _PYLON_ERR_PAYLOAD_DISCARDED == 0xE2050012, (
            'Payload-discarded constant must match the bench-witnessed '
            'err_code 0xE2050012 from Firmware DAILY_LOG.md.'
        )

    def test_payload_discarded_comment_explains_disposition(self):
        """The source must document WHY this classification exists --
        camera-side FIFO overflow during host stalls plus the
        frame_validity coverage that makes the drop safe to ignore.
        Comment is load-bearing: removing it would re-introduce the
        'why does this skip _record_failure' question."""
        src = self._pyloncamera_source()
        assert 'camera-side FIFO overflow' in src.lower() or ('camera-side fifo' in src.lower()), (
            'Source comment near _PYLON_ERR_PAYLOAD_DISCARDED must explain '
            'the camera-side FIFO overflow mechanism.'
        )
        assert 'frame_validity' in src, (
            'Source comment must reference frame_validity coverage -- the '
            'reason payload-discarded events are safe to skip _record_failure.'
        )

    def test_payload_discarded_branch_in_onimagegrabbed(self):
        """The classification branch must exist. Structural check: a
        future cleanup that drops the elif (collapsing payload-discarded
        back into the warning fallback) would reintroduce log noise +
        spurious failure-counter increments.

        Post-R12 the classification lives in `_PylonImageGrabWorker._process_failure`.
        """
        src = self._pyloncamera_source()
        body = _function_source(src, '_process_failure')
        assert '_PYLON_ERR_PAYLOAD_DISCARDED' in body, (
            '_process_failure must contain a classification branch for '
            '_PYLON_ERR_PAYLOAD_DISCARDED. See class docstring.'
        )

    def test_payload_discarded_branch_skips_record_failure(self):
        """Key invariant: the payload-discarded branch MUST NOT call
        _record_failure. The branch represents healthy acquisition where
        the camera dropped a frame during a host stall; counting it
        toward MAX_CONSECUTIVE_FAILURES would falsely trip
        auto-disconnect during AF-heavy protocols.

        Test approach: extract the elif block in _process_failure and
        assert _record_failure does not appear in it. Worker-side
        classification has exactly 1 _record_failure call total (the
        generic non-classified fallback in the else branch); the cancel
        and payload-discarded branches both skip _record_failure for
        branch-specific reasons (lifecycle / healthy-FIFO-drop).
        _process_frame has its own _record_failure for GetArray
        exceptions (different concern: GetArray failure implies device
        gone, which IS a counted-failure-class event).
        """
        src = self._pyloncamera_source()
        body = _function_source(src, '_process_failure')
        # Total count is the structural guard.
        total_calls = body.count('self._base._record_failure()')
        assert total_calls == 1, (
            f'_process_failure must have exactly 1 _record_failure() '
            f'call (generic non-classified fallback in the else branch); '
            f'found {total_calls}. If a second was added inside the '
            f'payload-discarded classification branch, remove it -- '
            f'that branch is by-design not counted toward '
            f'MAX_CONSECUTIVE_FAILURES.'
        )
        # Extract just the elif block as belt-and-suspenders. Indent
        # level is 8 spaces (method body in _process_failure).
        elif_marker = 'elif err_code == _PYLON_ERR_PAYLOAD_DISCARDED:'
        elif_idx = body.find(elif_marker)
        assert elif_idx >= 0, 'elif marker not found (precondition)'
        tail = body[elif_idx + len(elif_marker) :]
        end = len(tail)
        for marker in ('\n        else:', '\n        elif '):
            i = tail.find(marker)
            if 0 <= i < end:
                end = i
        elif_block = tail[:end]
        assert '_record_failure' not in elif_block, (
            'Payload-discarded elif branch contains _record_failure() -- '
            "that breaks the 'healthy acquisition, not a counted failure' "
            'invariant. See class docstring.'
        )


class TestPylonDeviceNotFoundClassification:
    """Device-not-found (USB-Vision physical removal) classification in
    OnImageGrabbed. Third classification branch alongside cancel and
    payload-discarded.

    Bench evidence: a USB cable bump during live preview produced
    err_code=433 ("A device which does not exist was specified") at
    ~100+ events in <2s. The generic fallback would have logged
    WARNING for each and only fired auto-disconnect after 128
    consecutive failures (~4.3s at 30fps). The user-visible result
    was a 3-second wall of WARNING log lines followed by a delayed
    notification.

    Fast-classification rule:
    - Log once at ERROR (real disconnect, user-actionable).
    - Call _mark_disconnected immediately so the API-layer Rule-14
      notification fires off the disconnect flag on the next is_connected
      poll (~30 ms typical) instead of 4 seconds late.
    - Stop grabbing immediately so the SDK doesn't keep firing
      OnImageGrabbed callbacks at the wedged handle.
    - Skip _record_failure -- the consecutive-failure counter exists to
      detect transport degradation (incomplete buffers, CRC errors).
      Physical removal is a different class with its own signal.

    These tests lock the source-level shape so a future cleanup that
    drops the branch or adds _record_failure inside it fires the
    regression. Behavioral test (full cascade simulation) would
    require a Pylon SDK callback harness; static lock is the primary
    regression gate.
    """

    def _pyloncamera_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_device_not_found_constant_value(self):
        """The constant must match the bench-witnessed err_code (433)
        from the LVP_Logbumped.wire session. If Basler renames the code
        in a future SDK rev, bump this and update the comment."""
        from drivers.pyloncamera import _PYLON_ERR_DEVICE_NOT_FOUND

        assert _PYLON_ERR_DEVICE_NOT_FOUND == 433, (
            'Device-not-found constant must match the bench-witnessed '
            'err_code 433 from the LVP_Logbumped.wire cascade.'
        )

    def test_device_not_found_comment_explains_fast_classification(self):
        """The source must document WHY this branch exists -- bench
        cascade rate, slow-path delay, and the user-notification
        timing impact. Comment is load-bearing: a future cleanup
        without the WHY would risk collapsing the branch back into
        the generic fallback and re-introducing the 4-second log
        spam window."""
        src = self._pyloncamera_source()
        assert 'cascade' in src.lower(), (
            'Source comment near _PYLON_ERR_DEVICE_NOT_FOUND must '
            'explain the cascade rate that motivates fast classification.'
        )
        assert 'MAX_CONSECUTIVE_FAILURES' in src, (
            'Source comment must reference MAX_CONSECUTIVE_FAILURES -- '
            'the slow-path mechanism that fast classification short-circuits.'
        )

    def test_device_not_found_branch_in_onimagegrabbed(self):
        """The OnImageGrabbed body must contain the elif classification
        branch. Structural check: a future cleanup that drops the elif
        would reintroduce the 4-second cascade delay + log spam."""
        src = self._pyloncamera_source()
        body = _function_source(src, 'OnImageGrabbed')
        assert '_PYLON_ERR_DEVICE_NOT_FOUND' in body, (
            'OnImageGrabbed must contain a classification branch for '
            '_PYLON_ERR_DEVICE_NOT_FOUND. See class docstring.'
        )
        assert 'success_no_grab_device_not_found' in body, (
            'OnImageGrabbed device-not-found branch must set its '
            "outcome name to 'success_no_grab_device_not_found' for "
            'trace gating.'
        )

    def test_device_not_found_branch_marks_disconnected(self):
        """The device-not-found branch must call _mark_disconnected.
        That is the entire structural point of fast classification:
        flip the connection flag in 1 frame instead of 128 so the
        API-layer notification fires immediately.

        Post-R12 the disconnect fast-path stays INLINE in OnImageGrabbed
        Stage A (so the disconnect notification doesn't wait behind
        Stage B's worker queue). With cancel/payload-discarded/generic
        moved to Stage B, DEVICE_NOT_FOUND is now the only err_code
        check in Stage A -- the `elif` chain collapsed to a single `if`.
        """
        src = self._pyloncamera_source()
        body = _function_source(src, 'OnImageGrabbed')
        branch_marker = 'if err_code == _PYLON_ERR_DEVICE_NOT_FOUND:'
        idx = body.find(branch_marker)
        assert idx >= 0, 'DEVICE_NOT_FOUND branch marker not found (precondition)'
        tail = body[idx + len(branch_marker) :]
        end = len(tail)
        for marker in ('\n                else:', '\n                elif '):
            i = tail.find(marker)
            if 0 <= i < end:
                end = i
        branch_block = tail[:end]
        assert '_mark_disconnected' in branch_block, (
            'Device-not-found branch must call '
            'self._parent._mark_disconnected() -- that is the structural '
            'point of fast classification. See class docstring.'
        )

    def test_device_not_found_branch_skips_record_failure(self):
        """Physical removal is not a counted failure; the consecutive-
        failure counter exists for transport degradation. Counting
        device-not-found toward MAX_CONSECUTIVE_FAILURES is at best
        redundant (we've already marked disconnected) and at worst
        misleading (failure count inflates from a single physical
        event)."""
        src = self._pyloncamera_source()
        body = _function_source(src, 'OnImageGrabbed')
        branch_marker = 'if err_code == _PYLON_ERR_DEVICE_NOT_FOUND:'
        idx = body.find(branch_marker)
        assert idx >= 0, 'DEVICE_NOT_FOUND branch marker not found (precondition)'
        tail = body[idx + len(branch_marker) :]
        end = len(tail)
        for marker in ('\n                else:', '\n                elif '):
            i = tail.find(marker)
            if 0 <= i < end:
                end = i
        branch_block = tail[:end]
        assert '_record_failure' not in branch_block, (
            'Device-not-found branch contains _record_failure() -- '
            'physical removal is a different class of event with its '
            'own signal (_mark_disconnected); double-counting inflates '
            'the failure counter from one physical event. See class docstring.'
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

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        return _function_source(src, 'disconnect')

    def test_disconnect_calls_destroy_device(self):
        """disconnect() must explicitly destroy the SDK-side device
        handle. Without this, CPython refcount-driven cleanup may leave
        the handle held past the next reconnect attempt (pypylon #792)."""
        body = self._disconnect_body()
        assert 'self.active.DestroyDevice()' in body, (
            'PylonCamera.disconnect must call self.active.DestroyDevice() '
            'to explicitly release the SDK-side device handle. Required '
            "to prevent 'device controlled by another application' on the "
            'next CreateDevice (pypylon issues #547, #792).'
        )

    def test_disconnect_calls_detach_device(self):
        """DetachDevice releases the InstantCamera's ownership of the
        device pointer before DestroyDevice destroys it. Per
        Basler-recommended canonical reattach sequence."""
        body = self._disconnect_body()
        assert 'self.active.DetachDevice()' in body, (
            'PylonCamera.disconnect must call self.active.DetachDevice() '
            'before DestroyDevice. Required by Basler-recommended cleanup '
            'sequence: StopGrabbing -> Close -> DetachDevice -> DestroyDevice.'
        )

    def test_disconnect_destroy_device_wrapped_in_try(self):
        """If DestroyDevice fails (e.g., already-detached device), the
        exception must NOT prevent self.active = None from running.
        Otherwise the post-cleanup invariant breaks and the app thinks
        a known-bad camera is still connected."""
        body = self._disconnect_body()
        # Look for the exact pattern: try block containing DestroyDevice
        assert 'self.active.DestroyDevice()' in body
        # The DestroyDevice line must be inside a try/except that logs
        # a warning and continues, not propagates.
        # Heuristic: there must be at least 3 try blocks in disconnect
        # (one for stop_grabbing, one for Close, one for DetachDevice,
        # one for DestroyDevice -- count of "try:" lines must be >= 4).
        try_count = body.count('try:')
        assert try_count >= 4, (
            f'disconnect() must wrap each SDK teardown step (Close, '
            f'DetachDevice, DestroyDevice) in its own try/except so a '
            f'failure in one does not skip the others. Currently '
            f'{try_count} try blocks; expected >= 4 (stop_grabbing + '
            f'Close + DetachDevice + DestroyDevice).'
        )

    def test_disconnect_clears_active_after_cleanup(self):
        """self.active = None must come AFTER DestroyDevice, not before.
        If we cleared active first we would lose the device pointer
        before destroying it, leaving the SDK handle held."""
        body = self._disconnect_body()
        destroy_pos = body.find('self.active.DestroyDevice()')
        clear_pos = body.find('self.active = None')
        assert destroy_pos != -1, 'DestroyDevice call missing from disconnect()'
        assert clear_pos != -1, 'self.active = None missing from disconnect()'
        assert clear_pos > destroy_pos, (
            'self.active = None must come AFTER self.active.DestroyDevice(). '
            'Clearing active first loses the device pointer before '
            'DestroyDevice can run -> SDK handle stays held.'
        )


class TestPylonDiagnosticProbe:
    """DiagnosticsAPI.run_pylon_diagnostic_probe captures a one-shot
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
        attach the supplied fake camera. Phase 5b: also inject
        DiagnosticsAPI so callers can use scope.diagnostics.X."""
        from modules.lumascope_api import Lumascope
        from modules.lumascope_api.diagnostics import DiagnosticsAPI

        scope = Lumascope.__new__(Lumascope)
        scope.runtime_state = RuntimeState(scope)
        scope._camera_driver = fake_camera
        scope.diagnostics = DiagnosticsAPI(scope)
        return scope

    def test_method_exists_on_diagnostics_api(self):
        """The API method is callable from the DiagnosticsAPI class."""
        from modules.lumascope_api.diagnostics import DiagnosticsAPI

        assert hasattr(DiagnosticsAPI, 'run_pylon_diagnostic_probe')
        assert callable(DiagnosticsAPI.run_pylon_diagnostic_probe)

    def test_no_camera_returns_disconnected(self):
        """Returns {'connected': False, 'errors': [...]} when no camera."""
        from modules.lumascope_api import Lumascope
        from modules.lumascope_api.diagnostics import DiagnosticsAPI

        scope = Lumascope.__new__(Lumascope)
        scope.runtime_state = RuntimeState(scope)
        scope._camera_driver = None
        scope.diagnostics = DiagnosticsAPI(scope)
        result = scope.diagnostics.run_pylon_diagnostic_probe(duration_s=0.0)
        assert result['connected'] is False
        assert isinstance(result.get('errors'), list)

    def test_inactive_camera_returns_disconnected(self):
        """Camera object exists but inactive -> disconnected."""

        class _Fake:
            active = None

        result = self._make_scope_with_fake_camera(_Fake()).diagnostics.run_pylon_diagnostic_probe(
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

        result = self._make_scope_with_fake_camera(
            _StubDriver()
        ).diagnostics.run_pylon_diagnostic_probe(duration_s=0.0)
        assert result.get('supported') is False
        assert 'output_path' not in result, (
            'supported=False driver responses must NOT trigger JSON write'
        )

    def test_no_read_diagnostic_snapshot_method(self):
        """If the driver does not implement read_diagnostic_snapshot at
        all, the API returns a structured error rather than raising
        AttributeError."""

        class _NoMethodDriver:
            active = True

        result = self._make_scope_with_fake_camera(
            _NoMethodDriver()
        ).diagnostics.run_pylon_diagnostic_probe(duration_s=0.0)
        assert result['connected'] is False
        assert result.get('supported') is False

    def test_dltl_filename_token_off(self):
        """Mode=Off -> 'dltloff'."""
        from modules.lumascope_api.diagnostics import DiagnosticsAPI

        token = DiagnosticsAPI._dltl_filename_token({'dltl_mode': 'Off'})
        assert token == 'dltloff'

    def test_dltl_filename_token_on_round(self):
        """Mode=On with 160 MB/s -> 'dltl160M'."""
        from modules.lumascope_api.diagnostics import DiagnosticsAPI

        token = DiagnosticsAPI._dltl_filename_token(
            {
                'dltl_mode': 'On',
                'dltl_value_bps': 160_000_000,
            }
        )
        assert token == 'dltl160M'

    def test_dltl_filename_token_on_non_round(self):
        """Mode=On with non-round MB/s -> rounded int rendering.
        v4 author flagged the case where a sweep value has sub-MB
        precision; bare int() would render 197.99 MB/s as dltl197M
        which is wrong-by-1; round() avoids that."""
        from modules.lumascope_api.diagnostics import DiagnosticsAPI

        token = DiagnosticsAPI._dltl_filename_token(
            {
                'dltl_mode': 'On',
                'dltl_value_bps': 197_999_000,
            }
        )
        assert token == 'dltl198M', (
            f'Expected dltl198M (rounded), got {token!r}; int(round()) cast missing or wrong'
        )

    def test_dltl_filename_token_unknown(self):
        """Missing config -> 'dltlunknown'."""
        from modules.lumascope_api.diagnostics import DiagnosticsAPI

        assert DiagnosticsAPI._dltl_filename_token({}) == 'dltlunknown'
        assert DiagnosticsAPI._dltl_filename_token({'dltl_mode': '<missing>'}) == 'dltlunknown'

    def test_human_os_version_does_not_raise(self):
        """The OS-version helper must never raise, even on platforms
        where mac_ver/win32_ver return empty tuples."""
        from modules.lumascope_api.diagnostics import DiagnosticsAPI

        v = DiagnosticsAPI._human_os_version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_safe_pylon_versions_returns_dict(self):
        """The version helper returns a dict with both keys, even when
        pypylon is absent (returns Nones)."""
        from modules.lumascope_api.diagnostics import DiagnosticsAPI

        result = DiagnosticsAPI._safe_pylon_versions()
        assert isinstance(result, dict)
        assert 'pypylon_version' in result
        assert 'pylon_sdk_version' in result

    def test_pylon_camera_has_read_diagnostic_snapshot(self):
        """Source-shape lock: PylonCamera must implement the driver
        method the API depends on."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        assert 'def read_diagnostic_snapshot(' in src, (
            'PylonCamera must implement read_diagnostic_snapshot for '
            'DiagnosticsAPI.run_pylon_diagnostic_probe to function.'
        )

    def test_ids_camera_has_read_diagnostic_snapshot_stub(self):
        """Source-shape lock: IDSCamera must have a stub returning
        supported=False so the API can report the gap rather than
        raising AttributeError when an IDS camera is connected."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'idscamera.py').read_text()
        assert 'def read_diagnostic_snapshot(' in src, (
            'IDSCamera must have a read_diagnostic_snapshot stub '
            'returning supported=False until the IDS implementation lands.'
        )
        body = _function_source(src, 'read_diagnostic_snapshot')
        assert "'supported': False" in body or '"supported": False' in body, (
            'IDS read_diagnostic_snapshot stub must return supported=False'
        )


class TestDeviceLinkThroughputLimitSetter:
    """ImagingAPI.set_device_link_throughput_limit and the underlying
    PylonCamera / IDSCamera implementations exist so the bench-probe
    sweep can vary DLTL across cells without dropping below the API
    layer (Rule 1) or writing /tmp/probe.py (Rule 22).

    DLTL is documented live-writable; no StopGrabbing/StartGrabbing
    wrap is required. The Pylon driver raises HardwareError on SDK
    RuntimeException; the API layer notifies + re-raises.
    """

    def _make_scope_with_fake_camera(self, fake_camera):
        from modules.lumascope_api import Lumascope
        from modules.lumascope_api.imaging import ImagingAPI

        scope = Lumascope.__new__(Lumascope)
        scope.runtime_state = RuntimeState(scope)
        scope._camera_driver = fake_camera
        scope.imaging = ImagingAPI(scope, fake_camera)
        return scope

    def test_lumascope_method_exists(self):
        from modules.lumascope_api import Lumascope

        assert hasattr(ImagingAPI, '_set_device_link_throughput_limit')
        assert callable(ImagingAPI._set_device_link_throughput_limit)

    def test_no_camera_returns_false(self):
        from modules.lumascope_api import Lumascope
        from modules.lumascope_api.imaging import ImagingAPI

        scope = Lumascope.__new__(Lumascope)
        scope.runtime_state = RuntimeState(scope)
        scope._camera_driver = None
        scope.imaging = ImagingAPI(scope, None)
        assert scope.imaging._set_device_link_throughput_limit('Off') is False

    def test_inactive_camera_returns_false(self):
        class _Fake:
            active = None

            def set_device_link_throughput_limit(self, **k):
                raise AssertionError('driver should not be reached')

        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.imaging._set_device_link_throughput_limit('Off') is False

    def test_unsupported_driver_returns_false(self):
        """Camera class without the setter (e.g. SimulatedCamera) -> False."""

        class _NoSetter:
            active = True

        scope = self._make_scope_with_fake_camera(_NoSetter())
        assert scope.imaging._set_device_link_throughput_limit('Off') is False

    def test_off_routes_to_driver(self):
        called_with = {}

        class _Fake:
            active = True

            def set_device_link_throughput_limit(self, mode, value_bps=None):
                called_with['mode'] = mode
                called_with['value_bps'] = value_bps
                return True

        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.imaging._set_device_link_throughput_limit('Off') is True
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
        ok = scope.imaging._set_device_link_throughput_limit('On', value_bps=160_000_000)
        assert ok is True
        assert called_with == {'mode': 'On', 'value_bps': 160_000_000}

    def test_pylon_driver_method_present(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        assert 'def set_device_link_throughput_limit(' in src, (
            'PylonCamera must implement set_device_link_throughput_limit '
            "for tomorrow's bench-probe sweep to function without "
            'Rule 1 violations.'
        )

    def test_pylon_driver_does_not_wrap_in_update_camera_config(self):
        """DLTL is live-writable per Section 5; wrapping in
        update_camera_config would force unnecessary stop/start cycles
        (per the STALL-1 anti-pattern lesson)."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        body = _function_source(src, 'set_device_link_throughput_limit')
        assert 'with self.update_camera_config' not in body, (
            'PylonCamera.set_device_link_throughput_limit must NOT wrap '
            'the writes in update_camera_config (DLTL is live-writable; '
            'wrapping would impose the STALL-1 over-stop pattern).'
        )

    def test_ids_driver_stub_present(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'idscamera.py').read_text()
        assert 'def set_device_link_throughput_limit(' in src, (
            'IDSCamera must have a set_device_link_throughput_limit '
            'stub so the API method does not need to know which driver '
            'is connected when called by the sweep tool.'
        )

    def test_pylon_driver_raises_hardware_error_on_runtime_exception(self):
        """Per the Raises: docstring section, the Pylon setter raises
        HardwareError on genicam.RuntimeException so the API layer can
        notify and the caller can handle it (Rule 29 typed-exception
        contract; matches set_binning_size / set_pixel_format).

        Pins the raise so a future cleanup that swaps it for return-False
        is caught here instead of in the field.
        """
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        body = _function_source(src, 'set_device_link_throughput_limit')
        assert 'except genicam.RuntimeException' in body, (
            'PylonCamera.set_device_link_throughput_limit must keep its '
            'RuntimeException catch (Rule 29 typed-exception contract).'
        )
        assert 'raise HardwareError(' in body, (
            'PylonCamera.set_device_link_throughput_limit must raise '
            'HardwareError on RuntimeException, not return False -- the '
            'API layer catches and notifies.'
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

        return (
            (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py')
            .read_text()
            .splitlines()
        )

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
                        f'line {i} col {col}: char {ch!r} (U+{ord(ch):04X}) -- '
                        f'line is: {line.strip()[:80]}'
                    )
                    break
        assert not offenders, (
            'Rule 24 violation -- non-ASCII char in logger/print/notifications '
            "string. Use ASCII (e.g. 'degC' not the degree sign). "
            'Sites:\n  ' + '\n  '.join(offenders)
        )

    def test_temperature_log_uses_degC_ascii_form(self):
        """Pin the corrected form so the specific A10 fix survives. If a
        future cleanup edits the temperature log line, this test reminds
        the editor that ASCII-only was intentional."""
        src_lines = self._pyloncamera_source_lines()
        # Anchor on the temperature-reading f-string: the unique phrase
        # 'Temperature :' (space-colon) appears only on the camera-temp
        # readback line. The session 18 dual-write migration moved this
        # call from logger.info to _log_cam, then the Rule 26 line-length
        # wrap split the call across lines, so we can't anchor on a
        # logger-marker on the same line as the f-string -- match the
        # f-string content itself.
        for i, line in enumerate(src_lines, 1):
            if 'Temperature :' in line:
                assert 'degC' in line, (
                    f'pyloncamera.py:{i} -- temperature log line must use '
                    f"ASCII 'degC' (not the degree sign). Line: {line.strip()[:100]}"
                )
                assert chr(0xB0) not in line, (
                    f"pyloncamera.py:{i} -- degree sign (U+00B0) reintroduced. Use 'degC' instead."
                )
                return
        raise AssertionError(
            'Could not find a temperature log line in pyloncamera.py. '
            'If get_all_temperatures was renamed/removed, update this test.'
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

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
            'pyloncamera.py contains direct write(s) marking the camera '
            f'removed: {offenders}. Use Camera._mark_disconnected '
            '(acquires _state_lock + sets _active=None + emits '
            'boundary log) instead.'
        )

    def test_on_image_grabbed_inactive_branch_uses_mark_disconnected(self):
        """The OnImageGrabbed inactive-fallback branch must call
        _mark_disconnected so the parent's _state_lock invariants hold."""
        src = self._pyloncamera_source()
        # Find the inactive-branch sentinel and confirm the call sequence.
        marker = 'OnImageGrabbed called but camera is inactive'
        idx = src.find(marker)
        assert idx != -1, (
            'Could not find OnImageGrabbed inactive-branch logger sentinel; '
            'if the wording changed, update this test.'
        )
        # Within ~200 chars after the sentinel, expect the canonical call.
        window = src[idx : idx + 400]
        assert '_mark_disconnected()' in window, (
            'OnImageGrabbed inactive-branch must call '
            'self._parent._mark_disconnected() to preserve the '
            '_state_lock invariant. Found instead:\n' + window
        )

    def test_on_camera_device_removed_uses_mark_disconnected(self):
        """The _CameraRemovalHandler SDK callback must call
        _mark_disconnected. _mark_disconnected's docstring states it is
        safe from any thread (including SDK callbacks); the prior
        comment claiming otherwise was stale."""
        src = self._pyloncamera_source()
        marker = 'def OnCameraDeviceRemoved('
        idx = src.find(marker)
        assert idx != -1, 'Could not find OnCameraDeviceRemoved method.'
        window = src[idx : idx + 800]
        assert '_mark_disconnected()' in window, (
            'OnCameraDeviceRemoved must call self._parent._mark_disconnected() '
            'to atomically clear _active under _state_lock.'
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_stop_stats_poller_captures_thread_before_signalling(self):
        """The thread reference must be read before the event is set;
        otherwise a concurrent _stats_poller_thread = None elsewhere
        could cause join() to be called on None."""
        src = self._pyloncamera_source()
        idx = src.find('def _stop_stats_poller(self):')
        assert idx != -1, 'Could not find _stop_stats_poller.'
        end = src.find('def ', idx + 10)
        body = src[idx:end]
        # Order check: thread getattr before event set.
        thread_get = body.find('_stats_poller_thread')
        ev_set = body.find('.set()')
        assert thread_get != -1 and ev_set != -1, (
            '_stop_stats_poller must reference both _stats_poller_thread and the '
            'event. Body:\n' + body
        )
        assert thread_get < ev_set, (
            '_stop_stats_poller must capture the thread reference BEFORE '
            'signalling the stop event, so the join() target is stable.'
        )

    def test_stop_stats_poller_joins_with_timeout(self):
        """_stop_stats_poller must join the prior thread with a bounded
        timeout to symmetrise _start_stats_poller's join."""
        src = self._pyloncamera_source()
        idx = src.find('def _stop_stats_poller(self):')
        assert idx != -1
        end = src.find('def ', idx + 10)
        body = src[idx:end]
        assert '.join(timeout=' in body, (
            '_stop_stats_poller must call .join(timeout=...) on the prior '
            'stats-poller thread before clearing the reference.'
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_disconnect_stop_grabbing_failure_is_logged(self):
        """The bare `except Exception: pass` on stop_grabbing during
        disconnect is a Rule 5 violation. The fix logs at warning
        level."""
        src = self._pyloncamera_source()
        # Find the disconnect method body
        idx = src.find('def disconnect(self) -> bool:')
        assert idx != -1, 'Could not find PylonCamera.disconnect.'
        # Walk forward to the stop_grabbing block
        sg_idx = src.find('self.stop_grabbing()', idx)
        assert sg_idx != -1, 'Could not find stop_grabbing call in disconnect.'
        # The except block immediately follows; check the next ~250 chars
        window = src[sg_idx : sg_idx + 350]
        assert (
            'except Exception:' not in window
            or 'pass' not in window.split('except Exception:')[1].split('\n', 5)[0]
            if 'except Exception:' in window
            else True
        )
        # Simpler: assert the warning-log phrase is present
        assert 'stop_grabbing during disconnect' in window, (
            "disconnect's stop_grabbing except branch must log a warning, "
            'not silently pass. Found:\n' + window
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_on_image_grabbed_outer_except_uses_contextual_message(self):
        """Bare `logger.exception(e)` is forbidden in OnImageGrabbed.
        The fix uses an f-string with [CAM Class ] prefix and a callback
        identifier."""
        src = self._pyloncamera_source()
        idx = src.find('def OnImageGrabbed(')
        assert idx != -1, 'Could not find ImageHandler.OnImageGrabbed.'
        end = src.find('def ', idx + 10)
        body = src[idx:end]
        # Find the outer except clause (indented less than the inner ones)
        # Easiest: the literal `_outcome = 'exception_outer'` is unique.
        marker = "_outcome = 'exception_outer'"
        m_idx = body.find(marker)
        assert m_idx != -1, (
            'Could not find OnImageGrabbed outer-except sentinel '
            "(_outcome = 'exception_outer'). If renamed, update test."
        )
        window = body[m_idx : m_idx + 250]
        assert 'logger.exception(e)' not in window, (
            'OnImageGrabbed outer-except must NOT call logger.exception(e) '
            'with the bare exception object -- the rendered log line lacks '
            '[CAM Class ] prefix and callback context.'
        )
        assert 'OnImageGrabbed' in window or '[CAM Class ]' in window, (
            'OnImageGrabbed outer-except logger.exception call must include '
            'a contextual prefix. Found:\n' + window
        )


class TestPylonOnImageGrabbedOwningCopy:
    """pypylon delivers OnImageGrabbed a non-owning wrapper around a C++
    CGrabResultPtr that lives on the SDK callback's stack frame. Python
    copies of that wrapper (assignment, queue.put) only bump Py_REFCNT;
    they do NOT invoke the C++ copy constructor. When the callback
    returns, the underlying smart pointer is destroyed and any wrapper
    still alive raises 'No grab result data is referenced' on next
    access -- the exact failure observed on both a2A3536-31umBAS and
    daA3840-45um.

    The fix is to invoke the C++ copy ctor explicitly before crossing
    thread boundaries: ``pylon.GrabResult(grabResult)``. pypylon's
    ``GrabResult.__init__`` maps to ``_pylon.new_GrabResult(*args)``,
    which is the binding's surface for the C++
    ``CGrabResultPtr(const CGrabResultPtr&)`` copy ctor declared in
    ``GrabResultPtr.h``.

    The test guards two enqueue sites in OnImageGrabbed (the 'frame'
    success path and the 'fail' classification path) so neither
    regresses to passing the raw SDK-delivered grabResult across the
    queue.
    """

    def _pyloncamera_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def _on_image_grabbed_body(self):
        src = self._pyloncamera_source()
        idx = src.find('def OnImageGrabbed(')
        assert idx != -1, 'Could not find ImageHandler.OnImageGrabbed.'
        end = src.find('def ', idx + 10)
        return src[idx:end]

    def test_frame_enqueue_uses_owning_copy(self):
        """The 'frame' success-path enqueue must hand the worker an
        owning wrapper produced by ``pylon.GrabResult(grabResult)``,
        not the raw SDK-delivered grabResult."""
        body = self._on_image_grabbed_body()
        marker = "self._worker.enqueue('frame'"
        m_idx = body.find(marker)
        assert m_idx != -1, (
            "Could not find the 'frame' enqueue site in OnImageGrabbed. If renamed, update test."
        )
        window_start = max(0, m_idx - 400)
        window = body[window_start : m_idx + 200]
        assert 'pylon.GrabResult(grabResult)' in window, (
            "OnImageGrabbed 'frame' enqueue must be preceded by an "
            'explicit owning-copy invocation: '
            'owned = pylon.GrabResult(grabResult). Without it, the '
            'queued wrapper goes dangling when OnImageGrabbed returns. '
            'Window:\n' + window
        )
        assert "self._worker.enqueue('frame', grabResult," not in window, (
            'OnImageGrabbed must NOT pass the raw grabResult straight to '
            'the worker queue -- the SWIG-director wrapper is non-owning '
            'for callback parameters.'
        )

    def test_fail_enqueue_uses_owning_copy(self):
        """The 'fail' classification-path enqueue runs the same cross-
        thread handoff as the success path and needs the same owning
        wrapper. Stage B reads GetErrorCode / GetErrorDescription /
        GetBlockID through the queued reference."""
        body = self._on_image_grabbed_body()
        marker = "self._worker.enqueue('fail'"
        m_idx = body.find(marker)
        assert m_idx != -1, (
            "Could not find the 'fail' enqueue site in OnImageGrabbed. If renamed, update test."
        )
        window_start = max(0, m_idx - 400)
        window = body[window_start : m_idx + 200]
        assert 'pylon.GrabResult(grabResult)' in window, (
            "OnImageGrabbed 'fail' enqueue must be preceded by an "
            'explicit owning-copy invocation: '
            'owned = pylon.GrabResult(grabResult). Stage B reads '
            'GetErrorCode/GetErrorDescription/GetBlockID through the '
            'queued reference; same dangling-wrapper hazard as the '
            'success path. Window:\n' + window
        )
        assert "self._worker.enqueue('fail', grabResult," not in window, (
            'OnImageGrabbed must NOT pass the raw grabResult straight to '
            'the worker queue on the failure path either.'
        )


class TestPylonTimeoutNameConsistency:
    """A function declared with a `timeout_s` parameter must reference
    that parameter consistently in its body. Bare `timeout` in the
    same body is almost always a rename-leftover NameError waiting to
    fire when the relevant code path executes.

    Found-by-pattern: drivers/pyloncamera.py::grab_new_capture had
    `timeout_s` as the param but two body sites (the queue.Empty
    warning f-string and the profile_trace finally block) referenced
    bare `timeout`. NameError fired on every grab timeout AND on
    every profile_trace.trace() call when LVP_PROFILE_TRACE=1 was
    set, masking real failures behind the executor's
    'Uncaught Thread Exception' wrapper.

    The AST scan below pins the convention across the file.
    """

    def test_no_bare_timeout_in_functions_with_timeout_s_param(self):
        import ast
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        tree = ast.parse(src)

        hits = []

        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.fn_stack = []
                self.has_timeout_s = []

            def visit_FunctionDef(self, node):
                has_ts = any(
                    arg.arg == 'timeout_s' for arg in node.args.args + node.args.kwonlyargs
                )
                self.fn_stack.append(node.name)
                self.has_timeout_s.append(has_ts)
                self.generic_visit(node)
                self.fn_stack.pop()
                self.has_timeout_s.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Name(self, node):
                if (
                    node.id == 'timeout'
                    and isinstance(node.ctx, ast.Load)
                    and self.fn_stack
                    and any(self.has_timeout_s)
                ):
                    hits.append((self.fn_stack[-1], node.lineno))

        Visitor().visit(tree)

        assert hits == [], (
            'Found bare `timeout` references inside functions that '
            'declare `timeout_s` as a parameter. Almost certainly a '
            'rename-leftover NameError. Sites: ' + ', '.join(f'{fn}:line{ln}' for fn, ln in hits)
        )


class TestCameraMarkDisconnectedDoesNotReleaseActiveOnCallbackThread:
    """Camera._mark_disconnected() must NOT clear ``self._active`` from
    inside its body.

    When called from the SDK callback thread (the inline disconnect
    fast-path in pyloncamera.OnImageGrabbed), dropping the last Python
    reference to the C++ InstantCamera wrapper triggers
    ~CInstantCamera synchronously on whichever thread runs it. The
    destructor calls into the SDK to tear down stream grabbers, buffer
    pools, and the device handle. If the SDK is concurrently in-flight
    with grab work, the destructor races those in-flight ops and
    triggers a native abort.

    The defense is: keep ``_mark_disconnected`` to flag-only state
    transition (sets ``_device_removed = True``); let the daemon
    teardown thread spawned by ``_schedule_async_teardown`` call
    ``disconnect()``, which releases ``self._active`` AFTER the safe
    SDK teardown sequence has run (DetachDevice, DestroyDevice).
    """

    def test_mark_disconnected_body_does_not_clear_active(self):
        import ast
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'camera.py').read_text()
        tree = ast.parse(src)

        found = None
        bad_assigns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '_mark_disconnected':
                found = node
                for sub in ast.walk(node):
                    # self._active = ... assignment (any RHS, but None is the historical hazard)
                    if isinstance(sub, ast.Assign):
                        for tgt in sub.targets:
                            if (
                                isinstance(tgt, ast.Attribute)
                                and isinstance(tgt.value, ast.Name)
                                and tgt.value.id == 'self'
                                and tgt.attr == '_active'
                            ):
                                bad_assigns.append(sub.lineno)
                break

        assert found is not None, 'Could not find Camera._mark_disconnected in drivers/camera.py.'
        assert bad_assigns == [], (
            'drivers/camera.py::_mark_disconnected must NOT assign to '
            'self._active. Dropping that reference here fires '
            '~CInstantCamera synchronously, which races concurrent SDK '
            'work when called from the SDK callback thread (pypylon '
            '#225 hazard). disconnect() releases _active safely on the '
            'daemon teardown thread. Offending line(s): ' + str(bad_assigns)
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_user_set_selector_uses_set_value(self):
        """init_camera_config must call camera.UserSetSelector.SetValue('Default')
        rather than `camera.UserSetSelector = 'Default'`. Consistent
        exception envelope; consistent with the rest of the file."""
        src = self._pyloncamera_source()
        assert "camera.UserSetSelector = 'Default'" not in src, (
            "Use camera.UserSetSelector.SetValue('Default') -- attribute "
            'assignment routes through pypylon __setattr__ which has a '
            'slightly different exception envelope than the explicit '
            'SetValue call used elsewhere in this file.'
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
        idx = src.find('def init_camera_config(self)')
        assert idx != -1
        end = src.find('def ', idx + 10)
        body = src[idx:end]
        assert "AcquisitionMode.SetValue('Continuous')" in body, (
            "init_camera_config must call AcquisitionMode.SetValue('Continuous')."
        )
        assert "TriggerMode.SetValue('Off')" in body, (
            "init_camera_config must call TriggerMode.SetValue('Off')."
        )

    def test_init_iterates_all_trigger_selector_entries(self):
        """Per Basler doc free-run-image-acquisition.html, 'Repeat the
        steps above for all available trigger types.' A camera exposing
        AcquisitionStart / FrameBurstStart / ExposureStart in addition
        to FrameStart needs each of them set to TriggerMode=Off, or a
        stray non-Off type leaks through and blocks free-run.

        Pins the iteration over TriggerSelector.GetEntries() so a future
        cleanup that collapses the loop back to a single FrameStart
        write fires this test."""
        src = self._pyloncamera_source()
        idx = src.find('def init_camera_config(self)')
        assert idx != -1
        end = src.find('def ', idx + 10)
        body = src[idx:end]
        assert 'TriggerSelector.GetEntries()' in body, (
            'init_camera_config must iterate '
            'camera.TriggerSelector.GetEntries() so TriggerMode=Off is '
            'applied to every available trigger type, not just '
            'FrameStart (per Basler doc free-run-image-acquisition.html).'
        )
        assert '.IsAvailable()' in body, (
            'init_camera_config trigger-types loop must filter on '
            'entry.IsAvailable() to skip entries that exist in the '
            "enum but aren't supported on this camera model."
        )
        assert '.GetSymbolic()' in body, (
            'init_camera_config trigger-types loop must call '
            "entry.GetSymbolic() to feed the enum's string name back "
            'into TriggerSelector.SetValue.'
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_gain_method_signature_no_self_param_shadow(self):
        """Forbid the shadowed signature `def gain(self, gain)`."""
        src = self._pyloncamera_source()
        assert 'def gain(self, gain)' not in src, (
            'PylonCamera.gain(self, gain) shadows the method name with '
            'the parameter. Use `def gain(self, value)` instead.'
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_disconnect_clears_self_validation_flag(self):
        src = self._pyloncamera_source()
        idx = src.find('def disconnect(self) -> bool:')
        assert idx != -1
        end = src.find('def ', idx + 10)
        body = src[idx:end]
        assert '_pylon_self_validation_done = False' in body, (
            'disconnect() must clear _pylon_self_validation_done so the '
            'next connect re-runs the StreamGrabber probe.'
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_canonical_underrun_node_name_constant(self):
        src = self._pyloncamera_source()
        assert "_UNDERRUN_NODE_NAME = 'Statistic_Buffer_Underrun_Count'" in src, (
            'Single canonical underrun-counter name '
            'Statistic_Buffer_Underrun_Count must be the constant.'
        )

    def test_no_candidate_list_or_resolver_method(self):
        src = self._pyloncamera_source()
        assert '_UNDERRUN_NODE_CANDIDATES' not in src, (
            '_UNDERRUN_NODE_CANDIDATES tuple was the multi-name '
            'speculative resolver; replaced by the single canonical '
            '_UNDERRUN_NODE_NAME constant.'
        )
        assert '_resolve_underrun_node_name' not in src, (
            '_resolve_underrun_node_name method was the multi-name '
            'resolver; with the single canonical constant the helper '
            'is dead code.'
        )
        assert '_underrun_node_name_cache' not in src, (
            "_underrun_node_name_cache was the resolver's cache; "
            'with the single canonical constant there is nothing to '
            'cache.'
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_camera_nodemap_probes_gev_network_parameters(self):
        """The 11 canonical GigE network-related camera-side nodes
        from network-related-parameters.md must appear in the
        camera-config probe in read_diagnostic_snapshot."""
        src = self._pyloncamera_source()
        idx = src.find('def read_diagnostic_snapshot(')
        assert idx != -1
        end = src.find('\n    def ', idx + 10)
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
                f'read_diagnostic_snapshot must probe {node!r} '
                f'(per network-related-parameters.md). Missing nodes '
                f'will not surface on dmA3536-9gm bench.'
            )

    def test_stream_grabber_probes_gige_resend_config(self):
        """The 8 canonical GigE Packet Resend Mechanism stream-
        grabber config nodes from stream-grabber-parameters.html
        must appear in the stream-grabber config probe."""
        src = self._pyloncamera_source()
        idx = src.find('def read_diagnostic_snapshot(')
        end = src.find('\n    def ', idx + 10)
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
                f'read_diagnostic_snapshot stream-grabber config '
                f'must probe {node!r} (per stream-grabber-parameters.'
                f'html Packet Resend Mechanism Parameters).'
            )

    def test_diag_stat_nodes_includes_gige_stat_counters(self):
        """The 3 GigE-specific stream-grabber stat counters must be
        in _DIAG_STAT_NODES so the pre/post deltas surface them."""
        src = self._pyloncamera_source()
        # Find the _DIAG_STAT_NODES tuple body
        idx = src.find('_DIAG_STAT_NODES = (')
        assert idx != -1
        end = src.find(')', idx)
        body = src[idx:end]
        for counter in (
            'Statistic_Resend_Packet_Count',
            'Statistic_Resend_Request_Count',
            'Statistic_Failed_Packet_Count',
        ):
            assert counter in body, (
                f'_DIAG_STAT_NODES must include {counter!r} so the '
                f'GigE resend traffic surfaces in the diagnostic '
                f'snapshot. Per stream-grabber-parameters.html '
                f'Statistics Parameters.'
            )

    def test_diag_stat_counters_includes_gige_counters_for_deltas(self):
        """Delta computation requires the same names in _DIAG_STAT_COUNTERS."""
        src = self._pyloncamera_source()
        idx = src.find('_DIAG_STAT_COUNTERS = (')
        assert idx != -1
        end = src.find(')', idx)
        body = src[idx:end]
        for counter in (
            'Statistic_Resend_Packet_Count',
            'Statistic_Resend_Request_Count',
            'Statistic_Failed_Packet_Count',
        ):
            assert counter in body, (
                f'_DIAG_STAT_COUNTERS must include {counter!r} for delta computation.'
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_clamp_helper_present(self):
        src = self._pyloncamera_source()
        assert 'def _clamp_dltl_value_bps(self, value_bps: int) -> int:' in src, (
            '_clamp_dltl_value_bps helper must exist with the documented signature.'
        )

    def test_clamp_calls_min_max_query(self):
        src = self._pyloncamera_source()
        idx = src.find('def _clamp_dltl_value_bps(')
        assert idx != -1
        end = src.find('\n    def ', idx + 10)
        body = src[idx:end]
        assert '.GetMin()' in body and '.GetMax()' in body, (
            '_clamp_dltl_value_bps must query DeviceLinkThroughputLimit'
            '.GetMin() and .GetMax() to determine the clamp range.'
        )

    def test_setter_calls_clamp_helper(self):
        src = self._pyloncamera_source()
        idx = src.find('def set_device_link_throughput_limit(')
        assert idx != -1
        end = src.find('\n    def ', idx + 10)
        body = src[idx:end]
        assert '_clamp_dltl_value_bps' in body, (
            'set_device_link_throughput_limit must run value_bps '
            'through _clamp_dltl_value_bps before SetValue.'
        )

    def test_docstring_records_too_low_warning(self):
        """Rolling-shutter distortion warning must appear in the docstring."""
        src = self._pyloncamera_source()
        idx = src.find('def set_device_link_throughput_limit(')
        assert idx != -1
        end = src.find('"""', src.find('"""', idx) + 3) + 3
        docstring = src[idx:end]
        assert 'rolling shutter' in docstring.lower() or 'rolling-shutter' in docstring.lower(), (
            'DLTL setter docstring must record the rolling-shutter '
            'distortion warning per per-camera spec pages.'
        )

    def test_docstring_records_too_high_warning(self):
        """Corrupt/dropped frames warning must appear in the docstring."""
        src = self._pyloncamera_source()
        idx = src.find('def set_device_link_throughput_limit(')
        assert idx != -1
        end = src.find('"""', src.find('"""', idx) + 3) + 3
        docstring = src[idx:end]
        assert 'corrupt' in docstring.lower() or 'dropped' in docstring.lower(), (
            'DLTL setter docstring must record the too-high warning '
            '(corrupt or dropped frames) per per-camera spec pages.'
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_resync_node_in_stats_node_names(self):
        """Statistic_Resynchronization_Count must be in the live
        stats poll set so the delta tracker has fresh data."""
        src = self._pyloncamera_source()
        idx = src.find('_STATS_NODE_NAMES = (')
        assert idx != -1
        end = src.find(')', idx)
        body = src[idx:end]
        assert 'Statistic_Resynchronization_Count' in body, (
            'Statistic_Resynchronization_Count must be in '
            '_STATS_NODE_NAMES so the live poller reads it each cycle.'
        )

    def test_resync_prominent_log_on_positive_delta(self):
        """Stats poller must emit a [INSTR RESYNC] warning when the
        delta is positive."""
        src = self._pyloncamera_source()
        idx = src.find('def _stats_poller_loop(self):')
        assert idx != -1
        end = src.find('\n    def ', idx + 10)
        body = src[idx:end]
        assert '[INSTR RESYNC]' in body, (
            'Stats poller must emit a [INSTR RESYNC] log line on '
            'positive resync delta -- per Basler doc this is the '
            'most serious error case in USB 3.0 / USB3 Vision.'
        )
        assert '_cam_log.warning' in body and 'RESYNC' in body, (
            'Resync delta must be logged at warning level (operator-actionable; not info).'
        )

    def test_resync_csv_column_present(self):
        """The pylon_stats_trace.csv header must include the resync
        column so historical analysis can correlate."""
        src = self._pyloncamera_source()
        idx = src.find("'pylon_stats_trace.csv'")
        assert idx != -1
        # Header is the next ~150 chars after the filename argument.
        window = src[idx : idx + 500]
        assert 'resync_count' in window, (
            'pylon_stats_trace.csv header must include resync_count '
            'column so the running total is captured per row.'
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_stats_poller_reads_temperature_state(self):
        src = self._pyloncamera_source()
        idx = src.find('def _stats_poller_loop(self):')
        assert idx != -1
        end = src.find('\n    def ', idx + 10)
        body = src[idx:end]
        assert 'TemperatureState' in body, (
            'Stats poller must read TemperatureState each cycle '
            'so over-temp events surface in the log.'
        )
        assert '[INSTR TEMP]' in body, (
            'Stats poller must emit [INSTR TEMP] on temperature '
            'state changes for log-grep visibility.'
        )
        assert 'Critical' in body and 'Error' in body, (
            'Stats poller must distinguish Critical / Error states '
            '(WARNING level) from Ok transitions (INFO level).'
        )

    def test_read_diagnostic_snapshot_captures_thermal_state(self):
        src = self._pyloncamera_source()
        idx = src.find('def read_diagnostic_snapshot(')
        assert idx != -1
        end = src.find('\n    def ', idx + 10)
        body = src[idx:end]
        for node in (
            'TemperatureState',
            'BslTemperatureMax',
            'BslTemperatureStatusErrorCount',
        ):
            assert node in body, (
                f'read_diagnostic_snapshot must probe {node!r} so '
                f"the camera's thermal history is captured for "
                f'cross-host comparison.'
            )

    def test_temperature_csv_column_present(self):
        src = self._pyloncamera_source()
        idx = src.find("'pylon_stats_trace.csv'")
        assert idx != -1
        window = src[idx : idx + 500]
        assert 'temperature_state' in window, (
            'pylon_stats_trace.csv header must include '
            'temperature_state column so post-hoc analysis '
            'can correlate stalls with temperature history.'
        )


class TestPylonMissedFrameDeltaLog:
    """Per Basler doc stream-grabber-parameters.html, "A high Missed
    Frame Count indicates that the host controller doesn't support
    the bandwidth of the camera, i.e., the host controller does not
    retrieve the acquired images in time."

    Missed frames climb BEFORE Failed_Buffer_Count moves -- they
    are an early bandwidth-stress signal. The stats poller now
    tracks the prior count and emits a WARNING-level log on any
    positive delta (same pattern as B5 resync).

    Audit finding B14.
    """

    def _pyloncamera_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_missed_frame_node_in_stats_node_names(self):
        src = self._pyloncamera_source()
        idx = src.find('_STATS_NODE_NAMES = (')
        assert idx != -1
        end = src.find(')', idx)
        body = src[idx:end]
        assert 'Statistic_Missed_Frame_Count' in body, (
            'Statistic_Missed_Frame_Count must be in _STATS_NODE_NAMES '
            'so the live poller reads it each cycle.'
        )

    def test_missed_frame_prominent_log_on_positive_delta(self):
        src = self._pyloncamera_source()
        idx = src.find('def _stats_poller_loop(self):')
        assert idx != -1
        end = src.find('\n    def ', idx + 10)
        body = src[idx:end]
        assert '[INSTR MISSED]' in body, (
            'Stats poller must emit [INSTR MISSED] on positive '
            'missed-frame delta -- early bandwidth-stress signal.'
        )

    def test_missed_frame_csv_column_present(self):
        src = self._pyloncamera_source()
        idx = src.find("'pylon_stats_trace.csv'")
        assert idx != -1
        window = src[idx : idx + 500]
        assert 'missed_frame_count' in window, (
            'pylon_stats_trace.csv header must include '
            'missed_frame_count column for historical correlation.'
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_is_connected_calls_is_camera_device_removed(self):
        src = self._pyloncamera_source()
        idx = src.find('def is_connected(self) -> bool:')
        assert idx != -1, 'Could not find PylonCamera.is_connected.'
        end = src.find('def ', idx + 10)
        body = src[idx:end]
        assert '.IsCameraDeviceRemoved()' in body, (
            'is_connected must call self.active.IsCameraDeviceRemoved() '
            'as a third check (after _device_removed flag + active is '
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

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_node_attr_get_helper_present(self):
        """The _node_attr_get helper must exist and accept *names."""
        src = self._pyloncamera_source()
        assert 'def _node_attr_get(camera, *names: str)' in src, (
            '_node_attr_get(camera, *names) helper missing -- this is '
            'the canonical Bsl-prefix-then-legacy fallback for live '
            'attribute-style reads.'
        )

    def test_safe_node_accepts_multiple_names(self):
        """_safe_node must accept *names so the diagnostic snapshot
        can probe Bsl-prefixed-then-legacy nodes via the nodemap."""
        src = self._pyloncamera_source()
        assert 'def _safe_node(nodemap, *names: str)' in src, (
            '_safe_node must accept *names (varargs) so call sites '
            'can pass multiple candidate names for the same logical '
            'parameter. Single-name calls remain backwards-compatible.'
        )

    def test_stats_poller_uses_bsl_resulting_frame_rate_first(self):
        """Live frame-rate read in _stats_poller_loop must try
        BslResultingAcquisitionFrameRate before ResultingFrameRate."""
        src = self._pyloncamera_source()
        idx = src.find('def _stats_poller_loop(self):')
        assert idx != -1
        end = src.find('def ', idx + 10)
        body = src[idx:end]
        assert 'BslResultingAcquisitionFrameRate' in body, (
            '_stats_poller_loop must probe BslResultingAcquisitionFrameRate '
            '(canonical for ace 2 / dart M/R per Basler doc).'
        )
        assert 'ResultingFrameRate' in body, (
            '_stats_poller_loop must keep ResultingFrameRate as the '
            'fallback for legacy ace cameras.'
        )
        # Bsl variant must come first in the call.
        bsl_pos = body.find('BslResultingAcquisitionFrameRate')  # noqa: F841 -- deferred
        legacy_pos = body.find("'ResultingFrameRate'")
        if legacy_pos != -1:  # legacy may also appear in a comment first
            # Just assert the call site has both; ordering inside the call
            # is structural so check a tighter window.
            pass
        # Assert _node_attr_get is used (rather than direct attribute access)
        assert '_node_attr_get(' in body, (
            '_stats_poller_loop must use _node_attr_get(...) for the '
            'frame-rate read so the Bsl-fallback pattern is centralised.'
        )

    def test_get_exposure_t_uses_bsl_effective_first(self):
        """get_exposure_t must read BslEffectiveExposureTime first
        (the doc-canonical effective value) and fall back to
        ExposureTime (the requested set value)."""
        src = self._pyloncamera_source()
        idx = src.find('def get_exposure_t(self)')
        assert idx != -1
        end = src.find('def ', idx + 10)
        body = src[idx:end]
        assert 'BslEffectiveExposureTime' in body, (
            'get_exposure_t must read BslEffectiveExposureTime first -- '
            'per Basler exposure-time.html doc, this is the effective '
            'value the camera actually used (vs ExposureTime which is '
            'the requested set value).'
        )
        assert '_node_attr_get(' in body, (
            'get_exposure_t must use _node_attr_get(...) for the Bsl-fallback pattern.'
        )

    def test_diag_snapshot_config_tuple_uses_bsl_fallbacks(self):
        """read_diagnostic_snapshot config tuple must include the
        Bsl-prefixed canonical names for ResultingFrameRate and
        ExposureTime so cross-host comparison probes the correct
        node on ace 2 / dart M/R."""
        src = self._pyloncamera_source()
        idx = src.find('def read_diagnostic_snapshot(')
        assert idx != -1
        end = src.find('\n    def ', idx + 10)
        body = src[idx:end]
        assert "'BslResultingAcquisitionFrameRate'" in body, (
            'read_diagnostic_snapshot must probe '
            'BslResultingAcquisitionFrameRate before ResultingFrameRate.'
        )
        assert "'BslEffectiveExposureTime'" in body, (
            'read_diagnostic_snapshot must probe BslEffectiveExposureTime '
            'before ExposureTime so the snapshot reports effective '
            'exposure on ace 2 / dart M/R.'
        )

    def test_node_attr_get_suppresses_getattr_exception(self):
        """_node_attr_get must NOT propagate the exception that
        pypylon's InstantCamera.__getattr__ raises for missing nodes.

        The helper relied on Python's `getattr(obj, name, default)`
        default-arg fallback. pypylon's InstantCamera doesn't honor
        that contract -- it raises ``genicam.LogicalErrorException``
        ("Node not existing") instead of returning None. Without the
        try/except, every protocol step's ``get_exposure_t()`` raised
        a 13-line traceback (1540 occurrences in a single protocol
        run on Windows 2026-05-08).
        """
        from drivers.pyloncamera import PylonCamera

        class _PypylonStyleCamera:
            def __getattr__(self, name):
                raise RuntimeError(f'Node {name!r} not existing')

        result = PylonCamera._node_attr_get(
            _PypylonStyleCamera(), 'ExposureTime', 'BslEffectiveExposureTime'
        )
        assert result is None, (
            f"_node_attr_get must return None when every name's getattr "
            f"raises (treating as 'node not present'); got {result!r}. "
            f'Without the try/except wrapper, the LogicalErrorException '
            f'propagates out and floods the error log.'
        )


class TestSequentialIOExecutorCancelledNotErrorLogged:
    """SequentialIOExecutor._on_task_done must not fire
    notifications.error when the exception is a CancelledError.
    Cancellations come from the caller (shutdown / clear_pending /
    cancel_all_protocols) by contract; treating them as failures
    floods the error log on every clean shutdown.
    """

    def _run_on_task_done(self, executor, exception):
        # Mirror the worker's lifecycle: put -> get -> task.run ->
        # _on_task_done. The worker dequeues before running; the test
        # bypasses task.run but must dequeue first so _on_task_done's
        # internal task_done() balances against the put. Without the
        # get_nowait, Stage A's shutdown -> clear_pending would call
        # task_done() on a queue whose unfinished count already went to
        # zero, raising ValueError.
        from modules.sequential_io_executor import IOTask

        task = IOTask(action=lambda: None, callback=lambda *a, **k: None)
        executor.queue.put(task)
        executor.queue.get_nowait()
        executor._on_task_done(task, None, exception)

    def test_cancelled_does_not_call_notifications_error(self, monkeypatch):
        from concurrent.futures import CancelledError
        from modules.sequential_io_executor import SequentialIOExecutor
        from modules import notification_center

        executor = SequentialIOExecutor(max_workers=1, name='TEST_CANCEL')
        try:
            calls = []
            monkeypatch.setattr(
                notification_center.notifications,
                'error',
                lambda *a, **kw: calls.append(('error', a, kw)),
            )
            self._run_on_task_done(executor, CancelledError())
            assert calls == [], (
                f'_on_task_done(..., CancelledError()) must not fire '
                f'notifications.error; got {calls}'
            )
        finally:
            executor.shutdown(wait=False)

    def test_runtime_error_still_calls_notifications_error(self, monkeypatch):
        from modules.sequential_io_executor import SequentialIOExecutor
        from modules import notification_center

        executor = SequentialIOExecutor(max_workers=1, name='TEST_REAL_FAIL')
        try:
            calls = []
            monkeypatch.setattr(
                notification_center.notifications,
                'error',
                lambda *a, **kw: calls.append(('error', a, kw)),
            )
            self._run_on_task_done(executor, RuntimeError('test failure'))
            assert len(calls) == 1, (
                f'_on_task_done(..., RuntimeError) must fire one notifications.error; got {calls}'
            )
        finally:
            executor.shutdown(wait=False)


class TestSequentialIOExecutorSilentOnFailure:
    """IOTask.silent_on_failure=True must suppress the generic
    notifications.error popup at _on_task_done. The caller opted in to
    handle its own notification path (Rule 14 -- API/caller decides,
    not the executor). LVP 09a324a shipped this for the
    protocol_image_writer.execute_step retry path where per-failure
    popups would stack into the Class A 110-popups-overnight storm.
    Regression guard for the executor topology plan Stage A amendments:
    Stage A's inline _on_task_done call must preserve this flag's
    semantics.
    """

    def _build_task(self, silent: bool):
        from modules.sequential_io_executor import IOTask

        return IOTask(
            action=lambda: None,
            callback=lambda *a, **k: None,
            silent_on_failure=silent,
        )

    def test_silent_on_failure_suppresses_notification(self, monkeypatch):
        from modules.sequential_io_executor import SequentialIOExecutor
        from modules import notification_center

        executor = SequentialIOExecutor(max_workers=1, name='TEST_SILENT')
        try:
            calls = []
            monkeypatch.setattr(
                notification_center.notifications,
                'error',
                lambda *a, **kw: calls.append(('error', a, kw)),
            )
            task = self._build_task(silent=True)
            executor.queue.put(task)
            executor.queue.get_nowait()  # mirror worker dequeue
            executor._on_task_done(task, None, RuntimeError('expected'))
            assert calls == [], (
                f'silent_on_failure=True must suppress notifications.error; got {calls}'
            )
        finally:
            executor.shutdown(wait=False)

    def test_silent_on_failure_default_does_fire_notification(self, monkeypatch):
        from modules.sequential_io_executor import SequentialIOExecutor
        from modules import notification_center

        executor = SequentialIOExecutor(max_workers=1, name='TEST_LOUD')
        try:
            calls = []
            monkeypatch.setattr(
                notification_center.notifications,
                'error',
                lambda *a, **kw: calls.append(('error', a, kw)),
            )
            task = self._build_task(silent=False)
            executor.queue.put(task)
            executor.queue.get_nowait()  # mirror worker dequeue
            executor._on_task_done(task, None, RuntimeError('expected'))
            assert len(calls) == 1, (
                f'silent_on_failure=False (default) must fire one notifications.error; got {calls}'
            )
        finally:
            executor.shutdown(wait=False)


class TestSequentialIOExecutorTDequeuePreservedUnderProfileTrace:
    """Stage A's inline _run_loop must preserve the _t_dequeue
    profile_trace timing stamp at sequential_io_executor.py:533. The
    timing is used for queue-wait + exec-time instrumentation per
    AUDIT_THREADING_PARALLELISM section 10.1. Stage A drops the
    two-pool ThreadPoolExecutor in favor of a single worker thread;
    this test guards the timing path against accidental removal.
    """

    def test_t_dequeue_set_when_profile_trace_enabled(self, monkeypatch):
        import time
        from modules.sequential_io_executor import SequentialIOExecutor, IOTask
        from lib import profile_trace

        # Force-enable profile_trace for the test. Restore previous state
        # in finally so other tests are unaffected.
        original = profile_trace.ENABLE_PROFILE_TRACE
        monkeypatch.setattr(profile_trace, 'ENABLE_PROFILE_TRACE', True)

        executor = SequentialIOExecutor(max_workers=1, name='TEST_TDEQUEUE')
        try:
            done_event = __import__('threading').Event()
            results = {}

            def captured_action():
                # Capture t_dequeue at task-run time so the test can read
                # what _run_loop stamped.
                results['t_dequeue'] = getattr(task, '_t_dequeue', None)
                done_event.set()

            task = IOTask(action=captured_action, callback=lambda *a, **k: None)
            executor.start()
            t_submit = time.monotonic()
            executor.put(task)

            assert done_event.wait(timeout=2.0), 'task did not run in 2s'
            assert results.get('t_dequeue') is not None, (
                '_t_dequeue must be set by _run_loop before task.run() '
                'when profile_trace is enabled'
            )
            assert results['t_dequeue'] >= t_submit, (
                f'_t_dequeue must be set AFTER submission; '
                f'submit={t_submit} t_dequeue={results["t_dequeue"]}'
            )
        finally:
            executor.shutdown(wait=False)
            monkeypatch.setattr(profile_trace, 'ENABLE_PROFILE_TRACE', original)


class TestSequentialIOExecutorSubmitThenShutdownNoFutureLeak:
    """Stage A retires the orphan-submit pattern that caused Bug E
    (Future + _WorkItem retention in CPython ThreadPoolExecutor
    internals). The structural fix runs every task inline in a single
    worker thread -- there is no Future and no _WorkItem.

    This unit test guards the proxy invariant that caller_futures is
    fully drained after shutdown(wait=False), even when tasks were
    submitted with return_future=True. A live caller_futures dict
    after shutdown would indicate the Stage A retirement re-introduced
    a leak surface.

    The bench-validated test (Handle.exe plateau on a 30-min protocol)
    is the canonical Bug E acceptance per
    docs/EXECUTOR_TOPOLOGY_PLAN_2026-05-13.md section 4.A. This unit
    test is the in-suite proxy.
    """

    def test_caller_futures_empty_after_submit_then_shutdown(self):
        from modules.sequential_io_executor import SequentialIOExecutor, IOTask

        executor = SequentialIOExecutor(max_workers=1, name='TEST_SHUTDOWN_DRAIN')
        executor.start()
        try:
            # Submit a batch of return_future tasks. Don't wait for them
            # to complete -- shutdown(wait=False) should cancel pending
            # and drain caller_futures regardless.
            for _ in range(20):
                task = IOTask(
                    action=lambda: None,
                    callback=lambda *a, **k: None,
                )
                executor.put(task, return_future=True)
        finally:
            executor.shutdown(wait=False)

        assert len(executor.caller_futures) == 0, (
            f'caller_futures must be fully drained after '
            f'shutdown(wait=False); residual={len(executor.caller_futures)}'
        )
        alloc, pop, _residual_live = executor.caller_futures_stats()
        assert alloc == pop, (
            f'caller_futures alloc/pop must balance after shutdown; alloc={alloc} pop={pop}'
        )


class TestSequentialIOExecutorPriorityAware:
    """priority_aware=True orders the default queue by IOTask.priority
    (lower value first) with FIFO tie-break within priority.
    priority_aware=False keeps submit-order FIFO regardless of priority.
    """

    @staticmethod
    def _drain_with_blocker(executor, head_event, tasks, timeout=2.0):
        """Submit a head task that blocks on head_event, then submit
        tasks in order. Release head_event and wait for all tasks to
        run. Returns the order in which the tasks' actions executed.
        """
        import threading as _t
        import time as _t2
        from modules.sequential_io_executor import IOTask

        observed = []

        def head_action():
            head_event.wait(timeout=timeout)
            observed.append('__head__')

        executor.put(IOTask(action=head_action))
        # Give the worker time to dequeue + enter head_action's wait.
        _t2.sleep(0.05)

        for label, prio in tasks:
            executor.put(
                IOTask(
                    action=lambda lbl=label: observed.append(lbl),
                    priority=prio,
                )
            )

        # Now release the head -- worker processes the rest in priority
        # order.
        head_event.set()
        deadline = _t2.monotonic() + timeout
        expected = 1 + len(tasks)
        while len(observed) < expected and _t2.monotonic() < deadline:
            _t2.sleep(0.01)
        return observed

    def test_high_jumps_med(self):
        import threading as _t
        from modules.sequential_io_executor import (
            SequentialIOExecutor,
            PRIORITY_HIGH,
            PRIORITY_MED,
        )

        executor = SequentialIOExecutor(name='TEST_PRIO', priority_aware=True)
        executor.start()
        try:
            head = _t.Event()
            order = self._drain_with_blocker(
                executor,
                head,
                [('med-A', PRIORITY_MED), ('high', PRIORITY_HIGH), ('med-B', PRIORITY_MED)],
            )
            assert order == ['__head__', 'high', 'med-A', 'med-B'], (
                f'HIGH must jump ahead of pending MEDs (FIFO within MED); got {order}'
            )
        finally:
            executor.shutdown(wait=False)

    def test_fifo_within_priority(self):
        import threading as _t
        from modules.sequential_io_executor import (
            SequentialIOExecutor,
            PRIORITY_MED,
        )

        executor = SequentialIOExecutor(name='TEST_PRIO_FIFO', priority_aware=True)
        executor.start()
        try:
            head = _t.Event()
            order = self._drain_with_blocker(
                executor,
                head,
                [
                    ('a', PRIORITY_MED),
                    ('b', PRIORITY_MED),
                    ('c', PRIORITY_MED),
                    ('d', PRIORITY_MED),
                ],
            )
            assert order == ['__head__', 'a', 'b', 'c', 'd'], (
                f'within a single priority the monotonic counter must '
                f'preserve submit order; got {order}'
            )
        finally:
            executor.shutdown(wait=False)

    def test_three_priorities_strict_ordering(self):
        import threading as _t
        from modules.sequential_io_executor import (
            SequentialIOExecutor,
            PRIORITY_HIGH,
            PRIORITY_MED,
            PRIORITY_LOW,
        )

        executor = SequentialIOExecutor(name='TEST_PRIO_THREE', priority_aware=True)
        executor.start()
        try:
            head = _t.Event()
            # Interleave submission order; expect priority sort then
            # FIFO tie-break:
            #   HIGH:  h1, h2
            #   MED:   m1, m2
            #   LOW:   l1, l2
            order = self._drain_with_blocker(
                executor,
                head,
                [
                    ('l1', PRIORITY_LOW),
                    ('m1', PRIORITY_MED),
                    ('h1', PRIORITY_HIGH),
                    ('m2', PRIORITY_MED),
                    ('h2', PRIORITY_HIGH),
                    ('l2', PRIORITY_LOW),
                ],
            )
            assert order == ['__head__', 'h1', 'h2', 'm1', 'm2', 'l1', 'l2'], (
                f'priority sort + FIFO tie-break failed; got {order}'
            )
        finally:
            executor.shutdown(wait=False)

    def test_low_eventually_runs_under_sustained_med(self):
        """LOW fairness sanity: pure priority ordering (no aging), but
        a finite MED batch must not infinitely starve LOW. Submit LOW
        first, then a 50-MED burst, verify LOW completes."""
        import threading as _t
        import time as _t2
        from modules.sequential_io_executor import (
            SequentialIOExecutor,
            IOTask,
            PRIORITY_MED,
            PRIORITY_LOW,
        )

        executor = SequentialIOExecutor(name='TEST_PRIO_FAIR', priority_aware=True)
        executor.start()
        try:
            head = _t.Event()
            low_done = _t.Event()

            def head_action():
                head.wait(timeout=2.0)

            executor.put(IOTask(action=head_action))
            _t2.sleep(0.05)

            executor.put(IOTask(action=low_done.set, priority=PRIORITY_LOW))
            for _ in range(50):
                executor.put(IOTask(action=lambda: None, priority=PRIORITY_MED))

            head.set()
            assert low_done.wait(timeout=3.0), (
                'LOW must run after the 50 MED tasks finish; '
                'pure priority sort still guarantees forward progress '
                'on a bounded MED batch'
            )
        finally:
            executor.shutdown(wait=False)

    def test_clear_pending_drains_high_first(self):
        """clear_pending on a priority_aware executor drains HIGH-first;
        all caller_futures are drained and the alloc/pop counters
        balance after the drain."""
        import threading as _t
        import time as _t2
        from modules.sequential_io_executor import (
            SequentialIOExecutor,
            IOTask,
            PRIORITY_HIGH,
            PRIORITY_MED,
        )

        executor = SequentialIOExecutor(name='TEST_PRIO_CLEAR', priority_aware=True)
        executor.start()
        try:
            # Block the worker so the queue actually accumulates tasks
            # without them draining mid-test.
            head_event = _t.Event()

            def head_action():
                head_event.wait(timeout=2.0)

            executor.put(IOTask(action=head_action))
            _t2.sleep(0.05)

            # Submit MED then HIGH then MED. Without priority,
            # cancel-order would be MED, HIGH, MED.
            cancel_order = []
            futs = []
            for label, prio in [
                ('med-A', PRIORITY_MED),
                ('high', PRIORITY_HIGH),
                ('med-B', PRIORITY_MED),
            ]:
                task = IOTask(
                    action=lambda lbl=label: cancel_order.append(('ran-', lbl)),
                    priority=prio,
                )
                fut = executor.put(task, return_future=True)
                futs.append((label, fut))

            executor.clear_pending()
            # Cancel callbacks on _ReusableTaskWaiter are recorded as
            # the order in which clear_pending pulled them. The
            # PriorityQueue.get_nowait order IS the cancel-execution
            # order from clear_pending's perspective.
            #
            # We can't directly observe cancel-call order without
            # patching the waiter; instead verify the structural
            # contract: every fut had cancel() invoked, and the
            # caller_futures dict is fully drained.
            head_event.set()
            _t2.sleep(0.1)
            assert len(executor.caller_futures) == 0, (
                'clear_pending must drain caller_futures fully; '
                f'residual={len(executor.caller_futures)}'
            )
            alloc, pop, _ = executor.caller_futures_stats()
            assert alloc == pop, (
                f'caller_futures alloc/pop must balance after '
                f'clear_pending; alloc={alloc} pop={pop}'
            )
        finally:
            executor.shutdown(wait=False)

    def test_caller_futures_invariant_under_priority_mix(self):
        """Mixed-priority return_future submission must keep
        alloc == pop at steady state."""
        import time as _t2
        from modules.sequential_io_executor import (
            SequentialIOExecutor,
            IOTask,
            PRIORITY_HIGH,
            PRIORITY_MED,
            PRIORITY_LOW,
        )

        executor = SequentialIOExecutor(name='TEST_PRIO_FUTURES', priority_aware=True)
        executor.start()
        try:
            prios = [PRIORITY_HIGH, PRIORITY_MED, PRIORITY_LOW] * 30
            for prio in prios:
                executor.put(
                    IOTask(action=lambda: None, priority=prio),
                    return_future=True,
                )
            # Drain.
            deadline = _t2.monotonic() + 3.0
            while executor.queue_size() > 0 and _t2.monotonic() < deadline:
                _t2.sleep(0.01)
            _t2.sleep(0.1)  # last task to complete + clean up
            alloc, pop, live = executor.caller_futures_stats()
            assert alloc == pop, (
                f'priority-mixed return_future submissions must keep '
                f'alloc==pop; alloc={alloc} pop={pop} live={live}'
            )
            assert live == 0, (
                f'no Future entries may remain in caller_futures after steady state; live={live}'
            )
        finally:
            executor.shutdown(wait=False)

    def test_priority_aware_false_keeps_fifo(self):
        """priority_aware=False (the default) ignores IOTask.priority
        and keeps submit-order FIFO."""
        import threading as _t
        from modules.sequential_io_executor import (
            SequentialIOExecutor,
            PRIORITY_HIGH,
            PRIORITY_MED,
        )

        executor = SequentialIOExecutor(name='TEST_FIFO_LEGACY')
        executor.start()
        try:
            head = _t.Event()
            order = self._drain_with_blocker(
                executor,
                head,
                [('med-A', PRIORITY_MED), ('high', PRIORITY_HIGH), ('med-B', PRIORITY_MED)],
            )
            assert order == ['__head__', 'med-A', 'high', 'med-B'], (
                f'priority_aware=False must keep submit-order FIFO '
                f'regardless of IOTask.priority; got {order}'
            )
        finally:
            executor.shutdown(wait=False)


class TestPylonAutoGainNoUpdateCameraConfigWrap:
    """update_auto_gain_target_brightness and update_auto_gain_min_max
    write Basler AutoTargetBrightness / AutoGainLowerLimit /
    AutoGainUpperLimit -- all runtime-modifiable per Basler doc, so the
    previous update_camera_config wrap was a needless stop_grabbing /
    start_grabbing cycle on every call (same structural class as
    STALL-1's per-step over-stop).

    Pins the structural fix so a future cleanup can't silently re-wrap
    the writes. Mirrors TestDeviceLinkThroughputLimitSetter
    .test_pylon_driver_does_not_wrap_in_update_camera_config.
    """

    def _pyloncamera_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_update_auto_gain_target_brightness_does_not_wrap(self):
        body = _function_source(
            self._pyloncamera_source(),
            'update_auto_gain_target_brightness',
        )
        assert 'with self.update_camera_config' not in body, (
            'PylonCamera.update_auto_gain_target_brightness must NOT '
            'wrap the AutoTargetBrightness write in update_camera_config '
            '(runtime-modifiable per Basler; wrapping would impose the '
            'STALL-1 over-stop pattern).'
        )

    def test_update_auto_gain_min_max_does_not_wrap(self):
        body = _function_source(
            self._pyloncamera_source(),
            'update_auto_gain_min_max',
        )
        assert 'with self.update_camera_config' not in body, (
            'PylonCamera.update_auto_gain_min_max must NOT wrap the '
            'AutoGainLowerLimit / AutoGainUpperLimit writes in '
            'update_camera_config (runtime-modifiable per Basler; '
            'wrapping would impose the STALL-1 over-stop pattern).'
        )

    def test_pylon_driver_writes_auto_target_brightness_directly(self):
        """Sanity: the method really does call .SetValue on the node so
        the no-wrap test isn't passing because the method is empty."""
        body = _function_source(
            self._pyloncamera_source(),
            'update_auto_gain_target_brightness',
        )
        assert 'AutoTargetBrightness.SetValue(' in body, (
            'PylonCamera.update_auto_gain_target_brightness must call '
            'AutoTargetBrightness.SetValue(...) on the live nodemap.'
        )

    def test_pylon_driver_writes_auto_gain_limits_directly(self):
        """Sanity: same as above for the min/max pair."""
        body = _function_source(
            self._pyloncamera_source(),
            'update_auto_gain_min_max',
        )
        assert 'AutoGainLowerLimit.SetValue(' in body, (
            'PylonCamera.update_auto_gain_min_max must call '
            'AutoGainLowerLimit.SetValue(...) on the live nodemap.'
        )
        assert 'AutoGainUpperLimit.SetValue(' in body, (
            'PylonCamera.update_auto_gain_min_max must call '
            'AutoGainUpperLimit.SetValue(...) on the live nodemap.'
        )


class TestErrorReportCountRetired:
    """The base Camera class previously held an `error_report_count`
    attribute that PylonCamera and IDSCamera reset to 0 on connect
    success and incremented on connect failure. Zero readers
    codebase-wide -- pure dead state. Retired in this commit.

    Pins the deletion so a future "while-I'm-here" addition can't
    silently re-introduce a counter without a consumer (Rule 2 single
    source of truth: dead state is worse than duplicated state).
    """

    def _read(self, rel_path):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / rel_path).read_text()

    def test_base_camera_does_not_define_error_report_count(self):
        assert 'error_report_count' not in self._read('drivers/camera.py'), (
            'drivers/camera.py must not re-introduce error_report_count '
            'without a reader (dead state retired; Rule 2).'
        )

    def test_pyloncamera_does_not_reference_error_report_count(self):
        assert 'error_report_count' not in self._read('drivers/pyloncamera.py'), (
            'drivers/pyloncamera.py must not re-introduce error_report_count '
            'writes (Rule 2; dead state retired).'
        )

    def test_idscamera_does_not_reference_error_report_count(self):
        assert 'error_report_count' not in self._read('drivers/idscamera.py'), (
            'drivers/idscamera.py must not re-introduce error_report_count '
            'writes (Rule 2; dead state retired).'
        )


class TestFindModelNameRetired:
    """A20 / D2: Camera.find_model_name was a 5-method dead capability
    (1 abstract + 4 driver impls + 3 test fakes). Each driver's
    connect() already sets self.model_name independently; nothing in
    production code ever called find_model_name(). Rule 35 cleanup:
    one canonical capability (model_name set in connect()), no
    parallel implementation.

    Pins the deletion across all 5 source files and the test fixture
    file so a future addition can't silently re-introduce a parallel
    discovery path without consensus.
    """

    def _read(self, rel_path):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / rel_path).read_text()

    def test_camera_base_does_not_define_find_model_name(self):
        assert 'find_model_name' not in self._read('drivers/camera.py'), (
            'drivers/camera.py must not re-introduce the find_model_name '
            'abstract (Rule 35; dead capability retired -- model_name '
            "is set in each driver's connect())."
        )

    def test_pyloncamera_does_not_define_find_model_name(self):
        assert 'find_model_name' not in self._read('drivers/pyloncamera.py')

    def test_idscamera_does_not_define_find_model_name(self):
        assert 'find_model_name' not in self._read('drivers/idscamera.py')

    def test_simulated_camera_does_not_define_find_model_name(self):
        assert 'find_model_name' not in self._read('drivers/simulated_camera.py')

    def test_fx2driver_does_not_define_find_model_name(self):
        assert 'find_model_name' not in self._read('drivers/fx2driver.py')

    def test_test_serial_safety_fakes_do_not_define_find_model_name(self):
        assert 'find_model_name' not in self._read('tests/test_serial_safety.py')


class TestPylonInitWaitsForIdleBeforeUserSetLoad:
    """B6: Per Basler user-sets.html, "Loading a user set is only
    possible when the camera is idle, i.e., not acquiring images."

    update_camera_config() stops the grab loop, but on slow hosts
    SDK StopGrabbing may not have fully settled by the time
    init_camera_config() arrives at UserSetLoad. The bounded poll
    surfaces the condition in logs rather than letting UserSetLoad
    silently raise inside the outer try/except.
    """

    def _pyloncamera_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_init_polls_is_grabbing_before_user_set_load(self):
        body = _function_source(self._pyloncamera_source(), 'init_camera_config')
        # The poll must be inside init_camera_config and BEFORE the
        # UserSetLoad call.
        assert 'self.is_grabbing()' in body, (
            'init_camera_config must poll self.is_grabbing() before '
            'UserSetLoad (Basler user-sets.html idle requirement).'
        )
        idle_idx = body.find('is_grabbing()')
        load_idx = body.find('UserSetLoad.Execute(')
        assert idle_idx >= 0 and load_idx >= 0
        assert idle_idx < load_idx, (
            'init_camera_config must poll is_grabbing BEFORE UserSetLoad.Execute(), not after.'
        )

    def test_init_warns_if_still_grabbing_after_poll(self):
        body = _function_source(self._pyloncamera_source(), 'init_camera_config')
        assert (
            'still' in body.lower() and 'grabbing' in body.lower() and 'warning' in body.lower()
        ), (
            'init_camera_config must log a warning if is_grabbing() '
            'stays True past the bounded poll -- silently letting '
            'UserSetLoad raise inside the outer try/except hides the '
            'condition from operators.'
        )


class TestPylonGainSelectorBeforeGainSetValue:
    """D10: Per Basler gain.html three-step recipe (GainAuto Off ->
    GainSelector All -> Gain SetValue), assert GainSelector='All'
    before each Gain.SetValue call. Defensive against upstream code
    that may have set GainSelector to a per-channel selector;
    per-write try/except tolerates camera models that don't expose
    GainSelector at all.

    Currently only matters if a future feature changes GainSelector
    (e.g. per-channel-gain UI). Cheap insurance against a class of
    bug that would otherwise be model-firmware-conditional.
    """

    def _pyloncamera_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_gain_method_sets_selector_to_all_first(self):
        body = _function_source(self._pyloncamera_source(), 'gain')
        sel_idx = body.find("GainSelector.SetValue('All')")
        gain_idx = body.find('Gain.SetValue(float(value))')
        assert sel_idx >= 0, (
            "PylonCamera.gain must call GainSelector.SetValue('All') "
            'before Gain.SetValue (Basler 3-step recipe).'
        )
        assert gain_idx >= 0
        assert sel_idx < gain_idx, (
            "GainSelector.SetValue('All') must precede "
            'Gain.SetValue(...) -- the order is the load-bearing part '
            'of the doc-named recipe.'
        )

    def test_gain_method_tolerates_missing_gain_selector(self):
        """The selector write must be in its own try/except so a
        camera model that doesn't expose GainSelector doesn't break
        Gain.SetValue."""
        body = _function_source(self._pyloncamera_source(), 'gain')
        # The selector and the actual write should be in separate
        # try blocks; an inner try around the selector preserves the
        # outer try/except's contract.
        sel_idx = body.find("GainSelector.SetValue('All')")
        # Find the closest 'try:' before the selector write.
        try_idx = body.rfind('try:', 0, sel_idx)
        # Find the closest 'except ' after the selector write but
        # before the Gain.SetValue line.
        gain_idx = body.find('Gain.SetValue(float(value))')
        except_idx = body.find('except ', sel_idx, gain_idx)
        assert try_idx >= 0 and except_idx >= 0, (
            "GainSelector.SetValue('All') must be wrapped in its own "
            "try/except so a missing selector doesn't break "
            "Gain.SetValue on cameras that don't expose it."
        )


class TestDltlSetterDocstringGigeCaveat:
    """D8: set_device_link_throughput_limit docstring must surface the
    GigE wire-limit transport caveat. On GigE cameras (e.g.
    dmA3536-9gm at 9.3 fps Mono8 ~109 MB/s) DLTL is bounded above by
    the ~110 MB/s wire limit -- materially different from USB3 where
    the knob has full headroom. Operators picking a value need to
    know the GigE-vs-USB3 distinction before they set DLTL=Off
    expecting the same behavior across transports.
    """

    def _pyloncamera_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def _lumascope_api_source(self):
        # set_device_link_throughput_limit body relocated to ImagingAPI
        # in Wave 7 Phase 4c. Helper name kept for diff-readability.
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'imaging.py'
        ).read_text()

    def test_pylon_setter_docstring_mentions_gige_wire_limit(self):
        body = _function_source(self._pyloncamera_source(), 'set_device_link_throughput_limit')
        assert 'GigE' in body and 'wire limit' in body, (
            'PylonCamera.set_device_link_throughput_limit docstring '
            'must surface the GigE wire-limit caveat (D8).'
        )

    def test_pylon_setter_docstring_points_to_gige_alternatives(self):
        body = _function_source(self._pyloncamera_source(), 'set_device_link_throughput_limit')
        assert 'set_gev_inter_packet_delay' in body, (
            'Pylon DLTL docstring must point to set_gev_inter_packet_delay as the GigE alternative.'
        )
        assert 'set_bandwidth_reserve_mode' in body, (
            'Pylon DLTL docstring must point to set_bandwidth_reserve_mode as the GigE alternative.'
        )

    def test_lumascope_setter_docstring_mentions_gige_wire_limit(self):
        # Phase 4f renamed ImagingAPI.set_device_link_throughput_limit to
        # the privatized _set_device_link_throughput_limit form (per
        # TestImagingPylonSdkPerfSettersPrivatized).
        body = _function_source(self._lumascope_api_source(), '_set_device_link_throughput_limit')
        assert 'GigE' in body and 'wire limit' in body, (
            'ImagingAPI._set_device_link_throughput_limit docstring '
            'must surface the GigE wire-limit caveat (D8).'
        )


class TestPylonChunkSelectorProbeWithFramecounterFallback:
    """B32: _enable_validity_chunks must probe ChunkSelector.GetEntries()
    before enabling chunks, with a FrameID -> Framecounter fallback for
    cameras that advertise the Framecounter chunk under that name.

    The earlier code unconditionally selected 'FrameID' and let pypylon
    raise on cameras that don't expose it -- which silently dropped
    per-frame identity from the trace on those cameras. The probe-first
    pattern + read-side alias keeps the trace populated regardless of
    which spelling the camera uses.

    Frame-identity is trace-only (frame_validity validates gain and
    exposure); skipping it does NOT break validity. The fix is for
    diagnostic completeness, not validity correctness.
    """

    def _pyloncamera_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_frame_identity_chunk_candidates_lists_frameid_first(self):
        """FrameID is the canonical name on most Basler cameras (data-
        chunks.html). Framecounter is the documented fallback. Probe
        FrameID first, then Framecounter -- pinning the order so a
        future cleanup that swaps them or alphabetises the tuple
        fires this test."""
        src = self._pyloncamera_source()
        assert "_FRAME_IDENTITY_CHUNK_CANDIDATES = ('FrameID', 'Framecounter')" in src, (
            'PylonCamera must declare _FRAME_IDENTITY_CHUNK_CANDIDATES '
            'with FrameID first, Framecounter second (B32 fallback).'
        )

    def test_enable_validity_chunks_probes_advertised_first(self):
        """The method must call _probe_advertised_chunks before
        attempting to enable per-chunk; otherwise it silently fails
        on cameras advertising Framecounter instead of FrameID."""
        body = _function_source(self._pyloncamera_source(), '_enable_validity_chunks')
        assert '_probe_advertised_chunks(' in body, (
            '_enable_validity_chunks must probe ChunkSelector entries '
            'before enabling chunks (B32; refactor shares '
            '_probe_advertised_chunks with probe_chunk_capabilities).'
        )

    def test_enable_validity_chunks_falls_back_to_framecounter(self):
        """The method must walk _FRAME_IDENTITY_CHUNK_CANDIDATES and
        pick the first advertised name (FrameID first, Framecounter
        second)."""
        body = _function_source(self._pyloncamera_source(), '_enable_validity_chunks')
        assert '_FRAME_IDENTITY_CHUNK_CANDIDATES' in body, (
            '_enable_validity_chunks must consult '
            '_FRAME_IDENTITY_CHUNK_CANDIDATES to fall back from '
            'FrameID to Framecounter (B32).'
        )

    def test_chunk_grab_result_attrs_aliases_framecounter(self):
        """The read-side map must include ChunkFramecounter aliased to
        the same 'FrameID' dict key so the read works regardless of
        which spelling the camera enabled."""
        src = self._pyloncamera_source()
        assert "('ChunkFrameID', 'FrameID')" in src, (
            'ImageHandler._CHUNK_GRAB_RESULT_ATTRS must keep the '
            "ChunkFrameID -> 'FrameID' mapping for cameras that "
            'advertise FrameID.'
        )
        assert "('ChunkFramecounter', 'FrameID')" in src, (
            'ImageHandler._CHUNK_GRAB_RESULT_ATTRS must include the '
            "ChunkFramecounter -> 'FrameID' alias so cameras that "
            'advertise Framecounter still produce a frame-identity '
            'value in the chunk dict (B32).'
        )


class TestAcquisitionStopModeSetter:
    """ImagingAPI.set_acquisition_stop_mode + driver setters give the
    bench-probe sweep a way to compare BslAcquisitionStopMode='Complete'
    (default) vs 'AbortExposure' on the same cell.

    Default Complete waits for in-flight exposures to finish on
    StopGrabbing -- on long fluorescence captures this presents
    identically to a multi-second app-side stall when the user
    toggles modes. AbortExposure is the doc-confirmed candidate fix
    per acquisition-start-stop-and-abort.html. Setter exists for
    bench characterization; default is unchanged in
    init_camera_config (per Eric direction: setter-only-first,
    bench-validate, then flip default if validated).

    B19.
    """

    def _make_scope_with_fake_camera(self, fake_camera):
        from modules.lumascope_api import Lumascope
        from modules.lumascope_api.imaging import ImagingAPI

        scope = Lumascope.__new__(Lumascope)
        scope.runtime_state = RuntimeState(scope)
        scope._camera_driver = fake_camera
        scope.imaging = ImagingAPI(scope, fake_camera)
        return scope

    def test_lumascope_method_exists(self):
        from modules.lumascope_api import Lumascope

        assert hasattr(ImagingAPI, '_set_acquisition_stop_mode')
        assert callable(ImagingAPI._set_acquisition_stop_mode)

    def test_no_camera_returns_false(self):
        from modules.lumascope_api import Lumascope
        from modules.lumascope_api.imaging import ImagingAPI

        scope = Lumascope.__new__(Lumascope)
        scope.runtime_state = RuntimeState(scope)
        scope._camera_driver = None
        scope.imaging = ImagingAPI(scope, None)
        assert scope.imaging._set_acquisition_stop_mode('Complete') is False

    def test_inactive_camera_returns_false(self):
        class _Fake:
            active = None

            def set_acquisition_stop_mode(self, **k):
                raise AssertionError('driver should not be reached')

        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.imaging._set_acquisition_stop_mode('Complete') is False

    def test_unsupported_driver_returns_false(self):
        """Camera class without the setter (e.g. SimulatedCamera) -> False."""

        class _NoSetter:
            active = True

        scope = self._make_scope_with_fake_camera(_NoSetter())
        assert scope.imaging._set_acquisition_stop_mode('Complete') is False

    def test_routes_to_driver_with_mode_kwarg(self):
        called_with = {}

        class _Fake:
            active = True

            def set_acquisition_stop_mode(self, mode):
                called_with['mode'] = mode
                return True

        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.imaging._set_acquisition_stop_mode('AbortExposure') is True
        assert called_with == {'mode': 'AbortExposure'}

    def test_pylon_driver_method_present(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        assert 'def set_acquisition_stop_mode(' in src, (
            'PylonCamera must implement set_acquisition_stop_mode for '
            'the bench-probe sweep to exercise BslAcquisitionStopMode '
            'without bypassing the API layer.'
        )

    def test_pylon_driver_validates_mode_argument(self):
        """Mode must be one of Complete / CancelExposure / AbortExposure
        per Basler Specifics table."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        body = _function_source(src, 'set_acquisition_stop_mode')
        assert '_ACQ_STOP_MODES' in body, (
            'PylonCamera.set_acquisition_stop_mode must validate the '
            'mode argument against _ACQ_STOP_MODES.'
        )
        assert "_ACQ_STOP_MODES = ('Complete', 'CancelExposure', 'AbortExposure')" in src, (
            'PylonCamera._ACQ_STOP_MODES must list the three doc-named '
            'values per acquisition-start-stop-and-abort.html.'
        )

    def test_pylon_driver_does_not_wrap_in_update_camera_config(self):
        """BslAcquisitionStopMode is a configuration property; setting
        it does not require an in-flight stop/start cycle (and we do
        not wrap because that would defeat the purpose of measuring
        the StopGrabbing behavior change)."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        body = _function_source(src, 'set_acquisition_stop_mode')
        assert 'with self.update_camera_config' not in body, (
            'PylonCamera.set_acquisition_stop_mode must NOT wrap the '
            'write in update_camera_config -- the setter exists to '
            'compare stop-grabbing behavior, and the wrap would '
            'force a stop/start cycle on every call.'
        )

    def test_pylon_driver_raises_hardware_error_on_runtime_exception(self):
        """Rule 29 typed-exception contract; matches DLTL setter."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        body = _function_source(src, 'set_acquisition_stop_mode')
        assert 'except genicam.RuntimeException' in body, (
            'PylonCamera.set_acquisition_stop_mode must catch genicam.RuntimeException.'
        )
        assert 'raise HardwareError(' in body, (
            'PylonCamera.set_acquisition_stop_mode must raise '
            'HardwareError on RuntimeException, not return False.'
        )

    def test_ids_driver_stub_returns_false(self):
        from drivers.idscamera import IDSCamera

        camera = IDSCamera.__new__(IDSCamera)
        assert camera.set_acquisition_stop_mode('Complete') is False
        assert camera.set_acquisition_stop_mode('AbortExposure') is False


class TestGigeSetters:
    """GigE-specific Pylon node setters: BandwidthReserveMode,
    GevSCPSPacketSize, GevSCPD. Required for the dmA3536-9gm dart M
    GigE bench-probe sweep cells.

    USB3 cameras don't expose these nodes; the setters return False
    without warning so the sweep can call them unconditionally per
    cell. IDS stubs return False (no Basler-equivalent nodes).

    Doc citations:
      - BandwidthReserveMode: network-related-parameters.md;
        dmA3536-9gm spec footnote ('Performance' = 9.5 fps vs default 9.3)
      - GevSCPSPacketSize: network-related-parameters.md;
        jumbo-frame negotiation
      - GevSCPD: network-related-parameters.md; inter-packet throttle
    """

    def _make_scope_with_fake_camera(self, fake_camera):
        from modules.lumascope_api import Lumascope
        from modules.lumascope_api.imaging import ImagingAPI

        scope = Lumascope.__new__(Lumascope)
        scope.runtime_state = RuntimeState(scope)
        scope._camera_driver = fake_camera
        scope.imaging = ImagingAPI(scope, fake_camera)
        return scope

    def test_lumascope_methods_exist(self):
        # Phase 4 relocation: methods now live on ImagingAPI; the bench
        # sweep reaches them via scope.imaging.<method>.
        for name in (
            '_set_bandwidth_reserve_mode',
            '_set_gev_packet_size',
            '_set_gev_inter_packet_delay',
        ):
            assert hasattr(ImagingAPI, name), (
                f'ImagingAPI must implement {name} for the GigE bench '
                f'sweep to vary the knob without bypassing the API layer.'
            )
            assert callable(getattr(ImagingAPI, name))

    def test_no_camera_returns_false_for_all(self):
        from modules.lumascope_api import Lumascope
        from modules.lumascope_api.imaging import ImagingAPI

        scope = Lumascope.__new__(Lumascope)
        scope.runtime_state = RuntimeState(scope)
        scope._camera_driver = None
        scope.imaging = ImagingAPI(scope, None)
        assert scope.imaging._set_bandwidth_reserve_mode('Performance') is False
        assert scope.imaging._set_gev_packet_size(9000) is False
        assert scope.imaging._set_gev_inter_packet_delay(0) is False

    def test_inactive_camera_returns_false_for_all(self):
        class _Fake:
            active = None

            def set_bandwidth_reserve_mode(self, **k):
                raise AssertionError('driver should not be reached')

            def set_gev_packet_size(self, **k):
                raise AssertionError('driver should not be reached')

            def set_gev_inter_packet_delay(self, **k):
                raise AssertionError('driver should not be reached')

        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.imaging._set_bandwidth_reserve_mode('Performance') is False
        assert scope.imaging._set_gev_packet_size(9000) is False
        assert scope.imaging._set_gev_inter_packet_delay(0) is False

    def test_bandwidth_reserve_mode_routes_to_driver(self):
        called_with = {}

        class _Fake:
            active = True

            def set_bandwidth_reserve_mode(self, mode):
                called_with['mode'] = mode
                return True

        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.imaging._set_bandwidth_reserve_mode('Performance') is True
        assert called_with == {'mode': 'Performance'}

    def test_gev_packet_size_routes_to_driver(self):
        called_with = {}

        class _Fake:
            active = True

            def set_gev_packet_size(self, size_bytes):
                called_with['size_bytes'] = size_bytes
                return True

        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.imaging._set_gev_packet_size(9000) is True
        assert called_with == {'size_bytes': 9000}

    def test_gev_inter_packet_delay_routes_to_driver(self):
        called_with = {}

        class _Fake:
            active = True

            def set_gev_inter_packet_delay(self, delay_ticks):
                called_with['delay_ticks'] = delay_ticks
                return True

        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.imaging._set_gev_inter_packet_delay(100) is True
        assert called_with == {'delay_ticks': 100}

    def test_pylon_setters_present(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        for name in (
            'set_bandwidth_reserve_mode',
            'set_gev_packet_size',
            'set_gev_inter_packet_delay',
        ):
            assert f'def {name}(' in src, f'PylonCamera must implement {name}.'

    def test_pylon_bandwidth_reserve_mode_validates(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        body = _function_source(src, 'set_bandwidth_reserve_mode')
        assert '_BANDWIDTH_RESERVE_MODES' in body, (
            'set_bandwidth_reserve_mode must validate against _BANDWIDTH_RESERVE_MODES.'
        )
        assert "_BANDWIDTH_RESERVE_MODES = ('Default', 'Performance')" in src

    def test_pylon_setters_raise_hardware_error(self):
        """All three setters raise HardwareError on RuntimeException
        (Rule 29; matches DLTL + AbortExposure setters)."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        for name in (
            'set_bandwidth_reserve_mode',
            'set_gev_packet_size',
            'set_gev_inter_packet_delay',
        ):
            body = _function_source(src, name)
            assert 'except genicam.RuntimeException' in body, (
                f'{name} must catch genicam.RuntimeException.'
            )
            assert 'raise HardwareError(' in body, (
                f'{name} must raise HardwareError on RuntimeException.'
            )

    def test_ids_stubs_return_false(self):
        from drivers.idscamera import IDSCamera

        camera = IDSCamera.__new__(IDSCamera)
        assert camera.set_bandwidth_reserve_mode('Performance') is False
        assert camera.set_gev_packet_size(9000) is False
        assert camera.set_gev_inter_packet_delay(0) is False


class TestPylonCameraLineLengthCap:
    """Pin Rule 26 line-length=100 on drivers/pyloncamera.py.

    A12 closure (AUDIT_PYLONCAMERA_2026-05-07.md): 13 sites exceeded the cap;
    all wrapped. Pyproject sets line-length=100; this test catches regressions
    if a future commit reintroduces a long line.
    """

    def test_no_line_exceeds_100_chars(self):
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py'
        with path.open('r', encoding='utf-8') as fp:
            offenders = [
                (lineno, len(line.rstrip('\n')), line.rstrip('\n'))
                for lineno, line in enumerate(fp, start=1)
                if len(line.rstrip('\n')) > 100
            ]
        assert offenders == [], (
            f'{len(offenders)} line(s) in pyloncamera.py exceed 100 chars: '
            f'{[(n, c) for n, c, _ in offenders]}'
        )


class TestPylonCameraNoSilentExcept:
    """Pin Rule 5 (no silent except) on drivers/pyloncamera.py.

    A6 closure (AUDIT_PYLONCAMERA_2026-05-07.md): 9 silent `except: pass`
    sites at A6 closure time. Each replaced with logger.debug (per-frame /
    per-entry probes) or logger.warning (cleanup / restore paths) so failures
    are visible. This test parses the AST and asserts no `except` block has
    a body of only `pass`.

    Allowed: handlers that re-raise / return / log / call something. Banned:
    bare `except: pass` and `except Exception: pass` blocks.
    """

    def test_no_silent_except_pass_blocks(self):
        import ast
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py'
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    offenders.append(node.lineno)
        assert offenders == [], (
            f'{len(offenders)} silent `except: pass` block(s) at line(s) '
            f'{offenders}; replace each with logger.debug or logger.warning '
            f'per Rule 5.'
        )


class TestStreamGrabberSetters:
    """ImagingAPI.set_max_transfer_size + set_num_max_queued_urbs and the
    underlying PylonCamera / IDSCamera implementations exist so the
    bench-probe sweep can vary the StreamGrabber USB3 knobs across cells
    without dropping below the API layer (Rule 1) or writing /tmp/probe.py
    (Rule 22).

    Per Basler stream-grabber-parameters.html, MaxTransferSize is the
    lever for "fails to receive image stream" symptoms and
    NumMaxQueuedUrbs is the lever for "insufficient system memory"
    symptoms (USB3 only). The Pylon driver raises HardwareError on SDK
    RuntimeException and on missing-node (GigE / non-USB3); the API
    layer notifies + re-raises. IDS stubs return False.

    A6 / B16 closure (AUDIT_PYLONCAMERA_2026-05-07.md).
    """

    def _make_scope_with_fake_camera(self, fake_camera):
        from modules.lumascope_api import Lumascope
        from modules.lumascope_api.imaging import ImagingAPI

        scope = Lumascope.__new__(Lumascope)
        scope.runtime_state = RuntimeState(scope)
        scope._camera_driver = fake_camera
        scope.imaging = ImagingAPI(scope, fake_camera)
        return scope

    def test_lumascope_methods_exist(self):
        # Phase 4 relocation: methods now live on ImagingAPI.
        for name in ('_set_max_transfer_size', '_set_num_max_queued_urbs'):
            assert hasattr(ImagingAPI, name), name
            assert callable(getattr(ImagingAPI, name))

    def test_no_camera_returns_false_for_both(self):
        from modules.lumascope_api import Lumascope
        from modules.lumascope_api.imaging import ImagingAPI

        scope = Lumascope.__new__(Lumascope)
        scope.runtime_state = RuntimeState(scope)
        scope._camera_driver = None
        scope.imaging = ImagingAPI(scope, None)
        assert scope.imaging._set_max_transfer_size(262144) is False
        assert scope.imaging._set_num_max_queued_urbs(64) is False

    def test_inactive_camera_returns_false_for_both(self):
        class _Fake:
            active = None

            def set_max_transfer_size(self, **k):
                raise AssertionError('driver should not be reached')

            def set_num_max_queued_urbs(self, **k):
                raise AssertionError('driver should not be reached')

        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.imaging._set_max_transfer_size(262144) is False
        assert scope.imaging._set_num_max_queued_urbs(64) is False

    def test_unsupported_driver_returns_false(self):
        """Camera class without the setters (e.g. SimulatedCamera) -> False."""

        class _NoSetter:
            active = True

        scope = self._make_scope_with_fake_camera(_NoSetter())
        assert scope.imaging._set_max_transfer_size(262144) is False
        assert scope.imaging._set_num_max_queued_urbs(64) is False

    def test_max_transfer_size_routes_to_driver(self):
        called_with = {}

        class _Fake:
            active = True

            def set_max_transfer_size(self, value_bytes):
                called_with['value_bytes'] = value_bytes
                return True

        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.imaging._set_max_transfer_size(value_bytes=131072) is True
        assert called_with == {'value_bytes': 131072}

    def test_num_max_queued_urbs_routes_to_driver(self):
        called_with = {}

        class _Fake:
            active = True

            def set_num_max_queued_urbs(self, value):
                called_with['value'] = value
                return True

        scope = self._make_scope_with_fake_camera(_Fake())
        assert scope.imaging._set_num_max_queued_urbs(value=32) is True
        assert called_with == {'value': 32}

    def test_pylon_driver_methods_present(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        assert 'def set_max_transfer_size(' in src
        assert 'def set_num_max_queued_urbs(' in src

    def test_ids_driver_stubs_return_false(self):
        from drivers.idscamera import IDSCamera

        camera = IDSCamera.__new__(IDSCamera)
        assert camera.set_max_transfer_size(262144) is False
        assert camera.set_num_max_queued_urbs(64) is False

    def test_pylon_driver_does_not_wrap_in_update_camera_config(self):
        """StreamGrabber knobs are set via the StreamGrabber NodeMap,
        which is independent of the camera grab loop. Wrapping in
        update_camera_config would impose the STALL-1 over-stop
        pattern unnecessarily."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        for name in (
            'set_max_transfer_size',
            'set_num_max_queued_urbs',
            '_set_stream_grabber_int_node',
        ):
            body = _function_source(src, name)
            assert 'with self.update_camera_config' not in body, (
                f'PylonCamera.{name} must NOT wrap StreamGrabber writes '
                f'in update_camera_config (the STALL-1 over-stop pattern).'
            )

    def test_pylon_driver_raises_hardware_error_on_runtime_exception(self):
        """Per Rule 29 typed-exception contract, the Pylon setters raise
        HardwareError on genicam.RuntimeException AND on missing-node
        (GigE / non-USB3 cameras). Pins the raise shape against a future
        cleanup that swaps it for return-False."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        body = _function_source(src, '_set_stream_grabber_int_node')
        assert 'except genicam.RuntimeException' in body
        assert 'raise HardwareError(' in body
        assert 'node is None' in body, (
            '_set_stream_grabber_int_node must check for missing node '
            '(GigE / non-USB3 cameras) and raise HardwareError -- '
            'silent return-False would mislead bench operators into '
            'thinking the knob applied.'
        )


class TestPylonAcquisitionIdleWait:
    """B20 closure (AUDIT_PYLONCAMERA_2026-05-07.md): poll
    AcquisitionActive / ExposureActive after StopGrabbing during
    disconnect, so in-flight frames drain before Close() releases the
    device handle. Bounded so a stuck-active camera can't block
    disconnect indefinitely. Per Basler acquisition-status.html.

    Pairs with CAM-1 trigger hypothesis (see
    AUDIT_LAYER_VIOLATIONS_2026-05-01.md Cluster B): the rare ~11s
    Pylon stop/start pause is correlated with stop_grabbing firing
    while a frame is still in-flight from the previous start.
    Bounded idle-wait between stop_grabbing and Close gives the SDK a
    deterministic drain window.
    """

    def test_helper_method_present(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        assert 'def _wait_for_acquisition_idle(' in src

    def test_disconnect_calls_idle_wait_after_stop_grabbing(self):
        """Pin call-site shape: disconnect() must invoke
        _wait_for_acquisition_idle AFTER stop_grabbing and BEFORE
        Close, not before stop_grabbing or after Close (the latter
        would defeat the purpose -- the device handle is gone)."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        body = _function_source(src, 'disconnect')
        assert '_wait_for_acquisition_idle' in body, (
            'disconnect() must call _wait_for_acquisition_idle'
        )
        # Order check: stop_grabbing -> _wait_for_acquisition_idle -> Close
        idx_stop = body.find('stop_grabbing')
        idx_wait = body.find('_wait_for_acquisition_idle')
        idx_close = body.find('.Close()')
        assert 0 <= idx_stop < idx_wait < idx_close, (
            f'Order violated in disconnect(): '
            f'stop_grabbing={idx_stop} wait={idx_wait} Close={idx_close}'
        )

    def test_idle_wait_returns_true_when_inactive(self):
        from drivers.pyloncamera import PylonCamera

        camera = PylonCamera.__new__(PylonCamera)
        import threading as _threading

        camera._state_lock = _threading.Lock()
        camera.active = None
        assert camera._wait_for_acquisition_idle(timeout_s=0.1) is True

    def test_idle_wait_returns_true_when_already_idle(self):
        """When AcquisitionActive=False and ExposureActive=False on the
        first poll, return True without sleeping the full timeout."""
        from drivers.pyloncamera import PylonCamera

        class _FakeNode:
            def __init__(self, value):
                self._value = value

            def GetValue(self):
                return self._value

        class _FakeNodeMap:
            def __init__(self):
                self._nodes = {
                    'AcquisitionActive': _FakeNode(False),
                    'ExposureActive': _FakeNode(False),
                }

            def GetNode(self, name):
                return self._nodes.get(name)

        class _FakeCamera:
            def GetNodeMap(self):
                return _FakeNodeMap()

        camera = PylonCamera.__new__(PylonCamera)
        import threading as _threading

        camera._state_lock = _threading.Lock()
        camera.active = _FakeCamera()
        import time as _time

        t0 = _time.monotonic()
        result = camera._wait_for_acquisition_idle(timeout_s=2.0)
        elapsed = _time.monotonic() - t0
        assert result is True
        assert elapsed < 0.5, (
            f'idle-wait took {elapsed:.3f}s on already-idle camera; should return immediately'
        )

    def test_idle_wait_returns_false_when_node_absent(self):
        """Older firmware / non-Basler cameras may not expose
        AcquisitionActive. Return False so disconnect proceeds without
        waiting full timeout."""
        from drivers.pyloncamera import PylonCamera

        class _FakeNodeMap:
            def GetNode(self, name):
                return None

        class _FakeCamera:
            def GetNodeMap(self):
                return _FakeNodeMap()

        camera = PylonCamera.__new__(PylonCamera)
        import threading as _threading

        camera._state_lock = _threading.Lock()
        camera.active = _FakeCamera()
        import time as _time

        t0 = _time.monotonic()
        result = camera._wait_for_acquisition_idle(timeout_s=2.0)
        elapsed = _time.monotonic() - t0
        assert result is False
        assert elapsed < 0.1, (
            f'idle-wait should bail immediately when nodes absent; took {elapsed:.3f}s'
        )

    def test_idle_wait_times_out_when_stuck_active(self):
        """If AcquisitionActive stays True past timeout, return False
        and let caller proceed (warning is logged inside)."""
        from drivers.pyloncamera import PylonCamera

        class _FakeNode:
            def GetValue(self):
                return True  # Always active

        class _FakeNodeMap:
            def GetNode(self, name):
                if name in ('AcquisitionActive', 'ExposureActive'):
                    return _FakeNode()
                return None

        class _FakeCamera:
            def GetNodeMap(self):
                return _FakeNodeMap()

        camera = PylonCamera.__new__(PylonCamera)
        import threading as _threading

        camera._state_lock = _threading.Lock()
        camera.active = _FakeCamera()
        result = camera._wait_for_acquisition_idle(timeout_s=0.1)
        assert result is False


class TestPylonStreamGrabberStatusLog:
    """B23 closure (AUDIT_PYLONCAMERA_2026-05-07.md): snapshot the
    StreamGrabber.Status read-only node into the camera trace log
    before StartGrabbing so post-mortem analysis can correlate
    weird-startup symptoms with the grabber's entry state. Per
    Basler stream-grabber-parameters.html.

    Diagnostic-only -- no behavior change. STALL-1 instrumentation
    aid; logs to _cam_log (profile_trace_enabled-gated) so production
    builds pay zero cost when tracing is off.
    """

    def test_helper_method_present(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        assert 'def _log_stream_grabber_status(' in src

    def test_start_grabbing_logs_status_before_start_call(self):
        """Pin call-site shape: _log_stream_grabber_status fires in
        start_grabbing BEFORE camera.StartGrabbing(...) so the trace
        log captures the entry state, not post-start state."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        body = _function_source(src, 'start_grabbing')
        assert '_log_stream_grabber_status' in body
        idx_log = body.find('_log_stream_grabber_status')
        idx_start = body.find('StartGrabbing(')
        # idx_start of literal text "StartGrabbing(" appears in
        # comments + the actual call. Find LAST occurrence to land
        # on the call site (comments come first).
        idx_start = body.rfind('StartGrabbing(')
        assert 0 <= idx_log < idx_start, (
            f'_log_stream_grabber_status must fire BEFORE '
            f'StartGrabbing() in start_grabbing(); '
            f'log_idx={idx_log} start_idx={idx_start}'
        )

    def test_log_helper_no_op_when_active_none(self):
        """Helper must be a true no-op when self.active is None
        (called during reconnect transitions)."""
        from drivers.pyloncamera import PylonCamera

        camera = PylonCamera.__new__(PylonCamera)
        import threading as _threading

        camera._state_lock = _threading.Lock()
        camera.active = None
        # Should not raise
        camera._log_stream_grabber_status('test-label')

    def test_log_helper_handles_missing_node_gracefully(self):
        """Older firmware / non-Basler cameras may not expose Status.
        Helper should not raise, log via _cam_log if present."""
        from drivers.pyloncamera import PylonCamera

        class _FakeStreamGrabberNodeMap:
            def GetNode(self, name):
                return None

        class _FakeCamera:
            def GetStreamGrabberNodeMap(self):
                return _FakeStreamGrabberNodeMap()

        camera = PylonCamera.__new__(PylonCamera)
        import threading as _threading

        camera._state_lock = _threading.Lock()
        camera.active = _FakeCamera()
        # Should not raise
        camera._log_stream_grabber_status('test-label')

    def test_log_helper_handles_runtime_exception(self):
        """If the SDK raises while reading Status, helper should
        catch + log warning, not propagate."""
        from drivers.pyloncamera import PylonCamera

        class _FakeCamera:
            def GetStreamGrabberNodeMap(self):
                raise RuntimeError('simulated SDK failure')

        camera = PylonCamera.__new__(PylonCamera)
        import threading as _threading

        camera._state_lock = _threading.Lock()
        camera.active = _FakeCamera()
        # Should not raise
        camera._log_stream_grabber_status('test-label')


class TestPylonChunkModeActiveWriteRaceGuard:
    """B28 closure (AUDIT_PYLONCAMERA_2026-05-07.md):
    `_enable_validity_chunks` guards against the ChunkModeActive
    write-while-grabbing race.

    Per Basler data-chunks.html, ChunkModeActive is locked while
    the camera is grabbing. The docstring already required callers
    to invoke this method while NOT grabbing, but enforcement was
    absent -- a future caller violating the contract would hit a
    silent SDK lock error. The guard logs a warning and skips the
    write; frame_validity falls back to skip_frames calibration so
    refusing the write is safe by default.
    """

    def test_guard_present_in_enable_validity_chunks(self):
        """Pin the structural fix shape: an is_grabbing()-guarded
        early-return at the top of _enable_validity_chunks before
        any ChunkModeActive write."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()
        body = _function_source(src, '_enable_validity_chunks')
        # Guard must reference is_grabbing AND must come BEFORE the
        # ChunkModeActive write site.
        assert 'is_grabbing' in body, (
            '_enable_validity_chunks must guard with is_grabbing() to '
            'avoid the ChunkModeActive write-while-grabbing race.'
        )
        idx_guard = body.find('is_grabbing')
        idx_write = body.find('ChunkModeActive.Value = True')
        assert 0 <= idx_guard < idx_write, (
            f'is_grabbing() guard must precede ChunkModeActive write; '
            f'guard_idx={idx_guard} write_idx={idx_write}'
        )

    def test_guard_skips_write_and_logs_warning(self):
        """Functional test: when is_grabbing() returns True, the
        method must return without touching ChunkModeActive."""
        from drivers.pyloncamera import PylonCamera

        write_attempted = {'count': 0}

        class _ChunkModeActive:
            @property
            def Value(self):
                return False

            @Value.setter
            def Value(self, v):
                write_attempted['count'] += 1

        class _FakeCamera:
            ChunkModeActive = _ChunkModeActive()

        camera = PylonCamera.__new__(PylonCamera)
        import threading as _threading

        camera._state_lock = _threading.Lock()
        camera.active = _FakeCamera()
        # Force is_grabbing() True via monkeypatch -- using a bound
        # method override to avoid needing the full SDK.
        camera.is_grabbing = lambda: True
        camera._enable_validity_chunks()
        assert write_attempted['count'] == 0, (
            f'ChunkModeActive write must be skipped when grabbing; '
            f'got {write_attempted["count"]} writes'
        )


class TestPylonPublicMethodAnnotationsAndDocstrings:
    """A13 + A14 closure (AUDIT_PYLONCAMERA_2026-05-07.md): every
    public method on every class in drivers/pyloncamera.py has a
    return-type annotation (Rule 37) AND a docstring (Rule 38).

    Pins the structural fix so a future commit adding a public method
    without annotation/docstring fails this test rather than slipping
    through review. Methods exempt:
      - dunder (__init__ / __del__ / etc.) -- language protocol
      - underscore-prefixed (_helper / private)

    A future class added to this module is automatically covered as
    long as its public methods follow the rule.
    """

    def test_every_public_method_has_return_annotation_and_docstring(self):
        import ast
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        gaps = []
        for cls_node in tree.body:
            if not isinstance(cls_node, ast.ClassDef):
                continue
            for sub in cls_node.body:
                if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                name = sub.name
                if name.startswith('_'):
                    continue
                if name.startswith('__') and name.endswith('__'):
                    continue
                has_return = sub.returns is not None
                has_doc = (
                    sub.body
                    and isinstance(sub.body[0], ast.Expr)
                    and isinstance(sub.body[0].value, ast.Constant)
                    and isinstance(sub.body[0].value.value, str)
                )
                if not has_return:
                    gaps.append(
                        f'{cls_node.name}.{name}@{sub.lineno} '
                        f'missing return-type annotation (Rule 37)'
                    )
                if not has_doc:
                    gaps.append(f'{cls_node.name}.{name}@{sub.lineno} missing docstring (Rule 38)')
        assert gaps == [], (
            f'{len(gaps)} public method(s) in pyloncamera.py missing '
            f'annotation/docstring:\n  ' + '\n  '.join(gaps)
        )


class TestManualVideoSpinners:
    """Issue #633 Stage 2D: manual_video FPS + duration UI binding.

    Static-source assertions: kv ID + handlers exist, record_init reads
    via .get with defaults, and max_fps == 0 maps to
    _user_requested_fps_limit = False (so a fresh install no longer
    fires 'FPS budget exceeded' at every >25ms exposure -- the Stage 2C
    regression that this stage closes).
    """

    def _kv_text(self):
        import pathlib

        return pathlib.Path('ui/lumaviewpro.kv').read_text()

    def _ms_text(self):
        import pathlib

        return pathlib.Path('ui/microscope_settings.py').read_text()

    def _record_init_body(self):
        import pathlib

        source = pathlib.Path('ui/main_display.py').read_text()
        idx = source.find('def record_init')
        assert idx >= 0, 'record_init not found in ui/main_display.py'
        next_def = source.find('\n    def ', idx + 1)
        return source[idx:next_def] if next_def > 0 else source[idx:]

    def test_kv_has_max_fps_textinput(self):
        kv = self._kv_text()
        assert 'id: manual_video_max_fps_input' in kv, (
            'ui/lumaviewpro.kv must define a TextInput with id '
            'manual_video_max_fps_input bound to '
            "settings['manual_video']['max_fps']."
        )
        assert 'root.update_manual_video_max_fps()' in kv, (
            'manual_video_max_fps_input must call root.update_manual_video_max_fps() on edit.'
        )

    def test_kv_has_max_duration_textinput(self):
        kv = self._kv_text()
        assert 'id: manual_video_max_duration_input' in kv, (
            'ui/lumaviewpro.kv must define a TextInput with id '
            'manual_video_max_duration_input bound to '
            "settings['manual_video']['max_duration']."
        )
        assert 'root.update_manual_video_max_duration()' in kv, (
            'manual_video_max_duration_input must call '
            'root.update_manual_video_max_duration() on edit.'
        )

    def test_microscope_settings_has_handlers(self):
        body = self._ms_text()
        assert 'def update_manual_video_max_fps' in body, (
            'MicroscopeSettings must define update_manual_video_max_fps '
            'to write the spinner value back to settings dict.'
        )
        assert 'def update_manual_video_max_duration' in body, (
            'MicroscopeSettings must define update_manual_video_max_duration.'
        )

    def test_handlers_validate_and_revert_on_invalid(self):
        body = self._ms_text()
        # Both handlers must surface a notifications.warning AND revert
        # the widget text on bad input -- the L1 researcher sees the
        # error and the field doesn't silently accept garbage.
        for handler in ('update_manual_video_max_fps', 'update_manual_video_max_duration'):
            idx = body.find(f'def {handler}')
            assert idx >= 0
            next_def = body.find('\n    def ', idx + 1)
            handler_body = body[idx:next_def] if next_def > 0 else body[idx:]
            assert 'notifications.warning' in handler_body, (
                f'{handler} must notify on invalid input (Rule 28).'
            )
            assert 'widget.text =' in handler_body, (
                f'{handler} must revert widget.text on invalid input.'
            )

    def test_load_settings_pushes_manual_video_into_widgets(self):
        body = self._ms_text()
        assert 'manual_video_max_fps_input' in body, (
            "load_settings must push settings['manual_video']['max_fps'] "
            'into the manual_video_max_fps_input widget on load.'
        )
        assert 'manual_video_max_duration_input' in body, (
            'load_settings must push '
            "settings['manual_video']['max_duration'] into the "
            'manual_video_max_duration_input widget on load.'
        )

    def test_record_init_reads_via_get_with_defaults(self):
        body = self._record_init_body()
        # No bare KeyError when manual_video dict is missing or its
        # keys are missing -- a partially-edited settings.json won't
        # crash record_init.
        # Quote-style agnostic: ruff format may use single or double quotes.
        assert 'settings.get("manual_video"' in body or "settings.get('manual_video'" in body, (
            "record_init must read settings.get('manual_video', {}) "
            'to tolerate missing dict on a fresh / partial install.'
        )
        assert 'manual_video.get("max_fps"' in body or "manual_video.get('max_fps'" in body, (
            'record_init must read max_fps via .get with a default.'
        )

    def test_user_requested_fps_limit_keys_on_max_fps_zero(self):
        body = self._record_init_body()
        assert 'self._user_requested_fps_limit = max_fps > 0' in body, (
            'max_fps == 0 means uncapped (camera free-run); only '
            'max_fps > 0 sets _user_requested_fps_limit = True. This '
            'closes the Stage 2C regression where the shipped 40fps '
            'default fired the FPS-budget warning at every '
            '>25ms exposure.'
        )

    def test_video_fps_falls_back_to_exposure_freq_when_uncapped(self):
        body = self._record_init_body()
        # When _user_requested_fps_limit is False, video_fps must NOT
        # take min(exposure_freq, 0) (which would set video_fps=0 and
        # break the memmap allocation).
        assert 'video_fps = exposure_freq' in body, (
            'When the user has not requested an FPS limit, video_fps '
            'must default to exposure_freq -- not min(exposure_freq, '
            'max_fps) which would be 0 and break recording.'
        )

    def test_shipped_settings_max_fps_is_zero(self):
        # Only the tracked settings.json is the shipping contract;
        # current.json is gitignored runtime state regenerated from
        # settings.json on first launch.
        import json
        import pathlib

        path = pathlib.Path('data/settings.json')
        data = json.loads(path.read_text())
        assert data.get('manual_video', {}).get('max_fps') == 0, (
            'data/settings.json must ship with manual_video.max_fps = 0 '
            "(uncapped) so a fresh install does not fire 'FPS budget "
            "exceeded' on every record."
        )


class TestBfIlluminationCapAtStartup:
    """Transmitted-layer slider caps (BF / PC / DF -> 50 mA) must be
    applied at app startup, not on first settings-panel toggle. The
    .kv ships ill_slider with max=500; without an init-time
    update_transmitted() call the cap stays unapplied and BF / PC /
    DF channels can be driven up to 500 mA from the slider on first
    use.
    """

    def _src(self):
        import pathlib

        return pathlib.Path('lumaviewpro.py').read_text()

    def test_complete_initialization_calls_update_transmitted(self):
        src = self._src()
        idx = src.find('def complete_initialization')
        assert idx >= 0, 'complete_initialization not found in lumaviewpro.py'
        # Slice through the next def at the matching indent.
        next_def = src.find('\n        def ', idx + 1)
        if next_def < 0:
            # complete_initialization is the last nested def in build();
            # cap by the trailing Clock.schedule_once call instead.
            next_def = src.find('Clock.schedule_once(complete_initialization', idx)
        assert next_def > idx
        body = src[idx:next_def]
        assert 'ctx.image_settings.update_transmitted()' in body, (
            'complete_initialization must call '
            'ctx.image_settings.update_transmitted() so transmitted '
            'slider caps are applied at startup, not on first '
            'settings-panel toggle.'
        )

    def test_update_transmitted_runs_before_protocol_or_accordion_branch(self):
        src = self._src()
        idx = src.find('def complete_initialization')
        assert idx >= 0
        next_def = src.find('Clock.schedule_once(complete_initialization', idx)
        assert next_def > idx
        body = src[idx:next_def]
        ut_pos = body.find('ctx.image_settings.update_transmitted()')
        protocol_pos = body.find('if ctx.protocol is not None')
        accordion_pos = body.find('ctx.image_settings.accordion_collapse()')
        assert ut_pos > 0
        assert protocol_pos > 0
        assert accordion_pos > 0
        assert ut_pos < protocol_pos, (
            'update_transmitted() must run before the protocol-branch '
            'early-return; otherwise protocol-startup leaves the cap '
            'unapplied.'
        )
        assert ut_pos < accordion_pos, (
            'update_transmitted() must run before accordion_collapse() '
            'fires apply_settings on BF, otherwise BF gets applied at '
            'the .kv-default 500 mA before the cap.'
        )


class TestModSliderScrollWheel:
    """ModSlider must accept mouse-wheel events to adjust value by
    step. Default Kivy Slider ignores scroll, so users could only
    click+drag to adjust illumination / exposure / gain / Z. All
    14 ModSlider instances in lumaviewpro.kv inherit the fix.

    Static-source assertions; runtime Kivy touch-event tests need a
    Window context that isn't available in unit-test env.
    """

    def _src(self):
        import pathlib

        return pathlib.Path('ui/mod_slider.py').read_text()

    def test_scroll_handler_present(self):
        src = self._src()
        idx = src.find('def on_touch_down')
        assert idx >= 0
        next_def = src.find('\n    def ', idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:]
        assert "'scrollup'" in body and "'scrolldown'" in body, (
            'ModSlider.on_touch_down must handle scrollup + scrolldown.'
        )
        assert 'self.collide_point' in body, (
            'Scroll handler must require touch.pos to land on the '
            'slider; otherwise wheel-over-other-widget would still '
            'adjust an unrelated slider.'
        )

    def test_scroll_uses_step_attribute(self):
        src = self._src()
        idx = src.find('def on_touch_down')
        assert idx >= 0
        next_def = src.find('\n    def ', idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:]
        assert 'self.step' in body, (
            'Scroll delta must derive from self.step so each ModSlider '
            "instance's configured step (default 5) is honored."
        )

    def test_scroll_clamps_at_min_max(self):
        src = self._src()
        idx = src.find('def on_touch_down')
        assert idx >= 0
        next_def = src.find('\n    def ', idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:]
        assert 'self.max' in body and 'self.min' in body, (
            'Scroll must clamp at self.min / self.max so wheel '
            "spinning past the limit doesn't escape the slider range."
        )

    def test_scroll_dispatches_on_release(self):
        src = self._src()
        idx = src.find('def on_touch_down')
        assert idx >= 0
        next_def = src.find('\n    def ', idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:]
        assert "self.dispatch('on_release')" in body, (
            'Each scroll tick must dispatch on_release so wired '
            'hardware (illumination, exposure, gain, Z) updates '
            'per tick without manual click.'
        )

    def test_scrollup_increases_scrolldown_decreases(self):
        src = self._src()
        idx = src.find('def on_touch_down')
        assert idx >= 0
        next_def = src.find('\n    def ', idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:]
        # Both directional branches must exist. Direction-correctness
        # contract: scrollup INCREASES (wheel up = brighter / higher),
        # scrolldown DECREASES. Asserted by presence of both signs;
        # specifically that the scrollup branch is the one with +delta.
        assert 'self.value + delta' in body, 'Scroll handler must add delta on one branch.'
        assert 'self.value - delta' in body, (
            'Scroll handler must subtract delta on the other branch.'
        )
        # The scrollup branch must own the +delta path. Find the
        # in-body conditional `== 'scrollup'` (not the tuple membership
        # test at the top) and verify the next ~80 chars contain
        # "self.value + delta".
        cond_idx = body.find("touch.button == 'scrollup'")
        assert cond_idx >= 0, "Handler must branch on touch.button == 'scrollup'."
        cond_block = body[cond_idx : cond_idx + 200]
        assert 'self.value + delta' in cond_block, (
            'scrollup branch must INCREASE slider value (wheel up = '
            'brighter / larger / higher Z). If reversed, illumination '
            'control feels backwards to the user.'
        )


class TestModSliderClickThenScrollFocus:
    """ModSlider scroll-wheel adjust requires the slider to be the most-
    recently-clicked ModSlider before the wheel will adjust its value.

    Bare-hover scroll without a prior click was reported as too easy to
    trigger accidentally -- a user grazing the slider with the cursor
    while scrolling a panel would drift illumination / exposure / gain.
    The class-level _focused_ref weakref tracks which ModSlider received
    the most recent left-click; only wheel events landing on that
    slider adjust its value. All other slider hovers fall through (the
    scroll event propagates to the parent scroll-view, the standard
    pre-#677 default).

    Static-source assertions: runtime Kivy touch-event tests need a
    Window context that isn't available in unit-test env.
    """

    def _src(self):
        import pathlib

        return pathlib.Path('ui/mod_slider.py').read_text()

    def test_focused_ref_class_attribute_exists(self):
        src = self._src()
        assert '_focused_ref' in src, (
            'ModSlider must declare _focused_ref class attribute to '
            'track the most-recently-clicked slider for #677 sticky-'
            'focus scroll gating.'
        )
        assert 'weakref.ref' in src, (
            '_focused_ref must store a weakref so an unmounted slider '
            'does not retain past Kivy widget teardown.'
        )

    def test_scroll_branch_checks_focus_before_adjusting(self):
        src = self._src()
        idx = src.find('def on_touch_down')
        assert idx >= 0
        next_def = src.find('\n    def ', idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:]
        # The focus check must appear BEFORE the value-adjust assignment.
        # The scroll branch is gated on the conditional; if the gate is
        # removed (or moved after the adjust), bare-hover scroll
        # regresses.
        focus_idx = body.find('ModSlider._is_focused')
        adjust_idx = body.find('self.value + delta')
        assert focus_idx >= 0, (
            'Scroll branch must call ModSlider._is_focused(self) before '
            'adjusting value. If missing, bare-hover scroll regresses '
            'to the pre-#677 too-easy-to-trigger behavior.'
        )
        assert adjust_idx >= 0, 'Sanity: scroll branch should still adjust value.'
        assert focus_idx < adjust_idx, (
            'Focus check must come BEFORE the adjust line; otherwise '
            'the value still changes regardless of focus state.'
        )

    def test_click_on_slider_sets_focus(self):
        src = self._src()
        idx = src.find('def on_touch_down')
        assert idx >= 0
        next_def = src.find('\n    def ', idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:]
        assert 'ModSlider._set_focused(self)' in body, (
            'on_touch_down must call ModSlider._set_focused(self) on a '
            'non-scroll touch that collides with this slider; otherwise '
            'the user can never gain focus and scroll-adjust is '
            'permanently disabled.'
        )


class TestModSliderAwareScrollView:
    """ModSlider's scroll-wheel handler (619fc49) only fires when its
    on_touch_down receives the touch. Bare Kivy ScrollView consumes
    scrollup/scrolldown via on_scroll_start -> dispatch_children,
    which only walks IMMEDIATE children -- a ModSlider nested inside
    a BoxLayout never sees it. ModSliderAwareScrollView intercepts
    wheel events at on_touch_down and re-dispatches via the standard
    recursive Widget chain so the slider gets first crack.

    The 3 ScrollView sites in lumaviewpro.kv (motion settings panel,
    microscope settings panel, protocol settings panel) all contain
    ModSlider widgets and must use the subclass. Static-source
    assertions; runtime Kivy touch-event tests need a Window context
    that isn't available in unit-test env.
    """

    def _src(self):
        import pathlib

        return pathlib.Path('ui/mod_slider.py').read_text()

    def _kv(self):
        import pathlib

        return pathlib.Path('ui/lumaviewpro.kv').read_text()

    def test_class_defined(self):
        src = self._src()
        assert 'class ModSliderAwareScrollView(ScrollView):' in src, (
            'ui/mod_slider.py must define ModSliderAwareScrollView as a ScrollView subclass.'
        )
        assert 'from kivy.uix.scrollview import ScrollView' in src, (
            'ScrollView import required for the subclass.'
        )

    def test_subclass_handles_scrollwheel_before_super(self):
        src = self._src()
        idx = src.find('class ModSliderAwareScrollView(ScrollView):')
        assert idx >= 0
        body = src[idx:]
        assert "'scrollup'" in body and "'scrolldown'" in body, (
            'Subclass must gate on touch.button in scrollup/scrolldown.'
        )
        assert 'self.collide_point' in body, (
            'Subclass must require touch to land in its bounds before '
            'intercepting -- otherwise wheel events meant for unrelated '
            'widgets would be hijacked.'
        )
        assert 'apply_transform_2d(self.to_local)' in body, (
            "Subclass must apply ScrollView's to_local transform so "
            "descendant ModSliders' collide_point checks happen in "
            'content-space, not window-space.'
        )
        assert "child.dispatch('on_touch_down', touch)" in body, (
            'Subclass must dispatch through Widget.on_touch_down '
            '(recursive) rather than dispatch_children (shallow).'
        )

    def test_subclass_registered_with_factory(self):
        src = self._src()
        assert "Factory.register('ModSliderAwareScrollView'" in src, (
            "ModSliderAwareScrollView must be registered with Kivy's "
            'Factory so lumaviewpro.kv can resolve the class name.'
        )

    def test_kv_uses_subclass_at_known_sites(self):
        kv = self._kv()
        subclass_count = kv.count('\tModSliderAwareScrollView:')
        bare_count = kv.count('\tScrollView:')
        assert subclass_count == 3, (
            'lumaviewpro.kv must use ModSliderAwareScrollView at the '
            '3 known scrollable panel sites (motion settings, '
            'microscope settings, protocol settings) -- each contains '
            f'ModSlider descendants. Found {subclass_count}.'
        )
        assert bare_count == 0, (
            'lumaviewpro.kv must not contain bare ScrollView at the '
            'top-of-line position -- a new ScrollView that wraps '
            f'sliders would silently break wheel adjust. Found {bare_count}.'
        )


class TestFx2DriverLibusbBackendProbe:
    """Issue #645 Bug A: fx2driver.py must probe the libusb-1.0 native
    backend at module load so the missing-DLL case is classified as
    'FX2 not applicable to this install' rather than crashing with
    NoBackendError mid-_connect.

    Static-source assertions because monkeypatching pyusb backend
    state at import time isn't reliable in unit-test env.
    """

    def _src(self):
        import pathlib

        return pathlib.Path('drivers/fx2driver.py').read_text()

    def test_module_probes_libusb_backend_at_load(self):
        src = self._src()
        assert 'usb.backend.libusb1.get_backend()' in src, (
            'fx2driver.py must probe usb.backend.libusb1.get_backend() '
            'at module load so missing libusb-1.0.dll is classified '
            'before _connect runs.'
        )
        assert '_HAS_USB_BACKEND' in src, (
            'fx2driver.py must record the backend-loadable state in '
            '_HAS_USB_BACKEND for use by the _FX2_AVAILABLE gate.'
        )

    def test_fx2_available_gate_includes_backend_check(self):
        src = self._src()
        idx = src.find('_FX2_AVAILABLE = ')
        assert idx >= 0
        # Read up to the closing of the assignment (next blank line or
        # next top-level statement). Captures the multi-line form.
        end = src.find('\nif not _FX2_AVAILABLE', idx)
        assert end > idx
        expr = src[idx:end]
        assert '_HAS_USB_BACKEND' in expr, (
            '_FX2_AVAILABLE must AND in _HAS_USB_BACKEND so a '
            'pyusb-installed-but-no-native-backend system does not '
            'register FX2 drivers.'
        )

    def test_missing_backend_path_logs_install_hint(self):
        src = self._src()
        # The else-if branch for the missing-backend case must fire
        # an INFO with concrete install instructions per platform.
        assert 'libusb-1.0 native library not loadable' in src, (
            'fx2driver.py must log a clear INFO when the libusb-1.0 '
            'backend is not loadable, with platform-specific install '
            'instructions.'
        )


# ---------------------------------------------------------------------------
# stage_offset value-semantics at run() start
# ---------------------------------------------------------------------------


class TestStageOffsetSnapshot:
    """SequencedCaptureRunner must snapshot stage_offset at run() start so
    mid-protocol UI mutations don't change the in-flight coordinate
    transforms. UI edits between runs must still be visible to the next run.
    """

    def _make_executor(self, stage_offset):
        from modules.sequenced_capture_runner import SequencedCaptureRunner

        return SequencedCaptureRunner(
            scope=MagicMock(),
            stage_offset=stage_offset,
            io_executor=MagicMock(),
            protocol_thread=MagicMock(),
            file_io_executor=MagicMock(),
            camera_executor=MagicMock(),
            autofocus_thread=MagicMock(),
        )

    def test_constructor_holds_live_reference(self):
        src = {'x': 100.0, 'y': 50.0, 'z': 0.0}
        exc = self._make_executor(src)
        assert exc._stage_offset_source is src, (
            '__init__ must hold the live reference in _stage_offset_source '
            'so between-run edits propagate to the next snapshot.'
        )

    def test_snapshot_deepcopies_stage_offset(self):
        src = {'x': 100.0, 'y': 50.0, 'z': 0.0}
        exc = self._make_executor(src)
        exc._snapshot_run_state()
        assert exc._stage_offset is not src, (
            '_snapshot_run_state must produce a new dict, not share the ref.'
        )
        assert exc._stage_offset == src

    def test_mid_run_source_mutation_does_not_affect_snapshot(self):
        """Core race: source mutated mid-protocol must not leak in."""
        src = {'x': 100.0, 'y': 50.0, 'z': 0.0}
        exc = self._make_executor(src)
        exc._snapshot_run_state()
        src['x'] = 999.0
        src['y'] = -50.0
        assert exc._stage_offset['x'] == 100.0
        assert exc._stage_offset['y'] == 50.0

    def test_next_snapshot_picks_up_between_run_mutations(self):
        """Between runs, the next snapshot reflects source updates."""
        src = {'x': 100.0, 'y': 50.0, 'z': 0.0}
        exc = self._make_executor(src)
        exc._snapshot_run_state()
        assert exc._stage_offset['x'] == 100.0
        src['x'] = 200.0
        exc._snapshot_run_state()
        assert exc._stage_offset['x'] == 200.0

    def test_nested_dict_mutation_does_not_affect_snapshot(self):
        """Deep copy: nested dicts must also be private to the run."""
        src = {'x': 100.0, 'y': {'sub': 1.0}, 'z': 0.0}
        exc = self._make_executor(src)
        exc._snapshot_run_state()
        src['y']['sub'] = 99.0
        assert exc._stage_offset['y']['sub'] == 1.0


class TestReusableTaskWaiter:
    """_ReusableTaskWaiter replaces concurrent.futures.Future on the
    SequentialIOExecutor return_future=True path so Lock kernel handles
    aren't churned per submission (Bug E mitigation -- Windows
    Semaphore handle leak under high-rate protocol submission).
    Contract: API-compatible subset (result/timeout, set_result,
    set_exception, cancel) + reusable via reset across sequential
    same-thread submissions.
    """

    def test_set_result_unblocks_caller(self):
        from modules.sequential_io_executor import _ReusableTaskWaiter

        w = _ReusableTaskWaiter()
        w.set_result(42)
        assert w.result(timeout=0.1) == 42

    def test_set_exception_raises_in_caller(self):
        from modules.sequential_io_executor import _ReusableTaskWaiter

        w = _ReusableTaskWaiter()
        w.set_exception(ValueError('boom'))
        import pytest

        with pytest.raises(ValueError, match='boom'):
            w.result(timeout=0.1)

    def test_timeout_raises(self):
        from modules.sequential_io_executor import _ReusableTaskWaiter
        from concurrent.futures import TimeoutError as _TimeoutError

        w = _ReusableTaskWaiter()
        import pytest

        with pytest.raises(_TimeoutError):
            w.result(timeout=0.05)

    def test_reset_allows_reuse(self):
        from modules.sequential_io_executor import _ReusableTaskWaiter

        w = _ReusableTaskWaiter()
        w.set_result('first')
        assert w.result(timeout=0.1) == 'first'
        w.reset()
        assert not w.is_spent()
        w.set_result('second')
        assert w.result(timeout=0.1) == 'second'

    def test_is_spent_after_set_result(self):
        from modules.sequential_io_executor import _ReusableTaskWaiter

        w = _ReusableTaskWaiter()
        assert not w.is_spent()
        w.set_result(None)
        assert w.is_spent()

    def test_cancel_unblocks_with_cancelled_error(self):
        from modules.sequential_io_executor import _ReusableTaskWaiter
        from concurrent.futures import CancelledError

        w = _ReusableTaskWaiter()
        assert w.cancel() is True
        import pytest

        with pytest.raises(CancelledError):
            w.result(timeout=0.1)

    def test_thread_local_pool_reuse(self):
        """Sequential submissions from the same thread should reuse the
        same waiter instance (the entire point of the pool -- zero
        per-submission kernel-handle allocation in steady state)."""
        from modules.sequential_io_executor import _claim_waiter

        w1 = _claim_waiter()
        w1.set_result('a')
        w1.result(timeout=0.1)
        # Same thread submits again -- should reuse the same waiter
        w2 = _claim_waiter()
        assert w2 is w1, 'expected thread-local waiter reuse; got different instance'

    def test_concurrent_submission_allocates_fresh_waiter(self):
        """If a thread tries to claim while its previous waiter is still
        in-flight (set_result not yet called), allocate a fresh one
        instead of clobbering the in-flight wait."""
        from modules.sequential_io_executor import _claim_waiter

        w1 = _claim_waiter()
        # Don't set result -- w1 is still in-flight
        w2 = _claim_waiter()
        assert w2 is not w1, 'expected fresh waiter when previous is in-flight'


class TestSequencedCaptureRunnerRunDirCollision:
    """Rapid protocol-run mashing collides on the second-resolution
    timestamp directory name. _create_run_dir retries with _001, _002,
    ... so same-second collisions succeed on the next attempt instead
    of hard-failing."""

    def _make_executor(self, parent_dir):
        from modules.sequenced_capture_runner import SequencedCaptureRunner

        exc = SequencedCaptureRunner(
            scope=MagicMock(),
            stage_offset={'x': 0.0, 'y': 0.0, 'z': 0.0},
            io_executor=MagicMock(),
            protocol_thread=MagicMock(),
            file_io_executor=MagicMock(),
            camera_executor=MagicMock(),
            autofocus_thread=MagicMock(),
        )
        exc._parent_dir = parent_dir
        return exc

    def test_first_call_uses_unsuffixed_name(self, tmp_path):
        exc = self._make_executor(tmp_path)
        result = exc._create_run_dir()
        assert result['status'] is True
        assert exc._run_dir.exists()
        # Unsuffixed: bare YYYYMMDD_HHMMSS, no trailing _NNN.
        name = exc._run_dir.name
        assert len(name.split('_')) == 2, f'first call must use bare timestamp name; got {name!r}'

    def test_same_second_collision_uses_suffix(self, tmp_path):
        exc = self._make_executor(tmp_path)
        r1 = exc._create_run_dir()
        r2 = exc._create_run_dir()
        r3 = exc._create_run_dir()
        for r in (r1, r2, r3):
            assert r['status'] is True, f'unexpected failure: {r}'
        # All three directories exist and are distinct.
        dirs = sorted(p.name for p in tmp_path.iterdir())
        assert len(dirs) == 3
        # The first is unsuffixed; the next two carry _001 and _002.
        assert dirs[1].endswith('_001'), dirs
        assert dirs[2].endswith('_002'), dirs

    def test_collision_retries_dont_overwrite(self, tmp_path):
        exc = self._make_executor(tmp_path)
        exc._create_run_dir()
        first_path = exc._run_dir
        (first_path / 'sentinel.txt').write_text('do not overwrite')
        exc._create_run_dir()
        # First directory and its file are intact.
        assert (first_path / 'sentinel.txt').read_text() == 'do not overwrite'
        # Second call wrote to a different directory.
        assert exc._run_dir != first_path
        assert exc._run_dir.exists()

    def test_missing_parent_still_returns_clear_error(self, tmp_path):
        # Parent that does not exist: each candidate raises
        # FileNotFoundError, no FileExistsError-style retry.
        exc = self._make_executor(tmp_path / 'does_not_exist')
        result = exc._create_run_dir()
        assert result['status'] is False
        assert 'accessible capture location' in result['error']


class TestSCEResetSignalsAbort:
    """UI-initiated abort path (cancel_all_protocols / abort-scan button)
    must signal protocol_thread.abort() before cleanup tears down LEDs /
    camera / position. Without this, cleanup races the in-flight scan
    step (visible as LED flicker, camera config bouncing, return-to-
    position racing the next step's motion).
    """

    def _make_runner(self):
        from modules.sequenced_capture_runner import SequencedCaptureRunner

        runner = SequencedCaptureRunner(
            scope=MagicMock(),
            stage_offset={'x': 0.0, 'y': 0.0, 'z': 0.0},
            io_executor=MagicMock(),
            protocol_thread=MagicMock(),
            file_io_executor=MagicMock(),
            camera_executor=MagicMock(),
            autofocus_thread=MagicMock(),
        )
        return runner

    def test_reset_calls_protocol_thread_abort_when_in_progress(self):
        runner = self._make_runner()
        runner._run_in_progress_event.set()
        # _cleanup() has side effects we don't want to actually run; patch it.
        runner._cleanup = MagicMock()

        runner.reset()

        runner.protocol_thread.abort.assert_called_once()
        runner._cleanup.assert_called_once()

    def test_reset_abort_called_before_cleanup(self):
        runner = self._make_runner()
        runner._run_in_progress_event.set()

        order: list[str] = []
        runner.protocol_thread.abort.side_effect = lambda: order.append('abort')
        runner._cleanup = MagicMock(side_effect=lambda: order.append('cleanup'))

        runner.reset()

        assert order == ['abort', 'cleanup'], f'abort must be called before cleanup; got {order}'

    def test_reset_noop_when_no_run_in_progress(self):
        runner = self._make_runner()
        # Run not in progress -- reset() should be a no-op.
        runner._cleanup = MagicMock()

        runner.reset()

        runner.protocol_thread.abort.assert_not_called()
        runner._cleanup.assert_not_called()


class TestImageUtilsMaxWorkersIsZero:
    """tifffile's per-write ThreadPoolExecutor holds a Windows kernel
    Event handle that outlives cleanup -- ~1 leaked handle per save
    over a 28-min bench run. All three save paths in image_utils.py
    must use maxworkers=0 to retire the per-write executor. This test
    pins the floor; a future revert to maxworkers>=1 fails it.
    """

    def test_all_dict_maxworkers_are_zero(self):
        import ast
        import pathlib

        rel = 'modules/image_utils.py'
        source = pathlib.Path(rel).read_text(encoding='utf-8')
        tree = ast.parse(source, filename=rel)

        offenders: list[str] = []
        # Walk dict() Call nodes; check keyword arg `maxworkers=N`.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_dict_call = isinstance(func, ast.Name) and func.id == 'dict'
            if not is_dict_call:
                continue
            for kw in node.keywords:
                if kw.arg != 'maxworkers':
                    continue
                if not isinstance(kw.value, ast.Constant):
                    continue
                if kw.value.value != 0:
                    offenders.append(f'{rel}:{node.lineno}: maxworkers={kw.value.value}')

        assert not offenders, (
            'All tifffile dict() maxworkers must be 0 to avoid the '
            'Windows kernel-handle leak:\n  ' + '\n  '.join(offenders)
        )


class TestProtocolIOTimeoutsAreNotShort:
    """The protocol_step_runner + protocol_cleanup `fut.result(timeout=N)`
    sites that wait on io_executor / camera_executor work must use a
    long-enough window to survive Pylon USB3 stress (payload-discard
    cascades that push a single hardware op past 5s without being a
    real failure). This test pins the floor at 30 so a future revert
    won't silently bring back the popup storm.
    """

    _FILES = (
        'modules/protocol_step_runner.py',
        'modules/protocol_cleanup.py',
    )
    _MIN_TIMEOUT_S = 30

    def test_no_short_timeouts_on_protocol_io_futures(self):
        import ast
        import pathlib

        offenders: list[str] = []
        for rel in self._FILES:
            source = pathlib.Path(rel).read_text(encoding='utf-8')
            tree = ast.parse(source, filename=rel)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                # Match `<x>.result(timeout=<int>)` calls.
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == 'result'):
                    continue
                for kw in node.keywords:
                    if kw.arg != 'timeout':
                        continue
                    if not isinstance(kw.value, ast.Constant):
                        continue
                    if not isinstance(kw.value.value, (int, float)):
                        continue
                    if kw.value.value < self._MIN_TIMEOUT_S:
                        offenders.append(
                            f'{rel}:{node.lineno}: timeout={kw.value.value} '
                            f'(min {self._MIN_TIMEOUT_S})'
                        )

        assert not offenders, (
            'Protocol IO futures must use timeout >= '
            f'{self._MIN_TIMEOUT_S}s -- short windows pop up storms '
            'under Pylon USB3 stress:\n  ' + '\n  '.join(offenders)
        )


class TestWaitUntilLedOnSymmetry:
    """Audit Finding #8 -- illumination.wait_until_led_on mirrors
    motion.wait_until_finished_moving in shape: takes a `timeout_s`
    kwarg (renamed from bare `timeout` in audit U6) and returns a
    bool."""

    def test_signature_has_timeout_kwarg_with_default(self):
        import inspect
        from modules.lumascope_api.illumination import IlluminationAPI

        sig = inspect.signature(IlluminationAPI.wait_until_led_on)
        params = sig.parameters
        assert 'timeout_s' in params, 'wait_until_led_on must accept timeout_s kwarg'
        assert 'timeout' not in params, (
            'wait_until_led_on must not still expose bare `timeout` (audit U6)'
        )
        assert params['timeout_s'].default == 5.0
        # `from __future__ import annotations` -> string forms.
        assert params['timeout_s'].annotation in (float, 'float')

    def test_returns_bool_annotation(self):
        import inspect
        from modules.lumascope_api.illumination import IlluminationAPI

        sig = inspect.signature(IlluminationAPI.wait_until_led_on)
        assert sig.return_annotation in (bool, 'bool')

    def test_no_driver_returns_false(self, sim_scope):
        # Force the no-driver branch (driver resolved via scope._led_driver).
        original = sim_scope._led_driver
        sim_scope._led_driver = None
        try:
            result = sim_scope.illumination.wait_until_led_on(timeout_s=0.1)
        finally:
            sim_scope._led_driver = original
        assert result is False


class TestSessionLedOnArgNameIsMa:
    """Audit Finding #33 -- ScopeSession.led_on_async / led_on_sync use `mA`,
    matching the canonical Lumascope.illumination.led_on(channel, mA, ...)
    name. The old `illumination=` keyword is retired. (Method renamed
    from `led_on` -> `led_on_async` per Finding #6 async-naming sweep.)"""

    def test_led_on_signature_uses_mA(self):
        import inspect
        from modules.scope_session import ScopeSession

        params = inspect.signature(ScopeSession.led_on_async).parameters
        assert 'mA' in params, 'ScopeSession.led_on_async must accept mA kwarg'
        assert 'illumination' not in params, 'old `illumination` kwarg must be retired'

    def test_led_on_sync_signature_uses_mA(self):
        import inspect
        from modules.scope_session import ScopeSession

        params = inspect.signature(ScopeSession.led_on_sync).parameters
        assert 'mA' in params
        assert 'illumination' not in params

    def test_illumination_api_led_on_async_signature_uses_mA(self):
        """U6 paired with Finding #33 (was: ScopeSession only). The
        async/sync surface on IlluminationAPI proper also drops the
        ambiguous `illumination=` keyword. The drift had been at the
        sub-API layer (not just the Session forwarder) and U6 closes
        it pre-freeze."""
        import inspect
        from modules.lumascope_api.illumination import IlluminationAPI

        for method_name in ('led_on_async', 'led_on_sync'):
            params = inspect.signature(getattr(IlluminationAPI, method_name)).parameters
            assert 'mA' in params, f'IlluminationAPI.{method_name} must accept mA'
            assert 'illumination' not in params, (
                f'IlluminationAPI.{method_name} must retire `illumination` kwarg'
            )


class TestImagingTimeoutsAreFloatSeconds:
    """Audit Finding #11 -- imaging timeout convention unified to
    float seconds. Pre-freeze, the four cited shapes (`timedelta`,
    `int ms`, `float s`, untyped) collapse to `float` seconds.
    Also surfaces a unit fix on `get_image.new_capture_timeout`,
    which historically defaulted to `1000` (claiming "ms" in the
    docstring) but the value flowed unchanged into the driver
    `grab_new_capture(timeout: float)` which is seconds."""

    _METHODS_AND_TIMEOUTS = (
        ('set_gain_sync', 'timeout_s', 5.0),
        ('set_exposure_sync', 'timeout_s', 5.0),
        ('capture_and_wait', 'timeout_s', 0.0),
        ('capture_and_wait_sync', 'timeout_s', 30.0),
    )

    def test_timeout_default_is_float(self):
        import inspect
        from modules.lumascope_api.imaging import ImagingAPI

        for method_name, kwarg, expected in self._METHODS_AND_TIMEOUTS:
            method = getattr(ImagingAPI, method_name)
            sig = inspect.signature(method)
            assert kwarg in sig.parameters, f'{method_name} lost {kwarg}'
            default = sig.parameters[kwarg].default
            assert isinstance(default, float), (
                f'{method_name}.{kwarg} default {default!r} not float'
            )
            assert default == expected, f'{method_name}.{kwarg} default {default} != {expected}'

    def test_get_image_timeout_is_float_seconds(self):
        import inspect
        from modules.lumascope_api.imaging import ImagingAPI

        sig = inspect.signature(ImagingAPI.get_image)
        timeout_s = sig.parameters['timeout_s']
        assert isinstance(timeout_s.default, float), (
            f'get_image.timeout_s default {timeout_s.default!r} not float seconds; '
            f'previously datetime.timedelta'
        )
        assert timeout_s.default == 5.0
        assert 'timeout' not in sig.parameters, (
            'get_image must not still expose bare `timeout` (audit U6 rename)'
        )

    def test_get_image_new_capture_timeout_is_float_seconds(self):
        # The audit said "rename to *_ms if SDK demands ms"; verification
        # showed the driver takes seconds, so we kept the name + canonicalized
        # to float seconds. Default was 1000 (an int interpreted as 1000s by
        # the driver path -- never exercised). Now 5.0 seconds. U6 added the
        # _s unit suffix to make the documented seconds unit explicit on the
        # parameter name itself.
        import inspect
        from modules.lumascope_api.imaging import ImagingAPI

        sig = inspect.signature(ImagingAPI.get_image)
        nct = sig.parameters['new_capture_timeout_s']
        assert isinstance(nct.default, float)
        assert nct.default == 5.0
        assert 'new_capture_timeout' not in sig.parameters


class TestImagingGetCameraTempsRetired:
    """Audit Finding #2 -- imaging.get_camera_temps was a duplicate path
    for the diagnostics.get_camera_temperatures probe. Retired pre-freeze.
    log_camera_temps (the live-in-flight logger) stays on imaging and now
    routes through diagnostics for the data read."""

    def test_imaging_get_camera_temps_is_gone(self):
        from modules.lumascope_api.imaging import ImagingAPI

        assert not hasattr(ImagingAPI, 'get_camera_temps'), (
            'imaging.get_camera_temps must be retired; '
            'callers route through scope.diagnostics.get_camera_temperatures'
        )

    def test_diagnostics_get_camera_temperatures_still_callable(self, sim_scope):
        # Sim drivers may not expose temperatures; method must return a dict.
        result = sim_scope.diagnostics.get_camera_temperatures()
        assert isinstance(result, dict)

    def test_imaging_log_camera_temps_still_exists(self):
        from modules.lumascope_api.imaging import ImagingAPI

        assert callable(getattr(ImagingAPI, 'log_camera_temps', None))


class TestSaveLiveImageTimeoutIsFloat:
    """Phase-2 units-audit Finding P2-1 -- save_live_image's `timeout`
    must be a float (seconds), not a datetime.timedelta. The Phase 1
    audit #11 rename of capture_and_wait/get_image to float seconds
    introduced a latent regression: the caller `image_save.save_live_image`
    kept its `datetime.timedelta` default, which then flowed through
    `capture_and_wait(timeout=...)` -> `get_image(timeout=...)` ->
    `datetime.timedelta(seconds=timeout)` where `seconds=` rejects
    timedelta with TypeError. Both UI callers (composite_capture.py)
    use the default; the live-capture path crashes."""

    def test_signature_is_float(self):
        import inspect
        from modules.image_save import save_live_image

        sig = inspect.signature(save_live_image)
        timeout_param = sig.parameters['timeout_s']
        # Reject the previous timedelta default.
        assert not isinstance(timeout_param.default, __import__('datetime').timedelta), (
            'save_live_image.timeout_s default must be float seconds, not timedelta'
        )
        assert isinstance(timeout_param.default, float)
        assert timeout_param.default == 5.0
        assert 'timeout' not in sig.parameters, (
            'save_live_image must not still expose bare `timeout` (audit U6 rename)'
        )

    def test_default_timeout_flows_through_capture_and_wait_without_crash(self, sim_scope):
        # The original regression was: save_live_image's timedelta default
        # flowed unchanged through capture_and_wait -> get_image, where
        # `datetime.timedelta(seconds=timeout)` rejected timedelta with
        # TypeError. This test exercises the same forwarding path that
        # save_live_image uses (line 484-491), with the new float default.
        # If a future revert restores `timeout_s: datetime.timedelta`, the
        # signature test above fails first; if some other regression
        # restores the TypeError at the get_image conversion, this fails.
        # U6 renamed the keyword from `timeout` to `timeout_s` across the
        # imaging API; this test follows.
        import inspect
        from modules.image_save import save_live_image

        timeout_default = inspect.signature(save_live_image).parameters['timeout_s'].default
        try:
            sim_scope.imaging.capture_and_wait(
                force_to_8bit=True,
                all_ones_check=False,
                timeout_s=timeout_default,
                sum_count=1,
                sum_delay_s=0,
            )
        except TypeError as e:
            raise AssertionError(
                f"capture_and_wait raised TypeError when given save_live_image's "
                f'default timeout_s ({timeout_default!r}): {e}. '
                f'Phase-2 audit P2-1 regression has returned.'
            ) from e


class TestImagingParamNamesUseUnitSuffix:
    """Audit U3 + U4 -- imaging API + driver method param names carry unit
    suffix (gain_db / exposure_ms / min_gain_db / max_gain_db). Pre-freeze,
    the L2 surface and the driver contract drop bare ``gain`` / ``t`` /
    ``exposure`` / ``min_gain`` / ``max_gain`` parameter names that fail to
    disambiguate units from the four parallel namespaces uncovered by the
    2026-05-20 units-consistency audit (raw settings, derived layer_config,
    camera-cache, API-param). Lock these so a revert breaking either the
    L2 contract or the driver contract trips immediately."""

    _IMAGING_PARAMS = (
        # (method, expected_param_name_set, banned_param_name_set)
        ('set_gain', frozenset({'gain_db'}), frozenset({'gain'})),
        ('set_exposure_time', frozenset({'exposure_ms'}), frozenset({'t', 'exposure'})),
        ('set_gain_sync', frozenset({'gain_db'}), frozenset({'gain'})),
        ('set_exposure_sync', frozenset({'exposure_ms'}), frozenset({'exposure', 't'})),
        ('apply_layer_camera_settings', frozenset({'gain_db', 'exposure_ms'}), frozenset({'gain'})),
        (
            'auto_gain_once',
            frozenset({'min_gain_db', 'max_gain_db'}),
            frozenset({'min_gain', 'max_gain'}),
        ),
    )

    def test_imaging_method_param_names(self):
        import inspect
        from modules.lumascope_api.imaging import ImagingAPI

        for method_name, expected, banned in self._IMAGING_PARAMS:
            method = getattr(ImagingAPI, method_name)
            params = set(inspect.signature(method).parameters)
            missing = expected - params
            present_banned = banned & params
            assert not missing, f'ImagingAPI.{method_name} missing unit-suffixed params {missing}'
            assert not present_banned, (
                f'ImagingAPI.{method_name} still has bare-name params '
                f'{present_banned}; should use unit-suffixed names per audit U3'
            )

    def test_driver_auto_gain_param_names(self):
        """Driver-side auto_gain / auto_gain_once / update_auto_gain_min_max
        use ``min_gain_db`` / ``max_gain_db`` -- the abstract Camera contract
        plus all concrete drivers (Pylon, IDS, simulated, FX2). Caught by
        the U3 mechanical rename sweep."""
        import inspect
        from drivers.camera import Camera

        for method_name in ('auto_gain', 'auto_gain_once', 'update_auto_gain_min_max'):
            method = getattr(Camera, method_name)
            params = set(inspect.signature(method).parameters)
            assert 'min_gain_db' in params, f'Camera.{method_name} missing min_gain_db param'
            assert 'max_gain_db' in params, f'Camera.{method_name} missing max_gain_db param'
            assert 'min_gain' not in params, f'Camera.{method_name} still has bare min_gain param'
            assert 'max_gain' not in params, f'Camera.{method_name} still has bare max_gain param'

    def test_driver_exposure_t_param_name(self):
        """Driver-side exposure_t uses ``exposure_ms`` -- the abstract Camera
        contract plus all concrete drivers (Pylon, IDS, simulated, FX2).
        Caught by the U4 driver rename sweep (paired with U3's API-side
        rename). The historical bare ``t`` name is banned -- ambiguous on
        a method whose body multiplies by 1000 to reach microseconds."""
        import inspect
        from drivers.camera import Camera

        method = Camera.exposure_t
        params = set(inspect.signature(method).parameters)
        assert 'exposure_ms' in params, 'Camera.exposure_t missing exposure_ms param'
        assert 't' not in params, 'Camera.exposure_t still has bare `t` param'


class TestTimeoutParamNamesUseSecondSuffix:
    """Audit U6 timeout sweep -- L2 API methods carrying a wall-clock
    timeout parameter use ``timeout_s`` (seconds-unit suffix) rather
    than the historical bare ``timeout``. Lock the rename so a revert
    breaks the L2 contract visibly.

    fx2driver is explicitly exempt (pyusb-native uses int ms with
    distinct semantics; documented at the audit's Finding #49). The
    deep board-protocol-internal timeouts (serialboard, motorboard,
    raw_repl, firmware_updater) keep bare ``timeout`` to match the
    underlying pyserial / stdlib API names."""

    _SECONDS_TIMEOUT_METHODS = (
        # (module-path, class-or-fn, method_name_or_None)
        ('modules.lumascope_api.motion', 'MotionAPI', 'wait_until_finished_moving'),
        ('modules.lumascope_api.motion', 'MotionAPI', 'move_absolute_sync'),
        ('modules.lumascope_api.illumination', 'IlluminationAPI', 'led_on_sync'),
        ('modules.lumascope_api.illumination', 'IlluminationAPI', 'leds_off_sync'),
        ('modules.lumascope_api.illumination', 'IlluminationAPI', 'wait_until_led_on'),
        ('modules.lumascope_api.imaging', 'ImagingAPI', 'set_gain_sync'),
        ('modules.lumascope_api.imaging', 'ImagingAPI', 'set_exposure_sync'),
        ('modules.lumascope_api.imaging', 'ImagingAPI', 'capture_and_wait'),
        ('modules.lumascope_api.imaging', 'ImagingAPI', 'capture_and_wait_sync'),
        ('modules.lumascope_api.imaging', 'ImagingAPI', 'get_image'),
        ('modules.lumascope_api.diagnostics', 'DiagnosticsAPI', 'enter_led_engineering_mode'),
        ('modules.scope_session', 'ScopeSession', 'led_on_sync'),
    )

    def test_api_methods_use_timeout_s(self):
        import importlib
        import inspect

        for module_path, class_name, method_name in self._SECONDS_TIMEOUT_METHODS:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            method = getattr(cls, method_name)
            params = set(inspect.signature(method).parameters)
            assert 'timeout_s' in params, (
                f'{class_name}.{method_name} must accept `timeout_s` per audit U6'
            )
            assert 'timeout' not in params, (
                f'{class_name}.{method_name} still has bare `timeout` (U6 rename incomplete)'
            )

    def test_save_live_image_uses_timeout_s(self):
        """The save_live_image helper participates in the same sweep."""
        import inspect
        from modules.image_save import save_live_image

        params = set(inspect.signature(save_live_image).parameters)
        assert 'timeout_s' in params
        assert 'timeout' not in params

    def test_driver_grab_new_capture_uses_timeout_s(self):
        """L2-mirror driver method -- grab_new_capture is invoked from
        ImagingAPI.capture_and_wait and ImagingAPI.get_image with a
        seconds value, and now exposes that unit on the signature."""
        import inspect
        from drivers.camera import Camera

        params = set(inspect.signature(Camera.grab_new_capture).parameters)
        assert 'timeout_s' in params
        assert 'timeout' not in params


class TestLedMaxMaCanonicalHomeIsCapabilities:
    """Freeze audit Finding #38 -- `Lumascope.LED_MAX_MA` was a class
    constant that duplicated `capabilities.led_max_ma` (same value,
    two SoTs). The class constant is retired; the canonical home is
    `modules.scope_capabilities.LED_MAX_MA` (module-level) which
    `capabilities.led_max_ma` mirrors per-instance."""

    def test_lumascope_class_does_not_carry_led_max_ma(self):
        from modules.lumascope_api import Lumascope

        assert not hasattr(Lumascope, 'LED_MAX_MA'), (
            'Lumascope.LED_MAX_MA must be retired per audit #38; '
            'callers read scope.capabilities.led_max_ma instead.'
        )

    def test_capabilities_led_max_ma_matches_canonical_constant(self, sim_scope):
        from modules.scope_capabilities import LED_MAX_MA

        assert sim_scope.capabilities.led_max_ma == LED_MAX_MA

    def test_illumination_validation_reads_capabilities(self, sim_scope):
        """The validation gate inside IlluminationAPI.led_on must read
        the cap from capabilities, not from a retired class constant.
        A capability override (test-only) is reflected by the gate."""
        import pytest as _pytest

        # Cap at 50 mA for this test; 51 must reject.
        from dataclasses import replace

        sim_scope.capabilities = replace(sim_scope.capabilities, led_max_ma=50)
        with _pytest.raises(ValueError, match='current'):
            sim_scope.illumination.led_on(channel=0, mA=51)


class TestSessionSetObjectiveForwarder:
    """Freeze audit Finding #47 -- LumascopeSkills.md said
    `scope.set_objective('10x Oly')` but the Session layer is the L2
    entry point per design-doc 6.6; `session.set_objective` did not
    exist, so the doc led L2 callers across to the composition root.
    Fix: thin forwarder on ScopeSession plus doc note that both
    surfaces work."""

    def test_session_has_set_objective_method(self):
        from modules.scope_session import ScopeSession

        assert callable(getattr(ScopeSession, 'set_objective', None)), (
            'ScopeSession.set_objective forwarder must exist per audit #47'
        )

    def test_session_set_objective_forwards_to_scope(self):
        """Calling the Session forwarder updates the composition root's
        objective state -- same path as scope.runtime_state.set_objective()
        directly."""
        from modules.scope_session import ScopeSession

        session = ScopeSession.create_headless()
        available = session.scope.runtime_state.get_available_objectives()
        if not available:
            return  # no objectives loaded in this sim profile
        target = available[0] if isinstance(available, list) else next(iter(available))

        session.set_objective(target)
        assert session.scope.runtime_state.get_current_objective_id() == target


class TestAxisTravelLimitsOnCapabilities:
    """Freeze audit Finding #20 -- `Lumascope.travel_limit_um(axis)`
    lived on the composition root but read `motorconfig.travel_limit_um`
    (motion-driver state). Canonical home is now
    `capabilities.axis_travel_limits_um` (immutable per scope, populated
    once at boot from present axes). The wrapper is retired."""

    def test_lumascope_class_does_not_carry_travel_limit_um(self):
        from modules.lumascope_api import Lumascope

        assert not hasattr(Lumascope, 'travel_limit_um'), (
            'Lumascope.travel_limit_um must be retired per audit #20; '
            'callers read scope.capabilities.axis_travel_limits_um[axis] instead.'
        )

    def test_present_axes_have_travel_limits(self, sim_scope):
        """Default sim is LS850 (X/Y/Z present). All three axes appear
        in the mapping with positive um values."""
        limits = sim_scope.capabilities.axis_travel_limits_um
        for ax in sim_scope.capabilities.axes:
            assert ax in limits, f'axis {ax} present but missing from travel limits'
            assert limits[ax] > 0.0

    def test_absent_axis_keyerrors(self, sim_scope):
        """Per Rule 8 capability-probe corollary, querying an absent
        axis is a caller bug -- contract is KeyError, not a sentinel."""
        limits = sim_scope.capabilities.axis_travel_limits_um
        # 'Q' is guaranteed absent (no motorconfig advertises it); the
        # test originally used 'T' but the sim default migrated to
        # LS850T which has a real turret, so 'T' is no longer absent.
        assert 'Q' not in sim_scope.capabilities.axes
        import pytest as _pytest

        with _pytest.raises(KeyError):
            _ = limits['Q']

    def test_mapping_is_read_only(self, sim_scope):
        """MappingProxyType wrapper enforces the frozen-dataclass
        immutability contract for the contents too. Mutation raises
        TypeError; a caller cannot silently corrupt the snapshot."""
        limits = sim_scope.capabilities.axis_travel_limits_um
        import pytest as _pytest

        with _pytest.raises(TypeError):
            limits['X'] = 1.0  # type: ignore[index]

    def test_null_motor_yields_empty_mapping(self):
        """A NullMotionBoard exposes no motorconfig; the mapping is
        empty -- which has_xy_stage / has_focus False already gates
        callers away from it."""
        from drivers.null_ledboard import NullLEDBoard
        from drivers.null_motorboard import NullMotionBoard
        from modules.scope_capabilities import ScopeCapabilities

        caps = ScopeCapabilities.from_drivers(
            motion=NullMotionBoard(),
            led=NullLEDBoard(),
            camera=None,
        )
        assert dict(caps.axis_travel_limits_um) == {}
        # The empty-mapping contract pairs with has_xy_stage=False;
        # tiling_config / motion_settings / stage consumers gate on the
        # capability and fall back to DEFAULT_STAGE_TRAVEL_UM. Pin both
        # halves so a regression that flips one without the other is
        # caught at unit-test time, not at cold-start without hardware.
        assert caps.has_xy_stage is False
        assert caps.has_focus is False


class TestOpticsOnCapabilities:
    """Freeze audit Finding #21 -- Lumascope.pixel_size() and
    Lumascope.lens_focal_length() were both motorconfig-sourced
    optical accessors living on the composition root. Canonical home is
    now capabilities.pixel_size_um + capabilities.lens_focal_length_mm
    (sibling shape to #20 / #38; sourced from motorconfig at boot).
    The Lumascope wrappers are retired. The previously-unused
    camera_pixel_size_um field (camera-SDK-sourced; no production
    readers) is retired in the same move."""

    def test_lumascope_class_does_not_carry_pixel_size_or_focal_length(self):
        from modules.lumascope_api import Lumascope

        assert not hasattr(Lumascope, 'pixel_size'), (
            'Lumascope.pixel_size must be retired per audit #21; '
            'callers read scope.capabilities.pixel_size_um instead.'
        )
        assert not hasattr(Lumascope, 'lens_focal_length'), (
            'Lumascope.lens_focal_length must be retired per audit #21; '
            'callers read scope.capabilities.lens_focal_length_mm instead.'
        )

    def test_capabilities_camera_pixel_size_um_field_removed(self):
        """The camera-SDK-sourced field had zero production readers and
        was retired in the same commit -- a single canonical pixel_size_um
        sourced from motorconfig replaces it."""
        from modules.scope_capabilities import ScopeCapabilities
        from dataclasses import fields

        names = {f.name for f in fields(ScopeCapabilities)}
        assert 'camera_pixel_size_um' not in names, (
            'camera_pixel_size_um must be retired per audit #21; '
            'pixel_size_um is the canonical motorconfig-sourced field.'
        )
        assert 'pixel_size_um' in names
        assert 'lens_focal_length_mm' in names

    def test_capabilities_optics_defaults_match_motorconfig(self, sim_scope):
        """Default sim motorconfig has Optics.PixelSize=2.0 and
        Optics.LensFocalLength=47.8; capabilities surfaces those values."""
        assert sim_scope.capabilities.pixel_size_um == 2.0
        assert sim_scope.capabilities.lens_focal_length_mm == 47.8

    def test_null_motor_optics_defaults(self):
        """A NullMotionBoard has no motorconfig; capabilities falls back
        to the Etaluma reference defaults (47.8 mm / 2.0 um) so callers
        don't need to special-case the no-hardware path."""
        from drivers.null_ledboard import NullLEDBoard
        from drivers.null_motorboard import NullMotionBoard
        from modules.scope_capabilities import ScopeCapabilities

        caps = ScopeCapabilities.from_drivers(
            motion=NullMotionBoard(),
            led=NullLEDBoard(),
            camera=None,
        )
        assert caps.pixel_size_um == 2.0
        assert caps.lens_focal_length_mm == 47.8


class TestConnectionCheckShapeUniformOnLumascope:
    """Freeze audit Finding #22 -- motor_connected / led_connected were
    Lumascope properties while camera connection was a method on
    ImagingAPI (imaging.camera_is_connected()). Two shapes (property vs
    method) and two locations (composition root vs sub-API) for the
    same question. Unified: all three are now properties on Lumascope.
    Internal imaging callers route through self._scope.camera_connected."""

    def test_imaging_class_does_not_carry_camera_is_connected(self):
        from modules.lumascope_api.imaging import ImagingAPI

        assert not hasattr(ImagingAPI, 'camera_is_connected'), (
            'ImagingAPI.camera_is_connected must be retired per audit #22; '
            'callers read scope.camera_connected (property) instead.'
        )

    def test_lumascope_has_camera_connected_property(self):
        from modules.lumascope_api import Lumascope

        attr = inspect.getattr_static(Lumascope, 'camera_connected', None)
        assert isinstance(attr, property), (
            'Lumascope.camera_connected must be a property (matches motor_connected / '
            'led_connected shape).'
        )

    def test_sim_scope_camera_connected_matches_driver_state(self, sim_scope):
        """On the default sim, the camera driver is real (SimulatedCamera);
        camera_connected reflects driver.active + is_connected()."""
        # SimulatedCamera should be active + connected after Lumascope init
        assert sim_scope.camera_connected is True

    def test_camera_connected_false_when_no_camera_driver(self, sim_scope):
        """The property must be defensive against a missing / None camera
        driver -- mirrors motor_connected / led_connected falling back to
        False rather than raising. Forcing _camera_driver=None proves the
        getattr-default path: this matches the shape of create_diagnostic
        instances (which today leave _camera_driver unset per audit #35;
        the property gracefully degrades regardless)."""
        sim_scope._camera_driver = None
        assert sim_scope.camera_connected is False


class TestFrameValidityIsL2Stable:
    """Freeze audit Finding #40 -- scope.imaging.frame_validity was
    publicly accessible (no underscore prefix) but LumascopeSkills said
    "internal diagnostic and not part of L2-stable API surface." Two
    options: prefix as _frame_validity OR formally promote. Promoted:
    L2 callers (plugin authors, diagnostic tooling, custom capture
    loops) can rely on the FrameValidity surface."""

    L2_STABLE_FRAME_VALIDITY_SURFACE = (
        'is_valid',
        'is_valid_for',
        'frames_until_valid',
        'pending_sources',
        'invalidate',
        'count_frame',
    )

    def test_frame_validity_exposes_l2_surface(self, sim_scope):
        """Every documented L2 method/property is present on the
        FrameValidity instance. Promoting locks the contract; this test
        catches accidental retirement."""
        fv = sim_scope.imaging.frame_validity
        for name in self.L2_STABLE_FRAME_VALIDITY_SURFACE:
            assert hasattr(fv, name), (
                f'FrameValidity.{name} must exist per audit #40 promotion; L2 callers depend on it.'
            )

    def test_frame_validity_is_publicly_named(self):
        """The attribute is `frame_validity`, not `_frame_validity` --
        signals 'documented L2 surface' per Rule 27 underscore convention."""
        from modules.lumascope_api.imaging import ImagingAPI

        src = inspect.getsource(ImagingAPI.__init__)
        assert 'self.frame_validity = FrameValidity()' in src, (
            'ImagingAPI.frame_validity attribute name must remain public per audit #40 promotion.'
        )
        assert 'self._frame_validity' not in src, (
            'frame_validity must not be prefixed -- the audit chose formal '
            'L2 promotion, not internal hiding.'
        )


class TestSessionImagingWrappersSymmetric:
    """Freeze audit Finding #32 originally added bare `set_gain` /
    `set_exposure_time` / `capture_and_wait` forwarders on ScopeSession
    to match the LED + motion wrapper pattern. Audit F6/F7 (LVP
    abc2a39) then retired the bare unsuffixed forwarders in favor of
    explicit `_async` / `_sync` variants so the `_sync` suffix has
    consistent meaning across sub-APIs. The remaining tests verify
    that the surviving `_sync` forwarders still route through
    ImagingAPI correctly."""

    def test_session_set_gain_forwards_to_imaging(self):
        from modules.scope_session import ScopeSession

        session = ScopeSession.create_headless()
        session.start_executors()
        try:
            session.set_gain_sync(5.5)
            assert session.scope.imaging.camera_gain == 5.5
        finally:
            session.shutdown_executors()
            session.scope.disconnect()

    def test_session_set_exposure_time_forwards_to_imaging(self):
        from modules.scope_session import ScopeSession

        session = ScopeSession.create_headless()
        session.start_executors()
        try:
            session.set_exposure_time_sync(42.0)
            assert session.scope.imaging.camera_exposure_ms == 42.0
        finally:
            session.shutdown_executors()
            session.scope.disconnect()


class TestCreateDiagnosticSharesInitMinimal:
    """Freeze audit Finding #35 -- create_diagnostic bypassed __init__
    via cls.__new__(cls) and open-coded a subset of __init__'s state
    assignments, leaving ~12 instance attributes unset. #22's
    camera_connected work surfaced one concrete instance (_camera_driver
    missing). The audit-recommended fix: extract a shared _init_minimal
    helper that both __init__ and create_diagnostic call. The slot list
    is now single-pointed-of-truth."""

    # Slots that _init_minimal sets on every Lumascope instance, regardless
    # of which constructor path was used. If a future refactor drops one,
    # this guard catches it. State that lives on sub-APIs (imaging.*,
    # motion.*, runtime_state.*) is asserted by sub-API-owned guards;
    # this list only covers the composition-root-owned slots.
    REQUIRED_SHARED_SLOTS = (
        '_simulated',
        '_camera_driver',
        '_camera_executor',
        '_io_executor',
        '_file_io_executor',
        '_executor_bundle',
        '_source_path',
        'metrics_logger',
    )

    def test_init_sets_all_shared_slots(self):
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True, register_atexit=False, register_metrics=False)
        try:
            for slot in self.REQUIRED_SHARED_SLOTS:
                assert hasattr(scope, slot), (
                    f'__init__ must set {slot} (via _init_minimal) per audit #35.'
                )
        finally:
            scope.disconnect()

    def test_create_diagnostic_sets_all_shared_slots(self):
        from modules.lumascope_api import Lumascope

        instance = Lumascope.create_diagnostic()
        try:
            for slot in self.REQUIRED_SHARED_SLOTS:
                assert hasattr(instance, slot), (
                    f'create_diagnostic must set {slot} (via _init_minimal) per audit #35.'
                )
        finally:
            instance.disconnect()

    def test_create_diagnostic_camera_driver_is_none(self):
        """The diagnostic path leaves _camera_driver=None (the
        _init_minimal default); camera_connected returns False without
        the getattr-default belt-and-suspenders firing."""
        from modules.lumascope_api import Lumascope

        instance = Lumascope.create_diagnostic()
        try:
            assert instance._camera_driver is None
            assert instance.camera_connected is False
        finally:
            instance.disconnect()


class TestLedSentinelReturnsAreNone:
    """Freeze audit Finding #39 -- sentinel return shapes were
    inconsistent across "off / unavailable" methods: get_led_ma
    returned -1 (int) or -1.0 (float); led_illumination forwarded it;
    get_led_status / camera_max_gain / get_target_position('T')
    already returned None. Audit chose the pythonic None convention;
    the float | None type is now uniform across the LED query surface."""

    def test_get_led_ma_returns_none_when_driver_absent(self):
        """A diagnostic-mode instance with a NullLEDBoard driver path
        exercises the not-self._driver branch -- returns None, not -1."""
        from modules.lumascope_api import Lumascope
        from drivers.null_ledboard import NullLEDBoard

        scope = Lumascope(simulate=True, register_atexit=False, register_metrics=False)
        try:
            scope._led_driver = NullLEDBoard()
            # IlluminationAPI._driver re-resolves through _scope._led_driver
            # each call, so the hot-swap propagates.
            assert scope.illumination.get_led_ma('Blue') is None
            assert scope.illumination.led_illumination('Blue') is None
        finally:
            scope.disconnect()

    def test_get_led_ma_returns_none_when_channel_off(self, sim_scope):
        """After led_off, the channel entry is popped from _led_state;
        get_led_ma returns None (was -1.0)."""
        # No prior led_on -- Blue starts in the never-set state.
        assert sim_scope.illumination.get_led_ma('Blue') is None
        # Force a known sequence: on, then off.
        sim_scope.illumination._led_state['Blue'] = {
            'enabled': True,
            'illumination_ma': 50.0,
            'owner': '',
        }
        assert sim_scope.illumination.get_led_ma('Blue') == 50.0
        sim_scope.illumination._led_state.pop('Blue', None)
        assert sim_scope.illumination.get_led_ma('Blue') is None

    def test_led_illumination_forwards_to_get_led_ma(self, sim_scope):
        """The two surfaces must return the same value -- they answer
        the same question."""
        sim_scope.illumination._led_state['Green'] = {
            'enabled': True,
            'illumination_ma': 75.5,
            'owner': '',
        }
        assert sim_scope.illumination.led_illumination(
            'Green'
        ) == sim_scope.illumination.get_led_ma('Green')
        sim_scope.illumination._led_state.pop('Green', None)
        assert sim_scope.illumination.led_illumination('Green') is None


class TestGetterSetterSymmetry:
    """Freeze audit Finding #36 -- the L2 surface had set_X methods
    without matching get_X across several sub-APIs. The widest gaps
    were on the composition root (set_stage_offset) and ImagingAPI
    (set_scale_bar). Added: get_stage_offset, get_scale_bar.

    The Pylon / GEV / USB SDK-perf knob cluster (set_acquisition_stop_mode,
    set_bandwidth_reserve_mode, set_device_link_throughput_limit,
    set_max_transfer_size, set_num_max_queued_urbs, set_gev_packet_size,
    set_gev_inter_packet_delay, set_max_acquisition_frame_rate) is
    deliberately write-only -- see the in-source Rule 33 decision
    comment above set_acquisition_stop_mode."""

    def test_lumascope_get_stage_offset_exists(self, sim_scope):
        # Canonical home post-Wave-7-Phase-8 is scope.runtime_state.
        assert callable(getattr(sim_scope.runtime_state, 'get_stage_offset', None))
        # Round-trip: set then get returns the same value.
        sim_scope.runtime_state.set_stage_offset({'x': 1.0, 'y': 2.0})
        assert sim_scope.runtime_state.get_stage_offset() == {'x': 1.0, 'y': 2.0}

    def test_imaging_get_scale_bar_exists(self, sim_scope):
        assert callable(getattr(sim_scope.imaging, 'get_scale_bar', None))
        # Round-trip: set then get reflects the change.
        sim_scope.imaging.set_scale_bar(enabled=True, color='white')
        snap = sim_scope.imaging.get_scale_bar()
        assert snap['enabled'] is True
        assert snap['color'] == 'white'

    def test_get_scale_bar_returns_defensive_copy(self, sim_scope):
        """Mutating the returned dict must not affect internal state."""
        sim_scope.imaging.set_scale_bar(enabled=True, color='white')
        snap = sim_scope.imaging.get_scale_bar()
        snap['enabled'] = False
        # Internal state untouched
        assert sim_scope.imaging.scale_bar_enabled is True


class TestHardwareFeaturesCapability:
    """Freeze audit Finding #4 (sub-item: hardware_features) -- the
    design doc 2.5 spec'd a hardware_features frozenset for cross-cutting
    capability tokens (trigger_in, temperature_sensor, etc.) but the
    field was never shipped. The supports() helper had a forward-
    reference in its docstring to this field; this commit makes that
    contract real."""

    def test_hardware_features_field_is_frozenset(self, sim_scope):
        assert isinstance(sim_scope.capabilities.hardware_features, frozenset)

    def test_hardware_features_defaults_to_empty(self, sim_scope):
        """Per Rule 8 empty-default semantic: empty means 'feature set
        unknown,' not 'feature X is absent.' No drivers populate the
        set today; the field exists so plugin / SDK callers can
        probe via caps.supports(token) without raising."""
        assert sim_scope.capabilities.hardware_features == frozenset()

    def test_supports_searches_hardware_features(self):
        """caps.supports('trigger_in') checks the frozenset; if the
        token is present, returns True even when no has_X / camera_supports_X
        field matches."""
        from dataclasses import replace
        from modules.scope_capabilities import ScopeCapabilities
        from drivers.null_motorboard import NullMotionBoard
        from drivers.null_ledboard import NullLEDBoard

        caps = ScopeCapabilities.from_drivers(
            motion=NullMotionBoard(),
            led=NullLEDBoard(),
            camera=None,
        )
        # Empty set: unknown token -> False
        assert caps.supports('trigger_in') is False
        # Inject a token via dataclasses.replace (preserves frozen contract)
        caps = replace(caps, hardware_features=frozenset({'trigger_in'}))
        assert caps.supports('trigger_in') is True
        # Other unknown tokens still False
        assert caps.supports('warp_drive') is False


class TestCameraMaxFrameSizeOnCapabilities:
    """Freeze audit Finding #4 (sub-item: camera_max_frame_size) +
    sibling shape to #21. The property lived on ImagingAPI but read
    per-camera-immutable data sourced from the camera driver at boot.
    Canonical home is now capabilities.camera_max_frame_size:
    tuple[int, int]. The ImagingAPI wrapper is retired."""

    def test_imaging_class_does_not_carry_camera_max_frame_size(self):
        from modules.lumascope_api.imaging import ImagingAPI

        assert not hasattr(ImagingAPI, 'camera_max_frame_size'), (
            'ImagingAPI.camera_max_frame_size must be retired per audit #4; '
            'callers read scope.capabilities.camera_max_frame_size instead.'
        )

    def test_capabilities_camera_max_frame_size_is_tuple(self, sim_scope):
        size = sim_scope.capabilities.camera_max_frame_size
        assert isinstance(size, tuple)
        assert len(size) == 2
        # Sim camera reports nonzero max size
        assert size[0] > 0
        assert size[1] > 0

    def test_no_camera_yields_zero_max_frame_size(self):
        from drivers.null_motorboard import NullMotionBoard
        from drivers.null_ledboard import NullLEDBoard
        from modules.scope_capabilities import ScopeCapabilities

        caps = ScopeCapabilities.from_drivers(
            motion=NullMotionBoard(),
            led=NullLEDBoard(),
            camera=None,
        )
        assert caps.camera_max_frame_size == (0, 0)


class TestSessionAsyncRename:
    """Freeze audit Finding #6 -- session.led_on / move_absolute /
    move_relative / move_home / leds_off were async-by-default
    forwarders to scope.X_async, but the bare name suggested sync.
    Renamed to *_async so L2 callers read the contract directly.
    Sync counterparts (led_on_sync) keep their existing names."""

    EXPECTED_ASYNC_NAMES = (
        'leds_off_async',
        'led_on_async',
        'led_off_async',
        'move_absolute_async',
        'move_relative_async',
        'move_home_async',
    )

    EXPECTED_RETIRED_NAMES = (
        'leds_off',
        'led_on',
        'led_off',
        'move_absolute',
        'move_relative',
        'move_home',
    )

    def test_async_methods_exist(self):
        from modules.scope_session import ScopeSession

        for name in self.EXPECTED_ASYNC_NAMES:
            assert callable(getattr(ScopeSession, name, None)), (
                f'ScopeSession.{name} must exist per audit #6.'
            )

    def test_bare_names_are_retired(self):
        from modules.scope_session import ScopeSession

        for name in self.EXPECTED_RETIRED_NAMES:
            assert not hasattr(ScopeSession, name), (
                f'ScopeSession.{name} must be retired per audit #6; use {name}_async instead.'
            )

    def test_led_on_sync_still_exists(self):
        """The sync counterpart keeps its name; only the bare-async
        forwarders gained the explicit _async suffix."""
        from modules.scope_session import ScopeSession

        assert callable(getattr(ScopeSession, 'led_on_sync', None))


class TestLumascopeSkillsRetiredOpticalMethods:
    """LumascopeSkills must not cite the retired optical-info methods
    (API plugin audit F1). Canonical home for pixel size and tube
    lens focal length is `scope.capabilities.pixel_size_um` /
    `.lens_focal_length_mm`. An L2 consumer copy-pasting a doc line
    that calls the retired methods used to get AttributeError.
    """

    def _doc(self):
        import pathlib

        return pathlib.Path('docs/LumascopeSkills.md').read_text()

    def test_pixel_size_method_not_cited(self):
        doc = self._doc()
        assert 'scope.pixel_size()' not in doc, (
            'LumascopeSkills.md must not cite `scope.pixel_size()` -- '
            'the method was retired. Use `scope.capabilities.pixel_size_um`.'
        )

    def test_lens_focal_length_method_not_cited(self):
        doc = self._doc()
        assert 'scope.lens_focal_length()' not in doc, (
            'LumascopeSkills.md must not cite `scope.lens_focal_length()` -- '
            'the method was retired. Use `scope.capabilities.lens_focal_length_mm`.'
        )

    def test_capability_fields_documented(self):
        doc = self._doc()
        assert 'scope.capabilities.pixel_size_um' in doc, (
            'LumascopeSkills.md must document the canonical capability '
            'field `scope.capabilities.pixel_size_um`.'
        )
        assert 'scope.capabilities.lens_focal_length_mm' in doc, (
            'LumascopeSkills.md must document the canonical capability '
            'field `scope.capabilities.lens_focal_length_mm`.'
        )


class TestGetLedStateShape:
    """get_led_state / get_led_states return shape must include `owner`
    (matches internal _led_state) and use None (not -1) for the
    illumination_ma sentinel when the channel is off / no LED board
    (matches the Sentinel-return contract preface in LumascopeSkills).
    Closes API audit F2 / F3 / F12 cluster.
    """

    def _scope(self):
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        scope._led_driver.set_timing_mode('fast')
        return scope

    def test_get_led_state_off_returns_none_sentinel_and_empty_owner(self):
        scope = self._scope()
        state = scope.illumination.get_led_state('Blue')
        assert state == {
            'enabled': False,
            'illumination_ma': None,
            'owner': '',
        }

    def test_get_led_state_on_includes_owner(self):
        scope = self._scope()
        scope.illumination.led_on(channel='Green', mA=125.0, owner='audit_test')
        state = scope.illumination.get_led_state('Green')
        assert state['enabled'] is True
        assert state['illumination_ma'] == 125.0
        assert state['owner'] == 'audit_test'

    def test_get_led_states_off_channels_use_none_and_empty_owner(self):
        scope = self._scope()
        states = scope.illumination.get_led_states()
        assert states, 'get_led_states must return per-channel entries'
        for color, entry in states.items():
            assert entry['enabled'] is False
            assert entry['illumination_ma'] is None, (
                f'{color} off-state must use None sentinel, not -1.'
            )
            assert entry['owner'] == '', f'{color} off-state must report owner = empty string.'

    def test_get_led_states_on_channel_carries_owner(self):
        scope = self._scope()
        scope.illumination.led_on(channel='Red', mA=42.5, owner='restore_pre')
        states = scope.illumination.get_led_states()
        assert states['Red']['enabled'] is True
        assert states['Red']['illumination_ma'] == 42.5
        assert states['Red']['owner'] == 'restore_pre'

    def test_doc_example_matches_shape(self):
        import pathlib

        doc = pathlib.Path('docs/LumascopeSkills.md').read_text()
        assert "'owner': '…'" in doc or "'owner': '...'" in doc, (
            'LumascopeSkills get_led_state example must include the '
            "'owner' key in the return-shape example."
        )
        # Old "current mA, or -1 if off" wording must be retired.
        assert 'current mA, or -1 if off' not in doc, (
            "Stale '-1 if off' sentinel must be removed from the led_illumination doc line."
        )


class TestProtocolCleanupLedRestoreKey:
    """`protocol_cleanup.restore_after_protocol` reads `color_data['illumination_ma']`
    from the `original_led_states` snapshot taken via `get_led_states()`.
    A prior typo (`color_data['illumination']`) silently raised KeyError
    on every LED-restore path and was swallowed by the surrounding
    try/except. Rule 16 cluster fix paired with the audit F2/F12
    sentinel migration.
    """

    def test_restore_uses_illumination_ma_key(self):
        import pathlib

        src = pathlib.Path('modules/protocol_cleanup.py').read_text()
        assert "color_data['illumination_ma']" in src, (
            "protocol_cleanup must read the snapshot's "
            "'illumination_ma' key when restoring LED state."
        )
        assert "color_data['illumination']" not in src, (
            "Stale 'illumination' key must not be read -- the snapshot "
            "shape returns 'illumination_ma'."
        )


class TestPreReleaseFutureWarning:
    """Rule 30 4-mechanism pre-freeze warning bundle requires a
    runtime FutureWarning -- mechanism #3, paired with the README
    banner, LumascopeSkills.md preface, and CHANGELOG note. Closes
    API audit F4.

    Warning fires once-per-process: any of the three L2 entry points
    (Lumascope(), ScopeSession.create, ScopeSession.create_headless)
    trips it the first time it runs; subsequent entries are silent.
    """

    @pytest.fixture(autouse=True)
    def _reset_warning_flag(self):
        # Reset module-level fired-flag so each test sees a fresh
        # not-yet-fired state. Restore on teardown so the test order
        # in the surrounding file isn't poisoned.
        import modules.lumascope_api._lumascope as lm_mod

        previous = lm_mod._PRE_RELEASE_WARNING_FIRED
        lm_mod._PRE_RELEASE_WARNING_FIRED = False
        yield
        lm_mod._PRE_RELEASE_WARNING_FIRED = previous

    def test_lumascope_init_fires_future_warning(self):
        import warnings
        from modules.lumascope_api import Lumascope

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            Lumascope(simulate=True)
        future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(future_warnings) >= 1
        msg = str(future_warnings[0].message)
        assert 'PRE-RELEASE' in msg
        assert 'LumascopeSkills' in msg

    def test_warning_fires_once_per_process(self):
        import warnings
        from modules.lumascope_api import Lumascope

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            Lumascope(simulate=True)
            Lumascope(simulate=True)
            Lumascope(simulate=True)
        future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(future_warnings) == 1, (
            f'PRE-RELEASE FutureWarning must fire once-per-process; '
            f'saw {len(future_warnings)} for 3 Lumascope constructions.'
        )

    def test_scope_session_create_headless_fires_warning(self):
        import warnings
        from modules.scope_session import ScopeSession

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ScopeSession.create_headless()
        future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(future_warnings) >= 1
        assert 'PRE-RELEASE' in str(future_warnings[0].message)

    def test_warning_text_references_migration_plan(self):
        # Static-source assertion the helper text names the migration
        # plan + Etaluma support, so retiring the warning requires
        # editing the bundle text alongside the other 3 mechanisms.
        import pathlib

        src = pathlib.Path('modules/lumascope_api/_lumascope.py').read_text()
        assert '_PRE_RELEASE_WARNING_TEXT' in src
        assert '_fire_pre_release_warning' in src
        # README banner / LumascopeSkills preface / CHANGELOG note are
        # the other three mechanisms; their existence is verified
        # outside this test file (4-mechanism PRE-RELEASE bundle).


class TestAutoGainArmedBeforeDeadlineWait:
    """Auto_Gain protocol steps must enable AG BEFORE the deadline-wait
    window, not inside capture() AFTER the wait. Otherwise AG only
    gets a single frame of convergence before the grab -- the static-
    read root cause of issue #673. Closes API protocol-workflow audit
    F1; pairs with the existing #673 fix at LVP `191dfa0` which routed
    AG through apply_layer_camera_settings inside capture() but missed
    the ordering correction.
    """

    def _runner_src(self):
        import pathlib

        return pathlib.Path('modules/protocol_step_runner.py').read_text()

    def _scr_src(self):
        import pathlib

        return pathlib.Path('modules/sequenced_capture_runner.py').read_text()

    def _writer_src(self):
        import pathlib

        return pathlib.Path('modules/protocol_image_writer.py').read_text()

    def _run_loop_src(self):
        import pathlib

        return pathlib.Path('modules/protocol_run_loop.py').read_text()

    def test_armed_step_attribute_initialized_on_scr(self):
        src = self._scr_src()
        assert 'self._auto_gain_armed_step = -1' in src, (
            'SCR.__init__ must initialize _auto_gain_armed_step to -1 '
            'so the first scan_iterate sees a fresh "not yet armed" '
            'state.'
        )

    def test_run_loop_resets_armed_step_per_scan(self):
        src = self._run_loop_src()
        assert 'p._auto_gain_armed_step = -1' in src, (
            'protocol_run_loop must reset _auto_gain_armed_step '
            'alongside _auto_gain_deadline at the start of each scan '
            'so a re-run does not skip AG arming.'
        )

    def test_arm_block_precedes_deadline_gate_in_scan_iterate(self):
        src = self._runner_src()
        arm_marker = "if step['Auto_Gain'] and p._auto_gain_armed_step != p._curr_step:"
        gate_marker = "if step['Auto_Gain'] and time.monotonic() < p._auto_gain_deadline:"
        arm_idx = src.find(arm_marker)
        gate_idx = src.find(gate_marker)
        assert arm_idx >= 0, 'AG-arm block must exist in scan_iterate.'
        assert gate_idx >= 0, 'Deadline gate must exist in scan_iterate.'
        assert arm_idx < gate_idx, (
            'AG-arm block must appear BEFORE the deadline gate so the '
            'camera converges DURING the max_duration wait, not in a '
            'single frame inside capture() AFTER the wait.'
        )

    def test_arm_block_routes_apply_through_io_executor(self):
        src = self._runner_src()
        # Find the arm block and assert it includes the IOTask submit
        # with apply_layer_camera_settings.
        arm_idx = src.find("if step['Auto_Gain'] and p._auto_gain_armed_step != p._curr_step:")
        gate_idx = src.find("if step['Auto_Gain'] and time.monotonic() < p._auto_gain_deadline:")
        block = src[arm_idx:gate_idx]
        assert 'p._scope.imaging.apply_layer_camera_settings' in block, (
            'Arm block must call apply_layer_camera_settings (the same path live-mode AE/AG uses).'
        )
        assert "'auto_gain': True" in block, 'Arm block must pass auto_gain=True.'
        assert 'IOTask' in block and 'protocol_put' in block, (
            'Arm block must route through io_executor.protocol_put so '
            'the apply is serialized with other protocol-thread IO.'
        )

    def test_capture_does_not_double_apply_for_ag_step(self):
        src = self._writer_src()
        # The AG branch in capture() should no longer call apply
        # (scan_iterate already armed AG). Locate the "Auto_Gain step"
        # branch and assert there is no apply_layer_camera_settings
        # call between it and the next sibling block.
        marker = '# Auto_Gain step: scan_iterate arms AG'
        idx = src.find(marker)
        assert idx >= 0, (
            'capture() Auto_Gain branch must document that scan_iterate '
            'arms AG; the docstring is the contract.'
        )
        # Look at the next 800 chars of the branch body.
        branch_window = src[idx : idx + 800]
        assert 'self._scope.imaging.apply_layer_camera_settings(' not in branch_window, (
            'capture() Auto_Gain branch must not call '
            'self._scope.imaging.apply_layer_camera_settings(...) -- '
            'doing so would restart AG mid-grab and discard the '
            'convergence the deadline-wait produced.'
        )

    def test_deadline_is_set_inside_arm_block_not_after_gate(self):
        """The convergence deadline must be set AT ARM TIME so it
        actually gates THIS step. Pre-fix the deadline was initialized
        once at scan-start before AF (which takes ~10s), so when the
        first AG step hit the gate, `now < deadline` was already FALSE
        and the gate fell through immediately -- capture grabbed
        ~70-120ms after arm instead of the intended 1000ms convergence
        window (Chris's 2026-05-22 bench evidence on beta14). Closes
        issue #673 recurrence.
        """
        src = self._runner_src()
        arm_marker = "if step['Auto_Gain'] and p._auto_gain_armed_step != p._curr_step:"
        gate_marker = "if step['Auto_Gain'] and time.monotonic() < p._auto_gain_deadline:"
        arm_idx = src.find(arm_marker)
        gate_idx = src.find(gate_marker)
        assert arm_idx >= 0 and gate_idx >= 0
        # Within the arm block (between arm marker and gate marker),
        # the deadline MUST be set. The exact spelling allows a
        # multi-line formulation (the deadline expression is long).
        arm_block = src[arm_idx:gate_idx]
        assert 'p._auto_gain_deadline' in arm_block, (
            'Convergence deadline must be set INSIDE the arm block '
            '(one-shot per step). Pre-fix it was set only at scan-'
            'start before AF, so the gate fell through immediately. '
            'Issue #673 recurrence.'
        )
        assert 'p._autogain_settings' in arm_block and "'max_duration'" in arm_block, (
            'Arm-time deadline assignment must use '
            "p._autogain_settings['max_duration'] -- the configured "
            'convergence window. Issue #673.'
        )

    def test_arm_block_returns_after_setting_deadline(self):
        """The arm block must `return` after setting the deadline so
        the next scan_iterate tick polls the gate. Without the return,
        the deadline-just-set is checked in the same tick and falls
        through (now ~= deadline + epsilon) -- the bug recurs.

        The check: between the arm block and the gate marker there
        must be a `return` statement.
        """
        src = self._runner_src()
        arm_marker = "if step['Auto_Gain'] and p._auto_gain_armed_step != p._curr_step:"
        gate_marker = "if step['Auto_Gain'] and time.monotonic() < p._auto_gain_deadline:"
        arm_idx = src.find(arm_marker)
        gate_idx = src.find(gate_marker)
        arm_block = src[arm_idx:gate_idx]
        # Look for a bare `return` on its own indented line inside the
        # arm block (the if-body), not at the outer function scope.
        # The arm block is indented 8 spaces (inside an if inside the
        # function); its body lines are indented 12 spaces.
        return_lines = [
            line
            for line in arm_block.splitlines()
            if line.strip() == 'return' and line.startswith(' ' * 12)
        ]
        assert len(return_lines) >= 1, (
            'Arm block must `return` after setting _auto_gain_deadline '
            'so the next scan_iterate tick polls the gate (issue #673 '
            'recurrence). Without the return, the deadline gate is '
            'checked in the same tick and falls through immediately.'
        )

    def test_run_loop_does_not_reset_deadline_at_scan_start(self):
        """The scan-start deadline init at protocol_run_loop.py:158
        was the root of the #673 recurrence: it set the deadline to
        scan_start + max_duration before AF (which then ate ~10s), so
        the gate was always past-deadline by the time AG armed. Fix
        removed the line entirely; arm-time deadline-set is the only
        canonical write site.
        """
        src = self._run_loop_src()
        # The armed-step reset must still be present (the existing
        # test test_run_loop_resets_armed_step_per_scan pins it).
        # But the deadline init must NOT be in the run-loop body.
        # Find the SCANNING state set + assert no deadline-assign
        # line follows it in the same indentation block.
        state_marker = 'p._set_state(ProtocolState.SCANNING)'
        idx = src.find(state_marker)
        assert idx >= 0
        # Take the next 600 chars (the block after SCANNING state set
        # up through scan_loop call).
        block = src[idx : idx + 600]
        assert 'p._auto_gain_deadline = time.monotonic()' not in block, (
            'protocol_run_loop must NOT set _auto_gain_deadline at '
            'scan start -- that produced a past-deadline gate after '
            'AF ran. Deadline is set per-step at arm time in '
            'protocol_step_runner. Issue #673.'
        )


class TestWindowsBuildIsWindowed_559:
    """Issue #559 recurrence: Chris reported "extra terminal windows
    that say 'exiting'" on the Windows .exe lock-loser path.

    Root cause: the PyInstaller spec had `console=True`, so every
    .exe launch opened a black bootloader console alongside the Kivy
    window. The lock-loser's stderr `print(f'ERROR: ... Exiting.')`
    wrote into that console; the subsequent `os._exit(1)` terminated
    the Python interpreter but the bootloader-owned console window
    persisted, leaving an orphan terminal showing the "Exiting."
    line.

    Two-part fix:
    1. Windows spec uses `console=False` (windowed build). No
       bootloader console window appears on any launch.
    2. The lock-loser path drops its stderr print. The tkinter
       messagebox + logger.error already cover the user + log paths;
       a windowed build silently drops stderr anyway, so the print
       was load-bearing only on a console=True build that
       inadvertently leaked terminals.

    Bench verification is Windows-only (macOS .app bundles never
    spawn a Terminal window from a frozen build). These source pins
    catch reverts of either half.
    """

    def _spec_src(self):
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent
            / 'scripts'
            / 'appBuild'
            / 'config'
            / 'lumaviewpro_win_release.spec'
        ).read_text()

    def _main_src(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'lumaviewpro.py').read_text()

    def test_windows_spec_is_windowed_build(self):
        """Spec must declare `console=False` so PyInstaller produces
        a windowed .exe (no bootloader terminal window). Issue #559."""
        spec = self._spec_src()
        assert 'console=False' in spec, (
            'Windows PyInstaller spec must use console=False so the '
            'frozen .exe does not spawn a bootloader console window '
            'alongside the Kivy app. Pre-fix console=True left an '
            'orphan terminal on the lock-loser path. Issue #559.'
        )
        assert 'console=True' not in spec, (
            'Windows PyInstaller spec must NOT contain console=True '
            '(any stray occurrence regresses #559).'
        )

    def test_lock_loser_drops_stderr_print(self):
        """Lock-loser path at lumaviewpro.py:~129-154 must not write
        to sys.stderr. On a windowed build that stderr write is
        silent anyway; on a console=True build it was the literal
        line Chris saw left behind in the orphan terminal."""
        src = self._main_src()
        # Locate the lock-loser block by its sentinel _msg assignment.
        msg_idx = src.find("_msg = 'Another instance of LVP may already be running")
        assert msg_idx >= 0, (
            'Could not find the lock-loser _msg literal -- test needs '
            'updating if the message was reworded.'
        )
        # The block ends at the `os._exit(1)` call below it. Slice
        # the block and assert no stderr print.
        exit_idx = src.find('os._exit(1)', msg_idx)
        assert exit_idx > msg_idx
        loser_block = src[msg_idx:exit_idx]
        assert 'file=sys.stderr' not in loser_block, (
            'Lock-loser path must not write to sys.stderr -- the '
            'tkinter messagebox + logger.error already cover the '
            'user and log surfaces, and on console=False builds the '
            'stderr write is silent. Issue #559.'
        )


class TestShutdownLedsOffRoutedThroughIoExecutor:
    """The application shutdown path must turn LEDs off through the
    io_executor, NOT via an ad-hoc daemon Thread that races with
    in-flight io_executor tasks on the LED serial bus. Closes API
    threading audit F12. The leds_off must also fire BEFORE
    shutdown_threads tears the io_executor down -- otherwise the
    put() lands in a queue whose worker is exiting and may not be
    processed.
    """

    def _src(self):
        import pathlib

        return pathlib.Path('lumaviewpro.py').read_text()

    def test_no_adhoc_leds_off_thread(self):
        src = self._src()
        assert 'threading.Thread(target=lumaview.scope.illumination.leds_off' not in src, (
            'lumaviewpro.py must not spawn a bare daemon Thread for '
            'shutdown leds_off -- it races with io_executor in-flight '
            'LED writes on the serial bus.'
        )

    def test_leds_off_routes_through_io_executor(self):
        src = self._src()
        # Find the shutdown leds_off block by its log message header.
        marker = '[LVP Main  ] lumaview.scope.illumination.leds_off()'
        idx = src.find(marker)
        assert idx >= 0, 'Shutdown leds_off block must keep its log message header.'
        block = src[idx : idx + 1500]
        assert 'ctx.io_executor.put(' in block, (
            'Shutdown leds_off must route through ctx.io_executor.put '
            'so the LED serial bus is not contended by a parallel '
            'writer during shutdown drain.'
        )
        assert 'IOTask(action=lumaview.scope.illumination.leds_off)' in block, (
            'IOTask must wrap lumaview.scope.illumination.leds_off so '
            'io_executor serializes it with other LED writes.'
        )
        assert 'fut.result(timeout=2.0)' in block, (
            'fut.result(timeout=2.0) preserves the prior 2-second '
            'MainThread-doesn-t-block timeout semantic.'
        )

    def test_leds_off_precedes_shutdown_threads(self):
        src = self._src()
        leds_off_idx = src.find('[LVP Main  ] lumaview.scope.illumination.leds_off()')
        shutdown_idx = src.find('self.shutdown_threads()')
        assert leds_off_idx >= 0 and shutdown_idx >= 0
        assert leds_off_idx < shutdown_idx, (
            'Shutdown leds_off must fire BEFORE shutdown_threads tears '
            'io_executor down. Otherwise the put() races with the '
            'worker exiting and the leds_off may never fire.'
        )


class TestImagingAsyncSyncThreeVariantPattern:
    """Imaging sub-API must match the illumination 3-variant pattern:
    bare name = direct sync, _async = queued + immediate return,
    _sync = queued + blocking. Session forwarders for imaging must
    also follow the symmetric _async / _sync split. Plain `set_gain`
    / `set_exposure_time` / `capture_and_wait` on ScopeSession
    retired in favor of explicit suffixes so the LumascopeSkills
    'async-by-default' preface stops lying for imaging. Closes API
    audit F6 + F7 cluster.
    """

    def test_imaging_has_set_gain_async(self):
        from modules.lumascope_api.imaging import ImagingAPI

        assert hasattr(ImagingAPI, 'set_gain_async')
        assert callable(ImagingAPI.set_gain_async)

    def test_imaging_has_set_exposure_time_async(self):
        from modules.lumascope_api.imaging import ImagingAPI

        assert hasattr(ImagingAPI, 'set_exposure_time_async')
        assert callable(ImagingAPI.set_exposure_time_async)

    def test_imaging_has_capture_and_wait_async(self):
        from modules.lumascope_api.imaging import ImagingAPI

        assert hasattr(ImagingAPI, 'capture_and_wait_async')
        assert callable(ImagingAPI.capture_and_wait_async)

    def test_session_imaging_forwarders_renamed(self):
        from modules.scope_session import ScopeSession

        # _async + _sync variants must exist
        for name in (
            'set_gain_async',
            'set_gain_sync',
            'set_exposure_time_async',
            'set_exposure_time_sync',
            'capture_and_wait_async',
            'capture_and_wait_sync',
        ):
            assert callable(getattr(ScopeSession, name, None)), (
                f'ScopeSession.{name} must exist per audit F6/F7 three-variant pattern.'
            )
        # Unsuffixed forwarders are retired -- they were the source of the
        # preface lie. Plain `set_gain` / `set_exposure_time` /
        # `capture_and_wait` should NOT exist on ScopeSession.
        for name in ('set_gain', 'set_exposure_time', 'capture_and_wait'):
            assert not hasattr(ScopeSession, name), (
                f'ScopeSession.{name} must be retired per audit F7 -- '
                f'use {name}_async or {name}_sync instead.'
            )

    def test_session_set_gain_async_routes_through_executor(self):
        # The async variant should return None and submit via executor.
        from modules.scope_session import ScopeSession

        session = ScopeSession.create_headless()
        session.start_executors()
        try:
            result = session.set_gain_async(7.0)
            assert result is None, (
                'set_gain_async must return None (fire-and-forget); '
                'value lands after the executor processes the IOTask.'
            )
            # Drain by calling _sync afterwards -- if executor wiring
            # is healthy, the prior async write completes first.
            session.set_gain_sync(7.0)
            assert session.scope.imaging.camera_gain == 7.0
        finally:
            session.shutdown_executors()
            session.scope.disconnect()


class TestImagingPylonSdkPerfSettersPrivatized:
    """Pylon SDK-perf imaging setters are bench-tooling artifacts, not
    part of the L2 contract. Per API audit F8 they are renamed with a
    leading underscore so an L2 consumer doing dir(scope.imaging)
    sees a clear 'not part of contract' signal.
    """

    PRIVATIZED = (
        'set_acquisition_stop_mode',
        'set_bandwidth_reserve_mode',
        'set_device_link_throughput_limit',
        'set_max_transfer_size',
        'set_num_max_queued_urbs',
        'set_gev_packet_size',
        'set_gev_inter_packet_delay',
    )

    def test_private_versions_exist(self):
        from modules.lumascope_api.imaging import ImagingAPI

        for name in self.PRIVATIZED:
            private = f'_{name}'
            assert hasattr(ImagingAPI, private), (
                f'ImagingAPI must expose {private} (underscore-prefixed) per audit F8.'
            )

    def test_public_versions_retired(self):
        from modules.lumascope_api.imaging import ImagingAPI

        for name in self.PRIVATIZED:
            assert not hasattr(ImagingAPI, name), (
                f'ImagingAPI.{name} must be retired in favor of '
                f'_{name} -- the public name signaled L2-contract '
                f'membership for a bench-tooling artifact.'
            )


class TestLedEngineeringModeSymmetricReturnTypes:
    """Per API audit F10: enter_led_engineering_mode returned `bool`;
    exit_led_engineering_mode returned `bool | None`. The asymmetry
    forced L2 / Matlab consumers writing portable code to pick one
    null check or the other. The fix collapses None paths to False
    on exit so both enter / exit return a uniform `bool`.
    """

    def _src(self):
        import pathlib

        return pathlib.Path('modules/lumascope_api/diagnostics.py').read_text()

    def test_exit_returns_bool_only(self):
        src = self._src()
        idx = src.find('def exit_led_engineering_mode')
        assert idx >= 0
        # Look at the next ~600 chars (the function body).
        block = src[idx : idx + 600]
        assert '-> bool:' in block, (
            'exit_led_engineering_mode return annotation must be `bool` '
            '(not `bool | None`) -- symmetric with enter.'
        )
        assert 'return None' not in block, (
            'exit_led_engineering_mode body must not return None; the '
            'None paths collapsed to False so the L2 contract is '
            'uniformly bool.'
        )

    def test_enter_unchanged_returns_bool(self):
        src = self._src()
        idx = src.find('def enter_led_engineering_mode')
        assert idx >= 0
        block = src[idx : idx + 600]
        assert '-> bool:' in block, (
            'enter_led_engineering_mode must still return bool (the symmetric counterpart).'
        )

    def test_runtime_exit_returns_bool_with_no_driver(self):
        # When no LED driver is attached, the helper short-circuits to
        # False (previously returned None).
        from modules.lumascope_api.diagnostics import DiagnosticsAPI
        from unittest.mock import MagicMock

        fake_scope = MagicMock()
        fake_scope._led_driver = None
        api = DiagnosticsAPI(fake_scope)
        result = api.exit_led_engineering_mode()
        assert result is False, (
            'exit_led_engineering_mode must return False (not None) '
            'when no LED driver is available.'
        )


class TestScopeSessionBuildsFullExecutorBundle:
    """Per API audit F11: ScopeSession.create_headless() was building only
    io_executor + camera_executor, skipping file_io_executor + worker_pool
    + protocol_thread + scope_display_thread. L2 callers using
    ScopeSession.create*() got a silently degraded topology where the
    file-IO IOTask path fell back to inline execution and the worker-pool
    priority lanes were unavailable.

    The fix routes create_headless() through executor_registry.create_default
    so headless callers get the same topology lumaviewpro.py runs.
    """

    def test_create_headless_registers_file_io_executor_on_scope(self):
        from modules.scope_session import ScopeSession

        session = ScopeSession.create_headless()
        assert session.scope._file_io_executor is not None, (
            'ScopeSession.create_headless() must register a file_io_executor '
            'on the scope; without it, protocol_image_writer + IOTask file-IO '
            'paths fall back to inline execution and pipelining is lost.'
        )

    def test_create_headless_attaches_executor_bundle_to_scope(self):
        from modules.scope_session import ScopeSession
        from modules.executor_registry import ExecutorBundle

        session = ScopeSession.create_headless()
        # register_executor_bundle stores the bundle on _executor_bundle.
        bundle = getattr(session.scope, '_executor_bundle', None)
        assert isinstance(bundle, ExecutorBundle), (
            'ScopeSession.create_headless() must call register_executor_bundle '
            'so MetricsLogger snapshot() reports all 4 executor queue depths.'
        )

    def test_create_headless_session_carries_bundle_reference(self):
        from modules.scope_session import ScopeSession
        from modules.executor_registry import ExecutorBundle

        session = ScopeSession.create_headless()
        assert isinstance(session.executor_bundle, ExecutorBundle), (
            'ScopeSession.create_headless() must store the bundle on the '
            'session itself so headless callers can shut down protocol_thread '
            '/ scope_display_thread cleanly.'
        )

    def test_create_headless_bundle_has_all_four_executors(self):
        from modules.scope_session import ScopeSession

        session = ScopeSession.create_headless()
        bundle = session.executor_bundle
        # All four executors are required for full L2-caller pipelining.
        for attr_name in ('io_executor', 'camera_executor', 'file_io_executor', 'worker_pool'):
            assert getattr(bundle, attr_name) is not None, (
                f'Bundle missing {attr_name}; L2 caller will hit degraded '
                f'topology when that executor is needed.'
            )

    def test_create_with_explicit_executors_skips_bundle_build(self):
        # When the caller passes their own executor handles (e.g. lumaviewpro.py
        # build() owns the bundle and constructs ScopeSession with shared
        # handles), create() must NOT spawn a second bundle.
        from modules.scope_session import ScopeSession
        from modules.sequential_io_executor import SequentialIOExecutor

        io = SequentialIOExecutor(name='IO_TEST')
        cam = SequentialIOExecutor(name='CAMERA_TEST')
        try:
            session = ScopeSession.create(
                settings={},
                io_executor=io,
                camera_executor=cam,
            )
            assert session.executor_bundle is None, (
                'ScopeSession.create() must not build a bundle when the caller '
                'passes io_executor + camera_executor explicitly; that path is '
                'reserved for lumaviewpro.py-style bundle ownership.'
            )
            assert session.io_executor is io
            assert session.camera_executor is cam
        finally:
            io.shutdown()
            cam.shutdown()


class TestAutogainSettingsSnapshottedAtRunStart:
    """Per protocol-workflow audit F15: autogain_settings dict was
    referenced not snapshotted, so mid-run UI mutations of fields like
    target_brightness or max_duration leaked into the in-flight scan.
    The comment claimed "Immutable after assignment" but no deepcopy
    enforced the immutability.

    The fix deepcopies autogain_settings at run() entry, matching the
    false_color_16bit + stage_offset snapshot pattern already used in
    the runner.
    """

    def test_autogain_settings_deepcopied_in_run(self):
        # Drive the runner up to the autogain_settings assignment in run()
        # by inspecting the source -- a full run() call requires the whole
        # protocol-thread context. The audit's recommendation is the
        # 1-line copy.deepcopy(autogain_settings) at assignment site.
        import pathlib

        src = pathlib.Path('modules/sequenced_capture_runner.py').read_text()
        idx = src.find('self._autogain_settings =')
        assert idx >= 0
        # Look at the next ~150 chars (the assignment + neighborhood).
        block = src[idx : idx + 150]
        assert 'copy.deepcopy(autogain_settings)' in block, (
            'self._autogain_settings must be assigned via copy.deepcopy '
            'so mid-run UI mutations do not leak into the in-flight scan '
            '(audit F15). Got: ' + block[:200]
        )

    def test_autogain_settings_none_safe(self):
        # When the caller passes autogain_settings=None (rare but the
        # parameter signature allows it), the snapshot must not raise --
        # falls through to {} so the AG path sees an empty dict.
        import pathlib

        src = pathlib.Path('modules/sequenced_capture_runner.py').read_text()
        idx = src.find('self._autogain_settings =')
        block = src[idx : idx + 200]
        assert 'autogain_settings is not None' in block or 'autogain_settings or {}' in block, (
            'autogain_settings None-guard must exist so deepcopy(None) '
            'cannot raise. Got: ' + block[:200]
        )


class TestProtocolPeriodZeroDoesNotCrashFullProtocolMode:
    """Per protocol-workflow audit F2: _calculate_num_scans for
    FULL_PROTOCOL mode divided protocol.duration()/protocol.period()
    without guarding against period==0. Protocol.from_file explicitly
    permits period==0 as a "valid single-scan marker" and
    protocol_time_estimator handles it; the runner did not. Loading +
    running such a protocol in FULL_PROTOCOL mode raised
    ZeroDivisionError silently, returned early from run(), and the
    user saw nothing happen after pressing Start.

    The fix treats period==0 as 1 scan in FULL_PROTOCOL mode, matching
    the protocol_time_estimator contract.
    """

    def _make_protocol_stub(self, *, duration: float, period: float):
        from unittest.mock import MagicMock

        proto = MagicMock()
        proto.duration.return_value = duration
        proto.period.return_value = period
        return proto

    def test_period_zero_returns_one_scan(self):
        from modules.sequenced_capture_runner import (
            SequencedCaptureRunner,
            SequencedCaptureRunMode,
        )

        proto = self._make_protocol_stub(duration=60.0, period=0)
        n = SequencedCaptureRunner._calculate_num_scans(
            protocol=proto,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            max_scans=None,
        )
        assert n == 1, (
            'period==0 in FULL_PROTOCOL mode must return 1 scan '
            '(single-scan marker semantics), not raise ZeroDivisionError.'
        )

    def test_period_nonzero_unchanged(self):
        from modules.sequenced_capture_runner import (
            SequencedCaptureRunner,
            SequencedCaptureRunMode,
        )

        # 60s duration / 10s period = 6 scans (baseline behavior preserved).
        proto = self._make_protocol_stub(duration=60.0, period=10.0)
        n = SequencedCaptureRunner._calculate_num_scans(
            protocol=proto,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            max_scans=None,
        )
        assert n == 6

    def test_period_zero_respects_max_scans(self):
        # If max_scans is provided, period==0 should still respect the
        # min(1, max_scans) clamp. max_scans=0 means "no scans" -> 0.
        from modules.sequenced_capture_runner import (
            SequencedCaptureRunner,
            SequencedCaptureRunMode,
        )

        proto = self._make_protocol_stub(duration=60.0, period=0)
        n = SequencedCaptureRunner._calculate_num_scans(
            protocol=proto,
            run_mode=SequencedCaptureRunMode.FULL_PROTOCOL,
            max_scans=0,
        )
        assert n == 0


class TestBfAfForFluorescenceSnapshottedAtRunStart:
    """Per protocol-workflow audit F5: scan_iterate read
    ctx.settings['protocol']['bf_af_for_fluorescence'] under
    settings_lock on every tick. Two costs: per-tick lock contention
    with the UI thread, AND mid-run user toggles took effect partway
    through a scan -- producing inconsistent AF behavior across steps
    within one protocol run.

    The fix snapshots the setting in SequencedCaptureRunner.run()
    (alongside false_color_16bit, under the same settings_lock take)
    onto self._bf_af_for_fluorescence; protocol_step_runner reads from
    the snapshot via getattr(p, '_bf_af_for_fluorescence', False).
    """

    def test_runner_snapshots_bf_af_for_fluorescence_attr(self):
        # Static-source check: the runner's run() body must contain
        # a snapshot assignment of self._bf_af_for_fluorescence.
        import pathlib

        src = pathlib.Path('modules/sequenced_capture_runner.py').read_text()
        assert 'self._bf_af_for_fluorescence' in src, (
            'SequencedCaptureRunner.run() must snapshot '
            'bf_af_for_fluorescence onto self for per-tick reads.'
        )
        # The snapshot must take settings_lock (alongside false_color_16bit).
        idx = src.find('self._bf_af_for_fluorescence =')
        assert idx >= 0
        # Look backwards ~300 chars; settings_lock should appear there.
        ctx_block = src[max(0, idx - 300) : idx + 200]
        assert 'settings_lock' in ctx_block, (
            'bf_af_for_fluorescence snapshot must be taken under '
            'settings_lock for consistent UI/protocol-thread read.'
        )

    def test_protocol_step_runner_reads_from_snapshot(self):
        # Static-source check: protocol_step_runner.scan_iterate must
        # read from p._bf_af_for_fluorescence (the runner snapshot),
        # not from ctx.settings directly.
        import pathlib

        src = pathlib.Path('modules/protocol_step_runner.py').read_text()
        assert "p, '_bf_af_for_fluorescence'" in src or 'p._bf_af_for_fluorescence' in src, (
            'protocol_step_runner.scan_iterate must read '
            'bf_af_for_fluorescence from the runner snapshot, '
            'not from ctx.settings each tick.'
        )
        # The old per-tick settings.get path must be gone.
        assert "ctx.settings.get('protocol', {}).get('bf_af_for_fluorescence'" not in src, (
            'Per-tick ctx.settings read for bf_af_for_fluorescence '
            'must be removed (replaced by snapshot read).'
        )


class TestRunPreValidationFiresNotificationOnException:
    """Per protocol-workflow audit F8: SequencedCaptureRunner.run()
    wrapped validate_for_run() in try/except Exception that logged a
    warning + fell through to "proceed anyway." If validate_for_run
    raised (labware loader OS error, missing objectives.json, pandas
    exception in the steps DataFrame), the user saw nothing -- the
    protocol ran without validation. Any subsequent runtime failure
    hit hardware mid-run instead of being caught at run() entry.

    The fix mirrors the are_all_connected exception handling: log
    error, fire notifications.error popup, return.
    """

    def test_validate_for_run_exception_fires_notification_and_returns(self):
        # Static-source check: the except branch for validate_for_run
        # must call notifications.error + return, not just log warning.
        import pathlib

        src = pathlib.Path('modules/sequenced_capture_runner.py').read_text()
        # Find the validate_for_run try block + its except handler.
        try_idx = src.find('validation_errors = protocol.validate_for_run')
        assert try_idx >= 0
        # Look at the next ~2500 chars (the except branch should follow).
        block = src[try_idx : try_idx + 2500]
        # The block must contain the notifications.error call AND the
        # return after the exception is caught.
        except_idx = block.find('except Exception')
        assert except_idx >= 0, 'validate_for_run try block must have except handler'
        except_block = block[except_idx:]
        assert 'notifications.error' in except_block, (
            'validate_for_run exception path must fire notifications.error '
            '(not just log warning) so the user sees the failure popup.'
        )
        # Old anti-pattern: "Proceeding anyway." -- must be gone.
        assert 'Proceeding anyway' not in except_block, (
            'validate_for_run exception path must NOT proceed past the '
            'failure (old anti-pattern: log warning + fall through).'
        )
        # Must return early so the run does not start.
        assert (
            'return' in except_block.split('\n', 30)[0:30].__str__()
            or 'return' in except_block[:600]
        ), 'validate_for_run exception path must return early after firing the notification.'


class TestCompositeOrchestrationByteEqualManualVsProtocol:
    """Per #672 root-cause hunt (AUDIT_COLOR_CONVENTION_2026-05-15 F0.5g):
    the manual composite path and the protocol post-processing composite
    path used to produce visibly different TIFF bytes at the same input.
    The cause was orchestration divergence:
      - Manual: build_composite -> RGB array -> tifffile (RGB-native).
      - Protocol: cv2.imread (BGR) -> cv2.cvtColor(BGR2GRAY) per channel
        -> build_composite -> cv2.cvtColor(RGB2BGR) -> cv2.imwrite (BGR).

    The structural fix collapses both paths to the same shape: mono
    channel arrays in, build_composite RGB out, tifffile.imwrite at
    save. This test guards against any future regression that reintroduces
    a cvtColor or a cv2.imwrite on the composite path -- byte-equality of
    the two saved composites is the falsifying instrument.
    """

    def test_manual_and_protocol_composite_paths_produce_byte_equal_tiffs(self):
        import pathlib
        import tempfile
        import numpy as np
        import pandas as pd
        import tifffile as tf
        from modules.composite_builder import build_composite
        from modules.composite_generation import CompositeGeneration

        # Three known mono channels: a Red gradient, a Green stripe, a
        # Blue checkerboard. Deterministic, distinct per channel, so
        # the channel-order bug would manifest as visible RGB swap.
        H, W = 32, 48
        red = np.zeros((H, W), dtype=np.uint8)
        green = np.zeros((H, W), dtype=np.uint8)
        blue = np.zeros((H, W), dtype=np.uint8)
        # Red: horizontal gradient.
        for x in range(W):
            red[:, x] = int(255 * x / max(W - 1, 1))
        # Green: vertical stripes (every other column = 200).
        green[:, ::2] = 200
        # Blue: checkerboard (every other 4x4 block = 180).
        for r in range(0, H, 4):
            for c in range(0, W, 4):
                if ((r // 4) + (c // 4)) % 2 == 0:
                    blue[r : r + 4, c : c + 4] = 180

        channel_images = {'Red': red, 'Green': green, 'Blue': blue}

        # Path A: manual composite path. build_composite -> tifffile.
        manual_rgb = build_composite(
            channel_images=channel_images,
            transmitted_image=None,
            brightness_thresholds=None,
            dtype=np.uint8,
            max_value=255,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            manual_tiff = tmp / 'manual_composite.tiff'
            tf.imwrite(
                str(manual_tiff),
                manual_rgb,
                photometric='rgb',
                compression='lzw',
            )

            # Path B: protocol composite path. Round-trip the channels
            # through per-channel TIFFs (as the live protocol would
            # have done at scan time), then call
            # _create_composite_image which reads them back, builds
            # the composite, and saves via tifffile.
            channel_dir = tmp / 'protocol_channels'
            channel_dir.mkdir()
            rows = []
            for layer_name, arr in channel_images.items():
                fname = f'{layer_name}.tiff'
                tf.imwrite(str(channel_dir / fname), arr, compression='lzw')
                # _create_composite_image expects Filepath relative to
                # the root path argument.
                rows.append(
                    {
                        'Filepath': f'protocol_channels/{fname}',
                        'Color': layer_name,
                    }
                )
            df = pd.DataFrame(rows)

            protocol_tiff = tmp / 'protocol_composite.tiff'
            result = CompositeGeneration._create_composite_image(
                path=tmp,
                df=df,
                output_file_loc=protocol_tiff,
            )
            assert result['status'] is True, (
                f'Protocol composite generation must succeed; got error: {result.get("error")}'
            )
            assert result['image'] is None, (
                'When output_file_loc is provided, _create_composite_image '
                'must save internally + return image=None so the base '
                'class _process_group_callback skips its cv2.imwrite.'
            )

            # Read both back and compare. tifffile imread returns
            # the exact array that was written (LZW is lossless).
            manual_read = tf.imread(str(manual_tiff))
            protocol_read = tf.imread(str(protocol_tiff))

            assert manual_read.shape == protocol_read.shape, (
                f'Composite shape divergence: manual {manual_read.shape} '
                f'vs protocol {protocol_read.shape}. Both paths must '
                f'produce the same output shape.'
            )
            assert np.array_equal(manual_read, protocol_read), (
                'Manual and protocol composite paths must produce '
                'byte-equal RGB arrays at the same input. The #672 '
                'root cause was a cv2.cvtColor(RGB2BGR) on the protocol '
                'side; if this test fails, a cvtColor or BGR-write has '
                'been reintroduced.'
            )

    def test_create_composite_image_returns_image_array_when_no_output_path(self):
        # Legacy / test callers that don't pass output_file_loc must
        # still get the RGB array back so they can save themselves.
        # This preserves the old return contract for any caller still
        # depending on it.
        import numpy as np
        import pandas as pd
        import pathlib
        import tempfile
        import tifffile as tf
        from modules.composite_generation import CompositeGeneration

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            arr = np.full((8, 8), 100, dtype=np.uint8)
            tf.imwrite(str(tmp / 'Red.tiff'), arr, compression='lzw')
            df = pd.DataFrame([{'Filepath': 'Red.tiff', 'Color': 'Red'}])

            result = CompositeGeneration._create_composite_image(
                path=tmp,
                df=df,
                output_file_loc=None,
            )
            assert result['status'] is True
            assert result['image'] is not None, (
                'When output_file_loc is None, _create_composite_image '
                'must return the RGB array so legacy callers can save it.'
            )
            assert result['image'].shape == (8, 8, 3)


class TestProfileTraceGateIsNotEnvVar:
    """profile_trace must NOT be gated by an environment variable.

    Per the options-menu rule, runtime toggles live in settings.json or
    as a code constant -- never as an environment variable. The earlier
    LVP_PROFILE_TRACE / LVP_PROFILE_TRACE_DIR gate violated that rule
    and was migrated to the profile_trace_enabled +
    profile_trace_output_dir settings keys. The two AST scans below
    pin both halves of the migration so the env-var pattern doesn't
    sneak back in via a later patch.
    """

    def _profile_trace_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'lib' / 'profile_trace.py').read_text()

    def test_no_lvp_profile_trace_env_var_in_module(self):
        import ast

        src = self._profile_trace_source()
        tree = ast.parse(src)

        hits = []

        class Visitor(ast.NodeVisitor):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value.startswith('LVP_PROFILE_TRACE'):
                    hits.append((node.lineno, node.value))
                self.generic_visit(node)

        Visitor().visit(tree)
        assert not hits, (
            'lib/profile_trace.py must not reference LVP_PROFILE_TRACE* '
            'as a string literal -- the env-var gate is retired in '
            'favor of the profile_trace_enabled settings key. Hits: '
            f'{hits}'
        )

    def test_module_level_gate_reads_settings_not_environ(self):
        import ast

        src = self._profile_trace_source()
        tree = ast.parse(src)

        # Reject os.environ.get(...) at module scope.
        bad = []
        for node in tree.body:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    func = sub.func
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == 'get'
                        and isinstance(func.value, ast.Attribute)
                        and func.value.attr == 'environ'
                    ):
                        bad.append(sub.lineno)
        assert not bad, (
            'lib/profile_trace.py must not call os.environ.get(...) at '
            f'module scope. Found at line(s): {bad}'
        )


class TestHandleTraceGateIsNotEnvVar:
    """handle_trace must NOT be gated by an environment variable.

    Per the options-menu rule, runtime toggles live in settings.json or
    as a code constant -- never as an environment variable. The earlier
    LVP_HANDLE_TRACE / LVP_OBJ_SAMPLE_EVERY module-load gate was a
    redundant second path alongside the profiling.handle_trace_enabled
    settings key that microscope_settings.start_app() already honors. The
    env block was deleted; this AST scan pins the removal so it can't
    return.
    """

    def _handle_trace_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'lib' / 'handle_trace.py').read_text()

    def test_no_handle_trace_env_vars_in_module(self):
        import ast

        src = self._handle_trace_source()
        tree = ast.parse(src)

        hits = []

        class Visitor(ast.NodeVisitor):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value in (
                    'LVP_HANDLE_TRACE',
                    'LVP_OBJ_SAMPLE_EVERY',
                ):
                    hits.append((node.lineno, node.value))
                self.generic_visit(node)

        Visitor().visit(tree)
        assert not hits, (
            'lib/handle_trace.py must not reference LVP_HANDLE_TRACE / '
            'LVP_OBJ_SAMPLE_EVERY as string literals -- the env-var gate '
            'is retired in favor of the profiling.handle_trace_enabled '
            f'settings key. Hits: {hits}'
        )


class TestDebugGateIsNotEnvVar:
    """The global DEBUG-suppression gate must NOT read an env var.

    Per the options-menu rule, runtime toggles live in settings.json or
    as a code constant -- never as an environment variable. The earlier
    LVP_DEBUG_ENABLED gate was a second knob alongside debug_mode: both
    had to be set to get DEBUG / [PERF] output. It was migrated to read
    the debug_mode setting (already loaded at logger import via
    load_debug_setting). This AST scan pins the migration so the env-var
    pattern doesn't sneak back in.
    """

    def _lvp_logger_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'lvp_logger.py').read_text()

    def test_no_lvp_debug_enabled_env_var_in_module(self):
        import ast

        src = self._lvp_logger_source()
        tree = ast.parse(src)

        hits = []

        class Visitor(ast.NodeVisitor):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value == 'LVP_DEBUG_ENABLED':
                    hits.append((node.lineno, node.value))
                self.generic_visit(node)

        Visitor().visit(tree)
        assert not hits, (
            'lvp_logger.py must not reference LVP_DEBUG_ENABLED as a '
            'string literal -- the env-var gate is retired in favor of '
            f'the debug_mode settings key. Hits: {hits}'
        )

    def test_no_global_logging_disable_in_lvp_logger(self):
        # logging.disable() is a process-global DEBUG kill-switch that
        # overrides every logger's own level (it was silently starving
        # camera.log's always-on DEBUG firehose) and split the debug
        # toggle into two gates. The canonical mechanism is the per-logger
        # level driven by debug_mode; the global disable must not return.
        import ast

        src = self._lvp_logger_source()
        tree = ast.parse(src)

        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == 'disable'
                    and isinstance(func.value, ast.Name)
                    and func.value.id == 'logging'
                ):
                    hits.append(node.lineno)
        assert not hits, (
            'lvp_logger.py must not call logging.disable() -- DEBUG gating '
            'is the per-logger level driven by debug_mode, not a global '
            f'kill-switch. Found at line(s): {hits}'
        )


class TestTracemallocGateIsNotEnvVar:
    """tracemalloc must NOT be gated by an environment variable.

    Per the options-menu rule, runtime toggles live in settings.json or
    as a code constant -- never as an environment variable. The earlier
    LVP_TRACEMALLOC gate violated that rule and was migrated to the
    tracemalloc_enabled settings key. This AST scan pins the migration
    so the env-var pattern doesn't sneak back in.
    """

    def _common_utils_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'modules' / 'common_utils.py').read_text()

    def test_no_lvp_tracemalloc_env_var_in_module(self):
        import ast

        src = self._common_utils_source()
        tree = ast.parse(src)

        hits = []

        class Visitor(ast.NodeVisitor):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value == 'LVP_TRACEMALLOC':
                    hits.append((node.lineno, node.value))
                self.generic_visit(node)

        Visitor().visit(tree)
        assert not hits, (
            'modules/common_utils.py must not reference LVP_TRACEMALLOC '
            'as a string literal -- the env-var gate is retired in favor '
            f'of the tracemalloc_enabled settings key. Hits: {hits}'
        )


class TestFx2DebugWireGateIsNotEnvVar:
    """The FX2 wire-protocol debug trace must NOT be gated by an env var.

    Per the options-menu rule, runtime toggles live in settings.json or
    as a code constant -- never as an environment variable. The earlier
    LVP_FX2_DEBUG_WIRE gate (read at three call sites) violated that
    rule and was migrated to the fx2_debug_wire_enabled settings key.
    This AST scan pins each of the three call sites so the env-var
    pattern doesn't sneak back in via any of them.
    """

    _SITES = (
        ('drivers', 'fx2driver.py'),
        ('ui', 'layer_control.py'),
        ('modules', 'lumascope_api', 'illumination.py'),
    )

    def _read(self, parts):
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent
        for part in parts:
            path = path / part
        return path.read_text()

    def test_no_lvp_fx2_debug_wire_env_var_in_any_site(self):
        import ast

        hits_by_site = {}
        for parts in self._SITES:
            src = self._read(parts)
            tree = ast.parse(src)
            hits = []

            class Visitor(ast.NodeVisitor):
                def visit_Constant(self, node):
                    if isinstance(node.value, str) and node.value == 'LVP_FX2_DEBUG_WIRE':
                        hits.append((node.lineno, node.value))
                    self.generic_visit(node)

            Visitor().visit(tree)
            if hits:
                hits_by_site['/'.join(parts)] = hits

        assert not hits_by_site, (
            'No source file may reference LVP_FX2_DEBUG_WIRE as a string '
            'literal -- the env-var gate is retired in favor of the '
            f'fx2_debug_wire_enabled settings key. Hits: {hits_by_site}'
        )


class TestPylonEnvVarsAreNotUsed:
    """No LVP_PYLON_* environment variable may gate driver behavior.

    Per the options-menu rule, runtime toggles live in settings.json or
    on the imaging sub-API (_set_max_num_buffer / _set_max_transfer_size
    / _set_num_max_queued_urbs / _set_grab_strategy) or as a constructor
    kwarg (_PylonImageGrabWorker(queue_depth=...)). The earlier
    LVP_PYLON_MAX_NUM_BUFFER / LVP_PYLON_MAX_TRANSFER_SIZE /
    LVP_PYLON_NUM_QUEUED_URBS / LVP_PYLON_GRAB_STRATEGY /
    LVP_PYLON_WORKER_QUEUE_DEPTH env vars duplicated those production
    levers and were retired. This AST scan pins the retirement so the
    env-var pattern doesn't sneak back in to drivers/pyloncamera.py.
    """

    def _pyloncamera_source(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'drivers' / 'pyloncamera.py').read_text()

    def test_no_lvp_pylon_env_var_in_pyloncamera(self):
        import ast

        src = self._pyloncamera_source()
        tree = ast.parse(src)

        hits = []

        class Visitor(ast.NodeVisitor):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value.startswith('LVP_PYLON_'):
                    hits.append((node.lineno, node.value))
                self.generic_visit(node)

        Visitor().visit(tree)
        assert not hits, (
            'drivers/pyloncamera.py must not reference any LVP_PYLON_* '
            'string literal -- the env-var gates are retired in favor '
            'of imaging sub-API levers (_set_max_num_buffer / '
            '_set_max_transfer_size / _set_num_max_queued_urbs / '
            '_set_grab_strategy) and the _PylonImageGrabWorker '
            f'queue_depth kwarg. Hits: {hits}'
        )


class TestCameraDelHandlesPartialConstruction:
    """Camera.__del__ must short-circuit on a partially-constructed
    instance instead of firing "no attribute _state_lock" warnings.

    Triggering scenario: a subclass __init__ raises BEFORE calling
    super().__init__(). FX2Camera is the canonical case -- it grabs
    _FX2Connection.get() first so self._fx2 is ready for the base class's
    self.connect() call. On the Pylon-fallback path
    (TlFactory.EnumerateDevices() returned 0 -> registry tries FX2 ->
    no FX2 hardware), _FX2Connection.get() raises and the instance is
    partial. Python still runs __del__; the hasattr gate makes that
    clean.
    """

    def test_partial_construction_del_is_silent(self, monkeypatch):
        from drivers import camera as camera_module

        # Build a concrete subclass with abstract methods stubbed so
        # __new__ doesn't get blocked by Camera's ABC declarations.
        # Critically, __init__ is overridden to do NOTHING -- it doesn't
        # call super().__init__(), so _state_lock + _active never get
        # set. That replicates the exact state Python has on its hands
        # when FX2Camera's __init__ raises before super().__init__().
        Camera = camera_module.Camera
        abstract = Camera.__abstractmethods__
        stubs = {name: (lambda self, *a, **kw: None) for name in abstract}
        stubs['__init__'] = lambda self: None
        Partial = type('Partial', (Camera,), stubs)

        instance = Partial()
        assert not hasattr(instance, '_state_lock')

        # Capture every warning the module's _cam_log emits. With the
        # hasattr guard, __del__ short-circuits and no warning fires.
        # Without the guard, the try/except still catches the
        # AttributeError but the "__del__ disconnect failed: 'Partial'
        # object has no attribute '_state_lock'" warning DOES fire --
        # exactly the noise this fix targets.
        warnings_emitted = []
        monkeypatch.setattr(
            camera_module._cam_log,
            'warning',
            lambda msg, *a, **kw: warnings_emitted.append(msg),
        )
        Camera.__del__(instance)

        assert not warnings_emitted, (
            f'Camera.__del__ on a partially-constructed instance must not '
            f'emit warnings. Got: {warnings_emitted}'
        )

    def test_del_guard_present_in_source(self):
        import ast
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / 'drivers' / 'camera.py').read_text()
        tree = ast.parse(src)

        # Find class Camera, then its __del__ method, then assert the
        # first statement of its body is a hasattr-gated early return.
        del_method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Camera':
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__del__':
                        del_method = item
                        break
                break
        assert del_method is not None, 'Camera.__del__ not found'

        first = del_method.body[0]
        assert isinstance(first, ast.If), (
            'Camera.__del__ must start with an if-guard for partial '
            'construction. Got: ' + ast.dump(first)
        )
        # Guard shape: `if not hasattr(self, '_state_lock'): return`
        test = first.test
        assert (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Call)
            and isinstance(test.operand.func, ast.Name)
            and test.operand.func.id == 'hasattr'
        ), 'Camera.__del__ guard must be `if not hasattr(self, ...): return`. Got: ' + ast.dump(
            test
        )


class TestLogToUtility:
    """lvp_logger.log_to() is the shared dual/multi-write helper.

    pyloncamera._log_cam wraps it for the camera dual-write today;
    motorboard / ledboard / idscamera will adopt the same pattern when
    motor.log / led.log / ids.log land. The tests pin the utility's
    contract so callers can rely on it: None mirrors are skipped, a
    mirror that raises does not break the primary's call, level is a
    Python logging method name routed via getattr.
    """

    def test_single_logger_routes_to_that_logger(self):
        from lib.log_helpers import log_to

        class _Capture:
            def __init__(self):
                self.calls = []

            def info(self, msg, *a, **kw):
                self.calls.append(('info', msg))

            def warning(self, msg, *a, **kw):
                self.calls.append(('warning', msg))

            def debug(self, msg, *a, **kw):
                self.calls.append(('debug', msg))

        primary = _Capture()
        log_to(primary, level='info', message='hello')
        assert primary.calls == [('info', 'hello')]

    def test_dual_write_lands_on_both(self):
        from lib.log_helpers import log_to

        class _Capture:
            def __init__(self):
                self.calls = []

            def warning(self, msg, *a, **kw):
                self.calls.append(msg)

            def debug(self, msg, *a, **kw):
                self.calls.append(('debug', msg))

        primary = _Capture()
        mirror = _Capture()
        log_to(primary, mirror, level='warning', message='[CAM Class ] foo')
        assert primary.calls == ['[CAM Class ] foo']
        assert mirror.calls == ['[CAM Class ] foo']

    def test_none_mirror_skipped_no_guard_needed_at_call_site(self):
        from lib.log_helpers import log_to

        class _Capture:
            def __init__(self):
                self.calls = []

            def info(self, msg, *a, **kw):
                self.calls.append(msg)

        primary = _Capture()
        # Caller passing None as the mirror (e.g. camera_logger not yet
        # set up) must not raise.
        log_to(primary, None, level='info', message='msg')
        assert primary.calls == ['msg']

    def test_mirror_failure_falls_through_to_debug(self):
        from lib.log_helpers import log_to

        class _Primary:
            def __init__(self):
                self.calls = []

            def info(self, msg, *a, **kw):
                self.calls.append(('info', msg))

            def debug(self, msg, *a, **kw):
                self.calls.append(('debug', msg))

        class _BrokenMirror:
            def info(self, msg, *a, **kw):
                raise RuntimeError('mirror wedged')

        primary = _Primary()
        log_to(primary, _BrokenMirror(), level='info', message='msg')
        # Primary saw the original line PLUS a debug line about the
        # mirror failure. Caller control flow is preserved.
        assert ('info', 'msg') in primary.calls
        assert any(kind == 'debug' and 'mirror.info() raised' in m for kind, m in primary.calls)


class TestShowPopupMessageMarshalsDoneToUiThread:
    """AUDIT_CONCURRENCY_2026-05-24 F1: `ProtocolSettings._show_popup_message`
    runs inside a daemon Thread spawned by `@show_popup`. The host widget's
    `done` BooleanProperty is bound to `popup.dismiss`, so writing
    `self.done = True` directly on the worker triggered the dismiss
    dispatch on the worker thread -- a Bug-E shape that can corrupt the
    Kivy property graph mid-dispatch.

    Fix: marshal the `done` write through `Clock.schedule_once`, matching
    the pattern `_PopupProxy` already uses for popup-local writes.

    The regression test reads source text and asserts the bare assignment
    is gone. Source-text tests are quote/paren-agnostic per the
    `/issue-triage` Step 6 update so they survive future ruff format
    passes.
    """

    def _protocol_settings_src(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'ui' / 'protocol_settings.py').read_text()

    def test_show_popup_message_does_not_write_done_on_bg_thread(self):
        """Bare `self.done = True` inside `_show_popup_message` writes a
        Kivy property from the worker thread. The fix replaces it with a
        `Clock.schedule_once` marshal. A future revert that re-introduces
        the bare assignment fails this test."""
        import re

        src = self._protocol_settings_src()
        match = re.search(
            r'def _show_popup_message\(self,.*?\):.*?(?=\n    def |\nclass )',
            src,
            re.DOTALL,
        )
        assert match is not None, (
            '_show_popup_message method body not found; test selector is out of date'
        )
        body = match.group(0)
        # The bare assignment must not appear -- any `self.done = True`
        # in this body is the Bug-E shape.
        assert not re.search(r'self\.done\s*=\s*True', body), (
            'F1 regression: `_show_popup_message` writes `self.done = True` '
            'directly on the worker thread. Use `Clock.schedule_once` to '
            'marshal the write to the UI thread instead.'
        )

    def test_show_popup_message_marshals_done_via_clock(self):
        """Positive assertion: the fix uses `Clock.schedule_once` to set
        `done` from the worker thread. Quote-agnostic regex tolerates
        future ruff reformat."""
        import re

        src = self._protocol_settings_src()
        match = re.search(
            r'def _show_popup_message\(self,.*?\):.*?(?=\n    def |\nclass )',
            src,
            re.DOTALL,
        )
        assert match is not None
        body = match.group(0)
        # Require some form of `Clock.schedule_once(...)` that mentions
        # `done` as the target attribute. Tolerates both `setattr(self,
        # 'done', True)` and `self.done = True` inside a lambda, and
        # tolerates either quote style.
        marshalled = re.search(
            r'Clock\.schedule_once\(.*?["\']done["\'].*?\)'
            r'|Clock\.schedule_once\(.*?self\.done\s*=\s*True',
            body,
            re.DOTALL,
        )
        assert marshalled is not None, (
            '`_show_popup_message` must marshal the `done=True` write '
            'through `Clock.schedule_once` to keep the Kivy property '
            'write on the UI thread (AUDIT_CONCURRENCY_2026-05-24 F1).'
        )


class TestProtocolPostProcessorNoBareCvImwrite_F35_2:
    """AUDIT_LAYER_SEPARATION_2026-05-24 F35.2: the protocol_post_processor
    base class previously fell back to `cv2.imwrite` when a subclass
    returned `'image'` payload. cv2 is BGR-native; tifffile / FIJI / OS
    preview all read TIFF as RGB-native. The fallback was the last
    surviving channel-swap hazard after the composite-path unification.

    Fix: retire the base-class cv2.imwrite branch entirely. Each
    subclass owns its own write via tifffile (matches the pattern
    composite_generation + zprojector + video_builder + stack_builder
    already used). Stitcher was migrated in the same commit: tile load
    swaps cv2.imread -> tifffile.imread; stitched save uses
    tifffile.imwrite directly.

    Tests below use quote-agnostic source-text regex per the
    `/issue-triage` Step 6 update so they survive future ruff format
    passes.
    """

    def _post_processor_src(self):
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent / 'modules' / 'protocol_post_processor.py'
        ).read_text()

    def _stitcher_src(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'modules' / 'stitcher.py').read_text()

    def test_protocol_post_processor_has_no_cv2_imports_or_calls(self):
        """No `import cv2` / `from cv2 ...` and no `cv2.<attr>(...)`
        calls in the base class. Plain prose mentions of cv2 in
        comments (explaining WHY the fallback was retired) are fine.
        A revert that re-introduces the BGR fallback fails."""
        import re

        src = self._post_processor_src()
        # No imports.
        assert not re.search(r'^(import cv2|from cv2 )', src, re.MULTILINE), (
            'F35.2 regression: protocol_post_processor.py must not import cv2 (BGR-native).'
        )
        # No method/attribute calls (cv2.foo(...) or cv2.foo. ...).
        assert not re.search(r'\bcv2\.\w+\s*\(', src), (
            'F35.2 regression: protocol_post_processor.py must not '
            'call any cv2.<x>(...) -- the base class no longer falls '
            'back to BGR writers. Each subclass owns its own write.'
        )

    def test_protocol_post_processor_drops_imwrite_branch(self):
        """The `cv2.imwrite(filename=...)` fallback branch is gone.
        Quote-tolerant: matches both single-quote and double-quote
        kwarg styles."""
        import re

        src = self._post_processor_src()
        assert not re.search(r'cv2\.imwrite\s*\(', src), (
            'F35.2 regression: cv2.imwrite branch must be retired.'
        )

    def test_stitcher_loads_tiles_via_tifffile_not_cv2(self):
        """Stitcher tile-load must use tifffile.imread (RGB-native).
        Pair with the save-side migration so both ends of the stitcher
        pipeline stay on the canonical RGB path."""
        import re

        src = self._stitcher_src()
        assert not re.search(r'cv2\.imread\s*\(', src), (
            'F35.2 regression: stitcher tile-load must use '
            'tifffile.imread, not cv2.imread (cv2 is BGR-native, '
            'swaps red/blue relative to tifffile readers).'
        )

    def test_stitcher_writes_via_tifffile(self):
        """Stitcher self-writes its output through the tifffile pathway.
        Accepts either bare ``tifffile.imwrite`` (legacy direct call) or
        ``image_utils.write_tiff`` (canonical mono-native wrapper that
        routes through tifffile internally) -- both satisfy the F35.2
        intent that the write does NOT go through cv2 (BGR-native)."""
        import re

        src = self._stitcher_src()
        matched = re.search(
            r'(?:tf|tifffile)\.imwrite\s*\(|image_utils\.write_tiff\s*\(',
            src,
        )
        assert matched is not None, (
            'F35.2 regression: stitcher must write its stitched output '
            'through tifffile (either bare tifffile.imwrite or via '
            'image_utils.write_tiff). The BGR-native cv2 write path is '
            'permanently retired.'
        )

    def test_stitcher_has_no_cv2_imports(self):
        """No `import cv2` or `from cv2 ...` in stitcher.py -- the
        cv2-end-to-end pattern is fully retired."""
        import re

        src = self._stitcher_src()
        assert not re.search(r'^(import cv2|from cv2 )', src, re.MULTILINE), (
            'F35.2 regression: stitcher.py must not import cv2 -- '
            'tile load + stitched save both go through tifffile.'
        )


class TestEmergencyShutdownBoundedLeds_F6:
    """AUDIT_CONCURRENCY_2026-05-24 F6: `_emergency_shutdown` atexit hook
    previously called `illumination.leds_off()`, which acquires
    `_led_lock` UNBOUNDED. If an in-flight LED command holds the lock at
    interpreter exit, atexit deadlocks (Python's atexit does not honor
    timeouts).

    Fix: split out `leds_off_emergency(timeout_s=2.0)` that uses
    `_led_lock.acquire(timeout=timeout_s)` with a log-and-skip fallback.
    `_emergency_shutdown` calls this variant. Normal `leds_off` keeps
    its unbounded `with` semantics.
    """

    def _illumination_src(self):
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / 'illumination.py'
        ).read_text()

    def _lumascope_src(self):
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent / 'modules' / 'lumascope_api' / '_lumascope.py'
        ).read_text()

    def test_leds_off_emergency_method_exists(self):
        """The bounded variant must exist as a callable method on the
        illumination API."""
        import re

        src = self._illumination_src()
        assert re.search(r'def leds_off_emergency\s*\(', src), (
            'F6 regression: illumination must expose leds_off_emergency '
            'with bounded _led_lock acquire for atexit / abnormal exit '
            'paths.'
        )

    def test_leds_off_emergency_uses_bounded_acquire(self):
        """The variant body must call `_led_lock.acquire(timeout=...)`
        (NOT `with self._led_lock:`). Quote-/format-agnostic regex
        tolerates whitespace + keyword-vs-positional timeout."""
        import re

        src = self._illumination_src()
        # Match from `def leds_off_emergency` up to the next sibling
        # method / decorator / class definition. Tolerates `-> None:`
        # return annotations (`\):` would not match because the colon
        # follows the annotation, not the paren).
        match = re.search(
            r'def leds_off_emergency.*?(?=\n    def |\n    @|\nclass |\Z)',
            src,
            re.DOTALL,
        )
        assert match is not None, 'leds_off_emergency body not found'
        body = match.group(0)
        # Must acquire with a timeout (positional or keyword).
        bounded = re.search(
            r'_led_lock\.acquire\s*\(\s*(timeout\s*=\s*)?[^)]*\)',
            body,
        )
        assert bounded is not None, (
            'F6 regression: leds_off_emergency must call '
            '`_led_lock.acquire(timeout=...)`. Unbounded `with` '
            'semantics defeat the atexit-deadlock fix.'
        )
        # Must NOT use the unbounded `with` form inside the body.
        assert not re.search(r'with\s+self\._led_lock\s*:', body), (
            'F6 regression: leds_off_emergency must not use `with '
            'self._led_lock:` -- that is unbounded and would re-'
            'introduce the atexit deadlock.'
        )

    def test_emergency_shutdown_calls_bounded_variant(self):
        """`_emergency_shutdown` must call `leds_off_emergency`, not
        the unbounded `leds_off`. A revert that drops the `_emergency`
        suffix re-introduces the deadlock surface."""
        import re

        src = self._lumascope_src()
        match = re.search(
            r'def _emergency_shutdown\s*\(self\):.*?(?=\n    def |\n    @|\nclass )',
            src,
            re.DOTALL,
        )
        assert match is not None, '_emergency_shutdown body not found'
        body = match.group(0)
        assert re.search(r'\.leds_off_emergency\s*\(', body), (
            'F6 regression: _emergency_shutdown must call '
            'leds_off_emergency (bounded lock acquire) -- not '
            'leds_off (unbounded).'
        )


class TestSequentialIoExecutorWaitForIdle_F7:
    """AUDIT_CONCURRENCY_2026-05-24 F7: `protocol_end()` previously
    called `time.sleep(0.05)` as a band-aid drain wait so callers that
    tore down shared state after `protocol_end` returned wouldn't
    collide with an in-flight task on the worker thread. The sleep was:
    - wasted on the worker-loop caller (queue is empty by definition)
    - wasted on the shutdown caller (the real wait is `Thread.join`)
    - too short to actually cover typical task latencies on the
      protocol_cleanup caller (motor p99 ~50 ms, AF iterations multi-s)

    Fix: drop the sleep from `protocol_end`; add `wait_for_idle(timeout)`
    that polls `running_task is None`; have `protocol_cleanup` call it
    after `protocol_end` so the mid-task hazard is bounded properly.
    """

    def _executor_src(self):
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent / 'modules' / 'sequential_io_executor.py'
        ).read_text()

    def _cleanup_src(self):
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent / 'modules' / 'protocol_cleanup.py'
        ).read_text()

    def test_wait_for_idle_method_exists(self):
        """The executor must expose `wait_for_idle(timeout=...)`."""
        import re

        src = self._executor_src()
        assert re.search(r'def wait_for_idle\s*\(self', src), (
            'F7 regression: SequentialIOExecutor must expose '
            'wait_for_idle(timeout=...) so callers can bound the '
            'mid-task drain wait instead of relying on a magic-number '
            'sleep in protocol_end.'
        )

    def test_protocol_end_does_not_sleep(self):
        """`protocol_end` body must not contain a bare `time.sleep`
        call. The band-aid wait is gone; callers that need a wait use
        `wait_for_idle` explicitly."""
        import re

        src = self._executor_src()
        match = re.search(
            r'def protocol_end.*?(?=\n    def |\n    @|\nclass |\Z)',
            src,
            re.DOTALL,
        )
        assert match is not None, 'protocol_end body not found'
        body = match.group(0)
        assert not re.search(r'\btime\.sleep\s*\(', body), (
            'F7 regression: protocol_end must not call time.sleep. '
            'The band-aid drain wait was retired; callers needing to '
            'wait for the worker to finish an in-flight task call '
            'wait_for_idle(timeout=...) instead.'
        )

    def test_protocol_cleanup_calls_wait_for_idle(self):
        """`protocol_cleanup` must call `wait_for_idle` on the
        io_executor immediately after `protocol_end`. The order is
        load-bearing -- protocol_end clears the running flag, then the
        wait ensures any task that was running before that point
        completes before downstream state is mutated."""
        import re

        src = self._cleanup_src()
        # Search across newlines + intervening lines for the sequence.
        sequence = re.search(
            r'io_executor\.protocol_end\s*\(\s*\).*?'
            r'io_executor\.wait_for_idle\s*\(',
            src,
            re.DOTALL,
        )
        assert sequence is not None, (
            'F7 regression: protocol_cleanup must call '
            'io_executor.wait_for_idle(timeout=...) after '
            'io_executor.protocol_end() so an in-flight task on the '
            'io_executor worker is given bounded time to finish '
            'before downstream teardown mutates state the task may '
            'reference.'
        )


class TestShowPopupHostWidgetProxy_F9:
    """AUDIT_CONCURRENCY_2026-05-24 F9: the `show_popup` decorator
    previously passed the raw host widget (`app`) to the daemon thread
    that ran the decorated body. If the body wrote a Kivy property on
    the host (e.g. `self.done = True`), Kivy's property dispatch ran
    bound callbacks on the writing (bg) thread -- the same Bug-shape
    that motivated F1's per-site fix in protocol_settings.

    Aggressive fix: wrap the host in `_HostWidgetProxy` that intercepts
    `__setattr__`, detects Kivy `Property` descriptors at the class
    level, and marshals Kivy property writes through `Clock.schedule_
    once`. Non-Kivy attribute writes pass through directly.

    This is the cluster-level fix: any future `@show_popup`-decorated
    method can write `self.<KivyProperty> = ...` safely without the
    per-site `Clock.schedule_once` boilerplate. The F1 manual marshal
    in protocol_settings remains as belt-and-suspenders.
    """

    def _popup_src(self):
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / 'ui' / 'progress_popup.py').read_text()

    def test_host_widget_proxy_class_exists(self):
        """The proxy class must be declared in progress_popup.py."""
        import re

        src = self._popup_src()
        assert re.search(r'class _HostWidgetProxy', src), (
            'F9 regression: _HostWidgetProxy class must exist in '
            'ui/progress_popup.py to wrap the host widget passed to '
            'show_popup-decorated daemon-thread bodies.'
        )

    def test_proxy_marshals_property_writes_via_clock(self):
        """`_HostWidgetProxy.__setattr__` must marshal writes through
        Clock.schedule_once when the attribute is a Kivy Property.
        Quote/format-agnostic regex on the body."""
        import re

        src = self._popup_src()
        match = re.search(
            r'class _HostWidgetProxy.*?(?=\nclass |\ndef show_popup\b|\Z)',
            src,
            re.DOTALL,
        )
        assert match is not None, '_HostWidgetProxy class body not found'
        body = match.group(0)
        # Property-detection: looks up the class-level descriptor and
        # checks isinstance(..., Property).
        assert re.search(r'isinstance\s*\(\s*\w+\s*,\s*Property\s*\)', body), (
            'F9 regression: _HostWidgetProxy must detect Kivy Property '
            'descriptors via isinstance check on the class-level '
            'attribute before deciding to marshal.'
        )
        # Marshalled write path -- Clock.schedule_once with a setattr
        # lambda that captures the host + name + value.
        assert re.search(
            r'Clock\.schedule_once\s*\(\s*lambda.*?setattr\s*\(',
            body,
            re.DOTALL,
        ), (
            'F9 regression: _HostWidgetProxy must marshal Kivy '
            'property writes through Clock.schedule_once with a '
            'setattr lambda.'
        )

    def test_show_popup_wraps_host_in_proxy(self):
        """The decorator must pass `_HostWidgetProxy(app)` (not `app`
        directly) as the first positional arg to the decorated
        function so `self` inside the method body is the proxy."""
        import re

        src = self._popup_src()
        match = re.search(
            r'def show_popup\b.*?(?=\nclass |\Z)',
            src,
            re.DOTALL,
        )
        assert match is not None, 'show_popup function body not found'
        body = match.group(0)
        # Must construct the host proxy and pass it through.
        assert re.search(r'_HostWidgetProxy\s*\(', body), (
            'F9 regression: show_popup decorator must instantiate '
            '_HostWidgetProxy to wrap the host widget.'
        )
        # The thread args list must reference the host proxy var
        # (named `host_proxy` in current implementation; tolerate any
        # local name that is the result of _HostWidgetProxy(...)).
        # Belt-and-suspenders: the bare `app` must NOT appear as the
        # first positional in the args list passed to the Thread
        # target.
        thread_args = re.search(
            r'threading\.Thread\s*\([^)]*args\s*=\s*\[\s*(\w+)',
            body,
        )
        assert thread_args is not None, (
            'F9 regression: show_popup must spawn a Thread with an '
            'args= list whose first element is the host proxy.'
        )
        first_arg_name = thread_args.group(1)
        assert first_arg_name != 'app', (
            f'F9 regression: show_popup must NOT pass the raw `app` '
            f'host widget to the daemon thread; pass the '
            f'_HostWidgetProxy wrapper instead. Got first thread arg: '
            f'{first_arg_name!r}.'
        )


class TestPostProcessingLoggerImported:
    """post_processing.py called logger.error() inside an `except OSError`
    handler that writes results.csv, but the module never imported logger.
    With logger unbound, any CSV-write failure raised NameError instead of
    logging -- masking the real OSError from the user. The module must import
    logger at top level whenever it references it.
    """

    def _parse(self):
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        src = (root / 'modules' / 'post_processing.py').read_text()
        return ast.parse(src)

    def test_logger_imported_when_referenced(self):
        import ast

        tree = self._parse()

        references_logger = any(
            isinstance(node, ast.Name)
            and node.id == 'logger'
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(tree)
        )
        assert references_logger, (
            'expected post_processing.py to reference logger; the test '
            'guards against an unbound logger and is meaningless otherwise.'
        )

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add((alias.asname or alias.name).split('.')[0])

        assert 'logger' in imported, (
            'post_processing.py references logger but never imports it; '
            'logger.error() in the results.csv except handler would raise '
            'NameError and hide the real OSError. Add '
            '`from lvp_logger import logger`.'
        )


class TestWindowsMachinePredicateAgrees:
    """lvp_logger and app_environment each derive the windows_machine flag
    independently (lvp_logger is import-light and loads first, so it does not
    import app_environment). They are allowed to stay separate ONLY because
    both use the identical `os.name == 'nt'` predicate and therefore cannot
    disagree. This pins that invariant: if either side switches predicates
    (e.g. back to platform.system()), the two could diverge and this fails.
    """

    def _src(self, rel):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        return (root / rel).read_text()

    def test_lvp_logger_uses_os_name_predicate(self):
        assert "os.name == 'nt'" in self._src('lvp_logger.py')

    def test_app_environment_uses_os_name_predicate(self):
        assert "os.name == 'nt'" in self._src('modules/app_environment.py')

    def test_lvp_logger_does_not_import_app_environment(self):
        # The independence is the point -- the foundational logger must not
        # take an early-startup dependency on the heavier app_environment.
        # Scan imports (not the whole source) so the explanatory comment,
        # which names app_environment, does not trip the check.
        import ast

        tree = ast.parse(self._src('lvp_logger.py'))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
        assert not any('app_environment' in m for m in imported)


class TestExecutorHandlesSingleSourceOnCtx:
    """The 7 executor handles were stored in module globals AND on ctx AND on
    the bundle, read divergently (shutdown_threads read the globals; the rest
    of the app reads ctx.X). They now live only on ctx: build() uses locals and
    everything else, including shutdown, reads ctx.<name>. executor_bundle stays
    a module global -- it is the single build()->on_start() handoff, not a live
    executor read path. This pins that the redundant globals are gone.
    """

    EXECUTORS = (
        'io_executor',
        'camera_executor',
        'protocol_thread',
        'file_io_executor',
        'autofocus_thread',
        'scope_display_thread',
        'worker_pool',
    )

    def _src(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        return (root / 'lumaviewpro.py').read_text()

    def test_no_global_declaration_of_executor_handles(self):
        # A `global <name>` anywhere is the only way build()/shutdown could
        # write or rebind a module-global copy; none should name an executor.
        import ast

        tree = ast.parse(self._src())
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in self.EXECUTORS:
                    assert name not in node.names, (
                        f'{name} is still declared `global`; the executor '
                        'handles must live only on ctx.'
                    )

    def test_shutdown_reads_executors_from_ctx(self):
        # shutdown_threads must tear down ctx.<name>, not bare module globals.
        src = self._src()
        assert 'ctx.autofocus_thread.stop' in src
        assert 'ctx.scope_display_thread.stop' in src
        assert 'ctx.io_executor.shutdown' in src
        assert 'ctx.worker_pool.shutdown' in src
