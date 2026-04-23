# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Pytest configuration for LumaViewPro tests.

Conftest is loaded by pytest before any test module is collected, so the
mock installation here happens before any test file's imports. Test files
can therefore import driver modules at module level without each one
re-installing its own MagicMock deps.

Hardware-test opt-in flags
--------------------------
    --run-hardware        firmware/serial board hardware (legacy flag)
    --run-ids-hardware    real IDS Peak SDK + connected camera
    --run-pylon-hardware  real Pylon SDK + connected camera

When a hardware flag is set, the corresponding SDK is NOT mocked so the
real module loads. Hardware tests are gated by markers (`ids_hardware`,
`pylon_hardware`) — see `pytest_collection_modifyitems` below.
"""
import os
import sys
from unittest.mock import MagicMock

import pytest

# Keep Kivy from writing anything to ~/.kivy/logs/ during tests. App code
# sets these in lumaviewpro.py + lvp_logger.py, but pytest may import Kivy
# before either runs, so set them here as well.
os.environ.setdefault("KIVY_NO_CONSOLELOG", "1")
os.environ.setdefault("KIVY_NO_FILELOG", "1")

# SerialBoard fires a per-command latency fingerprint at connect() (see
# drivers/serial_latency.py). That adds ~0.5 s per connect which is
# meaningful for tests that measure connect time or exercise many
# reconnects. Opt out by default; tests that want to cover the bench
# path unset this explicitly.
os.environ.setdefault("LVP_SKIP_CONNECT_BENCH", "1")


# ---------------------------------------------------------------------------
# Path setup — make `from drivers.x import Y` work from tests/
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Hardware-flag detection (must run before mock install)
# ---------------------------------------------------------------------------
# pytest_addoption hasn't been called yet at conftest import time, so we
# sniff sys.argv directly. Tolerant of `--flag` and `--flag=1` forms.
def _flag_in_argv(name):
    return any(a == name or a.startswith(f'{name}=') for a in sys.argv)


_HARDWARE_FLAG_MOCKS = {
    '--run-ids-hardware': [
        'ids_peak',
        'ids_peak.ids_peak',
        'ids_peak.ids_peak_ipl_extension',
        'ids_peak_ipl',
    ],
    '--run-pylon-hardware': [
        'pypylon',
        'pypylon.pylon',
        'pypylon.genicam',
    ],
}
_skip_mocks = set()
for _flag, _mods in _HARDWARE_FLAG_MOCKS.items():
    if _flag_in_argv(_flag):
        _skip_mocks.update(_mods)


# ---------------------------------------------------------------------------
# Centralized mock installation
# ---------------------------------------------------------------------------
# Test files used to duplicate this block at module level. Now they don't
# have to — conftest installs the union before any test is collected.
# Idempotent (uses setdefault) so files that still call install_mock_deps()
# are no-ops.

def install_mock_deps():
    """Install MagicMock entries for heavy deps not present on dev machines.

    Idempotent. Skips SDK mocks when the corresponding --run-*-hardware
    flag is set, so the real SDK can load.
    """
    mock_logger = MagicMock()
    mock_lvp_logger = MagicMock()
    mock_lvp_logger.logger = mock_logger
    mock_lvp_logger.version = "test"
    mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
    mock_lvp_logger.unpause_thread = MagicMock()
    mock_lvp_logger.pause_thread = MagicMock()

    deps = {
        # General heavy deps
        'platformdirs': MagicMock(),
        'lvp_logger': mock_lvp_logger,
        'requests': MagicMock(),
        'requests.structures': MagicMock(),
        'psutil': MagicMock(),
        'kivy': MagicMock(),
        'kivy.clock': MagicMock(),
        'kivy.base': MagicMock(),
        # FX2 / libusb (no hardware-test gate yet — always mocked)
        'usb': MagicMock(),
        'usb.core': MagicMock(),
        'usb.util': MagicMock(),
        'usb1': MagicMock(),
        # Camera SDKs — skipped when their --run-*-hardware flag is set
        'pypylon': MagicMock(),
        'pypylon.pylon': MagicMock(),
        'pypylon.genicam': MagicMock(),
        'ids_peak': MagicMock(),
        'ids_peak.ids_peak': MagicMock(),
        'ids_peak.ids_peak_ipl_extension': MagicMock(),
        'ids_peak_ipl': MagicMock(),
    }
    for name, mock_mod in deps.items():
        if name in _skip_mocks:
            continue
        sys.modules.setdefault(name, mock_mod)


# Run at conftest import time — before any test file is collected.
install_mock_deps()


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    """Register hardware-test opt-in flags."""
    def _safe(*args, **kwargs):
        try:
            parser.addoption(*args, **kwargs)
        except (ValueError, Exception):
            pass  # already registered by another plugin/conftest

    _safe("--run-hardware", action="store_true", default=False,
          help="Run hardware serial tests (firmware boards via SerialBoard)")
    _safe("--run-ids-hardware", action="store_true", default=False,
          help="Run IDS Peak hardware tests (real SDK + connected camera)")
    _safe("--run-pylon-hardware", action="store_true", default=False,
          help="Run Pylon hardware tests (real SDK + connected camera)")
    _safe("--run-timing-sensitive", action="store_true", default=False,
          help="Run wall-clock timing-sensitive tests (can be flaky under load)")


def pytest_configure(config):
    """Register custom markers used by hardware tests."""
    config.addinivalue_line(
        "markers",
        "ids_hardware: requires real IDS Peak SDK + connected camera "
        "(only runs with --run-ids-hardware)",
    )
    config.addinivalue_line(
        "markers",
        "pylon_hardware: requires real Pylon SDK + connected camera "
        "(only runs with --run-pylon-hardware)",
    )
    config.addinivalue_line(
        "markers",
        "timing_sensitive: measures wall-clock timing and can be flaky "
        "under CI/load (only runs with --run-timing-sensitive)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip hardware-marked tests unless the matching opt-in flag is set."""
    gates = [
        ("ids_hardware",    "--run-ids-hardware"),
        ("pylon_hardware",  "--run-pylon-hardware"),
        ("timing_sensitive", "--run-timing-sensitive"),
    ]
    for marker, flag in gates:
        if config.getoption(flag, default=False):
            continue
        skip = pytest.mark.skip(reason=f"needs {flag}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_scope():
    """Lumascope with simulated hardware in fast timing mode."""
    from modules.lumascope_api import Lumascope
    s = Lumascope(simulate=True)
    s.led.set_timing_mode('fast')
    s.motion.set_timing_mode('fast')
    s.camera.set_timing_mode('fast')
    s.camera.load_cycle_images()
    s.camera.start_grabbing()
    yield s
    s.camera.stop_grabbing()
    s.disconnect()


@pytest.fixture
def board_with_fake_transport(monkeypatch):
    """Factory — real SerialBoard wired to an in-memory FakeTransport.

    Exercises the full production raw-REPL path (SerialBoard._lock,
    enter/exit_raw_repl state machine, MpremoteSession.write_file's
    atomic-`.tmp` + SHA-256 + `.bak` sequence) against a fake in-memory
    filesystem — no pyserial, no device. Swaps in a base MpremoteSession
    (not _ManagedSession) so exit uses the fake's transport.exit_raw_repl
    instead of the pyserial-level _send_exit_sequence.

    Returns a factory: `make(board_type, initial_files=None) -> (board, fake)`.
    """
    def _factory(board_type, initial_files=None):
        from drivers.firmware_updater import BOARD_CONFIGS
        from drivers.serialboard import SerialBoard
        from drivers.mpremote_transport import MpremoteSession
        from tests.fake_transport import FakeTransport

        cfg = BOARD_CONFIGS[board_type]
        board = SerialBoard(
            vid=cfg.vid,
            pid=cfg.pid,
            label=cfg.board_type.name,
            port='/dev/fake',
        )
        fake = FakeTransport(initial_files=initial_files or {})

        # SerialBoard.enter_raw_repl calls _open_serial (if driver None)
        # then _close_driver; exit_raw_repl reopens via _open_serial.
        # Stub both to toggle a MagicMock driver so the non-None check
        # passes without touching pyserial.
        def _fake_open():
            board.driver = MagicMock()
        def _fake_close():
            board.driver = None
        monkeypatch.setattr(board, '_open_serial', _fake_open)
        monkeypatch.setattr(board, '_close_driver', _fake_close)

        # Redirect the mpremote session factory to wrap our FakeTransport.
        def _make_session(device_path, baudrate=115200):
            return MpremoteSession(fake)
        monkeypatch.setattr(
            'drivers.serialboard._create_mpremote_session', _make_session
        )

        # Post-exit firmware verification queries the live driver with
        # INFO — that's a pyserial-level operation outside mpremote's
        # abstraction (adapter docstring explicitly scopes it out).
        # Default to "firmware responding"; individual tests can flip
        # to None to exercise the failure path.
        monkeypatch.setattr(
            board, 'verify_firmware_running',
            lambda timeout=10: 'ok'
        )
        return board, fake

    return _factory
