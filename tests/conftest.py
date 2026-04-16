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
        'userpaths': MagicMock(),
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


def pytest_collection_modifyitems(config, items):
    """Skip hardware-marked tests unless the matching opt-in flag is set."""
    gates = [
        ("ids_hardware",   "--run-ids-hardware"),
        ("pylon_hardware", "--run-pylon-hardware"),
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
