# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Pytest configuration for LumaViewPro tests."""
import sys
import os
from unittest.mock import MagicMock

import pytest

# Add project root to path so 'from ledboard import LEDBoard' works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared pytest options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    """Register custom command-line options for hardware tests."""
    try:
        parser.addoption("--run-hardware", action="store_true",
                         default=False, help="Run hardware serial tests")
    except Exception:
        pass  # option already registered by plugin or another conftest


# ---------------------------------------------------------------------------
# Shared mock dependency installer
# ---------------------------------------------------------------------------

def install_mock_deps():
    """Install mock modules for heavy dependencies (lvp_logger, kivy, etc.).

    Safe to call multiple times -- uses setdefault so existing entries
    (from test files that mock at module level) are not overwritten.

    NOTE: 8 test files currently duplicate this logic at module level
    instead of using this function. New tests should use this function
    via the _mock_heavy_deps fixture instead of duplicating.
    See: test_integration.py, test_protocol_execution.py, test_scope_api.py,
    test_simulators.py, test_regression_p2.py, test_serial_safety.py,
    test_null_motorboard.py, test_notification_center.py
    """
    mock_logger = MagicMock()
    mock_lvp_logger = MagicMock()
    mock_lvp_logger.logger = mock_logger
    mock_lvp_logger.version = "test"
    mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
    mock_lvp_logger.unpause_thread = MagicMock()
    mock_lvp_logger.pause_thread = MagicMock()

    deps = {
        'userpaths': MagicMock(),
        'lvp_logger': mock_lvp_logger,
        'requests': MagicMock(),
        'requests.structures': MagicMock(),
        'psutil': MagicMock(),
        'kivy': MagicMock(),
        'kivy.clock': MagicMock(),
        'pypylon': MagicMock(),
        'pypylon.pylon': MagicMock(),
        'pypylon.genicam': MagicMock(),
        'ids_peak': MagicMock(),
        'ids_peak.ids_peak': MagicMock(),
        'ids_peak.ids_peak_ipl_extension': MagicMock(),
        'ids_peak_ipl': MagicMock(),
    }
    for name, mock_mod in deps.items():
        sys.modules.setdefault(name, mock_mod)


# ---------------------------------------------------------------------------
# Shared simulator fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_scope():
    """Create a Lumascope with simulated hardware in fast timing mode.

    Requires that heavy deps (lvp_logger, kivy, etc.) are already mocked —
    either by the test file's module-level sys.modules.setdefault calls,
    or by calling install_mock_deps() first.

    Yields the scope and disconnects on teardown.
    """
    install_mock_deps()
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
