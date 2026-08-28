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
`pylon_hardware`) -- see `pytest_collection_modifyitems` below.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest

# Keep Kivy from writing anything to ~/.kivy/logs/ during tests. App code
# sets these in lumaviewpro.py + lvp_logger.py, but pytest may import Kivy
# before either runs, so set them here as well.
os.environ.setdefault('KIVY_NO_CONSOLELOG', '1')
os.environ.setdefault('KIVY_NO_FILELOG', '1')

# ---------------------------------------------------------------------------
# Path setup -- make `from drivers.x import Y` work from tests/
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Typed pypylon stand-in (real handler bases + exception types). Needs the
# repo-root path insert above.
from tests import pypylon_stub as _pypylon_stub


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
    '--run-fx2-hardware': [
        'usb',
        'usb.core',
        'usb.util',
        'usb1',
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
# have to -- conftest installs the union before any test is collected.
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
    mock_lvp_logger.version = 'test'
    # Real writable path: production code (e.g. diagnostics probes) now reads
    # log_dir to place output under the log folder. Without this it would be an
    # auto-MagicMock, and Path(mock) -> Path('MagicMock/mock.log_dir/<id>'),
    # which mkdir() leaks into the repo root.
    mock_lvp_logger.log_dir = tempfile.gettempdir()
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
        'kivy.app': MagicMock(),
        'kivy.uix': MagicMock(),
        'kivy.uix.scrollview': MagicMock(),
        # FX2 / libusb -- skipped when --run-fx2-hardware is set.
        'usb': MagicMock(),
        'usb.core': MagicMock(),
        'usb.util': MagicMock(),
        'usb1': MagicMock(),
        # Camera SDKs -- skipped when their --run-*-hardware flag is set.
        # pypylon gets a typed stub (real subclassable handler bases +
        # exception types) instead of a blanket MagicMock so the driver's
        # ImageHandler / _CameraRemovalHandler classes can be instantiated
        # and their callbacks driven directly in unit tests.
        'pypylon': _pypylon_stub.pypylon,
        'pypylon.pylon': _pypylon_stub.pylon,
        'pypylon.genicam': _pypylon_stub.genicam,
        'ids_peak': MagicMock(),
        'ids_peak.ids_peak': MagicMock(),
        'ids_peak.ids_peak_ipl_extension': MagicMock(),
        'ids_peak_ipl': MagicMock(),
    }
    for name, mock_mod in deps.items():
        if name in _skip_mocks:
            continue
        sys.modules.setdefault(name, mock_mod)


# Run at conftest import time -- before any test file is collected.
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

    _safe(
        '--run-hardware',
        action='store_true',
        default=False,
        help='Run hardware serial tests (firmware boards via SerialBoard)',
    )
    _safe(
        '--run-ids-hardware',
        action='store_true',
        default=False,
        help='Run IDS Peak hardware tests (real SDK + connected camera)',
    )
    _safe(
        '--run-pylon-hardware',
        action='store_true',
        default=False,
        help='Run Pylon hardware tests (real SDK + connected camera)',
    )
    _safe(
        '--run-fx2-hardware',
        action='store_true',
        default=False,
        help='Run FX2 hardware tests (pyusb/libusb1 + connected LS620/LS560)',
    )
    _safe(
        '--run-timing-sensitive',
        action='store_true',
        default=False,
        help='Run wall-clock timing-sensitive tests (can be flaky under load)',
    )
    _safe(
        '--driver-log',
        action='store_true',
        default=False,
        help='Route the (normally mocked) driver loggers to a REAL DEBUG log '
        'file + stdout, so a test run records the driver internals (poll / '
        'DeviceLost / wedge classification / grab detail) instead of discarding '
        'them into the MagicMock. Works on any test; essential for diagnosing a '
        '--run-ids-hardware bench-test failure.',
    )


def pytest_configure(config):
    """Register custom markers used by hardware tests."""
    config.addinivalue_line(
        'markers',
        'ids_hardware: requires real IDS Peak SDK + connected camera '
        '(only runs with --run-ids-hardware)',
    )
    config.addinivalue_line(
        'markers',
        'pylon_hardware: requires real Pylon SDK + connected camera '
        '(only runs with --run-pylon-hardware)',
    )
    config.addinivalue_line(
        'markers',
        'fx2_hardware: requires pyusb/libusb1 + connected FX2 scope '
        '(only runs with --run-fx2-hardware)',
    )
    config.addinivalue_line(
        'markers',
        'timing_sensitive: measures wall-clock timing and can be flaky '
        'under CI/load (only runs with --run-timing-sensitive)',
    )

    if config.getoption('--driver-log', default=False):
        _enable_driver_logging(config)


def _enable_driver_logging(config):
    """Point the mocked lvp_logger's `logger`/`camera_logger` at a REAL DEBUG
    logger writing to a timestamped file + stdout.

    The suite mocks lvp_logger, so every driver `logger.*` / `_cam_log.*` call is
    normally swallowed by the MagicMock -- worthless when a --run-ids-hardware
    bench test fails and the driver's internal narrative is exactly what's
    needed. Runs in pytest_configure, before collection imports the drivers, so
    their module-level `from lvp_logger import logger` binds to the real logger.
    Off by default (flag-gated), so the normal mocked suite is unchanged.
    """
    import logging
    import time

    real = logging.getLogger('lvp_driver_test')
    real.setLevel(logging.DEBUG)
    real.propagate = False
    real.handlers.clear()

    # Under logs/, which is gitignored -- writing to the repo root left these
    # sitting beside the source tree, one per --driver-log run, forever.
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'driver_test_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    real.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('[driver] %(message)s'))
    real.addHandler(stream_handler)

    lvp = sys.modules.get('lvp_logger')
    if lvp is not None:
        lvp.logger = real
        lvp.camera_logger = real
        # Without this, modules bound to the protocol lane
        # (protocol_image_writer) log into the swallowing MagicMock and
        # their narrative vanishes from --driver-log bench captures.
        lvp.protocol_logger = real

    config._driver_log_path = log_path
    config._driver_log_handlers = (file_handler, stream_handler)
    config._driver_logger = real
    print(f'\n[--driver-log] driver logging -> {log_path}')


def pytest_unconfigure(config):
    """Close the --driver-log handlers and restate the file path."""
    handlers = getattr(config, '_driver_log_handlers', None)
    if not handlers:
        return
    logger = config._driver_logger
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    print(f'\n[--driver-log] driver log written to {config._driver_log_path}')


def pytest_collection_modifyitems(config, items):
    """Skip hardware-marked tests unless the matching opt-in flag is set."""
    gates = [
        ('ids_hardware', '--run-ids-hardware'),
        ('pylon_hardware', '--run-pylon-hardware'),
        ('fx2_hardware', '--run-fx2-hardware'),
        ('timing_sensitive', '--run-timing-sensitive'),
    ]
    for marker, flag in gates:
        if config.getoption(flag, default=False):
            continue
        skip = pytest.mark.skip(reason=f'needs {flag}')
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
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    s._camera_driver.load_cycle_images()
    s.imaging.start_streaming()
    yield s
    s.imaging.stop_streaming()
    s.disconnect()


@pytest.fixture
def scale_ctx(monkeypatch):
    """Install an app context whose scope reports a known image scale.

    ``common_utils.get_pixel_size`` / ``get_field_of_view`` read the pixel
    size and tube focal length from ``app_context.ctx.scope.capabilities``;
    production has no hardcoded fallback, so a test that needs a real scale
    (scale bar, tiling, field-of-view readout) must supply a scope that
    reports one. The values match Etaluma's Classic optics so geometry
    assertions written against the previous default stay valid.
    """
    import threading
    from types import SimpleNamespace

    import modules.app_context as app_context

    scope = SimpleNamespace(
        capabilities=SimpleNamespace(pixel_size_um=2.0, lens_focal_length_mm=47.8)
    )
    # A real AppContext (not a bare namespace) so code paths that gate on
    # "ctx is not None" -- e.g. the save-encoding resolver's settings_lock --
    # find the services they expect, not a half-built stand-in. The settings
    # store lives on the session, so the context needs one to reach it.
    monkeypatch.setattr(
        app_context,
        'ctx',
        app_context.AppContext(
            scope=scope,
            session=SimpleNamespace(settings={}, settings_lock=threading.Lock()),
        ),
    )
    return scope
