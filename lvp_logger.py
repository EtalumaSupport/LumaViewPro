#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""
lvp_logger.py configures a standard python logger for LumaViewPro.
"""

import os

# Suppress Kivy's own file/console logging before any Kivy import can fire.
# LVP routes the `kivy` logger into its file_handler / error_handler below,
# so Kivy diagnostics still land in the main LVP logs -- just not in
# ~/.kivy/logs/. The app entry point (lumaviewpro.py) sets the same vars,
# but lvp_logger is imported earlier by most code paths (tests, scripts,
# standalone imports), so set them here too.
os.environ.setdefault('KIVY_NO_CONSOLELOG', '1')
os.environ.setdefault('KIVY_NO_FILELOG', '1')

import logging
from logging.handlers import RotatingFileHandler
import sys
import ctypes
import platformdirs
import threading

global windows_machine

windows_machine = False

# Thread-local storage for tracking paused threads
_paused_threads = threading.local()

if os.name == 'nt':
    windows_machine = True

abspath = os.path.abspath(__file__)
basename = os.path.basename(__file__)
script_path = abspath[: -len(basename)]

# version.txt format:
#   Line 1: version string (e.g., "4.0.0-beta2") - used in folder names, must be path-safe
#   Line 2: build timestamp (e.g., "2026-03-27 18:52") - displayed in title bar only
version = ''
build_timestamp = ''
try:
    with open(os.path.join(script_path, 'version.txt')) as f:
        lines = f.readlines()
        version = lines[0].strip() if len(lines) > 0 else ''
        build_timestamp = lines[1].strip() if len(lines) > 1 else ''
except FileNotFoundError:
    pass  # Expected when running from source without version.txt
except Exception as e:
    print(f'[lvp_logger] WARNING: Failed to read version.txt: {e}', file=sys.stderr)

# Under PyInstaller (sys.frozen=True), this module's __file__ points
# into the bundle's extract dir -- _MEI<random> (onefile mode) or
# <install>/_internal (onedir 6+) -- NOT the install root where the
# WiX MSI drops marker.lvpinstalled. version.txt above works because
# it's bundled into the same dir via the .spec datas list; the marker
# is intentionally NOT bundled (it exists to distinguish "MSI-installed
# build" from "PyInstaller dev build"). Use sys.executable's directory
# when frozen so the probe lands on the install root.
if getattr(sys, 'frozen', False):
    _marker_dir = os.path.dirname(os.path.abspath(sys.executable))
else:
    _marker_dir = script_path.rstrip(os.sep) or '.'

try:
    with open(os.path.join(_marker_dir, 'marker.lvpinstalled')) as f:
        lvp_installed = True
except FileNotFoundError:
    lvp_installed = False  # Expected when running from source
except Exception as e:
    print(f'[lvp_logger] WARNING: Failed to read marker.lvpinstalled: {e}', file=sys.stderr)
    lvp_installed = False

if windows_machine and lvp_installed:
    documents_folder = platformdirs.user_documents_dir()
    lvp_appdata = os.path.join(documents_folder, f'LumaViewPro {version}')

    # Do NOT os.chdir() here -- it changes global CWD as a side effect of import.
    # Use absolute paths instead.
    pass

else:
    lvp_appdata = script_path

from modules.settings_init import load_debug_setting

try:
    debug = load_debug_setting(lvp_appdata)
except Exception as e:
    print(
        f'[lvp_logger] WARNING: Failed to load debug setting, defaulting to False: {e}',
        file=sys.stderr,
    )
    debug = False


log_dir = os.path.join(lvp_appdata, 'logs', 'LVP_Log')
os.makedirs(log_dir, exist_ok=True)

# files to which messages are logged
LOG_FILE = os.path.join(log_dir, 'lumaviewpro.log')

ERRORS_LOG_FILE = os.path.join(log_dir, 'lumaviewpro_errors.log')

REST_API_LOG_FILE = os.path.join(log_dir, 'lumaviewpro_rest_api.log')
SERIAL_LOG_FILE = os.path.join(log_dir, 'serial.log')
AUTOFOCUS_LOG_FILE = os.path.join(log_dir, 'autofocus.log')
API_LOG_FILE = os.path.join(log_dir, 'api.log')
CAMERA_LOG_FILE = os.path.join(log_dir, 'camera.log')
GUI_LOG_FILE = os.path.join(log_dir, 'gui_interactions.log')
METRICS_LOG_FILE = os.path.join(log_dir, 'metrics.log')
PROTOCOL_LOG_FILE = os.path.join(log_dir, 'protocol.log')


# CustomFormatter class enables change in log format depending on log level
class CustomFormatter(logging.Formatter):
    # if level is DEBUG/WARNING/ERROR/CRITICAL, log the level, message, time, and filename
    def __init__(
        self,
        fmt='[%(levelname)s] [%(threadName)s] %(asctime)s.%(msecs)03d - %(filename)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
    ):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        # if record.levelno == logging.INFO:
        #     # if INFO level, only log the message
        #     return record.getMessage()
        return logging.Formatter.format(self, record)


def minimize_logger_window():
    if sys.platform == 'win32':
        try:
            console_window = ctypes.windll.kernel32.GetConsoleWindow()
            if console_window:
                # Setting the found console window to a minimized state (state 6)
                ctypes.windll.user32.ShowWindow(console_window, 6)
                logger.info('[Logger  ] Console window minimized')
            else:
                logger.warning('[Logger  ] Console window not found.')
        except Exception as e:
            logger.error(f'[Logger  ] Failed to minimize console window: {e}')


def pause_thread():
    """Pause logging for the current thread. Logs will not be recorded until unpause_thread is called."""
    _paused_threads.paused = True


def unpause_thread():
    """Resume logging for the current thread."""
    _paused_threads.paused = False


def is_thread_paused():
    """Check if logging is paused for the current thread."""
    return getattr(_paused_threads, 'paused', False)


class ThreadPauseFilter(logging.Filter):
    """Filter that prevents logging from paused threads."""

    def filter(self, record):
        # Allow the log if the thread is not paused
        return not getattr(_paused_threads, 'paused', False)


# Log traceback if we have a crash to tell us more info on what happened.
# The tuple form `exc_info=(type, value, tb)` does not render through
# CustomFormatter, which silently dropped tracebacks in the field. Embed
# the formatted traceback directly in the message string so it renders
# regardless of stdlib exc_info handling.
import traceback as _traceback


def custom_except_hook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.critical('Logger ] Keyboard interrupt quit.')
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    tb_text = ''.join(_traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.critical(f'Logger ] CRASH - Uncaught Exception:\n{tb_text}')


# ensures logger is specific to the file importing lvp_logger
logger = logging.getLogger(__name__)

# Single source of truth for the verbosity floor: debug_mode picks DEBUG vs
# INFO, and that one level is the ONLY thing gating DEBUG output. The level is
# applied to both logger trees that write to the main log -- the 'LVP' parent
# (all LVP.* child loggers, incl. the preview [PERF] line) and lvp_logger's own
# logger. Loggers that want a fixed floor regardless of debug_mode set their own
# level (e.g. LVP.camera stays DEBUG -- the always-on camera.log firehose).
# Note: do NOT reintroduce a global logging.disable() here -- it overrides every
# logger's own level (it was silently starving camera.log of DEBUG) and split
# the toggle into two gates. The per-logger level is the canonical mechanism.
_log_level = logging.DEBUG if debug else logging.INFO

_lvp_parent = logging.getLogger('LVP')
_lvp_parent.setLevel(_log_level)
logger.setLevel(_log_level)

# Prevent logs from propagating to root (and the console)
if not debug:
    logger.propagate = False
    _lvp_parent.propagate = False

# obtains name of the module (file) importing lvp_logger
filename = f'{__file__}'
file_handler = RotatingFileHandler(
    LOG_FILE,
    mode='a',
    maxBytes=20 * 1024 * 1024,
    backupCount=5,
    encoding=None,
    delay=False,
)
file_handler.namer = lambda name: name.replace('.log', '') + '.log'
file_handler.setFormatter(CustomFormatter())
file_handler.addFilter(ThreadPauseFilter())

# Additional rotating file handler for errors and critical logs only
error_file_handler = RotatingFileHandler(
    ERRORS_LOG_FILE,
    mode='a',
    maxBytes=20 * 1024 * 1024,
    backupCount=5,
    encoding=None,
    delay=False,
)
# keep the same filename pattern for rotations
error_file_handler.namer = lambda name: name.replace('.log', '') + '.log'
error_file_handler.setFormatter(CustomFormatter())
error_file_handler.addFilter(ThreadPauseFilter())

# Accept all levels on the handler but filter to ERROR+ or forced records
error_file_handler.setLevel(logging.NOTSET)

"""

Example of forcing a log record to also go to the errors log file:

logger.info("Info message that should also go to errors file", extra={'force_error': True})

"""


class ErrorOrForcedFilter(logging.Filter):
    """Allows records that are ERROR/CRITICAL or explicitly marked via extra={'force_error': True}."""

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        return bool(getattr(record, 'force_error', False))


error_file_handler.addFilter(ErrorOrForcedFilter())

# REST API log handler -- captures records marked with extra={'api_request': True}
rest_api_handler = RotatingFileHandler(
    REST_API_LOG_FILE,
    mode='a',
    maxBytes=20 * 1024 * 1024,
    backupCount=5,
    encoding=None,
    delay=True,  # Don't create file until first REST API log message
)
rest_api_handler.namer = lambda name: name.replace('.log', '') + '.log'
rest_api_handler.setFormatter(CustomFormatter())
rest_api_handler.addFilter(ThreadPauseFilter())


class RestAPIFilter(logging.Filter):
    """Only allows records explicitly marked as REST API traffic."""

    def filter(self, record):
        return bool(getattr(record, 'api_request', False))


rest_api_handler.addFilter(RestAPIFilter())

# Serial log -- dedicated file for all serial command/response traffic with timing.
# Uses its own logger (LVP.serial) with propagate=False so serial traffic
# does NOT appear in the main log.  Errors still go to the errors log.
serial_logger = logging.getLogger('LVP.serial')
serial_logger.setLevel(logging.INFO)
serial_logger.propagate = False  # Keep serial traffic out of the main log


class SerialFormatter(logging.Formatter):
    """Compact format for serial log: timestamp board command -> response (timing)."""

    def __init__(self):
        super().__init__(
            fmt='%(asctime)s.%(msecs)03d %(message)s',
            datefmt='%H:%M:%S',
        )


# Firehose trace logs (serial, camera, protocol, api) stay fully verbose
# through the beta, but their rotation footprint is capped at 5MB x 3 files
# so a long-soak support bundle stays small. Revisit the verbosity itself
# (demote routine per-command traffic to DEBUG) at the 4.0.0 GA release.
serial_file_handler = RotatingFileHandler(
    SERIAL_LOG_FILE,
    mode='a',
    maxBytes=5 * 1024 * 1024,
    backupCount=2,
    encoding=None,
    delay=False,
)
serial_file_handler.namer = lambda name: name.replace('.log', '') + '.log'
serial_file_handler.setFormatter(SerialFormatter())
serial_file_handler.addFilter(ThreadPauseFilter())
serial_logger.addHandler(serial_file_handler)
# Also send serial errors/warnings to the errors log
serial_logger.addHandler(error_file_handler)

# Camera log -- dedicated file for all camera SDK command traffic with
# timing. Same shape as serial.log but for Pylon / IDS / FX2 / simulator
# camera drivers. Captures every meaningful SDK call (Gain, ExposureTime,
# StartGrabbing, StopGrabbing, PixelFormat, Binning, Width/Height, etc).
# Per-frame callback events are NOT logged here -- they're in the per-frame
# pylon_callback_trace.csv when profile_trace_enabled is set in
# settings.json. Always-on (not engineering-gated) to match serial.log
# behavior.
camera_logger = logging.getLogger('LVP.camera')
# DEBUG level (was INFO): camera.log is the firehose for camera debugging.
# Every per-frame check, every SDK-call return value, every state
# transition lands here. Main log stays uncluttered via propagate=False;
# the dual-write helper `_log_cam` in pyloncamera.py mirrors load-bearing
# events (identity, connect, GetArray failures) to both files.
camera_logger.setLevel(logging.DEBUG)
camera_logger.propagate = False  # Keep camera traffic out of the main log


class CameraFormatter(logging.Formatter):
    """Compact format for camera log: timestamp [thread] message."""

    def __init__(self):
        super().__init__(
            fmt='%(asctime)s.%(msecs)03d [%(threadName)s] %(message)s',
            datefmt='%H:%M:%S',
        )


camera_file_handler = RotatingFileHandler(
    CAMERA_LOG_FILE,
    mode='a',
    maxBytes=5 * 1024 * 1024,
    backupCount=2,
    encoding=None,
    delay=False,
)
camera_file_handler.namer = lambda name: name.replace('.log', '') + '.log'
camera_file_handler.setFormatter(CameraFormatter())
camera_file_handler.addFilter(ThreadPauseFilter())
camera_logger.addHandler(camera_file_handler)
# Also send camera errors/warnings to the errors log
camera_logger.addHandler(error_file_handler)

# Re-exported from lib.log_helpers so callers can `from lvp_logger
# import log_to` next to `logger` without learning a separate import
# path. The pure implementation lives in lib/ because conftest mocks
# lvp_logger wholesale during pytest, and tests need to exercise the
# real log_to.
from lib.log_helpers import log_to  # noqa: E402

# Metrics log -- dedicated file for periodic runtime-health snapshots
# (system metrics, handle/GC counts, buffer churn, frame-interval
# percentiles). Routed here instead of errors.log so errors.log stays
# signal-only. Uses standard CustomFormatter so existing log-parsing
# scripts continue to work against this file.
metrics_logger = logging.getLogger('LVP.metrics')
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False  # Keep metrics out of the main log

metrics_file_handler = RotatingFileHandler(
    METRICS_LOG_FILE,
    mode='a',
    maxBytes=20 * 1024 * 1024,
    backupCount=5,
    encoding=None,
    delay=False,
)
metrics_file_handler.namer = lambda name: name.replace('.log', '') + '.log'
metrics_file_handler.setFormatter(CustomFormatter())
metrics_file_handler.addFilter(ThreadPauseFilter())
metrics_logger.addHandler(metrics_file_handler)
# Metrics errors/warnings still hit the errors log
metrics_logger.addHandler(error_file_handler)

# Protocol log -- dedicated file for the per-step protocol-execution
# narrative (step records, per-channel LED/illumination, image-captured
# events). A long protocol soak emits tens of thousands of these per run;
# routing them here keeps the main log readable while preserving the full
# run history. propagate=False keeps protocol detail out of the main log;
# warnings/errors still mirror to the errors log.
protocol_logger = logging.getLogger('LVP.protocol')
protocol_logger.setLevel(logging.INFO)
protocol_logger.propagate = False  # Keep protocol detail out of the main log

protocol_file_handler = RotatingFileHandler(
    PROTOCOL_LOG_FILE,
    mode='a',
    maxBytes=5 * 1024 * 1024,
    backupCount=2,
    encoding=None,
    delay=False,
)
protocol_file_handler.namer = lambda name: name.replace('.log', '') + '.log'
protocol_file_handler.setFormatter(CustomFormatter())
protocol_file_handler.addFilter(ThreadPauseFilter())
protocol_logger.addHandler(protocol_file_handler)
# Protocol errors/warnings still hit the errors log
protocol_logger.addHandler(error_file_handler)

# Autofocus log -- dedicated file for AF sweep data, scores, timing.
# Engineering mode only -- handler attached via enable_engineering_logs().
af_logger = logging.getLogger('LVP.autofocus')
af_logger.setLevel(logging.INFO)
af_logger.propagate = False  # Keep AF data out of the main log
# Always send AF errors to the errors log
af_logger.addHandler(error_file_handler)


class AFFormatter(logging.Formatter):
    """Compact format for autofocus log."""

    def __init__(self):
        super().__init__(
            fmt='%(asctime)s.%(msecs)03d %(message)s',
            datefmt='%H:%M:%S',
        )


_af_file_handler = RotatingFileHandler(
    AUTOFOCUS_LOG_FILE,
    mode='a',
    maxBytes=20 * 1024 * 1024,
    backupCount=5,
    encoding=None,
    delay=True,  # Don't create file until first write
)
_af_file_handler.namer = lambda name: name.replace('.log', '') + '.log'
_af_file_handler.setFormatter(AFFormatter())
_af_file_handler.addFilter(ThreadPauseFilter())

# API log -- internal Lumascope API calls (state-changing operations).
# Engineering mode only -- handler attached via enable_engineering_logs().
api_logger = logging.getLogger('LVP.api')
api_logger.setLevel(logging.INFO)
api_logger.propagate = False  # Keep API traffic out of the main log
# Always send API errors to the errors log
api_logger.addHandler(error_file_handler)


class APIFormatter(logging.Formatter):
    """Compact format for API log."""

    def __init__(self):
        super().__init__(
            fmt='%(asctime)s.%(msecs)03d %(message)s',
            datefmt='%H:%M:%S',
        )


_api_file_handler = RotatingFileHandler(
    API_LOG_FILE,
    mode='a',
    maxBytes=5 * 1024 * 1024,
    backupCount=2,
    encoding=None,
    delay=True,  # Don't create file until first write
)
_api_file_handler.namer = lambda name: name.replace('.log', '') + '.log'
_api_file_handler.setFormatter(APIFormatter())
_api_file_handler.addFilter(ThreadPauseFilter())


def enable_engineering_logs(enabled: bool):
    """Attach/detach file handlers for engineering-mode-only logs.

    Called once after engineering mode is determined. When disabled,
    the loggers exist but have no file handler -- logging calls are
    essentially free (no I/O).

    WORKAROUND: During beta releases, always enable engineering logs
    for maximum debugging visibility. Remove this override after
    beta stabilization and gate behind engineering mode again.
    """
    # WORKAROUND: Force-enable during beta for field debugging
    if 'beta' in version.lower():
        enabled = True

    if enabled:
        if _af_file_handler not in af_logger.handlers:
            af_logger.addHandler(_af_file_handler)
        if _api_file_handler not in api_logger.handlers:
            api_logger.addHandler(_api_file_handler)
        logger.info('[Logger  ] Engineering logs enabled (autofocus.log, api.log)')
    else:
        if _af_file_handler in af_logger.handlers:
            af_logger.removeHandler(_af_file_handler)
        if _api_file_handler in api_logger.handlers:
            api_logger.removeHandler(_api_file_handler)


logger.addHandler(file_handler)
logger.addHandler(error_file_handler)
logger.addHandler(rest_api_handler)

# GUI interaction log -- every user action for crash forensics
# WORKAROUND: INFO level during beta. Move to DEBUG once stable.
gui_handler = RotatingFileHandler(
    GUI_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2, encoding='utf-8'
)
gui_handler.setFormatter(CustomFormatter())
gui_handler.setLevel(logging.INFO)
gui_logger = logging.getLogger('LVP.gui_interactions')
gui_logger.addHandler(gui_handler)
gui_logger.propagate = False

# Route Kivy framework errors to LVP main log + errors log
kivy_logger = logging.getLogger('kivy')
kivy_logger.addHandler(file_handler)
kivy_logger.addHandler(error_file_handler)
kivy_logger.propagate = False

# Give LVP.* loggers the same file handlers so their output is captured
_lvp_parent.addHandler(file_handler)
_lvp_parent.addHandler(error_file_handler)

# Best-effort: remove any existing console/stream handlers from root to reduce terminal noise
if not debug:
    try:
        root_logger = logging.getLogger()
        for h in list(root_logger.handlers):
            if isinstance(h, logging.StreamHandler):
                root_logger.removeHandler(h)
    except Exception as e:
        logger.warning(f'[Logger  ] Failed to remove console handler: {e}')

sys.excepthook = custom_except_hook


def log_environment_banner(source_path: str, version_str: str):
    """Emit the standard launch-time environment fingerprint.

    Logs git hash, run time, host/OS, Python interpreter + version, Kivy,
    and camera SDK versions (pypylon binding + Pylon SDK runtime,
    ids_peak). Every entry point that ships should call this on startup
    so support bundles always identify the exact environment that
    produced the log.

    Centralized here so REST API, headless test runner, CLI tools all
    get the same fingerprint without copy-paste.
    """
    import sys as _sys

    logger.info('[LVP Main  ] -----------------------------------------')
    logger.info(f'[LVP Main  ] Version:   {version_str}')

    # Build identity: branch + commit timestamp + build GUID from
    # version.txt. The pre-commit hook rewrites lines 2-4 on every commit
    # so these are always present in source clones, ZIP downloads, and
    # installer bundles alike. Triage chains:
    #   - `git log -S "<guid>" -- version.txt` finds the exact commit
    #     by GUID (works in any distribution).
    #   - `git log --before=<Built>+1m <Branch>` finds it by timestamp.
    #   - `.git_archival.txt` carries the actual SHA in GitHub ZIPs.
    _built = ''
    _branch = ''
    _build_guid = ''
    try:
        with open(os.path.join(source_path, 'version.txt')) as _vf:
            _lines = _vf.read().splitlines()
            if len(_lines) >= 2:
                _built = _lines[1].strip()
            if len(_lines) >= 3:
                _branch = _lines[2].strip()
            if len(_lines) >= 4:
                _build_guid = _lines[3].strip()
    except Exception:
        pass
    logger.info(f'[LVP Main  ] Built:     {_built or "unknown"}')
    logger.info(f'[LVP Main  ] Branch:    {_branch or "unknown"}')
    logger.info(f'[LVP Main  ] BuildGUID: {_build_guid or "unknown"}')

    # Runtime: distinguish installed .exe from running directly from a
    # source clone. The presence of marker.lvpinstalled means the MSI
    # ran (the marker is dropped by the installer). Without it, this is
    # a developer running `python lumaviewpro.py` from a clone.
    logger.info(f'[LVP Main  ] Runtime:   {"installed exe" if lvp_installed else "source / dev"}')

    # SHA lookup precedence:
    #   1) .git_archival.txt -- GitHub ZIP downloads substitute the
    #      $Format:%H$ placeholder with the real SHA at archive time.
    #      In a local git clone the placeholder is unsubstituted ($Format
    #      prefix); in a ZIP it has been replaced with the 40-char SHA.
    #   2) `git rev-parse --short HEAD` -- works in local clones with
    #      .git present. Installer builds wipe .git so this returns
    #      nothing.
    # Either path that yields a real value wins; otherwise fall back to
    # Branch + Built + BuildGUID for triage.
    _git_hash = None
    try:
        with open(os.path.join(source_path, '.git_archival.txt')) as _af:
            for _line in _af:
                if _line.startswith('node: ') and not _line.startswith('node: $Format'):
                    _git_hash = _line.split(': ', 1)[1].strip()[:12]
                    break
    except Exception:
        pass
    if not _git_hash:
        try:
            import subprocess

            _git_hash = (
                subprocess.check_output(
                    ['git', 'rev-parse', '--short', 'HEAD'],
                    cwd=source_path,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                .decode()
                .strip()
            )
        except Exception:
            pass
    logger.info(
        f'[LVP Main  ] Git:       {_git_hash or "unknown (use BuildGUID or Branch + Built)"}'
    )

    # debug_mode gates all DEBUG-level output (including the preview [PERF]
    # lines). State the resolved value AND which file it came from so a
    # support bundle alone answers "was debug on, and where is it set?" --
    # the live value is read from current.json once that exists, so editing
    # settings.json has no effect on an established install.
    from modules import settings_init as _si

    logger.info(
        f'[LVP Main  ] Debug:     debug_mode={debug} '
        f'(from {_si.debug_setting_source or "default -- settings file unread"})'
    )

    # Host + OS + Python + key library versions.
    try:
        import platform as _platform

        logger.info(f'[LVP Main  ] Host: {_platform.node()}')
        logger.info(f'[LVP Main  ] OS: {_platform.platform()}')
    except Exception as e:
        logger.info(f'[LVP Main  ] OS: unavailable ({e})')
    logger.info(f'[LVP Main  ] Python: {_sys.version.split()[0]} ({_sys.executable})')
    try:
        import kivy as _kivy

        logger.info(f'[LVP Main  ] Kivy: {_kivy.__version__}')
    except Exception as e:
        logger.info(f'[LVP Main  ] Kivy: unavailable ({e})')

    # Camera SDKs -- log both the Python binding version AND the
    # underlying SDK runtime. Binding/SDK mismatch has bitten us before.
    try:
        import importlib.metadata as _imeta

        _pypylon_binding = _imeta.version('pypylon')
    except Exception:
        _pypylon_binding = 'unknown'
    try:
        from pypylon import pylon as _pylon

        # Prefer the dotted string (e.g. "10.2.1.0471") over the raw
        # list form GetPylonVersion() returns -- the list renders as
        # `[10, 2, 1, 471]` in logs, which looks like a bug report
        # waiting to happen.
        try:
            _pylon_ver = _pylon.GetPylonVersionString()
        except Exception:
            _v = _pylon.GetPylonVersion()
            _pylon_ver = '.'.join(str(x) for x in _v)
        logger.info(f'[LVP Main  ] pypylon binding: {_pypylon_binding} / Pylon SDK: {_pylon_ver}')
    except Exception as e:
        logger.info(f'[LVP Main  ] Pylon SDK: unavailable ({e})')
    try:
        import importlib.metadata as _imeta

        _ids_ver = _imeta.version('ids_peak')
        logger.info(f'[LVP Main  ] ids_peak: {_ids_ver}')
    except Exception:
        logger.info('[LVP Main  ] ids_peak: not installed')

    logger.info('[LVP Main  ] -----------------------------------------')


# Also catch unhandled exceptions in worker threads (Python 3.8+).
# Same traceback-rendering workaround as custom_except_hook above.
def _thread_except_hook(args):
    if issubclass(args.exc_type, KeyboardInterrupt):
        return
    tb_text = ''.join(
        _traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
    )
    logger.critical(
        f"Logger ] CRASH - Uncaught Exception in thread '{args.thread.name}':\n{tb_text}"
    )


threading.excepthook = _thread_except_hook
minimize_logger_window()
