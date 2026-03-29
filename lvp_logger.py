#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

'''
lvp_logger.py configures a standard python logger for LumaViewPro.
'''

import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import ctypes
import userpaths
import threading

global windows_machine 

windows_machine = False

# Thread-local storage for tracking paused threads
_paused_threads = threading.local()

if os.name == "nt":
    windows_machine = True

abspath = os.path.abspath(__file__)
basename = os.path.basename(__file__)
script_path = abspath[:-len(basename)]

# version.txt format:
#   Line 1: version string (e.g., "4.0.0-beta2") - used in folder names, must be path-safe
#   Line 2: build timestamp (e.g., "2026-03-27 18:52") - displayed in title bar only
version = ""
build_timestamp = ""
try:
    with open(os.path.join(script_path, "version.txt")) as f:
        lines = f.readlines()
        version = lines[0].strip() if len(lines) > 0 else ""
        build_timestamp = lines[1].strip() if len(lines) > 1 else ""
except FileNotFoundError:
    pass  # Expected when running from source without version.txt
except Exception as e:
    print(f"[lvp_logger] WARNING: Failed to read version.txt: {e}", file=sys.stderr)

try:
    with open(os.path.join(script_path, "marker.lvpinstalled")) as f:
        lvp_installed = True
except FileNotFoundError:
    lvp_installed = False  # Expected when running from source
except Exception as e:
    print(f"[lvp_logger] WARNING: Failed to read marker.lvpinstalled: {e}", file=sys.stderr)
    lvp_installed = False

if windows_machine and lvp_installed:

    documents_folder = userpaths.get_my_documents()
    lvp_appdata = os.path.join(documents_folder, f"LumaViewPro {version}")

    # Do NOT os.chdir() here — it changes global CWD as a side effect of import.
    # Use absolute paths instead.
    pass

else:
    lvp_appdata = script_path

from modules.settings_init import load_debug_setting
try:
    debug = load_debug_setting(lvp_appdata)
except Exception as e:
    print(f"[lvp_logger] WARNING: Failed to load debug setting, defaulting to False: {e}", file=sys.stderr)
    debug = False



log_dir = os.path.join(lvp_appdata, "logs", "LVP_Log")
os.makedirs(log_dir, exist_ok=True)

# files to which messages are logged
LOG_FILE = os.path.join(log_dir, 'lumaviewpro.log')

ERRORS_LOG_FILE = os.path.join(log_dir, 'lumaviewpro_errors.log')

REST_API_LOG_FILE = os.path.join(log_dir, 'lumaviewpro_rest_api.log')
SERIAL_LOG_FILE = os.path.join(log_dir, 'serial.log')
AUTOFOCUS_LOG_FILE = os.path.join(log_dir, 'autofocus.log')
API_LOG_FILE = os.path.join(log_dir, 'api.log')
GUI_LOG_FILE = os.path.join(log_dir, 'gui_interactions.log')

# CustomFormatter class enables change in log format depending on log level 
class CustomFormatter(logging.Formatter):
    # if level is DEBUG/WARNING/ERROR/CRITICAL, log the level, message, time, and filename
    def __init__(self, 
                 fmt = '[%(levelname)s] [%(threadName)s] %(asctime)s.%(msecs)03d - %(filename)s - %(message)s', 
                 datefmt ='%m/%d/%Y %H:%M:%S'):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        # if record.levelno == logging.INFO:
        #     # if INFO level, only log the message
        #     return record.getMessage()
        return logging.Formatter.format(self, record)
    
def minimize_logger_window():
    if sys.platform == "win32":
        try:
            console_window = ctypes.windll.kernel32.GetConsoleWindow()
            if console_window:
                # Setting the found console window to a minimized state (state 6)
                ctypes.windll.user32.ShowWindow(console_window, 6)
                logger.info("[Logger  ] Console window minimized")
            else:
                logger.warning("[Logger  ] Console window not found.")
        except Exception as e:
            logger.error(f"[Logger  ] Failed to minimize console window: {e}")

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

# Log traceback if we have a crash to tell us more info on what happened
def custom_except_hook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.critical("Logger ] Keyboard interrupt quit.")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical("Logger ] CRASH - Uncaught Exception: ", exc_info=(exc_type, exc_value, exc_traceback))

# ensures logger is specific to the file importing lvp_logger
logger = logging.getLogger(__name__)

# Set up the 'LVP' parent logger so all LVP.* child loggers (used throughout the
# codebase) inherit handlers and don't propagate to root/Kivy console.
_lvp_parent = logging.getLogger('LVP')
_lvp_parent.setLevel(logging.INFO)

# Prevent logs from propagating to root (and the console)
if not debug:
    logger.propagate = False
    _lvp_parent.propagate = False

# determines lowest level of messages to log (DEBUG < INFO < WARNING < ERROR < CRITICAL)
logger.setLevel(logging.INFO)

# obtains name of the module (file) importing lvp_logger
filename = '%s' % __file__
file_handler = RotatingFileHandler(
    LOG_FILE,
    mode='a',
    maxBytes=20*1024*1024,
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
    maxBytes=20*1024*1024,
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

# REST API log handler — captures records marked with extra={'api_request': True}
rest_api_handler = RotatingFileHandler(
    REST_API_LOG_FILE,
    mode='a',
    maxBytes=20*1024*1024,
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

# Serial log — dedicated file for all serial command/response traffic with timing.
# Uses its own logger (LVP.serial) with propagate=False so serial traffic
# does NOT appear in the main log.  Errors still go to the errors log.
serial_logger = logging.getLogger('LVP.serial')
serial_logger.setLevel(logging.INFO)
serial_logger.propagate = False  # Keep serial traffic out of the main log

class SerialFormatter(logging.Formatter):
    """Compact format for serial log: timestamp board command → response (timing)."""
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s.%(msecs)03d %(message)s',
            datefmt='%H:%M:%S',
        )

serial_file_handler = RotatingFileHandler(
    SERIAL_LOG_FILE,
    mode='a',
    maxBytes=20*1024*1024,
    backupCount=5,
    encoding=None,
    delay=False,
)
serial_file_handler.namer = lambda name: name.replace('.log', '') + '.log'
serial_file_handler.setFormatter(SerialFormatter())
serial_file_handler.addFilter(ThreadPauseFilter())
serial_logger.addHandler(serial_file_handler)
# Also send serial errors/warnings to the errors log
serial_logger.addHandler(error_file_handler)

# Autofocus log — dedicated file for AF sweep data, scores, timing.
# Engineering mode only — handler attached via enable_engineering_logs().
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
    maxBytes=20*1024*1024,
    backupCount=5,
    encoding=None,
    delay=True,  # Don't create file until first write
)
_af_file_handler.namer = lambda name: name.replace('.log', '') + '.log'
_af_file_handler.setFormatter(AFFormatter())
_af_file_handler.addFilter(ThreadPauseFilter())

# API log — internal Lumascope API calls (state-changing operations).
# Engineering mode only — handler attached via enable_engineering_logs().
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
    maxBytes=20*1024*1024,
    backupCount=5,
    encoding=None,
    delay=True,  # Don't create file until first write
)
_api_file_handler.namer = lambda name: name.replace('.log', '') + '.log'
_api_file_handler.setFormatter(APIFormatter())
_api_file_handler.addFilter(ThreadPauseFilter())

def enable_engineering_logs(enabled: bool):
    """Attach/detach file handlers for engineering-mode-only logs.

    Called once after engineering mode is determined. When disabled,
    the loggers exist but have no file handler — logging calls are
    essentially free (no I/O).

    WORKAROUND: During beta releases, always enable engineering logs
    for maximum debugging visibility. Remove this override after
    beta stabilization and gate behind engineering mode again.
    """
    # WORKAROUND: Force-enable during beta for field debugging
    import re
    try:
        with open(os.path.join(os.path.dirname(__file__), 'version.txt')) as _vf:
            _ver = _vf.read().strip()
        if 'beta' in _ver.lower():
            enabled = True
    except Exception:
        pass

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

# GUI interaction log — every user action for crash forensics
# WORKAROUND: INFO level during beta. Move to DEBUG once stable.
gui_handler = RotatingFileHandler(
    GUI_LOG_FILE, maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
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
        logger.warning(f"[Logger  ] Failed to remove console handler: {e}")

sys.excepthook = custom_except_hook

# Also catch unhandled exceptions in worker threads (Python 3.8+)
def _thread_except_hook(args):
    if issubclass(args.exc_type, KeyboardInterrupt):
        return
    logger.critical(
        f"Logger ] CRASH - Uncaught Exception in thread '{args.thread.name}': ",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )
threading.excepthook = _thread_except_hook
minimize_logger_window()
logging.disable(logging.DEBUG)
