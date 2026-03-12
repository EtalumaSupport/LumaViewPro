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

os.chdir(script_path)

version = ""
try:
    with open("version.txt") as f:
        version = f.readlines()[0].strip()
except FileNotFoundError:
    pass  # Expected when running from source without version.txt
except Exception as e:
    print(f"[lvp_logger] WARNING: Failed to read version.txt: {e}", file=sys.stderr)

try:
    with open("marker.lvpinstalled") as f:
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
    maxBytes=5*1024*1024,
    backupCount=2,
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
    maxBytes=5*1024*1024,
    backupCount=2,
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
    maxBytes=5*1024*1024,
    backupCount=2,
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

logger.addHandler(file_handler)
logger.addHandler(error_file_handler)
logger.addHandler(rest_api_handler)

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
