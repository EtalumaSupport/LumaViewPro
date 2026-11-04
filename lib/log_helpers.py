# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The loggers the drivers and lib/ write to, by name, and their composition helpers.

No Kivy / platformdirs / config dependencies, so the lower layers import
their loggers from here without importing ``lvp_logger``, which reads the
settings layer to set itself up. A logger is a process-wide object found by
its name: ``lvp_logger`` takes these same objects and attaches the handlers,
levels and files to them, so a line written here lands where it always has.
The ``log_to`` helper is the shared dual / multi-write primitive that
subsystem-mirroring callers use (camera.log today; motor.log / led.log /
ids.log when those land).
"""

import logging
import threading

# Between INFO and DEBUG: the routine hardware traffic, a line per command (a
# serial exchange, a camera setter, an LED lit, a move made). INFO keeps the
# events and anomalies a field log needs; DEBUG keeps developer internals with
# no hardware fact in them. Registered here, the lowest module every writer and
# the settings' level check import, so "VERBOSE" is a level before anything
# names it. Written as ``log.log(VERBOSE, message)``.
VERBOSE = 15
logging.addLevelName(VERBOSE, 'VERBOSE')

# The names are literal, never ``__name__``: lvp_logger's main-log floor passes
# a record below INFO only from a logger named 'lvp_logger' or under 'LVP', and
# the session sets the main log's level by those names, so a logger named for
# this module would drop every driver's DEBUG line from lumaviewpro.log.
logger = logging.getLogger('lvp_logger')
camera_logger = logging.getLogger('LVP.camera')


def log_to(*loggers, level: str, message: str) -> None:
    """Mirror one message to multiple loggers (primary + N mirrors).

    Use this when a single event should land in BOTH the main log AND
    a per-subsystem dedicated log (camera.log today; motor.log /
    led.log / ids.log when those land). The first non-None logger
    argument is the primary; subsequent non-None ones are mirrors. If
    a mirror raises, the primary's call still landed and the failure
    is downgraded to a ``primary.debug()`` line so caller control flow
    stays intact.

    Args:
        *loggers: One or more logger-like objects exposing methods
            named by ``level`` (``logging.Logger`` instances qualify,
            as do test doubles). ``None`` entries are skipped so
            callers can pass an optional mirror without a guard. At
            least one non-None primary is required; with zero
            primaries the call is a silent no-op.
        level: ``'info'`` / ``'warning'`` / ``'error'`` / ``'debug'`` /
            ``'critical'`` / ``'exception'``. Same set as Python
            logging level-method names.
        message: The full log message. Convention: include the
            subsystem prefix (``'[CAM Class ]'``, ``'[XYZ Class ]'``,
            ``'[LED Class ]'``) so the line is grep-friendly across
            files.

    Canonical example -- the camera dual-write that
    ``pyloncamera._log_cam`` wraps over::

        log_to(logger, _cam_log, level='info',
               message='[CAM Class ] Disconnected from Pylon camera')
    """
    primary = None
    mirrors = []
    for lg in loggers:
        if lg is None:
            continue
        if primary is None:
            primary = lg
        else:
            mirrors.append(lg)
    if primary is None:
        return

    getattr(primary, level)(message)
    for mirror in mirrors:
        try:
            getattr(mirror, level)(message)
        except Exception:
            primary.debug(f'log_to: mirror.{level}() raised')


# Each trace logger and the file it writes, whose levels apply_session_level sets.
_trace_files: dict[logging.Logger, logging.Handler] = {}

_session_level = logging.INFO
_levels_lock = threading.Lock()


def write_trace_log(trace_logger: logging.Logger, file_handler: logging.Handler) -> None:
    """Give a trace logger its file, written at the session's level."""
    trace_logger.addHandler(file_handler)
    with _levels_lock:
        _trace_files[trace_logger] = file_handler
    apply_session_level(_session_level)


def apply_session_level(level: int | str) -> None:
    """Put the session's log level on every log that follows it.

    The 'LVP' parent (every LVP.* logger without a level of its own: the
    GUI interaction and protocol logs among them) and lvp_logger's own
    logger take it, as do the trace loggers.
    """
    global _session_level
    level = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    with _levels_lock:
        _session_level = level
        logging.getLogger('LVP').setLevel(level)
        logger.setLevel(level)
        for trace_logger, file_handler in _trace_files.items():
            trace_logger.setLevel(level)
            file_handler.setLevel(level)
