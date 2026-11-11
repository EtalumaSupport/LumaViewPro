# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The loggers the drivers and src/_lumascope/lib/ write to, by name, and their composition helpers.

No Kivy / platformdirs / config dependencies, so the lower layers import
their loggers from here without importing ``lvp_logger``, which loads the
settings layer. A logger is a process-wide object found by its name:
``lvp_logger``'s start takes these same objects and attaches the handlers,
levels and files to them, so a line written here lands where it always has.
The ``log_to`` helper is the shared dual / multi-write primitive that
subsystem-mirroring callers use (camera.log today; motor.log / led.log /
ids.log when those land).
"""

import collections
import enum
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


class _LinesBeforeAWarning(logging.Handler):
    """The recent VERBOSE lines a trace log did not write, written before its next warning.

    At INFO a trace log leaves out the routine command traffic, which is
    what a mid-run fault needs beside it: the commands that led there. This
    keeps the last ``capacity`` records the file's level kept out, and when
    a WARNING or above reaches the logger writes them to the file under one
    INFO header, before the warning itself (it is the logger's first
    handler), then holds nothing. At VERBOSE or DEBUG the file wrote every
    line already and nothing is held.
    """

    def __init__(self, file_handler: logging.Handler, capacity: int) -> None:
        super().__init__(level=VERBOSE)
        self._file = file_handler
        self._held: collections.deque[logging.LogRecord] = collections.deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING:
            if record.levelno < self._file.level:
                self._held.append(record)
            return
        if not self._held:
            return
        held = list(self._held)
        self._held.clear()
        header = logging.makeLogRecord(record.__dict__)
        header.levelno, header.levelname, header.args, header.exc_info, header.exc_text = (
            logging.INFO,
            'INFO',
            None,
            None,
            None,
        )
        header.msg = f'The {len(held)} VERBOSE lines before the {record.levelname} that follows:'
        self._file.handle(header)
        for line in held:
            self._file.handle(line)


class LogWindow(enum.Enum):
    """A stretch of the trace logs' traffic a support read opens first, kept in its own file.

    The rolling trace logs keep whatever the traffic after it left room
    for: on a long run, bring-up and the run's start are rotated away hours
    before a bundle is taken. Each window's file holds the serial, camera
    and api lines from its opening to its end, at VERBOSE whatever the
    session's level (DEBUG under debug_mode). The value is its file's name.
    """

    BRING_UP = 'bringup.log'


class _Window(logging.Handler):
    """One window: the trace loggers' records written to its file while it is open.

    It is open from ``open_window`` until ``end_window``, or until it has
    written ``budget_bytes``, whichever comes first, since an end can be far
    off or never come (a bring-up that never reaches its startup motion).
    A window that spends its budget writes one line saying so, the last of
    that window. Each opening writes a header naming what opened it. The
    file is appended to across openings and processes, and rotates at its
    cap as every log does.
    """

    def __init__(self, window: LogWindow, file_handler: logging.Handler, budget_bytes: int):
        super().__init__(level=logging.NOTSET)
        self._window = window
        self._file = file_handler
        self._budget_bytes = budget_bytes
        self._open = False
        self._written = 0

    def _write(self, message: str) -> None:
        self._file.handle(
            logging.makeLogRecord(
                {'name': 'LVP', 'msg': message, 'levelno': logging.INFO, 'levelname': 'INFO'}
            )
        )

    def open(self, opened_by: str) -> None:
        with self.lock:
            self._open = True
            self._written = 0
            self._write(f'{self._window.value} window opened: {opened_by}')

    def end(self) -> bool:
        """End the window; whether it was open."""
        with self.lock:
            was_open, self._open = self._open, False
            return was_open

    def emit(self, record: logging.LogRecord) -> None:
        if not self._open:
            return
        self._file.handle(record)
        self._written += len(self._file.format(record)) + 1
        if self._written >= self._budget_bytes:
            self._open = False
            self._write(
                f'{self._window.value} window closed: it wrote its budget of '
                f'{self._budget_bytes} bytes before its end'
            )


# Each trace logger and the file it writes, whose levels apply_session_level sets.
_trace_files: dict[logging.Logger, logging.Handler] = {}

# Each window this process writes; empty when logging was declined, so an
# opening or an end does nothing.
_windows: dict[LogWindow, _Window] = {}

_session_level = logging.INFO
_levels_lock = threading.Lock()


def write_trace_log(
    trace_logger: logging.Logger, file_handler: logging.Handler, *, held_lines: int
) -> None:
    """Give a trace logger its file, written at the session's level, with its held lines first."""
    trace_logger.addHandler(_LinesBeforeAWarning(file_handler, held_lines))
    trace_logger.addHandler(file_handler)
    with _levels_lock:
        _trace_files[trace_logger] = file_handler
    apply_session_level(_session_level)


def write_window(
    window: LogWindow,
    file_handler: logging.Handler,
    *,
    trace_loggers: tuple[logging.Logger, ...],
    budget_bytes: int,
) -> None:
    """Give a window its file, written from the trace loggers while the window is open.

    One handler on every trace logger, which do not propagate, so each
    record is written once, in the order it was logged.
    """
    keeper = _Window(window, file_handler, budget_bytes)
    for trace_logger in trace_loggers:
        trace_logger.addHandler(keeper)
    with _levels_lock:
        _windows[window] = keeper


def open_window(window: LogWindow, opened_by: str) -> None:
    """Write the trace loggers' lines to the window's file from here; nothing when it has none."""
    keeper = _windows.get(window)
    if keeper is not None:
        keeper.open(opened_by)


def end_window(window: LogWindow) -> bool:
    """Write no more to the window's file; whether it was open."""
    keeper = _windows.get(window)
    return keeper is not None and keeper.end()


def apply_session_level(level: int | str) -> None:
    """Put the session's log level on every log that follows it.

    The 'LVP' parent (every LVP.* logger without a level of its own: the
    GUI interaction and protocol logs among them) and lvp_logger's own
    logger take it, as do the trace logs' files. The trace loggers pass
    VERBOSE whatever the level, so a file at INFO still has its held lines
    and an open window (``LogWindow``) has every line.
    """
    global _session_level
    level = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    with _levels_lock:
        _session_level = level
        logging.getLogger('LVP').setLevel(level)
        logger.setLevel(level)
        for trace_logger, file_handler in _trace_files.items():
            trace_logger.setLevel(min(level, VERBOSE))
            file_handler.setLevel(level)


def begin_bring_up() -> None:
    """A session is being brought up: its trace traffic is kept in ``bringup.log`` until it ends."""
    open_window(LogWindow.BRING_UP, 'a session is being brought up')


def end_bring_up() -> None:
    """Bring-up has ended: ``bringup.log`` takes no more of it."""
    if end_window(LogWindow.BRING_UP):
        logger.info(
            f'[Logger    ] Bring-up has ended: its serial, camera and api lines are in '
            f'{LogWindow.BRING_UP.value}'
        )
