# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Centralized user-facing notification system.

Any thread can post a notification; UI subscribes and shows popups on the
main thread.  Replaces scattered ``show_notification_popup()`` calls with
a single bus that handles thread safety, deduplication, and severity
filtering.

Usage::

    from modules.notification_center import notifications

    # Producer (any thread):
    notifications.error("Motor", "Connection Lost", "Serial timeout on HOME")

    # Consumer (UI init, once):
    notifications.add_listener(my_callback, min_severity=Severity.WARNING)
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from enum import IntEnum

from drivers.exceptions import HardwareError
from lib import profile_trace
from modules.exceptions import (
    CaptureError,
    ConfigError,
    MoveNotCompletedError,
    ProtocolError,
    Quiet,
    Refusal,
)

logger = logging.getLogger('LVP.notifications')
# The reporter's own record -- what happened, with the traceback when it
# is a fault -- kept apart from the display line notify() writes, so the
# two are never read as one event twice.
_outcome_logger = logging.getLogger('LVP.outcomes')

# Faults whose message is written for the person. Any other exception's
# str() is a developer's words -- a Python class name, a repr -- so the
# person reads a generic sentence and the log carries the rest.
_TYPED_FAULTS = (CaptureError, ProtocolError, ConfigError, HardwareError, MoveNotCompletedError)

_UNTYPED_FAULT_BODY = 'The operation did not complete. Check the main log for details.'

# Set on an exception object once each half of its report is done, so the
# same object reported again -- by the lane that raised it and then by the
# caller that waited on it -- is logged once and shown at most once.
_LOGGED_MARK = '_lvp_outcome_logged'
_SHOWN_MARK = '_lvp_outcome_shown'


class Severity(IntEnum):
    """Notification severity levels (matches Python logging levels).

    NOTICE sits between INFO and WARNING: user-facing status that must
    reach the popup bridge (start/done of a long unattended operation)
    without misdeclaring itself as a fault. WARNING stays "something
    didn't work"; INFO stays log-only for normal users.
    """

    DEBUG = logging.DEBUG  # 10
    INFO = logging.INFO  # 20
    NOTICE = 25
    WARNING = logging.WARNING  # 30
    ERROR = logging.ERROR  # 40
    CRITICAL = logging.CRITICAL  # 50


# Name the custom level so log lines read 'NOTICE', not 'Level 25'.
logging.addLevelName(int(Severity.NOTICE), 'NOTICE')


@dataclass(frozen=True)
class Notification:
    """Immutable notification payload delivered to listeners."""

    severity: Severity
    category: str  # e.g. "Motor", "Camera", "FileIO", "Protocol"
    title: str  # short summary shown in popup title
    message: str  # detail shown in popup body
    timestamp: float = field(default_factory=time.monotonic)
    source: str = ''  # optional originating module/function
    fatal: bool = False  # reaches listeners even while a protocol suppresses popups
    # True when this notification ANSWERS a request that just arrived --
    # a refusal of a button press or an API call. Both suppression rules
    # below rest on a premise it falsifies: the unattended mute assumes
    # nobody is watching, and dedup assumes "already shown recently",
    # but someone asked, and asking twice is asking twice. Distinct from
    # fatal, which is about the fault's severity: a fault that ends the
    # operation must reach a watching user, yet one fault repeating is
    # still one fault and must still dedup.
    solicited: bool = False
    # Names the operation this notification is about, when it is one of a
    # sequence describing the same piece of work -- a "starting" notice and
    # the "finished" or "failed" notice that answers it. A UI listener can
    # then replace the earlier message instead of stacking a second one on
    # top of it. Empty for the ordinary standalone notification.
    operation_key: str = ''


class NotificationCenter:
    """Thread-safe notification bus.

    Producers call ``notify()`` (or convenience methods ``error()``, etc.)
    from any thread.  The call always logs via ``lvp_logger`` so file
    logging is never lost.  Registered listeners are invoked inline on the
    producer's thread -- UI listeners must wrap work in
    ``Clock.schedule_once``.

    Deduplication: notifications with the same ``(category, title)`` are
    suppressed if they arrive within ``dedup_window_s`` of each other.
    The full message still goes to the log file.
    """

    def __init__(self, dedup_window_s: float = 10.0):
        self._lock = threading.Lock()
        self._listeners: list[tuple[Severity, callable]] = []
        self._dedup: dict[tuple[str, str], float] = {}
        self._dedup_window_s = dedup_window_s
        # Shutdown suppression flag. When True, notifications still
        # get LOGGED (so post-mortem diagnostics survive) but no
        # listeners are invoked -- prevents the 30+ error-notification
        # flood during close that fires when queued IO tasks fail en
        # masse after the motor/camera disconnects. Issue #622.
        self._shutting_down = False
        # Unattended-run suppression. While a run nobody is watching is in
        # flight, non-fatal notifications still LOG but raise no popup -- a
        # modal could stall the run, and transient faults would pile up in
        # front of an empty chair. Fatal notifications (lost connection, a
        # run-aborting fault) still reach listeners.
        #
        # ATTENDEDNESS, not "a run is in flight": the capture runner drives
        # short interactive operations too, and one of those suppressing its
        # own failure popup is exactly the bug this name now prevents. The
        # runner decides which kind it is and says so; this flag only obeys.
        self._unattended_run = False

    def set_shutting_down(self, value: bool = True) -> None:
        """Toggle suppression of listener dispatch. Call from on_stop
        BEFORE disconnecting hardware so teardown-induced task failures
        don't spam popups/toasts on their way out. Logs still capture
        everything."""
        with self._lock:
            self._shutting_down = bool(value)

    def set_unattended_run(self, value: bool = True) -> None:
        """Toggle suppression of NON-FATAL listener dispatch for a run nobody
        is watching. Fatal notifications still reach listeners; logs always
        capture everything. Pair with the run's start + every cleanup path so
        the flag cannot stick on and mute popups after the run ends.

        The caller passes attendedness, not "am I busy": an interactive
        operation that routes through the same runner must pass False, or it
        silences its own failure popup."""
        with self._lock:
            self._unattended_run = bool(value)

    # ------------------------------------------------------------------
    # Producer API (any thread)
    # ------------------------------------------------------------------

    def notify(
        self,
        severity: Severity,
        category: str,
        title: str,
        message: str,
        source: str = '',
        fatal: bool = False,
        operation_key: str = '',
        solicited: bool = False,
    ) -> None:
        """Post a notification.  Thread-safe.  Always logs.

        ``fatal`` notifications reach listeners even while a protocol
        suppresses non-fatal popups (set via ``set_unattended_run``).

        ``solicited`` notifications answer a request that just arrived, so
        neither suppression rule applies to them: the caller is present by
        construction, and a repeated request is a repeated question. Set it
        at the funnel that knows the notification is an answer, never at an
        emitter that cannot tell who asked.

        ``operation_key`` marks this as one of a sequence about a single piece
        of work, so a UI listener can replace the earlier message rather than
        stack on it.
        """
        # Always log at the matching level. Collapsed to one physical
        # line: message prose may span paragraphs, and raw continuation
        # lines carry no level/timestamp prefix.
        from modules import gui_logger

        logger.log(
            int(severity),
            f'[{category}] {gui_logger.one_line(title)}: {gui_logger.one_line(message)}',
        )

        # Forensics: every notification (independent of any UI popup
        # bridge that may suppress it post-shutdown) lands in
        # gui_interactions.log so post-mortem can see what messages
        # the user was looking at. Best-effort -- gui_logger import or
        # logging stack failures don't disrupt the notify path.
        # Failure surfaces at warning level in the main log so a
        # silently-broken forensic-log subsystem is visible during
        # post-mortem; stderr-print is intentionally NOT used because
        # frozen pyinstaller builds suppress stderr from L1 users.
        try:
            from modules import gui_logger

            gui_logger.notification(
                severity.name if hasattr(severity, 'name') else str(severity),
                f'{category}/{title}',
                message,
                source=source or '',
            )
        except Exception as e:
            logger.warning(f'notification forensic write failed: {type(e).__name__}: {e}')

        # Dedup check + shutdown suppression
        key = (category, title)
        now = time.monotonic()
        suppressed_reason = None
        with self._lock:
            if self._shutting_down:
                suppressed_reason = 'shutdown'  # logged above; suppressed during close
            elif self._unattended_run and not fatal and not solicited:
                # logged above; non-fatal popups suppressed on an unattended run
                suppressed_reason = 'unattended_run'
            else:
                last = self._dedup.get(key, 0.0)
                # The window still advances for a solicited notification, so a
                # later unsolicited repeat of the same (category, title) is
                # measured from the answer the user actually saw.
                if not solicited and (now - last) < self._dedup_window_s:
                    suppressed_reason = 'dedup'  # already shown recently
                else:
                    self._dedup[key] = now
                    listeners = list(self._listeners)
        if suppressed_reason is not None:
            # The forensic write above happens BEFORE this decision, so on its
            # own it says "posted", never "seen". Without this line a support
            # bundle cannot answer whether the user was ever shown a failure --
            # the popup is the only carrier, so a suppressed one would leave no
            # record anywhere that it happened. Unconditional, not behind the
            # profile-trace flag, because the question is asked of customer
            # logs captured long after the fact.
            logger.info(
                f'[{category}] {gui_logger.one_line(title)}: '
                f'not shown to the user (suppressed: {suppressed_reason})'
            )
            # Emitted outside the lock: the tracer takes its own module-wide
            # lock, and nesting the two would order a pair of locks for the
            # sake of a diagnostic. What the user never saw IS the
            # measurement here -- the popup is currently the only carrier for
            # these failures, so a suppressed one otherwise leaves no record
            # anywhere that it happened.
            if profile_trace.ENABLE_PROFILE_TRACE:
                profile_trace.trace(
                    'notification_suppressed_trace.csv',
                    'ts_ms,reason,severity,category,title,fatal',
                    [
                        f'{time.time() * 1000.0:.3f}',
                        suppressed_reason,
                        getattr(severity, 'name', severity),
                        category,
                        title,
                        int(bool(fatal)),
                    ],
                    recording_id=profile_trace.NO_RECORDING,
                )
            return

        n = Notification(
            severity=severity,
            category=category,
            title=title,
            message=message,
            timestamp=now,
            source=source,
            fatal=fatal,
            operation_key=operation_key,
            solicited=solicited,
        )
        for min_sev, cb in listeners:
            if severity >= min_sev:
                try:
                    cb(n)
                except Exception as ex:
                    logger.debug(f'notification listener error: {ex}')

    def report_outcome(
        self,
        exception: BaseException,
        *,
        solicited: bool,
        category: str,
        log_only: bool = False,
        fault_title: str = 'Operation failed',
    ) -> None:
        """Log an outcome once and show it at most once, as its type says.

        The one place an exception that ended its flight becomes a log record
        and a notification. What it is -- a refusal (``Refusal``), a quiet
        outcome (``Quiet``, or a by-contract cancel) or a fault (anything
        else) -- and the words, title and level all come from the exception's
        type; the caller says only whether a person just asked (``solicited``),
        which ``category`` it belongs to, and, with ``log_only``, that no one
        is to be shown it.

        A fault is logged at ERROR with its traceback; a quiet outcome at INFO;
        a refusal that is not shown at WARNING, with no traceback. A shown
        outcome's display line is ``notify()``'s own, so a shown refusal is one
        WARNING line and a shown fault is its traceback line and that one. A
        refusal is shown as a warning under its ``title``; a fault as an error,
        in its own words when its type writes them for a person and in a
        generic sentence when it does not, under its ``title`` or
        ``fault_title``. A quiet outcome is never shown.

        Each half happens once per exception object, whoever reports it and
        from whichever thread.
        """
        refusal = isinstance(exception, Refusal)
        quiet = isinstance(exception, (Quiet, CancelledError))
        # Check-and-mark only: notify() takes this same lock, so logging and
        # notifying happen after it is released.
        with self._lock:
            do_log = not getattr(exception, _LOGGED_MARK, False)
            do_show = not log_only and not quiet and not getattr(exception, _SHOWN_MARK, False)
            if do_log:
                setattr(exception, _LOGGED_MARK, True)
            if do_show:
                setattr(exception, _SHOWN_MARK, True)

        kind = type(exception).__name__
        if do_log:
            if quiet:
                _outcome_logger.info(f'[{category}] {kind}: {exception}')
            elif refusal:
                if not do_show:
                    reason = getattr(exception, 'reason', None)
                    because = f', {reason}' if reason else ''
                    _outcome_logger.warning(f'[{category}] refused ({kind}{because}): {exception}')
            else:
                _outcome_logger.error(
                    f'[{category}] raised {kind}: {exception}', exc_info=exception
                )
        if not do_show:
            return
        if refusal:
            self.warning(
                category,
                exception.title,
                str(exception),
                solicited=solicited,
                operation_key=REFUSAL_OPERATION_KEY,
            )
            return
        body = (
            str(exception)
            if isinstance(exception, _TYPED_FAULTS) and str(exception)
            else _UNTYPED_FAULT_BODY
        )
        title = getattr(exception, 'title', None) or fault_title
        self.error(category, title, body, solicited=solicited)

    # Convenience methods
    def debug(self, category: str, title: str, message: str, **kw) -> None:
        self.notify(Severity.DEBUG, category, title, message, **kw)

    def info(self, category: str, title: str, message: str, **kw) -> None:
        self.notify(Severity.INFO, category, title, message, **kw)

    def notice(self, category: str, title: str, message: str, **kw) -> None:
        self.notify(Severity.NOTICE, category, title, message, **kw)

    def warning(self, category: str, title: str, message: str, **kw) -> None:
        self.notify(Severity.WARNING, category, title, message, **kw)

    def error(self, category: str, title: str, message: str, **kw) -> None:
        self.notify(Severity.ERROR, category, title, message, **kw)

    def critical(self, category: str, title: str, message: str, **kw) -> None:
        # App-level failures are fatal: they reach listeners even while a
        # protocol suppresses non-fatal popups, unless a caller overrides.
        kw.setdefault('fatal', True)
        self.notify(Severity.CRITICAL, category, title, message, **kw)

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def add_listener(self, callback, min_severity: Severity = Severity.WARNING) -> None:
        """Register a listener.  Called on the producer's thread."""
        with self._lock:
            self._listeners.append((min_severity, callback))

    def remove_listener(self, callback) -> None:
        """Unregister a listener."""
        with self._lock:
            self._listeners = [(s, cb) for s, cb in self._listeners if cb is not callback]

    # ------------------------------------------------------------------
    # Testing / introspection
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all state (for testing)."""
        with self._lock:
            self._listeners.clear()
            self._dedup.clear()


# The operation this key names is "the answer to the user's last refused
# request". One key for every refusal, deliberately: the newest refusal is
# the true answer, so the bridge's supersession replaces the dialog on a
# second press instead of stacking one per press -- which is what bounds the
# dialogs now that a solicited notification no longer dedups.
REFUSAL_OPERATION_KEY = 'run_refusal'


# Module-level singleton -- import this in producers and consumers.
notifications = NotificationCenter()
