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
from dataclasses import dataclass, field
from enum import IntEnum

from lib import profile_trace

logger = logging.getLogger('LVP.notifications')


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
        # Protocol-running suppression. While a protocol runs unattended,
        # non-fatal notifications still LOG but raise no popup -- no one is
        # watching, a modal could stall the run, and transient faults would
        # pile up. Fatal notifications (lost connection, a run-aborting fault)
        # still reach listeners. Set by the protocol runner; cleared on every
        # cleanup path.
        self._protocol_running = False

    def set_shutting_down(self, value: bool = True) -> None:
        """Toggle suppression of listener dispatch. Call from on_stop
        BEFORE disconnecting hardware so teardown-induced task failures
        don't spam popups/toasts on their way out. Logs still capture
        everything."""
        with self._lock:
            self._shutting_down = bool(value)

    def set_protocol_running(self, value: bool = True) -> None:
        """Toggle suppression of NON-FATAL listener dispatch while a protocol
        runs unattended. Fatal notifications still reach listeners; logs always
        capture everything. Pair with the run's start + every cleanup path so
        the flag cannot stick on and mute popups after the run ends."""
        with self._lock:
            self._protocol_running = bool(value)

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
    ) -> None:
        """Post a notification.  Thread-safe.  Always logs.

        ``fatal`` notifications reach listeners even while a protocol
        suppresses non-fatal popups (set via ``set_protocol_running``).
        """
        # Always log at the matching level
        logger.log(int(severity), f'[{category}] {title}: {message}')

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
            elif self._protocol_running and not fatal:
                # logged above; non-fatal popups suppressed mid-protocol
                suppressed_reason = 'protocol_running'
            else:
                last = self._dedup.get(key, 0.0)
                if (now - last) < self._dedup_window_s:
                    suppressed_reason = 'dedup'  # already shown recently
                else:
                    self._dedup[key] = now
                    listeners = list(self._listeners)
        if suppressed_reason is not None:
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
        )
        for min_sev, cb in listeners:
            if severity >= min_sev:
                try:
                    cb(n)
                except Exception as ex:
                    logger.debug(f'notification listener error: {ex}')

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


# Module-level singleton -- import this in producers and consumers.
notifications = NotificationCenter()
