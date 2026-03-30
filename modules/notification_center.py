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

logger = logging.getLogger('LVP.notifications')


class Severity(IntEnum):
    """Notification severity levels (matches Python logging levels)."""
    DEBUG = logging.DEBUG        # 10
    INFO = logging.INFO          # 20
    WARNING = logging.WARNING    # 30
    ERROR = logging.ERROR        # 40
    CRITICAL = logging.CRITICAL  # 50


@dataclass(frozen=True)
class Notification:
    """Immutable notification payload delivered to listeners."""
    severity: Severity
    category: str       # e.g. "Motor", "Camera", "FileIO", "Protocol"
    title: str          # short summary shown in popup title
    message: str        # detail shown in popup body
    timestamp: float = field(default_factory=time.monotonic)
    source: str = ""    # optional originating module/function


class NotificationCenter:
    """Thread-safe notification bus.

    Producers call ``notify()`` (or convenience methods ``error()``, etc.)
    from any thread.  The call always logs via ``lvp_logger`` so file
    logging is never lost.  Registered listeners are invoked inline on the
    producer's thread — UI listeners must wrap work in
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

    # ------------------------------------------------------------------
    # Producer API (any thread)
    # ------------------------------------------------------------------

    def notify(
        self,
        severity: Severity,
        category: str,
        title: str,
        message: str,
        source: str = "",
    ) -> None:
        """Post a notification.  Thread-safe.  Always logs."""
        # Always log at the matching level
        logger.log(int(severity), f"[{category}] {title}: {message}")

        # Dedup check
        key = (category, title)
        now = time.monotonic()
        with self._lock:
            last = self._dedup.get(key, 0.0)
            if (now - last) < self._dedup_window_s:
                return  # suppressed — already shown recently
            self._dedup[key] = now
            listeners = list(self._listeners)

        n = Notification(
            severity=severity,
            category=category,
            title=title,
            message=message,
            timestamp=now,
            source=source,
        )
        for min_sev, cb in listeners:
            if severity >= min_sev:
                try:
                    cb(n)
                except Exception:
                    pass  # listener failure must not break producer

    # Convenience methods
    def debug(self, category: str, title: str, message: str, **kw) -> None:
        self.notify(Severity.DEBUG, category, title, message, **kw)

    def info(self, category: str, title: str, message: str, **kw) -> None:
        self.notify(Severity.INFO, category, title, message, **kw)

    def warning(self, category: str, title: str, message: str, **kw) -> None:
        self.notify(Severity.WARNING, category, title, message, **kw)

    def error(self, category: str, title: str, message: str, **kw) -> None:
        self.notify(Severity.ERROR, category, title, message, **kw)

    def critical(self, category: str, title: str, message: str, **kw) -> None:
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


# Module-level singleton — import this in producers and consumers.
notifications = NotificationCenter()
