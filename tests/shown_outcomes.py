# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the one outcome reporter shows a person, captured for a test."""

from __future__ import annotations

from modules import notification_center
from modules.notification_center import NotificationCenter, Severity


def capture_shown(monkeypatch) -> list:
    """Point the reporter at a fresh centre; return the notifications it delivers."""
    centre = NotificationCenter(dedup_window_s=10.0)
    shown = []
    centre.add_listener(shown.append, min_severity=Severity.INFO)
    monkeypatch.setattr(notification_center, 'notifications', centre)
    return shown
