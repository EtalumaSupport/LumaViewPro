"""Regression: camera-init failure popups speak researcher voice.

A camera-not-initialized popup shipped the raw
`ValueError: <kind> registry has no real drivers and no null fallback.`
text -- exception class name, registry / null-fallback internals, and a
doubled period (the exception message already ends in '.') -- straight to
the user. The exception detail is already captured by the caller's
logger.error; the popup body must not duplicate it.

Exercises the production `_notify_camera_failure` (Rule 3), capturing the
message that reaches the notification center.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.lumascope_api._lumascope import _notify_camera_failure


@pytest.fixture
def captured(monkeypatch):
    seen: dict = {}

    import modules.notification_center as nc

    def _capture(category, title, message, *args, **kwargs):
        seen['category'] = category
        seen['title'] = title
        seen['message'] = message

    monkeypatch.setattr(nc.notifications, 'warning', _capture)
    return seen


class TestCameraNotifyVoice:
    def test_registry_valueerror_body_is_clean(self, captured):
        exc = ValueError('camera registry has no real drivers and no null fallback.')
        _notify_camera_failure(exc)
        body = captured['message']
        assert body  # non-empty, actionable
        assert 'ValueError' not in body
        assert 'null fallback' not in body
        assert 'registry' not in body
        assert '..' not in body  # no doubled period

    def test_camera_in_use_body_is_clean(self, captured):
        # pypylon RuntimeException is matched by type-name string.
        exc = type('RuntimeException', (Exception,), {})('device busy 0xdead')
        _notify_camera_failure(exc)
        body = captured['message']
        assert 'RuntimeException' not in body
        assert '0xdead' not in body

    def test_permission_error_body_is_clean(self, captured):
        _notify_camera_failure(PermissionError('COM7 access denied by winerror 5'))
        body = captured['message']
        assert 'winerror' not in body
        assert 'COM7 access denied' not in body

    def test_file_not_found_body_is_clean(self, captured):
        _notify_camera_failure(FileNotFoundError('/dev/video0 missing'))
        body = captured['message']
        assert '/dev/video0' not in body
