"""Regression: camera-init failure popups speak researcher voice.

A camera-not-initialized popup shipped the raw
`ValueError: <kind> registry has no real drivers and no null fallback.`
text -- exception class name, registry / null-fallback internals, and a
doubled period (the exception message already ends in '.') -- straight to
the user. The exception detail belongs to the log, behind the typed
outcome as its cause; the body the person reads must not duplicate it.

Exercises the production classification of a camera's connect failure and
the words its typed outcome carries.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.exceptions import CameraNotAvailableError
from modules.lumascope_api._lumascope import _camera_failure_cause


def _body(exc) -> str:
    return str(CameraNotAvailableError(_camera_failure_cause(exc)))


class TestCameraNotifyVoice:
    def test_registry_valueerror_body_is_clean(self):
        exc = ValueError('camera registry has no real drivers and no null fallback.')
        body = _body(exc)
        assert body  # non-empty, actionable
        assert 'ValueError' not in body
        assert 'null fallback' not in body
        assert 'registry' not in body
        assert '..' not in body  # no doubled period

    def test_camera_in_use_body_is_clean(self):
        # pypylon RuntimeException is matched by type-name string.
        exc = type('RuntimeException', (Exception,), {})('device busy 0xdead')
        body = _body(exc)
        assert 'RuntimeException' not in body
        assert '0xdead' not in body

    def test_permission_error_body_is_clean(self):
        body = _body(PermissionError('COM7 access denied by winerror 5'))
        assert 'winerror' not in body
        assert 'COM7 access denied' not in body

    def test_file_not_found_body_is_clean(self):
        body = _body(FileNotFoundError('/dev/video0 missing'))
        assert '/dev/video0' not in body
