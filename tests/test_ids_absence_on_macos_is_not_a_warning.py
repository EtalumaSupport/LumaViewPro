# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""On macOS the IDS driver's absence is a fact of the host, not a warning.

IDS peak ships Windows and Linux builds only. Every launch on a Mac, the
simulator included, logged `IDS camera driver unavailable: No module named
'ids_peak_ipl'` at WARNING, a line that can never be otherwise there and
that buried the warnings that matter. On Windows and Linux the WARNING
stays: a frozen build missing the IDS library once made an IDS scope look
like it had no camera.
"""

import sys
from unittest.mock import MagicMock

from modules.lumascope_api import _lumascope


def _attempt(monkeypatch, platform):
    log = MagicMock()
    monkeypatch.setattr(_lumascope, 'logger', log)
    # The import fails the way it does on a host without the IDS SDK.
    monkeypatch.setitem(sys.modules, 'drivers.idscamera', None)
    return _lumascope._register_ids_camera(platform), log


def test_on_macos_the_absence_is_info_and_the_driver_is_not_attempted(monkeypatch):
    camera, log = _attempt(monkeypatch, 'darwin')
    assert camera is None
    log.warning.assert_not_called()
    (message,), _ = log.info.call_args
    assert 'not supported on macOS' in message


def test_on_windows_and_linux_a_missing_driver_is_still_a_warning(monkeypatch):
    for platform in ('win32', 'linux'):
        camera, log = _attempt(monkeypatch, platform)
        assert camera is None
        (message,), _ = log.warning.call_args
        assert 'IDS camera driver unavailable' in message
