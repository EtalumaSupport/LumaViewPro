# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Bring-up's completion line says whether the camera is streaming.

The line said "camera streaming" on every bring-up, so a launch with no
camera logged a camera that was not there. It reads the imaging API now.
"""

import modules.scope_session as scope_session_mod
from tests.log_capture import capture_module_log, messages
from tests.test_bring_up_is_a_record import _bring_up


def _completion_line(records):
    lines = [m for m in messages(records) if 'bring-up complete' in m]
    assert len(lines) == 1, lines
    return lines[0]


def test_a_bring_up_without_a_camera_does_not_say_it_streams(monkeypatch, tmp_path):
    records = capture_module_log(monkeypatch, scope_session_mod)
    _bring_up(monkeypatch, tmp_path, camera=FileNotFoundError('no camera'))
    assert _completion_line(records).endswith('camera not streaming')


def test_a_bring_up_with_a_camera_says_it_streams(monkeypatch, tmp_path):
    records = capture_module_log(monkeypatch, scope_session_mod)
    session = _bring_up(monkeypatch, tmp_path)
    assert session.scope.imaging.is_streaming()
    assert _completion_line(records).endswith('camera streaming')
