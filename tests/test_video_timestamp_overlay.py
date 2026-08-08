# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The timestamp overlay is a user option, not a hardcoded encode behavior.

video.timestamp_overlay (shipped default: on) decides whether each video
frame gets the capture timestamp burned in. The choice is snapshotted
with the rest of the recording config and travels to the writer as a
required argument, so no encode path can silently decide it.
"""

import json
import pathlib
import queue
from typing import ClassVar

import pytest

from modules.video_capture import VideoCaptureResult, write_video


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_shipped_default_is_overlay_on():
    data = json.loads((REPO_ROOT / 'data' / 'settings.json').read_text())
    assert data['video']['timestamp_overlay'] is True, (
        'data/settings.json must ship video.timestamp_overlay = true: '
        'existing installs recorded with the overlay burned in, and the '
        'option must not silently change their output'
    )


class _OverlayCapturingWriter:
    captured: ClassVar[dict] = {}
    dropped_frames = 0

    def __init__(self, **kwargs):
        _OverlayCapturingWriter.captured = kwargs
        self.output_path = kwargs.get('output_path')

    def add_frame(self, **kwargs):
        pass

    def close(self):
        pass


@pytest.mark.parametrize('overlay', [True, False])
def test_write_video_hands_the_choice_to_the_writer(tmp_path, monkeypatch, overlay):
    monkeypatch.setattr('modules.video_capture.VideoWriter', _OverlayCapturingWriter)
    result = VideoCaptureResult(
        captured_frames=0,
        calculated_fps=10,
        video_images=queue.Queue(),
        duration_sec=0.1,
        dropped_frames=0,
    )
    write_video(
        result=result,
        save_folder=tmp_path,
        name='clip',
        video_as_frames=False,
        step={'Color': 'BF', 'False_Color': False},
        callbacks={},
        save_encoding='8bit',
        capture_depth=8,
        timestamp_overlay=overlay,
    )
    assert _OverlayCapturingWriter.captured['include_timestamp_overlay'] is overlay
