# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The timestamp overlay is a user option, not a hardcoded encode behavior.

video.timestamp_overlay (shipped default: on) decides whether each video
frame gets the capture timestamp burned in. The choice is snapshotted
with the rest of the recording config and travels to the writer as a
required argument, so no encode path can silently decide it.
"""

import json
import pathlib
import threading
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

import modules.protocol_recording as protocol_recording
from modules.protocol_recording import ProtocolVideoStep


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


def _video_step(monkeypatch, tmp_path, *, timestamp_overlay, false_color=False):
    """Drive one MP4 protocol video step whose camera delivers nothing.

    The writer is constructed before any frame arrives, so the config
    that reaches it is pinned even on a frame-less step.
    """
    monkeypatch.setattr(protocol_recording, 'VideoWriter', _OverlayCapturingWriter)
    monkeypatch.setattr(protocol_recording, 'check_disk_space_ok', lambda *a, **k: (True, 999999))

    scope = MagicMock()
    scope.imaging.frames_until_valid.return_value = 0
    scope.imaging.camera_active = False  # wait loop exits on its first tick
    scope.imaging.camera_identity = {
        'model': 'sim',
        'serial': '0',
        'timestamp_tick_frequency_hz': None,
    }
    scope.imaging.camera_frame_size = {'width': 64, 'height': 48}

    capture_config = MagicMock()
    capture_config.capture_depth = 8
    capture_config.save_encoding = '8bit'

    step = {
        'Video Config': {'fps': 5, 'duration': 1},
        'Color': 'Blue',
        'False_Color': false_color,
        'Auto_Gain': False,
        'Exposure': 10.0,
    }
    recorder = ProtocolVideoStep(
        scope=scope,
        step=step,
        save_folder=tmp_path,
        name='clip',
        video_as_frames=False,
        capture_config=capture_config,
        timestamp_overlay=timestamp_overlay,
        global_max_fps=0,
        autogain_settings={},
        callbacks={},
        aborted_event=threading.Event(),
        is_run_in_progress=lambda: True,
        abort_run_fatal=MagicMock(),
        abort_run_on_writer_death=MagicMock(),
        record_step_row=MagicMock(),
        record_dropped_capture=MagicMock(),
    )
    outcome = recorder.run_blocking()
    assert outcome == protocol_recording.NO_FRAMES
    return _OverlayCapturingWriter.captured


@pytest.mark.parametrize('overlay', [True, False])
def test_protocol_video_hands_the_choice_to_the_writer(tmp_path, monkeypatch, overlay):
    captured = _video_step(monkeypatch, tmp_path, timestamp_overlay=overlay)
    assert captured['include_timestamp_overlay'] is overlay


def test_grayscale_stream_encodes_as_none_not_bf(tmp_path, monkeypatch):
    """False-color off must encode true grayscale (color=None), the same
    contract the manual leg uses -- not the 'BF' gray-colormap the old
    protocol leg diverged with."""
    captured = _video_step(monkeypatch, tmp_path, timestamp_overlay=True, false_color=False)
    assert captured['color'] is None


def test_false_color_stream_carries_the_layer(tmp_path, monkeypatch):
    captured = _video_step(monkeypatch, tmp_path, timestamp_overlay=True, false_color=True)
    assert captured['color'] == 'Blue'
