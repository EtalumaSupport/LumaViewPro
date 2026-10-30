# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The timestamp overlay is a user option, not a hardcoded encode behavior.

video.timestamp_overlay (shipped default: on) decides whether each video
frame gets the capture timestamp burned in. The choice is snapshotted
with the rest of the recording config and travels to the writer as a
required argument, so no encode path can silently decide it.
"""

import json
import pathlib
from typing import ClassVar

import pytest

import modules.protocol_recording as protocol_recording
from modules.layer_record import GREY, release_color
from tests.stock_channels import LED0
from tests.video_step_drives import ElapsedClock, video_step_on


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
    on_frame: ClassVar = None
    dropped_frames = 0

    def __init__(self, **kwargs):
        _OverlayCapturingWriter.captured = kwargs
        self.output_path = kwargs.get('output_path')

    def add_frame(self, **kwargs):
        _OverlayCapturingWriter.on_frame()

    def close(self):
        pass


def _video_step(scope, monkeypatch, tmp_path, *, timestamp_overlay, false_color=False):
    """Drive one MP4 protocol video step and return the config its writer was made with.

    The first frame the writer takes ends the 1 s recording.
    """
    elapsed = ElapsedClock()
    monkeypatch.setattr(_OverlayCapturingWriter, 'on_frame', elapsed.pass_duration)
    monkeypatch.setattr(protocol_recording, 'VideoWriter', _OverlayCapturingWriter)
    monkeypatch.setattr(protocol_recording, 'check_disk_space_ok', lambda *a, **k: (True, 999999))

    recorder = video_step_on(
        scope,
        tmp_path,
        step={'False_Color': false_color},
        video_as_frames=False,
        timestamp_overlay=timestamp_overlay,
        elapsed_clock=elapsed,
    )
    assert recorder.run_blocking() == protocol_recording.COMPLETED
    assert recorder.wait_until_finished(timeout=10.0), 'the finish never ended'
    return _OverlayCapturingWriter.captured


@pytest.mark.parametrize('overlay', [True, False])
def test_protocol_video_hands_the_choice_to_the_writer(tmp_path, monkeypatch, overlay, sim_scope):
    captured = _video_step(sim_scope, monkeypatch, tmp_path, timestamp_overlay=overlay)
    assert captured['include_timestamp_overlay'] is overlay


def test_grayscale_stream_encodes_grey(tmp_path, monkeypatch, sim_scope):
    """False-color off must encode true grayscale (GREY), the same
    contract the manual leg uses."""
    captured = _video_step(
        sim_scope, monkeypatch, tmp_path, timestamp_overlay=True, false_color=False
    )
    assert captured['color'] == GREY


def test_false_color_stream_carries_the_layer(tmp_path, monkeypatch, sim_scope):
    captured = _video_step(
        sim_scope, monkeypatch, tmp_path, timestamp_overlay=True, false_color=True
    )
    assert captured['color'] == release_color(LED0)


def test_the_timestamp_is_drawn_on_a_copy_of_a_read_only_frame():
    """A camera's delivered frame is read-only and shared with every other
    consumer: the timestamp is drawn on a copy, and the frame keeps its pixels."""
    import numpy as np

    from modules.image_utils import add_timestamp

    image = np.zeros((100, 200), dtype=np.uint8)
    image.flags.writeable = False

    result = add_timestamp(image, '2026-01-01 00:00:00')

    assert result.any(), 'no timestamp was drawn'
    assert not image.any(), 'the frame the timestamp was drawn from was changed'
