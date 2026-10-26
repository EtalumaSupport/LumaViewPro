# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Create Video's 'auto' rate reads the recording's measured rate.

Encoding an existing frames folder used to run at whatever number sat
in the FPS box (guess 5 on failure) while the manifest's measured
capture rate sat unread beside the frames. 'auto' (frame_rate_fps=None)
now resolves at the build_from_folder dispatch: a manual recording
plays at its manifest's measured rate -- engine manifests first, the
legacy session manifest for pre-engine folders -- and folders with no
measured rate build at the named default. An explicit rate still
overrides everything (deliberate timelapse speedups stay possible).
"""

import json
import pathlib
from unittest.mock import patch

import pytest

from modules.video_builder import DEFAULT_BUILD_FPS, VideoBuilder
from modules.layer_record import release_colors
from tests.stock_channels import LED0, LED1, LED2


def _make_manual_folder(tmp_path, n=3):
    for i in range(n):
        (tmp_path / f'ManualVideo_Frame_{i:04}_2026-08-08_12-00-00-000.tiff').touch()
    return tmp_path


def _built_rate(tmp_path, frame_rate_fps):
    captured = {}

    def _capture(self, path, df, frame_rate_fps, **kwargs):
        captured['fps'] = frame_rate_fps
        return {'status': True, 'actual_output_file_loc': path / f'{path.name}.mp4'}

    with patch.object(VideoBuilder, '_create_video', _capture):
        builder = VideoBuilder(has_turret=False)
        result = builder.build_from_folder(
            tmp_path,
            pathlib.Path('unused_tiling.json'),
            frame_rate_fps=frame_rate_fps,
            enable_timestamp_overlay=False,
            channel_colors=release_colors(),
        )
    assert result['status'] is True
    return captured['fps']


def test_auto_uses_engine_manifest_measured_rate(tmp_path):
    _make_manual_folder(tmp_path)
    (tmp_path / 'recording_manifest.json').write_text(
        json.dumps({'measured_fps': 9.5, 'channel_color': LED0})
    )
    assert _built_rate(tmp_path, None) == pytest.approx(9.5)


def test_auto_uses_legacy_manifest_measured_rate(tmp_path):
    # A pre-engine folder's manifest, as an earlier release wrote it: old channel names.
    _make_manual_folder(tmp_path)
    (tmp_path / 'session_manifest.json').write_text(
        json.dumps({'recording': {'channel_color': 'Red', 'actual_fps': {'mean': 3.25}}})
    )
    assert _built_rate(tmp_path, None) == pytest.approx(3.25)


def test_auto_without_any_manifest_uses_named_default(tmp_path):
    _make_manual_folder(tmp_path)
    assert _built_rate(tmp_path, None) == DEFAULT_BUILD_FPS


def test_explicit_rate_overrides_the_manifest(tmp_path):
    _make_manual_folder(tmp_path)
    (tmp_path / 'recording_manifest.json').write_text(json.dumps({'measured_fps': 9.5}))
    assert _built_rate(tmp_path, 30) == 30


def test_engine_manifest_wins_over_legacy(tmp_path):
    _make_manual_folder(tmp_path)
    (tmp_path / 'recording_manifest.json').write_text(json.dumps({'measured_fps': 9.5}))
    (tmp_path / 'session_manifest.json').write_text(
        json.dumps({'recording': {'actual_fps': {'mean': 3.25}}})
    )
    assert _built_rate(tmp_path, None) == pytest.approx(9.5)


def test_a_manifest_names_its_channel_and_whether_it_was_false_coloured(tmp_path):
    """A recording records its channel and the false-colour flag; the video
    is coloured by the channel only when the flag was on."""
    builder = VideoBuilder(has_turret=False)
    read = []
    for false_color in (True, False):
        folder = tmp_path / str(false_color)
        folder.mkdir()
        (folder / 'recording_manifest.json').write_text(
            json.dumps({'channel': LED2, 'false_color': false_color, 'measured_fps': 2.0})
        )
        read.append(builder._read_recording_manifest(folder)['channel_color'])
    assert read == [LED2, None]


def test_channel_color_read_from_either_manifest_generation(tmp_path):
    builder = VideoBuilder(has_turret=False)
    (tmp_path / 'recording_manifest.json').write_text(
        json.dumps({'channel_color': LED1, 'measured_fps': 2.0})
    )
    info = builder._read_recording_manifest(tmp_path)
    assert info == {'channel_color': LED1, 'measured_fps': 2.0}

    legacy_dir = tmp_path / 'legacy'
    legacy_dir.mkdir()
    (legacy_dir / 'session_manifest.json').write_text(
        json.dumps({'recording': {'channel_color': 'Lumi', 'actual_fps': {'mean': 0.0}}})
    )
    info = builder._read_recording_manifest(legacy_dir)
    # A zero-sample mean is not a measured rate.
    assert info == {'channel_color': 'Lumi', 'measured_fps': None}


def test_frames_without_any_manifest_warn_loudly(tmp_path, monkeypatch):
    # The manifest is the sole carrier of the recording's channel color
    # and measured rate. Silently returning None fields made a LOST
    # manifest indistinguishable from deliberate grayscale: the build
    # quietly produced a colorless video at the default rate. The reader
    # now says so in the log (log-only: a pre-manifest legacy folder
    # takes the same path legitimately, so a popup would misfire).
    # Recorded logger, not caplog: the shared lvp_logger is
    # conftest-mocked, so caplog cannot observe it.
    from unittest.mock import MagicMock

    import modules.video_builder as video_builder_module

    logger_mock = MagicMock()
    monkeypatch.setattr(video_builder_module, 'logger', logger_mock)

    builder = VideoBuilder(has_turret=False)
    info = builder._read_recording_manifest(tmp_path)

    assert info == {'channel_color': None, 'measured_fps': None}
    assert logger_mock.warning.called, (
        'a folder with no readable manifest must warn about the degraded build'
    )
    message = logger_mock.warning.call_args[0][0]
    assert 'manifest' in message.lower() and 'grayscale' in message.lower()
