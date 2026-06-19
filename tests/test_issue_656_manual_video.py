# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Create Video from a manually recorded ("Frames") folder.

A manual recording writes ManualVideo_Frame_<NNNN>_<ts>.tiff frames plus a
session_manifest.json, but no protocol_record.tsv. The protocol post-processing
pipeline requires that record, so "Create Video" used to fail outright. These
tests cover the record-less dispatch (build_from_folder) and the timestamp
move from burned-into-pixels to a build-time, toggle-controlled overlay sourced
from each frame's own metadata.
"""

import datetime
import json
import pathlib

import cv2
import numpy as np
import pandas as pd
import tifffile as tf

import modules.image_utils as image_utils
from modules.video_builder import VideoBuilder


def _ts(frame_num: int) -> datetime.datetime:
    return datetime.datetime(2026, 5, 30, 14, 22, 0) + datetime.timedelta(seconds=frame_num)


def _find_video(folder, stem):
    # VideoWriter emits .mp4 when PyAV is present and falls back to .avi (cv2)
    # when it is not, so the encode tests stay backend-agnostic: locate whatever
    # was written and decode with cv2, which reads both. This lets the encode
    # path run locally on macOS (cv2 fallback, no PyAV/OpenCV ffmpeg dylib
    # collision) as well as on CI where PyAV is installed.
    for ext in ('.mp4', '.avi'):
        candidate = folder / f'{stem}{ext}'
        if candidate.is_file():
            return candidate
    raise AssertionError(f'no video output for {stem}: {sorted(folder.glob(stem + ".*"))}')


def _frame_count(video_path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    return count


def _first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    assert ok, f'could not decode first frame of {video_path}'
    return frame


def _write_manual_frame(folder, frame_num, *, include_iso=True, value=20000):
    """Write one frame the way the manual recording path does."""
    ts = _ts(frame_num)
    img = np.full((64, 64), value, dtype=np.uint16)
    metadata = {
        'datetime': ts.strftime('%Y:%m:%d %H:%M:%S'),
        'timestamp': ts.strftime('%Y:%m:%d %H:%M:%S.%f'),
        'frame_num': frame_num,
    }
    if include_iso:
        metadata['timestamp_iso'] = ts.isoformat(timespec='microseconds')
    ts_filename = ts.strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
    file_loc = folder / f'ManualVideo_Frame_{frame_num:04}_{ts_filename}.tiff'
    image_utils.write_tiff(
        data=img,
        file_loc=file_loc,
        metadata=metadata,
        ome=False,
        color='BF',
        video_frame=True,
        significant_bits=16,
    )
    return file_loc, ts


def _make_manual_folder(tmp_path, n_frames=3, include_hyperstack=True, channel_color=None):
    folder = tmp_path / 'MyRecording'
    folder.mkdir(parents=True)
    for i in range(n_frames):
        _write_manual_frame(folder, i)
    if include_hyperstack:
        # A multi-page OME stack container that build_from_folder must NOT
        # treat as a frame.
        stack = np.zeros((2, 64, 64), dtype=np.uint16)
        tf.imwrite(str(folder / 'ManualVideo_Frame_HyperStack.ome.tiff'), stack)
    manifest = {'manifest_version': 1}
    if channel_color is not None:
        manifest['recording'] = {'channel_color': channel_color}
    (folder / 'session_manifest.json').write_text(json.dumps(manifest))
    return folder


# ---------------------------------------------------------------------------
# read_frame_timestamp -- recovers the per-frame time from each metadata shape
# ---------------------------------------------------------------------------


def test_read_frame_timestamp_manual_iso(tmp_path):
    file_loc, ts = _write_manual_frame(tmp_path, 0, include_iso=True)
    assert image_utils.read_frame_timestamp(file_loc) == ts


def test_read_frame_timestamp_protocol_video_no_iso(tmp_path):
    # Protocol video-step frames carry 'timestamp' / 'datetime' but no iso key.
    file_loc, ts = _write_manual_frame(tmp_path, 7, include_iso=False)
    recovered = image_utils.read_frame_timestamp(file_loc)
    # 'timestamp' is microsecond precision; the recovered value matches exactly.
    assert recovered == ts


def test_read_frame_timestamp_no_metadata_returns_none(tmp_path):
    bare = tmp_path / 'bare.tiff'
    tf.imwrite(str(bare), np.zeros((8, 8), dtype=np.uint16))
    assert image_utils.read_frame_timestamp(bare) is None


# ---------------------------------------------------------------------------
# build_from_folder -- manual recordings route to the record-less build
# ---------------------------------------------------------------------------


def test_build_from_folder_manual_creates_video_excluding_hyperstack(tmp_path):
    folder = _make_manual_folder(tmp_path, n_frames=3, include_hyperstack=True)

    builder = VideoBuilder(has_turret=False)
    result = builder.build_from_folder(
        folder,
        tmp_path / 'tiling.json',
        None,
        frames_per_sec=5,
        enable_timestamp_overlay=False,
    )

    assert result['status'] is True
    out = _find_video(folder, folder.name)
    # 3 numbered frames; the HyperStack.ome.tiff must be excluded.
    assert _frame_count(out) == 3


def test_build_from_folder_empty_returns_status_false(tmp_path):
    folder = tmp_path / 'empty'
    folder.mkdir()
    builder = VideoBuilder(has_turret=False)
    result = builder.build_from_folder(folder, tmp_path / 'tiling.json', None)
    # No manual frames -> falls through to load_folder, which reports no
    # protocol data / no images rather than raising.
    assert result['status'] is False


def test_build_from_folder_routes_protocol_to_load_folder(tmp_path, monkeypatch):
    # A folder with no manual frames must take the protocol load_folder path.
    folder = tmp_path / 'protocol_run'
    folder.mkdir()
    tf.imwrite(str(folder / 'scan_step1_Blue.tiff'), np.zeros((8, 8), dtype=np.uint16))

    builder = VideoBuilder(has_turret=False)
    assert builder._is_manual_recording_folder(folder) is False

    called = {}

    def fake_load_folder(path, tiling_configs_file_loc, popup=None, **kwargs):
        called['path'] = path
        return {'status': True, 'message': 'Success'}

    monkeypatch.setattr(builder, 'load_folder', fake_load_folder)
    builder.build_from_folder(folder, tmp_path / 'tiling.json', None)
    assert called['path'] == folder


# ---------------------------------------------------------------------------
# Timestamp toggle -- the overlay is build-time and controllable
# ---------------------------------------------------------------------------


def test_timestamp_overlay_toggle_changes_output(tmp_path):
    off_folder = _make_manual_folder(tmp_path / 'off', n_frames=2, include_hyperstack=False)
    on_folder = _make_manual_folder(tmp_path / 'on', n_frames=2, include_hyperstack=False)

    builder = VideoBuilder(has_turret=False)
    builder.build_from_folder(
        off_folder, tmp_path / 'tiling.json', None, frames_per_sec=5, enable_timestamp_overlay=False
    )
    builder.build_from_folder(
        on_folder, tmp_path / 'tiling.json', None, frames_per_sec=5, enable_timestamp_overlay=True
    )

    # The overlay draws a timestamp into the bottom-left; the frames must differ.
    off_frame = _first_frame(_find_video(off_folder, off_folder.name))
    on_frame = _first_frame(_find_video(on_folder, on_folder.name))
    assert not np.array_equal(off_frame, on_frame)


# ---------------------------------------------------------------------------
# Channel color -- a false-colored recording must render in color, not gray
# (the color is recovered from session_manifest.json; the frames are mono)
# ---------------------------------------------------------------------------


def _channel_means_bgr(frame):
    # cv2 decodes BGR; return (B, G, R) channel means.
    return frame[:, :, 0].mean(), frame[:, :, 1].mean(), frame[:, :, 2].mean()


def test_manual_video_renders_channel_color_from_manifest(tmp_path):
    folder = _make_manual_folder(
        tmp_path, n_frames=2, include_hyperstack=False, channel_color='Red'
    )
    builder = VideoBuilder(has_turret=False)
    result = builder.build_from_folder(folder, tmp_path / 'tiling.json', None, frames_per_sec=5)
    assert result['status'] is True
    b, g, r = _channel_means_bgr(_first_frame(_find_video(folder, folder.name)))
    # 'Red' false-color -> red channel dominates; NOT grayscale (the bug).
    assert r > g + 10 and r > b + 10, f'expected red-dominant frame, got B={b} G={g} R={r}'


def test_manual_video_grayscale_without_channel_color(tmp_path):
    # No channel_color in the manifest (old recording / brightfield) -> gray.
    folder = _make_manual_folder(tmp_path, n_frames=2, include_hyperstack=False, channel_color=None)
    builder = VideoBuilder(has_turret=False)
    result = builder.build_from_folder(folder, tmp_path / 'tiling.json', None, frames_per_sec=5)
    assert result['status'] is True
    b, g, r = _channel_means_bgr(_first_frame(_find_video(folder, folder.name)))
    assert abs(r - g) < 8 and abs(g - b) < 8, f'expected gray frame, got B={b} G={g} R={r}'


def test_read_manifest_channel_color(tmp_path):
    builder = VideoBuilder(has_turret=False)
    folder = tmp_path / 'rec'
    folder.mkdir()
    # No manifest -> None.
    assert builder._read_manifest_channel_color(folder) is None
    # Manifest without the key -> None.
    (folder / 'session_manifest.json').write_text(json.dumps({'manifest_version': 1}))
    assert builder._read_manifest_channel_color(folder) is None
    # Manifest with the key -> the color.
    (folder / 'session_manifest.json').write_text(
        json.dumps({'recording': {'channel_color': 'Green'}})
    )
    assert builder._read_manifest_channel_color(folder) == 'Green'


def test_build_session_manifest_includes_channel_color():
    from modules.recording_manifest import build_session_manifest

    m = build_session_manifest(
        timestamps=[],
        chunks_per_frame=[],
        tick_freq_hz=None,
        captured_frames=0,
        video_duration=0.0,
        channel_color='Blue',
    )
    assert m['recording']['channel_color'] == 'Blue'


def test_create_video_missing_timestamp_no_crash(tmp_path):
    # Frames with no recoverable timestamp + an empty df Timestamp (the value
    # the loader fills for missing data). Overlay ON must degrade to no overlay
    # rather than crash on ''.to_pydatetime().
    folder = tmp_path / 'no_ts'
    folder.mkdir()
    for i in range(2):
        tf.imwrite(str(folder / f'frame_{i:04}.tiff'), np.full((64, 64), 12000, dtype=np.uint16))

    df = pd.DataFrame(
        {
            'Filepath': [f'frame_{i:04}.tiff' for i in range(2)],
            'Scan Count': range(2),
            'Timestamp': '',
            'Color': None,
        }
    )
    builder = VideoBuilder(has_turret=False)
    result = builder._create_video(
        path=folder,
        df=df,
        frames_per_sec=5,
        enable_timestamp_overlay=True,
        output_file_loc=pathlib.Path('out.mp4'),
        popup=None,
        total_groups=1,
        current_group=1,
    )
    assert result['status'] is True
    assert _find_video(folder, 'out').is_file()
