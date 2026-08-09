# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Frame-ordering regression tests for VideoBuilder's video_Frame sort.

The producer names frames ``{name}_Frame_{frame_num:04}`` -- four-digit
zero padding that grows to five digits at frame 10,000, so any
fixed-width or lexical ordering wraps there and scrambles the output
video. Both consumer legs must parse the frame number and sort
numerically: the protocol leg's _create_video sort, and the manual leg,
whose frame order comes from the directory listing.
"""

import pathlib
import random
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.video_builder import VideoBuilder


def _frame_name(n):
    # Mirrors the producer's f'{name}_Frame_{frame_num:04}' plus the
    # TIFF suffix the writer appends.
    return f'Well1_video_Frame_{n:04}.tiff'


def _consumed_order(tmp_path, frame_numbers):
    """Run the real _create_video sort; return frame numbers in the
    order the builder consumed them (writer and file reads stubbed)."""
    shuffled = list(frame_numbers)
    random.Random(42).shuffle(shuffled)
    df = pd.DataFrame({'Filepath': [_frame_name(n) for n in shuffled]})

    consumed = []

    def _record_frame(self, writer, image_path, **kwargs):
        consumed.append(int(image_path.name.rsplit('_', 1)[1].split('.')[0]))
        return True

    writer = MagicMock()
    writer.dropped_frames = 0
    with (
        patch.object(VideoBuilder, '_add_source_frame', _record_frame),
        patch('modules.video_builder.VideoWriter', return_value=writer),
    ):
        builder = VideoBuilder(has_turret=False)
        builder._create_video(
            path=tmp_path,
            df=df,
            frames_per_sec=10,
            enable_timestamp_overlay=False,
            output_file_loc=pathlib.Path('out.mp4'),
        )
    return consumed


def test_sub_10k_frame_ordering_is_numeric(tmp_path):
    frames = list(range(0, 200)) + list(range(9900, 10000))
    consumed = _consumed_order(tmp_path, frames)
    assert consumed == sorted(frames)


def test_frame_ordering_survives_10k_rollover(tmp_path):
    frames = list(range(9990, 10020))
    consumed = _consumed_order(tmp_path, frames)
    assert consumed == sorted(frames)


def test_unparseable_frame_name_fails_loudly(tmp_path):
    # A frame name with no numeric index cannot be ordered; a guessed key
    # would scramble the video silently, so the build must raise instead.
    df = pd.DataFrame({'Filepath': ['Well1_video_Frame_.tiff']})
    builder = VideoBuilder(has_turret=False)
    with pytest.raises(ValueError, match='frame number'):
        builder._create_video(
            path=tmp_path,
            df=df,
            frames_per_sec=10,
            enable_timestamp_overlay=False,
            output_file_loc=pathlib.Path('out.mp4'),
        )


def test_manual_leg_orders_frames_numerically_across_10k(tmp_path):
    # The manual leg's frame order comes from the directory listing, which
    # is lexical: frame 10000 sorts between 0999 and 1000. The builder must
    # re-sort numerically before assigning consumption order.
    frames = list(range(9990, 10020))
    for n in frames:
        (tmp_path / f'ManualVideo_Frame_{n:04}_2026-08-08_12-00-00-000.tiff').touch()

    captured = {}

    def _capture_df(self, path, df, **kwargs):
        captured['order'] = [int(name.split('_')[2]) for name in df['Filepath']]
        return {'status': True}

    with patch.object(VideoBuilder, '_create_video', _capture_df):
        builder = VideoBuilder(has_turret=False)
        result = builder._build_manual_recording_video(tmp_path, frames_per_sec=10)

    assert result['status'] is True
    assert captured['order'] == sorted(frames)


def _protocol_df(rows):
    """Post-processing df shaped like the helper's load, one row per file.

    rows: (filepath, scan_count) pairs. Carries every column the video
    grouping and filename generation read.
    """
    from modules.common_utils import PostFunction

    data = {
        'Filepath': [],
        'Scan Count': [],
    }
    for path, scan in rows:
        data['Filepath'].append(pathlib.Path(path))
        data['Scan Count'].append(scan)
    n = len(rows)
    data.update(
        {
            'Well': ['A1'] * n,
            'Label': [''] * n,
            'Color': ['BF'] * n,
            'Objective': [''] * n,
            'X': [1.0] * n,
            'Y': [2.0] * n,
            'Z': [3.0] * n,
            'Z-Slice': [''] * n,
            'Tile': [''] * n,
            'Custom Step': [False] * n,
            'Timestamp': [''] * n,
            'Raw': [True] * n,
        }
    )
    for column in PostFunction.list_values():
        data[column] = [False] * n
    return pd.DataFrame(data)


class TestMultiScanVideoGrouping:
    def test_each_scan_is_its_own_video_group(self):
        # Two scans of one video step share every positional column and
        # re-use frame numbers 0..N -- grouped together their sort keys
        # collide and the output interleaves both scans scrambled. Each
        # recording must build its own video.
        rows = []
        for scan in (0, 1):
            folder = f'BF/A1_BF_{scan:04}_video'
            for n in range(3):
                rows.append((f'{folder}/A1_BF_{scan:04}_video_Frame_{n:04}.tiff', scan))
        df = _protocol_df(rows)

        groups = VideoBuilder._get_groups(df)

        assert len(groups) == 2
        for _, group in groups:
            assert group['Scan Count'].nunique() == 1
            assert len(group) == 3

    def test_video_group_filenames_carry_the_scan_token(self):
        rows = []
        for scan in (0, 1):
            folder = f'BF/A1_BF_{scan:04}_video'
            for n in range(3):
                rows.append((f'{folder}/A1_BF_{scan:04}_video_Frame_{n:04}.tiff', scan))
        df = _protocol_df(rows)

        builder = VideoBuilder(has_turret=False)
        names = {
            builder._generate_filename(df=group.reset_index(drop=True), capture_root=None)
            for _, group in VideoBuilder._get_groups(df)
        }

        # One name per recording, mirroring the on-disk folder names.
        assert names == {'A1_BF_0000_video.mp4', 'A1_BF_0001_video.mp4'}

    def test_stills_timelapse_still_groups_across_scans(self):
        # Behavior-preservation guard: passes before and after the
        # grouping change. A still capture's time axis IS the scan axis,
        # so scans share one group and one output video.
        rows = [(f'BF/A1_BF_{scan:04}.tiff', scan) for scan in (0, 1, 2)]
        df = _protocol_df(rows)

        groups = VideoBuilder._get_groups(df)

        assert len(groups) == 1
        builder = VideoBuilder(has_turret=False)
        ((_, group),) = groups
        name = builder._generate_filename(df=group.reset_index(drop=True), capture_root=None)
        assert name == 'A1_BF_video.mp4'
