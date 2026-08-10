# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Per-(well, scan) video hyperstack grouping -- the customer defect.

A protocol video step drains hundreds of per-frame TIFFs whose
execution-record rows all share the recording's Scan Count. Grouped like
stills they form one non-rectangular group that _create_stack refuses --
so a video-step protocol yielded ZERO hyperstacks (and a multi-scan run
would have interleaved recordings if it hadn't). Video rows now group per
(well, scan): one OME-TIFF per recording, T = frame order within the
recording, C = channel, Z where present. Stills keep their cross-scan
grouping (T = scan), and a well holding both stills and video steps
yields both artifact families.
"""

import pathlib
import re

import numpy as np
import pandas as pd
import tifffile as tf

from modules import image_utils
from modules import recording_frames
from modules.common_utils import PostFunction
from modules.stack_builder import StackBuilder


def _frame_path(well, color, scan, n):
    folder = f'{color}/{well}_{color}_{scan:04}_video'
    return f'{folder}/{well}_{color}_{scan:04}_video_Frame_{n:04}.tiff'


def _stack_df(rows):
    """Post-processing df shaped like the helper's load, one row per file.

    rows: (filepath, scan, color, z_slice) tuples. Carries every column
    the hyperstack grouping, filename generation, and grid build read.
    """
    data = {
        'Filepath': [],
        'Scan Count': [],
        'Color': [],
        'Z-Slice': [],
    }
    for path, scan, color, z_slice in rows:
        data['Filepath'].append(pathlib.Path(path))
        data['Scan Count'].append(scan)
        data['Color'].append(color)
        data['Z-Slice'].append(z_slice)
    n = len(rows)
    data.update(
        {
            'Well': ['A1'] * n,
            'Name': [''] * n,
            'Label': [''] * n,
            'Objective': [''] * n,
            'X': [1.0] * n,
            'Y': [2.0] * n,
            'Z': [3.0] * n,
            'Tile': [''] * n,
            'Tile Group ID': [''] * n,
            'Custom Step': [False] * n,
            'Timestamp': [''] * n,
            'Raw': [True] * n,
        }
    )
    for column in PostFunction.list_values():
        data[column] = [False] * n
    return pd.DataFrame(data)


def _write_frames(tmp_path, df):
    """Write a real mono TIFF for every df row, valued by frame number so
    T-order is observable in the output."""
    for _, row in df.iterrows():
        loc = tmp_path / row['Filepath']
        loc.parent.mkdir(parents=True, exist_ok=True)
        try:
            value = recording_frames.frame_number(loc.name)
        except ValueError:
            value = 7
        tf.imwrite(loc, np.full((4, 4), value, dtype=np.uint8))


class TestVideoWellGrouping:
    def test_video_well_groups_per_scan(self):
        # Two recordings of one video step share every positional column;
        # each must form its own group (one OME-TIFF per (well, scan)).
        rows = []
        for scan in (0, 1):
            for n in range(3):
                rows.append((_frame_path('A1', 'BF', scan, n), scan, 'BF', 0))
        df = _stack_df(rows)

        groups = StackBuilder._get_groups(df)

        assert len(groups) == 2
        for _, group in groups:
            assert group['Scan Count'].nunique() == 1
            assert len(group) == 3

    def test_mixed_well_yields_both_artifact_families(self):
        # A well with BOTH stills and video steps: the stills keep their one
        # cross-scan group (their T axis IS the scan axis), and the video
        # frames form their own per-scan group beside it.
        rows = [(f'BF/A1_BF_{scan:04}.tiff', scan, 'BF', 0) for scan in (0, 1, 2)]
        rows += [(_frame_path('A1', 'BF', 1, n), 1, 'BF', 0) for n in range(3)]
        df = _stack_df(rows)

        groups = StackBuilder._get_groups(df)

        assert len(groups) == 2
        stills_groups = [
            group
            for _, group in groups
            if not recording_frames.is_video_frame(group.iloc[0]['Filepath'])
        ]
        assert len(stills_groups) == 1, 'expected one stills group beside the video group'
        assert stills_groups[0]['Scan Count'].nunique() == 3, (
            'the stills group must keep spanning scans'
        )

    def test_video_group_filenames_carry_the_scan_token(self):
        rows = []
        for scan in (0, 1):
            for n in range(3):
                rows.append((_frame_path('A1', 'BF', scan, n), scan, 'BF', 0))
        df = _stack_df(rows)

        builder = StackBuilder(has_turret=False)
        names = {
            builder._generate_filename(df=group.reset_index(drop=True), capture_root=None)
            for _, group in StackBuilder._get_groups(df)
        }

        assert len(names) == 2, f'each (well, scan) stack needs its own filename, got {names}'
        assert all('0000' in name or '0001' in name for name in names)


class TestVideoWellStackBuild:
    def test_video_well_builds_a_stack_with_frame_t_axis(self, tmp_path):
        # The customer's literal report: a video well produced zero stacks.
        # All frame rows share the recording's Scan Count; the build must
        # remap T to the frame ordinal and produce one plane per frame, in
        # frame order.
        rows = [(_frame_path('A1', 'BF', 0, n), 0, 'BF', 0) for n in range(3)]
        df = _stack_df(rows)
        _write_frames(tmp_path, df)

        builder = StackBuilder(has_turret=False)
        ((_, group),) = StackBuilder._get_groups(df)
        result = builder._group_algorithm(
            path=tmp_path,
            df=group.reset_index(drop=True),
            output_file_loc=pathlib.Path('out.ome.tiff'),
        )

        assert result.status, f'video-well stack build failed: {result.error}'
        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            data = tif.asarray().reshape(3, 4, 4)
        assert [int(data[t, 0, 0]) for t in range(3)] == [0, 1, 2], (
            'T axis must follow frame order within the recording'
        )

    def test_video_stack_planes_carry_delta_t_from_frame_timestamps(self, tmp_path):
        # PR-6 timing honesty: the recording's real per-frame capture
        # times land as per-plane DeltaT (seconds from the earliest
        # plane), so a variable-rate recording reads back with true
        # timing instead of an implied constant rate. Frames written via
        # the production frame writer carry the timestamps.
        base_s = 1_755_000_000.0
        rows = []
        for n in range(3):
            path = _frame_path('A1', 'BF', 0, n)
            rows.append((path, 0, 'BF', 0))
            loc = tmp_path / path
            loc.parent.mkdir(parents=True, exist_ok=True)
            metadata, _ = recording_frames.tiff_frame_metadata(
                timestamp_s=base_s + n * 0.5,
                frame_number=n,
                chunks=None,
                tick_freq_hz=None,
            )
            image_utils.write_tiff(
                data=np.full((4, 4), n, dtype=np.uint8),
                file_loc=loc,
                metadata=metadata,
                ome=False,
                color='BF',
                significant_bits=8,
                save_encoding='8bit',
                video_frame=True,
            )
        df = _stack_df(rows)

        builder = StackBuilder(has_turret=False)
        ((_, group),) = StackBuilder._get_groups(df)
        result = builder._group_algorithm(
            path=tmp_path,
            df=group.reset_index(drop=True),
            output_file_loc=pathlib.Path('out.ome.tiff'),
        )
        assert result.status, f'stack build failed: {result.error}'

        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            ome_xml = tif.ome_metadata or ''
        deltas = sorted(float(m) for m in re.findall(r'DeltaT="([\d.]+)"', ome_xml))
        assert deltas == [0.0, 0.5, 1.0], (
            f'per-plane DeltaT must carry the real frame offsets, got {deltas}'
        )

    def test_stack_without_timestamps_omits_delta_t(self, tmp_path):
        # All-or-nothing: a stack whose inputs carry no readable capture
        # time makes no timing claim at all -- a partial or invented
        # DeltaT list would read downstream as a measurement.
        rows = [(_frame_path('A1', 'BF', 0, n), 0, 'BF', 0) for n in range(3)]
        df = _stack_df(rows)
        _write_frames(tmp_path, df)

        builder = StackBuilder(has_turret=False)
        ((_, group),) = StackBuilder._get_groups(df)
        result = builder._group_algorithm(
            path=tmp_path,
            df=group.reset_index(drop=True),
            output_file_loc=pathlib.Path('out.ome.tiff'),
        )
        assert result.status, f'stack build failed: {result.error}'

        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            ome_xml = tif.ome_metadata or ''
        assert 'DeltaT' not in ome_xml

    def test_unequal_channel_frame_counts_refused(self, tmp_path):
        # Behavior-preservation guard (refuses before and after): two
        # channels of one (well, scan) with unequal frame counts cannot
        # form a rectangular T x Z x C grid; refuse loudly, never pad.
        rows = [(_frame_path('A1', 'Green', 0, n), 0, 'Green', 0) for n in range(3)]
        rows += [(_frame_path('A1', 'Red', 0, n), 0, 'Red', 0) for n in range(2)]
        df = _stack_df(rows)
        _write_frames(tmp_path, df)

        builder = StackBuilder(has_turret=False)
        ((_, group),) = StackBuilder._get_groups(df)
        result = builder._group_algorithm(
            path=tmp_path,
            df=group.reset_index(drop=True),
            output_file_loc=pathlib.Path('out.ome.tiff'),
        )

        assert not result.status
        assert 'A1' in str(result.error)
