# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Z-Projection / Composite / Stitch must not ingest video-frame rows.

These tools' input filters excluded only derived-OUTPUT rows (the built
MP4s and stacks), so the raw per-frame TIFFs of a video recording flowed
into their groups: hundreds of same-Z frames would "z-project" as if
they were Z-slices, and composite groups would absorb whole recordings
per channel -- mislabeled artifacts presented as real. Video frames
belong to exactly two consumers: Create Video and the per-(well, scan)
hyperstack.
"""

import pathlib

import pandas as pd
import pytest

from modules.common_utils import PostFunction
from modules.composite_generation import CompositeGeneration
from modules.stitcher import Stitcher
from modules.zprojector import ZProjector


def _mixed_df():
    """One still row and two protocol video-frame rows, all raw."""
    paths = [
        'BF/A1_BF_0000.tiff',
        'BF/A1_BF_0000_video/A1_BF_0000_video_Frame_0000.tiff',
        'BF/A1_BF_0000_video/A1_BF_0000_video_Frame_0001.tiff',
    ]
    n = len(paths)
    data = {
        'Filepath': [pathlib.Path(p) for p in paths],
        'Raw': [True] * n,
    }
    for column in PostFunction.list_values():
        data[column] = [False] * n
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    'processor_cls', [ZProjector, CompositeGeneration, Stitcher], ids=lambda c: c.__name__
)
def test_empty_input_keeps_its_columns(processor_cls):
    # An upstream filter can empty the frame before this one runs; on an
    # empty frame a mapped predicate has no dtype, and indexing with a
    # non-bool mask degrades to column selection -- the downstream groupby
    # then dies on its own group keys.
    processor = processor_cls(has_turret=False)
    empty = _mixed_df().iloc[0:0]

    filtered = processor._filter_ignored_types(empty)

    assert list(filtered.columns) == list(empty.columns)
    assert len(filtered) == 0


@pytest.mark.parametrize(
    'processor_cls', [ZProjector, CompositeGeneration, Stitcher], ids=lambda c: c.__name__
)
def test_video_frames_filtered_out(processor_cls):
    processor = processor_cls(has_turret=False)
    filtered = processor._filter_ignored_types(_mixed_df())

    kept = [str(p) for p in filtered['Filepath']]
    assert kept == ['BF/A1_BF_0000.tiff'], (
        f'{processor_cls.__name__} must drop raw video-frame rows '
        f'(a recording is not Z-slices, channels, or tiles), kept {kept}'
    )
